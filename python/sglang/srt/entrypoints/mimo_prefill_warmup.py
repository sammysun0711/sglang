"""Representative MiMo prefill requests shared by startup and evaluation."""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path

# Multi-tokenizer Uvicorn workers configure the HTTP logger, not the root.
logger = logging.getLogger("sglang.srt.entrypoints.http_server.mimo_prefill")

# The first four-request batch warms non-singleton embedding masks. The
# remaining pairs cover the measured MiMo FP8 GEMM tile-order/reduction modes.
# Asymmetric pairs cover short SWA shards while the other DP group keeps SP
# active. The final pairs cross the 16K chunk boundary to exercise cached halos.
# Coverage is specific to the measured DP2 workload, not arbitrary topologies.
INPUT_BATCHES = (
    (128, 128, 128, 128),
    (257, 6656),
    (64, 6656),
    (6656, 6656),
    (2049, 2049),
    (6656, 64),
    (6656, 257),
    (10240, 10240),
    (6145, 6145),
    (8191, 8191),
    (12288, 12288),
    (15744, 15744),
    (15873, 15873),
    (16385, 20480),
    (20480, 16385),
)


def warmup_mimo_prefill_memory_kernels(args, model_config, pool, max_batch):
    """Compile allocation signatures without executing kernels or touching KV.

    The representative requests exercise model shapes, but their two-request
    batches cannot cover every allocator batch bucket used by serving traffic.
    Warm these small dtype/batch families separately in each scheduler process.
    """
    from sglang.srt.environ import envs

    if not envs.SGLANG_ENABLE_MIMO_SWA_SP.get():
        return
    from sglang.srt.configs.model_config import MIMO_V2_MODEL_ARCHS

    if (
        not any(
            a in MIMO_V2_MODEL_ARCHS
            for a in (model_config.hf_config.architectures or [])
        )
        or args.disaggregation_mode == "decode"
        or args.page_size != 64
        or args.kv_cache_dtype != "bf16"
    ):
        return
    import torch

    table = pool.req_to_token
    if torch.version.hip is None or table.device.type != "cuda":
        return
    if "gfx942" not in torch.cuda.get_device_properties(table.device).gcnArchName:
        return
    from sglang.srt.mem_cache.triton_ops.allocator import alloc_extend_kernel
    from sglang.srt.mem_cache.triton_ops.common import _get_last_loc_safe_kernel
    from sglang.srt.speculative.triton_ops.cache_locs import assign_req_to_token_pool

    count = 0
    with torch.cuda.device(table.device):
        for exponent in range((max(1, max_batch) - 1).bit_length() + 1):
            batch_upper = 1 << exponent
            for length_dtype in (torch.int32, torch.int64):
                for last_dtype in (torch.int32, torch.int64):
                    compiled = alloc_extend_kernel.warmup(
                        length_dtype,
                        length_dtype,
                        last_dtype,
                        torch.int64,
                        torch.int64,
                        batch_upper,
                        args.page_size,
                        grid=(1,),
                    )
                    # Materialize Triton's lazy launcher/module handles too.
                    # Retrieving run does not invoke it or execute a GPU kernel.
                    _ = compiled.run
                    count += 1
                if args.speculative_algorithm == "DFLASH":
                    compiled = assign_req_to_token_pool.warmup(
                        torch.int64,
                        table,
                        length_dtype,
                        length_dtype,
                        torch.int64,
                        table.shape[1],
                        batch_upper,
                        grid=(1,),
                    )
                    _ = compiled.run
                    count += 1
        # Triton specializes scalar counts equal to one and divisible by 16.
        # Cover those ABI families without invoking the lookup kernel.
        for num_tokens_hint in (1, 2, 16):
            if num_tokens_hint > max_batch:
                continue
            for length_dtype in (torch.int32, torch.int64):
                compiled = _get_last_loc_safe_kernel.warmup(
                    table,
                    torch.int64,
                    length_dtype,
                    torch.int32,
                    num_tokens_hint,
                    table.stride(0),
                    BLOCK_SIZE=256,
                    PREFIX_DTYPE_IS_I64=length_dtype == torch.int64,
                    grid=(1,),
                )
                _ = compiled.run
                count += 1
    logger.info(
        "MiMo prefill memory warmup: initialized %d signatures through batch %d",
        count,
        1 << (max(1, max_batch) - 1).bit_length(),
    )


def representative_batches(requests, context):
    if requests < 0 or not isinstance(context, int) or context < 2:
        raise ValueError("requests must be non-negative and context must be >= 2")
    completed = index = 0
    while completed < requests:
        lengths = INPUT_BATCHES[index % len(INPUT_BATCHES)][: requests - completed]
        yield {
            "input_ids": [[100] * min(length, context - 1) for length in lengths],
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
        }
        completed += len(lengths)
        index += 1


def validate_warmup_response(outputs, payload):
    inputs = payload["input_ids"]
    if not isinstance(outputs, list) or len(outputs) != len(inputs):
        raise RuntimeError("Warmup response does not match the requested batch")
    for output, tokens in zip(outputs, inputs):
        meta = output.get("meta_info", {})
        if (
            meta.get("prompt_tokens") != len(tokens)
            or meta.get("completion_tokens") != 1
            or meta.get("finish_reason", {}).get("type") == "abort"
        ):
            raise RuntimeError(
                "Warmup request did not complete with the expected token counts"
            )


async def _run_representative_prefill(tokenizer_manager):
    from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
    from sglang.srt.managers.io_struct import GenerateReqInput

    args = tokenizer_manager.server_args
    context = min(
        tokenizer_manager.context_len, tokenizer_manager.max_req_input_len + 1
    )
    started = time.monotonic()
    logger.info("MiMo representative startup warmup: starting 32 requests")
    completed = 0
    for payload in representative_batches(32, context):
        # Repeated token IDs must still exercise full prefill when radix caching
        # is enabled. No user requests are admitted during this startup phase.
        if not args.disable_radix_cache:
            result = await tokenizer_manager.flush_cache()
            if not result.success:
                raise RuntimeError("Could not flush cache before MiMo warmup")
        req = GenerateReqInput(**payload)
        if args.disaggregation_mode == "prefill":
            req.bootstrap_host = [FAKE_BOOTSTRAP_HOST] * len(payload["input_ids"])
            req.bootstrap_room = [
                (i % args.dp_size) * (2**63 // args.dp_size)
                + (completed + i) * args.tp_size
                + (i % args.tp_size)
                for i in range(len(payload["input_ids"]))
            ]
        outputs = None
        # Fully drain the generator: failures after a yield must fail startup.
        async for outputs in tokenizer_manager.generate_request(req, None):
            pass
        validate_warmup_response(outputs, payload)
        completed += len(payload["input_ids"])
        logger.info(
            "MiMo representative startup warmup: %d/32 requests, input lengths %s",
            completed,
            [len(ids) for ids in payload["input_ids"]],
        )
    result = await tokenizer_manager.flush_cache()
    if not result.success:
        raise RuntimeError("Could not flush cache after MiMo warmup")
    logger.info(
        "MiMo representative startup warmup: complete 32/32 requests in %.3fs",
        time.monotonic() - started,
    )


async def _warmup_once(tokenizer_manager, state_path):
    import fcntl

    # The lock and result belong to one server generation. All tokenizer
    # workers wait here before yielding their HTTP lifespan. A failed or killed
    # leader leaves a non-complete result, so followers cannot declare readiness.
    with state_path.open("a+") as state:
        while True:
            try:
                fcntl.flock(state, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.1)
        try:
            state.seek(0)
            status = state.read()
            if status == "complete":
                return
            if status:
                raise RuntimeError("MiMo startup warmup failed in another worker")
            state.write("running")
            state.flush()
            await _run_representative_prefill(tokenizer_manager)
            state.seek(0)
            state.truncate()
            state.write("complete")
            state.flush()
        finally:
            fcntl.flock(state, fcntl.LOCK_UN)


async def maybe_warmup_mimo_prefill(tokenizer_manager, *, main_pid=None):
    from sglang.srt.environ import envs

    if not envs.SGLANG_ENABLE_MIMO_SWA_SP.get():
        return
    from sglang.srt.configs.model_config import MIMO_V2_MODEL_ARCHS

    args = tokenizer_manager.server_args
    architectures = tokenizer_manager.model_config.hf_config.architectures or []
    if (
        not any(arch in MIMO_V2_MODEL_ARCHS for arch in architectures)
        or args.disaggregation_mode == "decode"
    ):
        return
    # PID start time distinguishes fresh launches even if the OS reuses a PID.
    # Keep the tiny result file in /tmp so a restarted HTTP worker in the same
    # server generation also observes completion without rerunning requests.
    import psutil

    pid = main_pid if main_pid is not None else os.getpid()
    generation = f"{pid}-{psutil.Process(pid).create_time():.6f}"
    path = Path(tempfile.gettempdir()) / f"sglang-mimo-prefill-{generation}.state"
    timeout = envs.SGLANG_WARMUP_TIMEOUT.get()
    await asyncio.wait_for(
        _warmup_once(tokenizer_manager, path), timeout=timeout if timeout > 0 else 1800
    )
