"""Representative MiMo prefill requests shared by startup and evaluation."""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Balanced pairs cover the measured MiMo FP8 GEMM tile-order/reduction modes.
# Two asymmetric pairs also cover short SWA shards while the other DP group
# keeps TP SP active. Reverse them so both DP groups initialize that variant.
# Coverage is specific to the measured DP2 workload, not arbitrary topologies.
INPUT_LENGTH_PAIRS = (
    (128, 128),
    (257, 257),
    (64, 6656),
    (6656, 6656),
    (2049, 2049),
    (6656, 64),
    (4095, 4095),
    (10240, 10240),
    (6145, 6145),
    (8191, 8191),
    (12288, 12288),
    (14784, 14784),
    (15744, 15744),
    (15873, 15873),
    (16383, 16383),
    (16384, 16384),
)


def representative_batches(requests, context):
    if requests < 0 or not isinstance(context, int) or context < 2:
        raise ValueError("requests must be non-negative and context must be >= 2")
    for index in range((requests + 1) // 2):
        count = min(2, requests - index * 2)
        lengths = INPUT_LENGTH_PAIRS[index % len(INPUT_LENGTH_PAIRS)][:count]
        yield {
            "input_ids": [[100] * min(length, context - 1) for length in lengths],
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
        }


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
    for index, payload in enumerate(representative_batches(32, context)):
        # Repeated token IDs must still exercise full prefill when radix caching
        # is enabled. No user requests are admitted during this startup phase.
        if not args.disable_radix_cache:
            result = await tokenizer_manager.flush_cache()
            if not result.success:
                raise RuntimeError("Could not flush cache before MiMo warmup")
        req = GenerateReqInput(**payload)
        if args.disaggregation_mode == "prefill":
            req.bootstrap_host = [FAKE_BOOTSTRAP_HOST] * 2
            req.bootstrap_room = [
                (i % args.dp_size) * (2**63 // args.dp_size) + index * args.tp_size + i
                for i in range(2)
            ]
        outputs = None
        # Fully drain the generator: failures after a yield must fail startup.
        async for outputs in tokenizer_manager.generate_request(req, None):
            pass
        validate_warmup_response(outputs, payload)
        logger.info(
            "MiMo representative startup warmup: %d/32 requests, input lengths %s",
            (index + 1) * 2,
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
