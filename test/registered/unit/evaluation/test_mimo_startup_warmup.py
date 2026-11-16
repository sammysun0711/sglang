"""CPU checks for gated, once-per-server representative prefill warmup."""

import asyncio
import contextlib
import io
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch

from sglang.srt.entrypoints import mimo_prefill_warmup as warmup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


class FakeManager:
    def __init__(self, *, radix=False, mode="null", fail=None):
        self.server_args = SimpleNamespace(
            disable_radix_cache=not radix,
            disaggregation_mode=mode,
            tp_size=8,
            dp_size=2,
        )
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["MiMoV2ForCausalLM"])
        )
        self.context_len = 1048576
        self.max_req_input_len = self.context_len - 1
        self.requests = []
        self.flush_cache = AsyncMock(return_value=SimpleNamespace(success=True))
        self.fail = fail

    async def generate_request(self, req, request):
        self.requests.append(req)
        yield [
            {
                "meta_info": {
                    "prompt_tokens": len(ids),
                    "completion_tokens": 0 if self.fail == "incomplete" else 1,
                    "finish_reason": {"type": "length"},
                }
            }
            for ids in req.input_ids
        ]
        if self.fail == "after_yield":
            raise RuntimeError("Failure after first yield")


class StartupWarmupTest(CustomTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_flag_and_model_gate(self):
        with patch.object(warmup, "_warmup_once", new_callable=AsyncMock) as run:
            # Unset and explicit zero must return before touching model metadata.
            with patch.dict(os.environ):
                os.environ.pop("SGLANG_ENABLE_MIMO_SWA_SP", None)
                await warmup.maybe_warmup_mimo_prefill(None)
                os.environ["SGLANG_ENABLE_MIMO_SWA_SP"] = "0"
                await warmup.maybe_warmup_mimo_prefill(None)
            with patch.dict(os.environ, SGLANG_ENABLE_MIMO_SWA_SP="1"):
                manager = FakeManager(mode="decode")
                await warmup.maybe_warmup_mimo_prefill(manager)
                manager.server_args.disaggregation_mode = "null"
                manager.model_config.hf_config.architectures = ["LlamaForCausalLM"]
                await warmup.maybe_warmup_mimo_prefill(manager)
                run.assert_not_awaited()
                manager.model_config.hf_config.architectures = ["MiMoV2ForCausalLM"]
                await warmup.maybe_warmup_mimo_prefill(manager)
                run.assert_awaited_once()

    async def test_exact_schedule_and_cache_flush(self):
        manager = FakeManager(radix=True, mode="prefill")
        await warmup._run_representative_prefill(manager)
        self.assertEqual(len(manager.requests), 15)
        self.assertEqual(
            [tuple(len(ids) for ids in req.input_ids) for req in manager.requests],
            list(warmup.INPUT_BATCHES),
        )
        self.assertEqual(sum(len(req.input_ids) for req in manager.requests), 32)
        self.assertEqual(manager.flush_cache.await_count, 16)
        rooms = [room for req in manager.requests for room in req.bootstrap_room]
        self.assertEqual(len(set(rooms)), 32)

    async def test_waiters_cannot_return_before_warmup_completes(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def run(manager):
            entered.set()
            await release.wait()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state"
            with patch.object(
                warmup, "_run_representative_prefill", side_effect=run
            ) as fn:
                leader = asyncio.create_task(warmup._warmup_once(None, path))
                await entered.wait()
                followers = [
                    asyncio.create_task(warmup._warmup_once(None, path))
                    for _ in range(7)
                ]
                await asyncio.sleep(0.02)
                self.assertFalse(any(task.done() for task in [leader, *followers]))
                release.set()
                await asyncio.gather(leader, *followers)
                # HTTP worker restart in the same server generation is also a no-op.
                await warmup._warmup_once(None, path)
                fn.assert_awaited_once()
                self.assertEqual(path.read_text(), "complete")

    async def test_failed_or_interrupted_leader_blocks_readiness(self):
        for failure in ("incomplete", "after_yield"):
            with self.subTest(
                failure=failure
            ), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "state"
                with self.assertRaises(RuntimeError):
                    await warmup._warmup_once(FakeManager(fail=failure), path)
                with self.assertRaisesRegex(RuntimeError, "another worker"):
                    await warmup._warmup_once(FakeManager(), path)


class MemoryWarmupTest(CustomTestCase):
    def test_startup_logs_follow_http_worker_configuration(self):
        parent = logging.getLogger("sglang.srt.entrypoints.http_server")
        root = logging.getLogger()
        saved = parent.level, parent.handlers[:], parent.propagate, root.level
        output = io.StringIO()
        try:
            root.setLevel(logging.WARNING)
            parent.setLevel(logging.INFO)
            parent.handlers = [logging.StreamHandler(output)]
            parent.propagate = False
            warmup.logger.info("representative startup completed")
            self.assertIn("representative startup completed", output.getvalue())
        finally:
            parent.setLevel(saved[0])
            parent.handlers = saved[1]
            parent.propagate = saved[2]
            root.setLevel(saved[3])

    def test_compiles_batch_and_dtype_families_without_launching(self):
        calls, initialized = [], []

        class Compiled:
            @property
            def run(self):
                initialized.append(True)

                def launch(*args, **kwargs):
                    raise AssertionError("Memory warmup must not execute kernels")

                return launch

        class Kernel:
            def __init__(self, name):
                self.name = name

            def warmup(self, *args, **kwargs):
                calls.append((self.name, args, kwargs))
                return Compiled()

        args = SimpleNamespace(
            page_size=64,
            kv_cache_dtype="bf16",
            disaggregation_mode="null",
            speculative_algorithm="DFLASH",
        )
        config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["MiMoV2ForCausalLM"])
        )
        table = SimpleNamespace(
            device=SimpleNamespace(type="cuda"),
            shape=(9, 32768),
            stride=lambda dim: 32768,
        )
        pool = SimpleNamespace(req_to_token=table)
        modules = {
            "sglang.srt.configs.model_config": SimpleNamespace(
                MIMO_V2_MODEL_ARCHS=("MiMoV2ForCausalLM",)
            ),
            "sglang.srt.mem_cache.triton_ops.common": SimpleNamespace(
                _get_last_loc_safe_kernel=Kernel("last_loc")
            ),
            "sglang.srt.mem_cache.triton_ops.allocator": SimpleNamespace(
                alloc_extend_kernel=Kernel("alloc")
            ),
            "sglang.srt.speculative.triton_ops.cache_locs": SimpleNamespace(
                assign_req_to_token_pool=Kernel("assign")
            ),
        }
        with (
            patch.dict(os.environ, SGLANG_ENABLE_MIMO_SWA_SP="1"),
            patch.dict(sys.modules, modules),
            patch.object(torch.version, "hip", "7.2"),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(gcnArchName="gfx942"),
            ),
            patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()),
        ):
            warmup.warmup_mimo_prefill_memory_kernels(args, config, pool, 5)
        allocation = [arguments for name, arguments, _ in calls if name == "alloc"]
        assignment = [arguments for name, arguments, _ in calls if name == "assign"]
        self.assertEqual({a[-2] for a in allocation}, {1, 2, 4, 8})
        self.assertEqual({a[-1] for a in assignment}, {1, 2, 4, 8})
        self.assertEqual(
            {(a[0], a[2]) for a in allocation},
            {
                (a, b)
                for a in (torch.int32, torch.int64)
                for b in (torch.int32, torch.int64)
            },
        )
        self.assertTrue(all(a[1] is table and a[-2] == 32768 for a in assignment))
        self.assertTrue(all(options["grid"] == (1,) for _, _, options in calls))
        lookups = [(a, opts) for name, a, opts in calls if name == "last_loc"]
        self.assertEqual({a[4] for a, _ in lookups}, {1, 2})
        self.assertTrue(
            all(
                opts["BLOCK_SIZE"] == 256
                and opts["PREFIX_DTYPE_IS_I64"] == (a[2] == torch.int64)
                for a, opts in lookups
            )
        )
        self.assertEqual(len(initialized), len(calls))
        self.assertIs(pool.req_to_token, table)

    def test_memory_warmup_is_gated_before_gpu_access(self):
        with patch.dict(os.environ, SGLANG_ENABLE_MIMO_SWA_SP="0"):
            warmup.warmup_mimo_prefill_memory_kernels(None, None, None, 5)
        args = SimpleNamespace(
            disaggregation_mode="decode", page_size=64, kv_cache_dtype="bf16"
        )
        config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["MiMoV2ForCausalLM"])
        )
        with patch.dict(os.environ, SGLANG_ENABLE_MIMO_SWA_SP="1"):
            warmup.warmup_mimo_prefill_memory_kernels(args, config, None, 5)
            args.disaggregation_mode = "null"
            config.hf_config.architectures = ["LlamaForCausalLM"]
            warmup.warmup_mimo_prefill_memory_kernels(args, config, None, 5)


if __name__ == "__main__":
    unittest.main()
