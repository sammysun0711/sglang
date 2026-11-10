"""CPU checks for gated, once-per-server representative prefill warmup."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.srt.entrypoints import mimo_prefill_warmup as warmup
from sglang.test.ci.ci_register import register_cpu_ci

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


class StartupWarmupTest(unittest.IsolatedAsyncioTestCase):
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
        self.assertEqual(len(manager.requests), 16)
        self.assertEqual(
            [len(req.input_ids[0]) for req in manager.requests],
            list(warmup.INPUT_LENGTHS),
        )
        self.assertEqual(sum(len(req.input_ids) for req in manager.requests), 32)
        self.assertEqual(manager.flush_cache.await_count, 17)
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


if __name__ == "__main__":
    unittest.main()
