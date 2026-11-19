"""Model-free ROCm TP4 regression; no CI registration.

Run from the repository root:
  PYTHONPATH=python:$PYTHONPATH python -m torch.distributed.run \
    --standalone --nproc_per_node=4 test/manual/amd/test_dflash_request_owner.py
"""

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist


class SamplingInfo(SimpleNamespace):
    def __len__(self):
        return self.temperatures.shape[0]


def check_case(group, batch, captured_batch):
    from sglang.srt.speculative.dflash_utils import (
        apply_dflash_verify_logits_adjustments,
        compute_dflash_correct_drafts_and_bonus,
        compute_dflash_sampling_correct_drafts_and_bonus,
    )
    from sglang.srt.speculative.request_owner_verify import (
        distribute_logits,
        restore_full_logits,
        verify_owned,
    )

    block, vocab = 8, 1024
    torch.manual_seed(12345)
    full = torch.randn(captured_batch * block, vocab, device="cuda") * 0.2
    peaks = torch.randint(vocab, (captured_batch * block, 1), device="cuda")
    full.scatter_(1, peaks, 7.0)
    full = full.to(torch.bfloat16)
    shard = vocab // group.world_size
    columns = slice(group.rank_in_group * shard, (group.rank_in_group + 1) * shard)
    local = full[:, columns].contiguous()

    def exchange():
        return distribute_logits(local, group, block_size=block, vocab_size=vocab)

    if captured_batch != batch:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            exchange()
            exchange()
        stream.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            owned, chunk = exchange()
        # Exercise captured communication with changed inputs and padded requests.
        full = full.roll(1, dims=1)
        local.copy_(full[:, columns])
        graph.replay()
    else:
        owned, chunk = exchange()

    padded = torch.zeros(group.world_size * chunk * block, vocab, device="cuda")
    padded[: full.shape[0]].copy_(full)
    start = group.rank_in_group * chunk * block
    torch.testing.assert_close(
        owned, padded[start : start + chunk * block], rtol=0, atol=0
    )
    dense = full[: batch * block].float()
    restored = restore_full_logits(
        owned, group, batch_size=batch, block_size=block, chunk=chunk
    )
    torch.testing.assert_close(restored, dense, rtol=0, atol=0)

    target = dense.argmax(-1).view(batch, block)
    candidates = torch.zeros_like(target)
    candidates[:, 1:] = target[:, :-1]
    # Greedy verification covers every possible correct-draft count at BS8.
    for request in range(batch):
        count = request % block
        if count < block - 1:
            candidates[request, count + 1] = (target[request, count] + 1) % vocab
    info = SamplingInfo(
        is_all_greedy=True,
        has_custom_logit_processor=False,
        temperatures=torch.linspace(0.8, 1.2, batch, device="cuda").view(-1, 1),
        top_ps=torch.full((batch,), 0.85, device="cuda"),
        top_ks=torch.full((batch,), vocab, device="cuda", dtype=torch.int32),
        need_top_k_sampling=False,
        need_top_p_sampling=True,
    )
    expected = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates, target_predict=target
    )
    expected = (expected[0].to(torch.int32), expected[1])
    actual = verify_owned(owned, candidates, info, group, chunk=chunk)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        actual[0],
        torch.arange(batch, device="cuda", dtype=torch.int32) % block,
        rtol=0,
        atol=0,
    )

    info.is_all_greedy = False
    for seed in (2026, 2027):
        torch.manual_seed(seed)
        expected = tuple(
            tensor.clone()
            for tensor in compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates, next_token_logits=dense, sampling_info=info
            )
        )
        expected_rng = torch.cuda.get_rng_state()
        torch.manual_seed(seed)
        actual = verify_owned(owned, candidates, info, group, chunk=chunk)
        assert actual is not None, "Sampling silently fell back instead of testing owners"
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), expected_rng)

    info.is_all_greedy = True
    info.penalizer_orchestrator = SimpleNamespace(is_required=True)
    info.acc_linear_penalties = torch.zeros(batch, vocab, device="cuda")
    info.acc_linear_penalties[:, 0] = 50
    assert verify_owned(owned, candidates, info, group, chunk=chunk) is None
    restored = restore_full_logits(
        owned, group, batch_size=batch, block_size=block, chunk=chunk
    )
    apply_dflash_verify_logits_adjustments(
        next_token_logits=restored, sampling_info=info, draft_token_num=block
    )
    expected_logits = (
        dense.view(batch, block, vocab) + info.acc_linear_penalties[:, None, :]
    ).view(-1, vocab)
    torch.testing.assert_close(restored, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(
        compute_dflash_correct_drafts_and_bonus(
            candidates=candidates, target_predict=restored.argmax(-1).view(batch, block)
        ),
        compute_dflash_correct_drafts_and_bonus(
            candidates=candidates,
            target_predict=expected_logits.argmax(-1).view(batch, block),
        ),
        rtol=0,
        atol=0,
    )
    torch.cuda.synchronize()
    dist.barrier()
    if group.rank_in_group == 0:
        print(
            f"PASS BS={batch}, captured BS={captured_batch}: logits, greedy, "
            "seeded top-p, RNG state, penalty fallback",
            flush=True,
        )


def main():
    if torch.version.hip is None or int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise RuntimeError("Run with torchrun --nproc_per_node=4 on ROCm")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    from sglang.srt.environ import envs
    from sglang.srt.speculative.dflash_utils import is_dflash_sampling_verify_available
    from sglang.srt.speculative.request_owner_scan import _extension

    group = SimpleNamespace(
        world_size=4,
        rank_in_group=dist.get_rank(),
        pynccl_comm=PyNcclCommunicator(dist.group.WORLD, rank),
    )
    assert group.pynccl_comm.available, "Direct RCCL communicator unavailable"
    try:
        with (
            envs.SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY.override(True),
            patch(
                "sglang.srt.server_args.get_global_server_args",
                return_value=SimpleNamespace(
                    speculative_accept_threshold_single=1.0,
                    speculative_accept_threshold_acc=1.0,
                ),
            ),
        ):
            assert is_dflash_sampling_verify_available()
            for batch, captured_batch in ((1, 1), (5, 5), (8, 8), (5, 8)):
                check_case(group, batch, captured_batch)
            assert _extension.cache_info().currsize == 1, "Owner scan was not exercised"
        if dist.get_rank() == 0:
            print("PASS: ROCm owner scan extension exercised on all TP4 ranks", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
