"""Bitwise and eight-rank graph checks for DP zero-fill/copy fusion."""

import argparse
import json
import os
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import torch
import torch.distributed as dist

import sglang.srt.layers.dp_attention as dp
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment, destroy_model_parallel, get_tp_group,
    graph_capture, init_distributed_environment, initialize_model_parallel,
    set_custom_all_reduce,
)


def bits_equal(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def local_checks(device):
    count = 0
    dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.int32, torch.int64, torch.float8_e4m3fnuz]
    for dtype in dtypes:
        for hidden in (7, 4096):
            if dtype.is_floating_point:
                source = torch.randn(9, hidden, device=device).to(dtype)
                if dtype != torch.float8_e4m3fnuz:
                    source[0, :3] = torch.tensor([-0.0, float("nan"), float("inf")], device=device, dtype=dtype)
            else:
                source = torch.randint(-10000, 10000, (9, hidden), device=device, dtype=dtype)
                if dtype == torch.int64:
                    source *= 2**40
            guarded = torch.full((64 + 17 * hidden + 64,), 17, device=device, dtype=torch.float32).to(dtype)
            destination = guarded[64:-64].view(17, hidden)
            offset = torch.tensor([0], device=device, dtype=torch.int64)
            size = torch.tensor([9], device=device, dtype=torch.int64)
            dp.pad_copy_triton(destination, source, offset, size)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                dp.pad_copy_triton(destination, source, offset, size)
            for start, length in ((0, 9), (3, 9), (15, 2), (5, 0), (1, 4)):
                offset.fill_(start)
                size.fill_(length)
                reference = torch.zeros((17, hidden), device=device, dtype=torch.float32).to(dtype)
                reference[start:start+length].copy_(source[:length])
                for _ in range(3):
                    graph.replay()
                    torch.cuda.synchronize()
                    assert bits_equal(destination, reference), (dtype, hidden, start, length)
                    guard = torch.full((64,), 17, device=device, dtype=torch.float32).to(dtype)
                    assert bits_equal(guarded[:64], guard) and bits_equal(guarded[-64:], guard)
                count += 1
            del graph
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    assert world == 8
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    set_custom_all_reduce(True)
    init_distributed_environment(world_size=world, rank=rank, local_rank=local,
                                 distributed_init_method="env://", backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=8, pipeline_model_parallel_size=1)
    group = get_tp_group()
    assert group.ca_comm is not None and not group.ca_comm.disabled
    local_count = local_checks(device) if rank == 0 else 0
    dist.barrier()
    # Exercise the actual gather implementation with two attention TP4 groups.
    dp.get_attention_dp_rank = lambda: rank // 4
    dp.get_attention_tp_rank = lambda: rank % 4
    rows = []
    for global_rows in (8, 128, 192, 256, 320, 512, 768, 832, 896, 960, 1024, 2048, 2560, 2880):
        local_rows = global_rows // 2
        torch.manual_seed(1000 + rank + global_rows)
        source = torch.randn((local_rows, 4096), device=device, dtype=torch.bfloat16)
        destination = torch.empty((global_rows, 4096), device=device, dtype=source.dtype)
        offset = torch.tensor([rank // 4 * local_rows], device=device, dtype=torch.int64)
        size = torch.tensor([local_rows], device=device, dtype=torch.int64)
        batch = SimpleNamespace(dp_local_start_pos=offset, dp_local_num_tokens=size)
        for partial in (False, True):
            graphs = {}
            for fused in (False, True):
                dp._FUSED_DP_PAD_COPY = fused
                for _ in range(3):
                    dp._dp_gather_via_all_reduce(destination, source, batch, partial)
                torch.cuda.synchronize()
                with graph_capture() as capture:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture.stream):
                        for _ in range(10):
                            dp._dp_gather_via_all_reduce(destination, source, batch, partial)
                graphs[fused] = graph
            for length in (local_rows, max(1, local_rows - 1), 0, local_rows):
                size.fill_(length)
                for _ in range(3):
                    graphs[False].replay()
                    torch.cuda.synchronize()
                    reference = destination.clone()
                    graphs[True].replay()
                    torch.cuda.synchronize()
                    assert bits_equal(destination, reference), (rank, global_rows, partial, length)
                expected = torch.zeros_like(destination, dtype=torch.float32)
                if partial or rank % 4 == 0:
                    expected[rank // 4 * local_rows:rank // 4 * local_rows + length] = source[:length].float()
                dist.all_reduce(expected, group=group.device_group)
                expected = expected.bfloat16()
                error = float((destination.float()-expected.float()).norm() / expected.float().norm().clamp_min(1e-6))
                assert torch.isfinite(destination).all() and error < .01, (rank, global_rows, partial, error)
            times = {}
            for fused in (False, True):
                samples = []
                for _ in range(7):
                    dist.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10): graphs[fused].replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000 / 100)
                gathered = [None] * world
                dist.all_gather_object(gathered, median(samples), group=group.cpu_group)
                times[str(fused)] = gathered
            if rank == 0:
                row = {"global_rows": global_rows, "hidden": 4096, "partial": partial,
                       "bit_exact_to_baseline": True, "rank_times_us": times,
                       "baseline_slowest_us": max(times["False"]), "fused_slowest_us": max(times["True"])}
                row["speedup"] = row["baseline_slowest_us"] / row["fused_slowest_us"]
                rows.append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"status": "running", "results": rows}, indent=2)+"\n")
                print(json.dumps(row), flush=True)
            del graphs
    dist.barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    if rank == 0:
        args.output.write_text(json.dumps({"status": "passed", "local_bitwise_cases": local_count,
            "results": rows, "scope": "Eight-rank TP8 communication with DP2/attention-TP4 semantics; graph replay changes valid lengths, including empty groups. Local copy checks include six dtypes, NaN/Inf/signed-zero, integer payloads, unaligned lengths and output guards. Component timings require later real-model and unprofiled serving validation."}, indent=2)+"\n")


if __name__ == "__main__":
    main()
