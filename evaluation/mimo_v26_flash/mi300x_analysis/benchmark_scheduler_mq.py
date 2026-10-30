"""Eight-process routing/overflow checks and transport-only timing; no GPU model."""

import argparse
from datetime import timedelta
import json
from pathlib import Path
import statistics
import time
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.shm_broadcast import MessageQueue
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.request_receiver import SchedulerRequestReceiver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8
    local_groups = [dist.new_group(ranks=list(range(start, start + 4)), backend="gloo") for start in (0, 4)]
    local = local_groups[rank // 4]

    def wrap(pg, ranks):
        group = SimpleNamespace(cpu_group=pg, ranks=ranks, rank=rank,
            rank_in_group=ranks.index(rank), world_size=len(ranks))
        group.mq_broadcaster = MessageQueue.create_from_process_group(pg, max_chunk_bytes=65536, max_chunks=2)
        group.broadcast_object = MethodType(GroupCoordinator.broadcast_object, group)
        return group

    global_group = wrap(dist.group.WORLD, list(range(8)))
    local_group = wrap(local, list(range(rank // 4 * 4, rank // 4 * 4 + 4)))
    records = []
    for layout in ("ordinary", "dpa_global_control", "dpa_local_control"):
        dpa = layout != "ordinary"
        local_control = layout == "dpa_local_control"
        tp_group = global_group if dpa else local_group
        receiver = SimpleNamespace(
            ps=SimpleNamespace(attn_tp_rank=rank % 4, attn_cp_rank=0, attn_tp_size=4, attn_cp_size=1,
                tp_size=8 if dpa else 4),
            server_args=SimpleNamespace(enable_dp_attention=dpa, enable_dp_attention_local_control_broadcast=local_control),
            tp_group=tp_group, tp_cpu_group=tp_group.cpu_group,
            attn_tp_group=local_group, attn_tp_cpu_group=local_group.cpu_group,
            _broadcast_in_group=SchedulerRequestReceiver._broadcast_in_group,
            _split_work_and_control_reqs=lambda data: (data[0], data[1]),
        )
        for mode in ("gloo", "mq", "mq_unavailable"):
            saved_queues = [global_group.mq_broadcaster, local_group.mq_broadcaster]
            if mode == "mq_unavailable":
                global_group.mq_broadcaster = local_group.mq_broadcaster = None
            for label, tokens, iterations in (("empty", 0, 100), ("small", 64, 30), ("overflow", 378880, 3)):
                timings = []
                ids = list(range(tokens))
                with envs.SGLANG_SCHEDULER_MQ_BROADCAST.override(mode != "gloo"):
                    for step in range(iterations):
                        work = [{"kind": "work", "dp": rank // 4, "step": step, "input_ids": ids}] if tokens else []
                        control = [{"kind": "control", "source": rank // 4 * 4, "step": step}] if tokens else []
                        sent = [work, control] if dpa else work + control
                        if rank % 4:
                            sent = None
                        expected_source = rank // 4 * 4 if local_control or not dpa else 0
                        expected = work + ([{"kind": "control", "source": expected_source, "step": step}] if tokens else [])
                        dist.barrier()
                        started = time.perf_counter()
                        received = SchedulerRequestReceiver._broadcast_reqs_across_ranks(receiver, sent)
                        elapsed_us = (time.perf_counter() - started) * 1e6
                        assert received == expected, (rank, layout, mode, label, step)
                        timings.append(elapsed_us)
                dist.barrier()
                gathered = [None] * world
                dist.all_gather_object(gathered, timings)
                if rank == 0:
                    slowest = [max(values[i] for values in gathered) for i in range(iterations)]
                    record = {"layout": layout, "transport": mode, "payload": label,
                        "iterations": iterations, "all_rank_content_checks_passed": True,
                        "median_slowest_rank_us": statistics.median(slowest),
                        "per_rank_median_us": [statistics.median(values) for values in gathered]}
                    records.append(record)
                    print(json.dumps(record), flush=True)
            global_group.mq_broadcaster, local_group.mq_broadcaster = saved_queues
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"passed": True, "world_size": world,
            "scope": "CPU transport only; preserves both TP4 groups, global/local DPA control, empty polls, large-message overflow and absent-queue fallback. Not a serving speedup.",
            "results": records}, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
