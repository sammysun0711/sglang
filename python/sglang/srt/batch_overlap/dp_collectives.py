"""Prefill DP collectives with child-owned geometry and explicit completion.

This experimental provider is not selected by the serving path yet. It uses
the communicator's allocating collective APIs and is intended for stage-level
AG/RS overlap in global TP / attention DP x TP, without adding copy kernels.
"""

from dataclasses import dataclass
from typing import Tuple

import torch

from sglang.srt.layers.dp_attention import (
    get_attention_cp_size,
    get_attention_dp_rank,
    get_attention_dp_size,
    get_attention_tp_group,
    get_attention_tp_size,
    get_moe_data_parallel_world_size,
    get_moe_expert_parallel_world_size,
    get_tp_group,
)


@dataclass(frozen=True)
class DPCollectivePlan:
    """Immutable CPU geometry for one child; it never reads GPU scalars."""

    rows_per_dp: Tuple[int, ...]
    attn_tp_size: int
    dp_rank: int

    @classmethod
    def from_rows(cls, rows_per_dp):
        rows = tuple(int(n) for n in rows_per_dp)
        tp = get_attention_tp_size()
        assert get_attention_cp_size() == 1
        assert get_moe_expert_parallel_world_size() == 1
        assert get_moe_data_parallel_world_size() == 1
        assert len(rows) == get_attention_dp_size()
        assert len(rows) > 1 and tp > 1
        assert get_tp_group().world_size == len(rows) * tp
        assert all(n > 0 and n % tp == 0 for n in rows)
        return cls(rows, tp, get_attention_dp_rank())

    @property
    def local_rows(self):
        return self.rows_per_dp[self.dp_rank]

    @property
    def global_rows(self):
        return sum(self.rows_per_dp)

    @property
    def shard_rows(self):
        return [
            n // self.attn_tp_size
            for n in self.rows_per_dp
            for _ in range(self.attn_tp_size)
        ]

    @property
    def local_start(self):
        return sum(self.rows_per_dp[: self.dp_rank])


@dataclass
class DPCollectiveWork:
    output: torch.Tensor
    complete: torch.cuda.Event
    plan: DPCollectivePlan

    def wait(self):
        stream = torch.cuda.current_stream(self.output.device)
        stream.wait_event(self.complete)
        # Outputs are allocated on the communication stream. Keep their
        # storage alive for downstream compute after this work object dies.
        self.output.record_stream(stream)
        return self.output

    def wait_local_view(self):
        """Return complete unnormalized local rows for the residual contract."""
        output = self.wait()
        assert output.shape[0] == self.plan.global_rows
        return output.narrow(0, self.plan.local_start, self.plan.local_rows)


def _stream_and_event(tensor, event_key):
    # Reuse the native single-thread TBO executor's stream/event ownership.
    # Lazy import avoids importing the TBO batch preparer at module import.
    from sglang.srt.batch_overlap.two_batch_overlap import (
        _get_tbo_tp_comm_stream,
        _get_tbo_tp_event,
    )

    stream = _get_tbo_tp_comm_stream(tensor.device)
    event = _get_tbo_tp_event(tensor.device, ("dp_collective", *event_key))
    stream.wait_stream(torch.cuda.current_stream(tensor.device))
    tensor.record_stream(stream)
    return stream, event


def launch_dp_partial_gather(local_partial, plan: DPCollectivePlan, *, event_key):
    """Reduce attention TP partials, then gather shards in exact SUM_LEN order.

    The caller adds the residual on the attention leader before calling this.
    Comm APIs allocate their outputs, retaining RCCL for the bulk gather even
    when the group's automatic allgather would select custom allgather.
    """
    assert local_partial.is_contiguous()
    assert local_partial.shape[0] == plan.local_rows
    stream, event = _stream_and_event(local_partial, event_key)
    with torch.cuda.stream(stream):
        shard = get_attention_tp_group().reduce_scatterv(local_partial)
        output = _single_tensor(
            get_tp_group().all_gatherv(shard, sizes=plan.shard_rows)
        )
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == plan.global_rows
        event.record(stream)
    return DPCollectiveWork(output, event, plan)


def launch_dp_combine(global_partial, plan: DPCollectivePlan, *, event_key):
    """Reduce global MoE partials, then replicate this attention group's rows.

    The residual is already complete/local and must not participate here.
    """
    assert global_partial.is_contiguous()
    assert global_partial.shape[0] == plan.global_rows
    stream, event = _stream_and_event(global_partial, event_key)
    with torch.cuda.stream(stream):
        shard = get_tp_group().reduce_scatterv(global_partial, sizes=plan.shard_rows)
        output = _single_tensor(get_attention_tp_group().all_gatherv(shard))
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == plan.local_rows
        event.record(stream)
    return DPCollectiveWork(output, event, plan)


def _single_tensor(output):
    # Current SGLang returns a one-element list even for one tensor input.
    if isinstance(output, list):
        assert len(output) == 1
        return output[0]
    return output
