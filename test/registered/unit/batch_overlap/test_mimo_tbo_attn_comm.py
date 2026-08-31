from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.batch_overlap.comm_stream as comm_stream
import sglang.srt.batch_overlap.operations as operations
import sglang.srt.batch_overlap.operations_strategy as operations_strategy
from sglang.srt.environ import envs
from sglang.srt.layers import communicator
from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _named_op(name):
    def op(*args, **kwargs):
        return None

    op.__name__ = name
    return op


def _fake_mimo_layer(*, supports_attn_comm_tbo=True, has_tbo_attn_all_gather=True):
    return SimpleNamespace(
        is_layer_sparse=True,
        layer_communicator=SimpleNamespace(
            supports_tbo_attn_communication=lambda: supports_attn_comm_tbo,
            has_tbo_attn_all_gather=lambda: has_tbo_attn_all_gather,
        ),
        self_attn=SimpleNamespace(
            op_prepare=_named_op("op_prepare"),
            op_core=_named_op("op_core"),
        ),
        mlp=SimpleNamespace(
            op_gate=_named_op("op_gate"),
            op_select_experts=_named_op("op_select_experts"),
            op_dispatch_a=_named_op("op_dispatch_a"),
            op_dispatch_b=_named_op("op_dispatch_b"),
            op_experts=_named_op("op_experts"),
            op_combine_a=_named_op("op_combine_a"),
            op_combine_b=_named_op("op_combine_b"),
            op_output=_named_op("op_output"),
        ),
        op_comm_prepare_attn=_named_op("op_comm_prepare_attn"),
        op_comm_prepare_attn_a=_named_op("op_comm_prepare_attn_a"),
        op_comm_prepare_attn_b=_named_op("op_comm_prepare_attn_b"),
        op_comm_prepare_mlp=_named_op("op_comm_prepare_mlp"),
        op_comm_prepare_mlp_a=_named_op("op_comm_prepare_mlp_a"),
        op_comm_prepare_mlp_b=_named_op("op_comm_prepare_mlp_b"),
        op_comm_postprocess_layer=_named_op("op_comm_postprocess_layer"),
    )


def _operation_names(strategy):
    return [
        "yield" if isinstance(op, operations.YieldOperation) else op.__name__
        for op in strategy.operations
    ]


def test_mimo_prefill_attn_collectives_use_existing_tbo_yields(monkeypatch):
    monkeypatch.setattr(operations_strategy, "_is_hip", True)
    monkeypatch.setattr(
        operations_strategy.DeepEPConfig,
        "get_instance",
        lambda: (_ for _ in ()).throw(
            AssertionError("ROCm MiMo TBO must not initialize CUDA DeepEPConfig")
        ),
    )
    monkeypatch.setattr(
        operations_strategy,
        "get_moe_a2a_backend",
        lambda: MoeA2ABackend.MORI,
    )
    monkeypatch.setattr(
        operations_strategy,
        "get_deepep_mode",
        lambda: DeepEPMode.NORMAL,
    )
    monkeypatch.setattr(
        operations_strategy.torch.cuda,
        "get_device_properties",
        lambda **_: SimpleNamespace(multi_processor_count=256),
    )

    with envs.SGLANG_MIMO_TBO_ATTN_COMM.override(True):
        strategy = operations_strategy._compute_moe_mimov2_prefill(_fake_mimo_layer())

    assert _operation_names(strategy) == [
        "op_comm_prepare_attn_a",
        "yield",
        "op_comm_prepare_attn_b",
        "op_prepare",
        "op_core",
        "op_comm_prepare_mlp_a",
        "yield",
        "op_comm_prepare_mlp_b",
        "op_gate",
        "op_select_experts",
        "op_dispatch_a",
        "yield",
        "op_dispatch_b",
        "op_experts",
        "op_combine_a",
        "yield",
        "op_combine_b",
        "op_output",
        "op_comm_postprocess_layer",
    ]


def test_mimo_prefill_first_sparse_layer_only_overlaps_reduce_scatter(monkeypatch):
    monkeypatch.setattr(operations_strategy, "_is_hip", True)
    monkeypatch.setattr(
        operations_strategy,
        "get_moe_a2a_backend",
        lambda: MoeA2ABackend.MORI,
    )
    monkeypatch.setattr(
        operations_strategy,
        "get_deepep_mode",
        lambda: DeepEPMode.NORMAL,
    )
    monkeypatch.setattr(
        operations_strategy.torch.cuda,
        "get_device_properties",
        lambda **_: SimpleNamespace(multi_processor_count=256),
    )

    with envs.SGLANG_MIMO_TBO_ATTN_COMM.override(True):
        strategy = operations_strategy._compute_moe_mimov2_prefill(
            _fake_mimo_layer(has_tbo_attn_all_gather=False)
        )

    assert _operation_names(strategy)[:8] == [
        "op_comm_prepare_attn",
        "op_prepare",
        "op_core",
        "op_comm_prepare_mlp_a",
        "yield",
        "op_comm_prepare_mlp_b",
        "op_gate",
        "op_select_experts",
    ]


def test_mimo_prefill_attn_collective_tbo_is_opt_in(monkeypatch):
    monkeypatch.setattr(operations_strategy, "_is_hip", True)
    monkeypatch.setattr(
        operations_strategy,
        "get_moe_a2a_backend",
        lambda: MoeA2ABackend.MORI,
    )
    monkeypatch.setattr(
        operations_strategy,
        "get_deepep_mode",
        lambda: DeepEPMode.NORMAL,
    )
    monkeypatch.setattr(
        operations_strategy.torch.cuda,
        "get_device_properties",
        lambda **_: SimpleNamespace(multi_processor_count=256),
    )

    with envs.SGLANG_MIMO_TBO_ATTN_COMM.override(False):
        strategy = operations_strategy._compute_moe_mimov2_prefill(_fake_mimo_layer())

    assert _operation_names(strategy)[:4] == [
        "op_comm_prepare_attn",
        "op_prepare",
        "op_core",
        "op_comm_prepare_mlp",
    ]


def test_mimo_prefill_attn_collective_tbo_rejects_non_mori(monkeypatch):
    monkeypatch.setattr(operations_strategy, "_is_hip", True)
    monkeypatch.setattr(
        operations_strategy,
        "get_moe_a2a_backend",
        lambda: MoeA2ABackend.DEEPEP,
    )
    monkeypatch.setattr(
        operations_strategy,
        "get_deepep_mode",
        lambda: DeepEPMode.NORMAL,
    )
    monkeypatch.setattr(
        operations_strategy.torch.cuda,
        "get_device_properties",
        lambda **_: SimpleNamespace(multi_processor_count=256),
    )

    with envs.SGLANG_MIMO_TBO_ATTN_COMM.override(True):
        with pytest.raises(RuntimeError, match="requires the MORI"):
            operations_strategy._compute_moe_mimov2_prefill(_fake_mimo_layer())


def test_mimo_prefill_attn_collective_tbo_rejects_low_latency_mori(monkeypatch):
    monkeypatch.setattr(operations_strategy, "_is_hip", True)
    monkeypatch.setattr(
        operations_strategy,
        "get_moe_a2a_backend",
        lambda: MoeA2ABackend.MORI,
    )
    monkeypatch.setattr(
        operations_strategy,
        "get_deepep_mode",
        lambda: DeepEPMode.LOW_LATENCY,
    )
    monkeypatch.setattr(
        operations_strategy.torch.cuda,
        "get_device_properties",
        lambda **_: SimpleNamespace(multi_processor_count=256),
    )

    with envs.SGLANG_MIMO_TBO_ATTN_COMM.override(True):
        with pytest.raises(RuntimeError, match="normal mode only"):
            operations_strategy._compute_moe_mimov2_prefill(_fake_mimo_layer())


def test_tbo_comm_stream_pool_reuses_stream_per_group(monkeypatch):
    streams = []
    events = []

    class FakeStream:
        pass

    class FakeEvent:
        pass

    def make_stream(priority):
        stream = FakeStream()
        streams.append((priority, stream))
        return stream

    def make_event(**kwargs):
        event = FakeEvent()
        events.append((kwargs, event))
        return event

    monkeypatch.setattr(comm_stream.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(comm_stream.torch.cuda, "Stream", make_stream)
    monkeypatch.setattr(comm_stream.torch.cuda, "Event", make_event)
    comm_stream.TboCommStreamPool._streams.clear()
    comm_stream.TboCommStreamPool._events.clear()

    group_a = object()
    group_b = object()
    first = comm_stream.TboCommStreamPool.get_stream_from_pool(group_a)
    second = comm_stream.TboCommStreamPool.get_stream_from_pool(group_a)
    third = comm_stream.TboCommStreamPool.get_stream_from_pool(group_b)

    assert first is second
    assert third is not first
    assert [priority for priority, _ in streams] == [0, 0]

    events_a0 = comm_stream.TboCommStreamPool.get_events(group_a, 0)
    events_a0_again = comm_stream.TboCommStreamPool.get_events(group_a, 0)
    events_a1 = comm_stream.TboCommStreamPool.get_events(group_a, 1)
    assert events_a0 is events_a0_again
    assert events_a1 is not events_a0
    assert len(events) == 4


def test_layer_communicator_detects_supported_tbo_collectives(monkeypatch):
    group = object()
    monkeypatch.setattr(communicator, "get_attention_tp_group", lambda: group)
    monkeypatch.setattr(communicator, "get_tp_group", lambda: group)

    layer_communicator = communicator.LayerCommunicator.__new__(
        communicator.LayerCommunicator
    )
    layer_communicator._communicate_simple_fn = (
        communicator.CommunicateSimpleFn._scattered_to_tp_attn_full
    )
    layer_communicator._communicate_with_all_reduce_and_layer_norm_fn = partial(
        communicator.CommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual,
        residual_input_mode=communicator.ScatterMode.SCATTERED,
    )

    assert layer_communicator.supports_tbo_attn_communication()
    assert layer_communicator.has_tbo_attn_all_gather()

    layer_communicator._communicate_simple_fn = (
        communicator.CommunicateSimpleFn._trivial
    )
    assert layer_communicator.supports_tbo_attn_communication()
    assert not layer_communicator.has_tbo_attn_all_gather()


def test_tbo_collective_uses_shared_comm_stream_and_gpu_events(monkeypatch):
    calls = []

    class FakeEvent:
        def __init__(self, *args, **kwargs):
            pass

        def record(self, stream):
            calls.append(("record", stream))

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            calls.append(("wait", self, event))

    compute = FakeStream("compute")
    comm = FakeStream("comm")
    events = comm_stream.TboCommEvents(
        compute_done=FakeEvent(),
        comm_done=FakeEvent(),
    )

    @contextmanager
    def use_stream(stream):
        calls.append(("enter", stream))
        yield
        calls.append(("exit", stream))

    monkeypatch.setattr(communicator.torch.cuda, "current_stream", lambda: compute)
    monkeypatch.setattr(communicator.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(communicator.torch.cuda, "stream", use_stream)
    monkeypatch.setattr(
        communicator.TboCommStreamPool,
        "get_stream_from_pool",
        lambda group: comm,
    )
    output = communicator.LayerCommunicator._run_tbo_collective(
        object(), events, lambda: "output"
    )

    assert output == "output"
    assert ("wait", comm, events.compute_done) in calls
    assert any(call[0] == "record" and call[1] is comm for call in calls)
    assert any(call[0] == "wait" and call[1] is compute for call in calls)


def test_prepare_attn_tbo_preallocates_output_before_collective(monkeypatch):
    calls = []
    group = object()
    compute_stream = object()
    output = object()
    events = comm_stream.TboCommEvents(
        compute_done=SimpleNamespace(
            record=lambda stream: calls.append(("compute_done", stream))
        ),
        comm_done=object(),
    )

    monkeypatch.setattr(communicator, "get_attention_tp_group", lambda: group)
    monkeypatch.setattr(
        communicator.torch.cuda, "current_stream", lambda: compute_stream
    )
    monkeypatch.setattr(
        communicator.TboCommStreamPool,
        "get_events",
        lambda actual_group, subbatch_index: events,
    )
    monkeypatch.setattr(
        communicator.CommunicateSimpleFn,
        "_allocate_scattered_to_tp_attn_full_output",
        staticmethod(
            lambda hidden_states, context: calls.append(
                ("allocate", hidden_states, context)
            )
            or output
        ),
    )

    layer_communicator = communicator.LayerCommunicator.__new__(
        communicator.LayerCommunicator
    )
    layer_communicator._context = object()
    layer_communicator.supports_tbo_attn_communication = lambda: True
    layer_communicator.has_tbo_attn_all_gather = lambda: True
    layer_communicator._prepare_attn_local = lambda *args: ("input", "residual")

    pending = layer_communicator.prepare_attn_tbo_a(
        "hidden_states",
        "residual",
        "forward_batch",
        tbo_subbatch_index=1,
    )

    assert pending.hidden_states == "input"
    assert pending.output_hidden_states is output
    assert calls == [
        ("allocate", "input", layer_communicator._context),
        ("compute_done", compute_stream),
    ]


def test_tbo_attn_all_gather_fills_preallocated_tuple(monkeypatch):
    local = tuple(object() for _ in range(3))
    output = tuple(object() for _ in range(3))
    calls = []

    monkeypatch.setattr(
        communicator,
        "attn_tp_all_gather_into_tensor",
        lambda gathered, local_: calls.append((gathered, local_)),
    )

    result = communicator.CommunicateSimpleFn._scattered_to_tp_attn_full_into(
        output,
        local,
    )

    assert result is output
    assert calls == list(zip(output, local, strict=True))


def test_tbo_attn_all_gather_tuple_allocation_uses_attention_group(monkeypatch):
    group = object()
    symmetric_memory_calls = []

    @contextmanager
    def use_symmetric_memory(actual_group, disabled):
        symmetric_memory_calls.append((actual_group, disabled))
        yield

    monkeypatch.setattr(communicator, "get_attention_tp_group", lambda: group)
    monkeypatch.setattr(communicator, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(communicator, "use_symmetric_memory", use_symmetric_memory)

    local = (
        torch.empty((2, 3), dtype=torch.float32),
        torch.empty((2, 1), dtype=torch.float16),
        torch.empty((2, 5), dtype=torch.bfloat16),
    )
    output = communicator.CommunicateSimpleFn._allocate_scattered_to_tp_attn_full_output(
        local,
        SimpleNamespace(attn_tp_size=4),
    )

    assert [item.shape for item in output] == [(8, 3), (8, 1), (8, 5)]
    assert [item.dtype for item in output] == [item.dtype for item in local]
    assert symmetric_memory_calls == [(group, True)] * 3


def test_prepare_mlp_tbo_splits_collective_from_layernorm(monkeypatch):
    group = object()
    ready_event = SimpleNamespace(record=lambda stream: None)
    events = comm_stream.TboCommEvents(
        compute_done=ready_event,
        comm_done=object(),
    )
    monkeypatch.setattr(communicator, "get_attention_tp_group", lambda: group)
    monkeypatch.setattr(communicator.torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(
        communicator.TboCommStreamPool,
        "get_events",
        lambda actual_group, subbatch_index: events,
    )

    reduce_scatter_calls = []

    def fake_reduce_scatter(output, input_):
        reduce_scatter_calls.append((output, input_))
        output.copy_(input_[: output.shape[0]])

    monkeypatch.setattr(
        communicator,
        "attn_tp_reduce_scatter_tensor",
        fake_reduce_scatter,
    )
    monkeypatch.setattr(
        communicator.LayerCommunicator,
        "_run_tbo_collective",
        staticmethod(lambda group, events, fn: fn()),
    )

    class FakeLayerNorm:
        def __call__(self, hidden_states, residual):
            return hidden_states + 1, residual + 2

    layer_communicator = communicator.LayerCommunicator.__new__(
        communicator.LayerCommunicator
    )
    layer_communicator._context = SimpleNamespace(
        attn_tp_size=2,
        attn_tp_rank=0,
        attn_dp_size=1,
    )
    layer_communicator.post_attention_layernorm = FakeLayerNorm()
    layer_communicator._communicate_with_all_reduce_and_layer_norm_fn = partial(
        communicator.CommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual,
        residual_input_mode=communicator.ScatterMode.TP_ATTN_FULL,
    )
    layer_communicator.supports_tbo_attn_communication = lambda: True

    hidden_states = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    residual = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    pending = layer_communicator.prepare_mlp_tbo_a(
        hidden_states,
        residual,
        tbo_subbatch_index=1,
    )

    assert reduce_scatter_calls == []
    assert pending.output_hidden_states.shape == (2, 2)
    assert pending.residual.shape == (2, 2)

    output, output_residual = layer_communicator.prepare_mlp_tbo_b(pending)

    assert len(reduce_scatter_calls) == 1
    assert torch.equal(output, hidden_states[:2] + 1)
    assert torch.equal(output_residual, residual[:2] + 2)
