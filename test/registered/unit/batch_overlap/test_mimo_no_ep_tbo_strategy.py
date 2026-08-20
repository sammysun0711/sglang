import gc
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sglang.srt.batch_overlap import two_batch_overlap
from sglang.srt.batch_overlap.operations import YieldOperation
from sglang.srt.batch_overlap.operations_strategy import (
    _compute_moe_mimov2_layer_operations_strategy_tbo,
)
from sglang.srt.managers.scheduler_components.dp_attn import SchedulerDPAttnAdapter
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    get_batch_sizes_to_capture,
    is_tbo_cuda_graph_enabled,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.models import mimo_v2


class _Ops:
    def op_prepare(self):
        pass

    def op_core(self):
        pass

    def op_gate(self):
        pass

    def op_select_experts(self):
        pass

    def op_tbo_no_ep_experts(self):
        pass

    def op_tbo_no_ep_all_reduce_launch(self):
        pass

    def op_tbo_no_ep_all_reduce_wait(self):
        pass

    def op_output(self):
        pass


class _Layer:
    is_layer_sparse = True
    self_attn = _Ops()
    mlp = _Ops()

    def op_comm_prepare_attn(self):
        pass

    def op_tbo_no_ep_attn_all_reduce_launch(self):
        pass

    def op_tbo_no_ep_attn_all_reduce_wait_prepare_mlp(self):
        pass

    def op_comm_postprocess_layer(self):
        pass


class _NoA2ABackend:
    @staticmethod
    def is_none():
        return True


@patch(
    "sglang.srt.batch_overlap.operations_strategy.get_moe_a2a_backend",
    return_value=_NoA2ABackend(),
)
def test_mimo_no_ep_prefill_strategy_has_two_collective_yields(_):
    strategy = _compute_moe_mimov2_layer_operations_strategy_tbo(
        _Layer(), ForwardMode.EXTEND
    )

    yield_indices = [
        index
        for index, operation in enumerate(strategy.operations)
        if isinstance(operation, YieldOperation)
    ]

    assert strategy.deep_gemm_num_sms is None
    assert strategy.tbo_delta_stages == 0
    assert yield_indices == [4, 10]
    assert strategy.operations[3].__name__ == "op_tbo_no_ep_attn_all_reduce_launch"
    assert (
        strategy.operations[5].__name__
        == "op_tbo_no_ep_attn_all_reduce_wait_prepare_mlp"
    )
    assert strategy.operations[9].__name__ == "op_tbo_no_ep_all_reduce_launch"
    assert strategy.operations[11].__name__ == "op_tbo_no_ep_all_reduce_wait"


class _FakeDevice:
    index = 0


class _FakeTensor:
    def __init__(self):
        self.device = _FakeDevice()
        self.recorded_streams = []

    def record_stream(self, stream):
        self.recorded_streams.append(stream)


class _FakeEvent:
    def __init__(self):
        self.recorded_streams = []

    def record(self, stream):
        self.recorded_streams.append(stream)


class _FakeStream:
    def __init__(self, name):
        self.name = name
        self.waited_events = []

    def wait_event(self, event):
        self.waited_events.append(event)


class _FakeGroup:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.inputs = []

    def all_reduce(self, input_tensor):
        self.inputs.append(input_tensor)
        return next(self.outputs)


def test_mimo_no_ep_tbo_reuses_stream_and_events_for_out_of_place_ar():
    two_batch_overlap._tbo_tp_comm_streams.clear()
    two_batch_overlap._tbo_tp_events.clear()
    compute_stream = _FakeStream("compute")
    comm_stream = _FakeStream("comm")
    events = []

    def make_event():
        event = _FakeEvent()
        events.append(event)
        return event

    inputs = [_FakeTensor(), _FakeTensor()]
    outputs = [_FakeTensor(), _FakeTensor()]
    group = _FakeGroup(outputs)

    with (
        patch.object(
            two_batch_overlap.torch.cuda,
            "Stream",
            return_value=comm_stream,
        ) as stream_ctor,
        patch.object(
            two_batch_overlap.torch.cuda,
            "Event",
            side_effect=make_event,
        ) as event_ctor,
        patch.object(
            two_batch_overlap.torch.cuda,
            "device",
            side_effect=lambda *_: nullcontext(),
        ),
        patch.object(
            two_batch_overlap.torch.cuda,
            "stream",
            side_effect=lambda *_: nullcontext(),
        ),
        patch.object(
            two_batch_overlap.torch.cuda,
            "current_stream",
            return_value=compute_stream,
        ),
    ):
        first = two_batch_overlap.launch_tbo_tp_all_reduce(
            inputs[0], group=group, event_key=("mimo_attention", 0)
        )
        assert two_batch_overlap.wait_tbo_tp_all_reduce(first) is outputs[0]
        second = two_batch_overlap.launch_tbo_tp_all_reduce(
            inputs[1], group=group, event_key=("mimo_attention", 0)
        )
        assert two_batch_overlap.wait_tbo_tp_all_reduce(second) is outputs[1]

    stream_ctor.assert_called_once()
    assert event_ctor.call_count == 2
    assert group.inputs == inputs
    assert comm_stream.waited_events == [events[0], events[0]]
    assert events[0].recorded_streams == [compute_stream, compute_stream]
    assert events[1].recorded_streams == [comm_stream, comm_stream]
    assert compute_stream.waited_events == [events[1], events[1]]
    assert inputs[0].recorded_streams == []
    assert inputs[1].recorded_streams == []
    assert outputs[0].recorded_streams == [compute_stream]
    assert outputs[1].recorded_streams == [compute_stream]

    two_batch_overlap._tbo_tp_comm_streams.clear()
    two_batch_overlap._tbo_tp_events.clear()


def test_mimo_no_ep_tbo_in_place_ar_does_not_record_stream():
    two_batch_overlap._tbo_tp_comm_streams.clear()
    two_batch_overlap._tbo_tp_events.clear()
    compute_stream = _FakeStream("compute")
    comm_stream = _FakeStream("comm")
    input_tensor = _FakeTensor()
    group = _FakeGroup([input_tensor])

    with (
        patch.object(two_batch_overlap.torch.cuda, "Stream", return_value=comm_stream),
        patch.object(two_batch_overlap.torch.cuda, "Event", side_effect=_FakeEvent),
        patch.object(
            two_batch_overlap.torch.cuda,
            "device",
            side_effect=lambda *_: nullcontext(),
        ),
        patch.object(
            two_batch_overlap.torch.cuda,
            "stream",
            side_effect=lambda *_: nullcontext(),
        ),
        patch.object(
            two_batch_overlap.torch.cuda,
            "current_stream",
            return_value=compute_stream,
        ),
    ):
        handle = two_batch_overlap.launch_tbo_tp_all_reduce(
            input_tensor, group=group, event_key=("mimo_expert", 0)
        )
        assert two_batch_overlap.wait_tbo_tp_all_reduce(handle) is input_tensor

    assert input_tensor.recorded_streams == []
    two_batch_overlap._tbo_tp_comm_streams.clear()
    two_batch_overlap._tbo_tp_events.clear()


def test_mimo_no_ep_tbo_handle_keeps_out_of_place_input_alive_until_wait():
    two_batch_overlap._tbo_tp_comm_streams.clear()
    two_batch_overlap._tbo_tp_events.clear()
    compute_stream = _FakeStream("compute")
    comm_stream = _FakeStream("comm")
    input_tensor = _FakeTensor()
    input_ref = weakref.ref(input_tensor)
    output_tensor = _FakeTensor()
    group = _FakeGroup([output_tensor])

    with (
        patch.object(two_batch_overlap.torch.cuda, "Stream", return_value=comm_stream),
        patch.object(two_batch_overlap.torch.cuda, "Event", side_effect=_FakeEvent),
        patch.object(
            two_batch_overlap.torch.cuda,
            "device",
            side_effect=lambda *_: nullcontext(),
        ),
        patch.object(
            two_batch_overlap.torch.cuda,
            "stream",
            side_effect=lambda *_: nullcontext(),
        ),
        patch.object(
            two_batch_overlap.torch.cuda,
            "current_stream",
            return_value=compute_stream,
        ),
    ):
        handle = two_batch_overlap.launch_tbo_tp_all_reduce(
            input_tensor, group=group, event_key=("mimo_attention", 1)
        )
        # The fake group records inputs for assertions in another test. Drop
        # that test-only reference so this check isolates the TBO handle.
        group.inputs.clear()
        del input_tensor
        gc.collect()
        assert input_ref() is not None
        assert two_batch_overlap.wait_tbo_tp_all_reduce(handle) is output_tensor
        del handle

    gc.collect()
    assert input_ref() is None
    assert output_tensor.recorded_streams == [compute_stream]
    two_batch_overlap._tbo_tp_comm_streams.clear()
    two_batch_overlap._tbo_tp_events.clear()


class _FakeNoEpBackend:
    @staticmethod
    def is_none():
        return True


def _make_forward_batch(forward_mode=ForwardMode.EXTEND, spec_info=None):
    children = [
        SimpleNamespace(tbo_padded_len=2048, forward_mode=forward_mode),
        SimpleNamespace(tbo_padded_len=2048, forward_mode=forward_mode),
    ]
    return SimpleNamespace(
        can_run_tbo=True,
        tbo_children=children,
        forward_mode=forward_mode,
        global_forward_mode=forward_mode,
        spec_info=spec_info,
    )


def _no_ep_tbo_reason(forward_batch):
    model = SimpleNamespace(pp_group=SimpleNamespace(world_size=1))
    server_args = SimpleNamespace(
        attn_cp_size=1,
        enable_quant_communications=False,
        enable_aiter_allreduce_fusion=False,
    )
    with (
        patch.object(mimo_v2, "get_global_server_args", return_value=server_args),
        patch.object(mimo_v2, "get_tensor_model_parallel_world_size", return_value=8),
        patch.object(mimo_v2, "get_attention_tp_size", return_value=8),
        patch.object(mimo_v2, "get_moe_expert_parallel_world_size", return_value=1),
        patch.object(mimo_v2, "is_dp_attention_enabled", return_value=False),
        patch.object(mimo_v2, "get_moe_a2a_backend", return_value=_FakeNoEpBackend()),
    ):
        return mimo_v2.MiMoV2Model._no_ep_tbo_ineligible_reason(model, forward_batch)


def test_mimo_no_ep_tbo_gate_accepts_only_ordinary_prefill():
    assert _no_ep_tbo_reason(_make_forward_batch()) is None

    target_verify = _make_forward_batch(ForwardMode.TARGET_VERIFY)
    assert "ordinary EXTEND" in _no_ep_tbo_reason(target_verify)

    speculative_extend = _make_forward_batch(spec_info=object())
    assert "speculative" in _no_ep_tbo_reason(speculative_extend)


def test_tbo_forces_scheduler_split_metadata_without_dp_mlp_sync():
    marker = object()
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, spec_info=None)
    adapter = SimpleNamespace(
        server_args=SimpleNamespace(
            enable_two_batch_overlap=True,
            moe_a2a_backend="none",
        ),
        get_require_mlp_sync=lambda: False,
        prepare_mlp_sync_batch=lambda batch: marker,
    )

    result = SchedulerDPAttnAdapter.maybe_prepare_mlp_sync_batch(
        adapter, batch, need_sync=False
    )

    assert result is marker


@pytest.mark.parametrize(
    ("forward_mode", "spec_info"),
    [
        (ForwardMode.DECODE, None),
        (ForwardMode.TARGET_VERIFY, object()),
        (ForwardMode.MIXED, None),
        (ForwardMode.EXTEND, object()),
    ],
)
def test_tbo_does_not_force_scheduler_split_metadata_for_unsupported_no_a2a_batch(
    forward_mode, spec_info
):
    batch = SimpleNamespace(
        forward_mode=forward_mode,
        spec_info=spec_info,
        global_num_tokens=[4096],
        global_num_tokens_for_logprob=[1],
        tbo_split_seq_index=1,
        global_forward_mode=ForwardMode.EXTEND,
        is_extend_in_batch=True,
    )
    adapter = SimpleNamespace(
        server_args=SimpleNamespace(
            enable_two_batch_overlap=True,
            moe_a2a_backend="none",
        ),
        get_require_mlp_sync=lambda: False,
        prepare_mlp_sync_batch=lambda _: pytest.fail(
            "unsupported TP-only TBO batches must not be split"
        ),
    )

    result = SchedulerDPAttnAdapter.maybe_prepare_mlp_sync_batch(
        adapter, batch, need_sync=False
    )

    assert result is batch
    assert batch.global_num_tokens is None
    assert batch.global_num_tokens_for_logprob is None
    assert batch.tbo_split_seq_index is None
    assert batch.global_forward_mode is None
    assert batch.is_extend_in_batch == forward_mode.is_extend()


@pytest.mark.parametrize(
    ("forward_mode", "spec_info"),
    [
        (ForwardMode.DECODE, None),
        (ForwardMode.TARGET_VERIFY, object()),
        (ForwardMode.MIXED, None),
        (ForwardMode.EXTEND, object()),
    ],
)
def test_tbo_split_preparer_rejects_unsupported_no_a2a_batch(forward_mode, spec_info):
    batch = SimpleNamespace(forward_mode=forward_mode, spec_info=spec_info)
    preparer = two_batch_overlap.TboDPAttentionPreparer()

    with (
        patch.object(two_batch_overlap, "is_tbo_enabled", return_value=True),
        patch.object(
            two_batch_overlap,
            "get_moe_a2a_backend",
            return_value=_FakeNoEpBackend(),
        ),
        patch.object(two_batch_overlap, "get_deepep_mode"),
    ):
        can_run_tbo, local_forward_mode = preparer.prepare_all_gather(batch)

    assert not can_run_tbo
    assert local_forward_mode == forward_mode.value
    assert preparer.local_tbo_split_seq_index is None


def test_tbo_does_not_force_scheduler_split_metadata_for_a2a_backend():
    batch = SimpleNamespace()
    adapter = SimpleNamespace(
        server_args=SimpleNamespace(
            enable_two_batch_overlap=True,
            moe_a2a_backend="deepep",
        ),
        get_require_mlp_sync=lambda: False,
        prepare_mlp_sync_batch=lambda _: pytest.fail(
            "A2A TBO must retain the existing need_sync=False behavior"
        ),
    )

    result = SchedulerDPAttnAdapter.maybe_prepare_mlp_sync_batch(
        adapter, batch, need_sync=False
    )

    assert result is batch


@pytest.mark.parametrize(
    ("moe_a2a_backend", "expected"),
    [("none", False), ("deepep", True)],
)
def test_tbo_cuda_graph_policy_depends_on_a2a_backend(moe_a2a_backend, expected):
    server_args = SimpleNamespace(
        enable_two_batch_overlap=True,
        moe_a2a_backend=moe_a2a_backend,
    )

    assert is_tbo_cuda_graph_enabled(server_args) is expected


def test_no_a2a_tbo_keeps_ordinary_decode_capture_sizes():
    server_args = SimpleNamespace(
        enable_two_batch_overlap=True,
        moe_a2a_backend="none",
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(bs=[1, 2, 3, 4]),
        ),
        enable_torch_compile=False,
    )
    model_runner = SimpleNamespace(
        server_args=server_args,
        req_to_token_pool=SimpleNamespace(size=4),
    )

    with (
        patch(
            "sglang.srt.model_executor.runner.base_cuda_graph_runner.require_gathered_buffer",
            return_value=False,
        ),
        patch(
            "sglang.srt.model_executor.runner.base_cuda_graph_runner.get_attention_cp_size",
            return_value=1,
        ),
    ):
        capture_bs, compile_bs = get_batch_sizes_to_capture(
            model_runner, num_tokens_per_bs=5
        )

    assert capture_bs == [1, 2, 3, 4]
    assert compile_bs == []


def _make_decode_graph_runner(enable_tbo_cuda_graph):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.require_mlp_tp_gather = False
    runner.enable_pdmux = False
    runner.disable_padding = False
    runner.max_bs = 4
    runner.require_mlp_sync = False
    runner.is_encoder_decoder = False
    runner.capture_hidden_mode = CaptureHiddenMode.NULL
    runner.enable_tbo_cuda_graph = enable_tbo_cuda_graph
    runner.num_tokens_per_bs = 1
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(
            is_ngram=lambda: False,
        )
    )
    return runner


def _make_decode_graph_forward_batch(forward_mode=ForwardMode.DECODE):
    return SimpleNamespace(
        replace_embeds=None,
        batch_size=1,
        can_run_tbo=False,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        spec_info=None,
        forward_mode=forward_mode,
    )


@pytest.mark.parametrize(
    "forward_mode",
    [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY],
)
def test_no_a2a_tbo_decode_graph_accepts_non_tbo_batch(forward_mode):
    runner = _make_decode_graph_runner(enable_tbo_cuda_graph=False)

    assert runner.can_run(_make_decode_graph_forward_batch(forward_mode))


def test_a2a_tbo_decode_graph_still_requires_tbo_batch():
    runner = _make_decode_graph_runner(enable_tbo_cuda_graph=True)

    assert not runner.can_run(_make_decode_graph_forward_batch())
