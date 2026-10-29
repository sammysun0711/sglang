"""Single-GPU BF16 FlyPA versus CK varlen, including every CK KV gather."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics

import torch
from aiter.ops.mha import mha_varlen_fwd
from kernels.flydsl_pa import PagedAttention
from kernels.gather import launch_gather_shuffle_5d_to_linear

HERE = Path(__file__).resolve().parent
CASES = [(32768, 32768), (32768, 65536), (32768, 262144), (18432, 378880)]
H, HK, DQ, DV, PAGE = 16, 1, 192, 128, 64


def output_hash(tensor):
    return hashlib.sha256(tensor.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def one_case(q_len, kv_len, samples):
    torch.manual_seed(12345)
    pages = (kv_len + PAGE - 1) // PAGE
    k = torch.randn(pages, PAGE, HK, DQ, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(pages, PAGE, HK, DV, device="cuda", dtype=torch.bfloat16)
    kc = k.view(pages, PAGE, HK, DQ // 8, 8).permute(0, 2, 3, 1, 4).contiguous()
    vc = v.view(pages, PAGE // 8, 8, HK, DV).permute(0, 3, 1, 4, 2).contiguous()
    fused_qkv = torch.randn(q_len, H * DQ + DQ + DV, device="cuda", dtype=torch.bfloat16)
    q = fused_qkv[:, :H * DQ].view(q_len, H, DQ)  # Preserve the original strided Q.
    outputs = {name: torch.full((q_len, H, DV), float("nan"), device="cuda", dtype=q.dtype)
               for name in ("flypa", "ck_with_gather")}
    cuq = torch.tensor([0, q_len], device="cuda", dtype=torch.int32)
    cuk = torch.tensor([0, kv_len], device="cuda", dtype=torch.int32)
    indptr = torch.tensor([0, pages], device="cuda", dtype=torch.int32)
    indices = torch.arange(pages + 256, device="cuda", dtype=torch.int32)
    indices[pages:].zero_()
    tail = torch.tensor([(kv_len - 1) % PAGE + 1], device="cuda", dtype=torch.int32)
    one = torch.ones(1, device="cuda")
    slots = torch.arange(kv_len, device="cuda", dtype=torch.int64)
    workspace = {}
    flypa = PagedAttention(H, HK, DQ, DV, PAGE, True, quant_query_mode="per-tensor",
                          cooperative_v=False, manual_schedule=False,
                          cta_multiplier=1, workgroup_waves=8)

    def run_flypa():
        flypa(q, kc, vc, cuq, cuk, indptr, indices, q_len, kv_len, True,
              one, one, one, tail, out=outputs["flypa"])

    def run_ck():
        # Keep the gather inside this callable and inside the captured graph.
        kl, vl = launch_gather_shuffle_5d_to_linear(kc, vc, slots, workspace=workspace)
        result = mha_varlen_fwd(q, kl, vl, cuq, cuk, q_len, kv_len,
            0, 0.0, DQ**-0.5, 0.0, False, True, -1, -1, 0,
            False, False, out=outputs["ck_with_gather"])
        # The first zero above is min_seqlen_q: a skip threshold. q_len would skip all work.
        assert result[0].data_ptr() == outputs["ck_with_gather"].data_ptr()

    kl, vl = launch_gather_shuffle_5d_to_linear(kc, vc, slots, workspace=workspace)
    torch.testing.assert_close(kl, k.reshape(-1, HK, DQ)[:kv_len], atol=0, rtol=0)
    torch.testing.assert_close(vl, v.reshape(-1, HK, DV)[:kv_len], atol=0, rtol=0)
    rows = torch.tensor(sorted({i for i in (0, 1, 31, q_len // 4, q_len // 2,
                                           q_len - 33, q_len - 2, q_len - 1)
                                if 0 <= i < q_len}), device="cuda")

    def reference():
        scores = (q[rows].float().reshape(-1, DQ) @ k.reshape(-1, DQ)[:kv_len].float().T)
        scores = scores.view(len(rows), H, kv_len) * DQ**-0.5
        visible = kv_len - q_len + rows + 1  # Bottom-right causal alignment.
        scores.masked_fill_(torch.arange(kv_len, device="cuda")[None, None, :]
                            >= visible[:, None, None], -float("inf"))
        return torch.softmax(scores, -1).reshape(-1, kv_len) @ v.reshape(-1, DV)[:kv_len].float()

    def check(name, expected):
        output = outputs[name]
        assert torch.isfinite(output).all(), f"{name}: unwritten/nonfinite output"
        actual = output[rows].float().reshape(-1, DV)
        torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
        return dict(max_abs_error=(actual - expected).abs().max().item(),
                    relative_rms=((actual - expected).square().mean() / expected.square().mean()).sqrt().item())

    functions = {"flypa": run_flypa, "ck_with_gather": run_ck}
    graphs, initial_hashes, errors = {}, {}, {}
    expected = reference()
    for name, run in functions.items():
        run()
        errors[name] = check(name, expected)
        initial_hashes[name] = output_hash(outputs[name])
        for _ in range(2):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        torch.cuda.synchronize()
        assert output_hash(outputs[name]) == initial_hashes[name]
        graphs[name] = graph
    for _ in range(3):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    timings = {name: [] for name in graphs}
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for sample in range(samples):
            order = list(graphs) if sample % 2 == 0 else list(reversed(graphs))
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                timings[name].append(start.elapsed_time(end))
    finally:
        if gc_enabled:
            gc.enable()
    for name in graphs:
        assert output_hash(outputs[name]) == initial_hashes[name]
    q.neg_()
    expected = reference()
    replay_errors = {}
    for name, graph in graphs.items():
        outputs[name].fill_(float("nan"))
        graph.replay()
        replay_errors[name] = check(name, expected)
        if kv_len > 1:
            assert output_hash(outputs[name]) != initial_hashes[name]
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return dict(query_tokens=q_len, kv_tokens=kv_len, q_stride=list(q.stride()),
                flypa_ms=medians["flypa"], ck_with_gather_ms=medians["ck_with_gather"],
                speedup=medians["flypa"] / medians["ck_with_gather"], samples_ms=timings,
                initial_errors=errors, changed_query_replay_errors=replay_errors,
                initial_output_sha256=initial_hashes, checked_query_rows=rows.cpu().tolist(),
                reference_atol=.02, reference_rtol=.02, gather_bit_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, type=int, metavar=("QUERY", "KV"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert args.samples > 0
    torch.cuda.set_device(args.device)
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    properties = torch.cuda.get_device_properties(args.device)
    assert properties.gcnArchName.startswith("gfx942"), properties.gcnArchName
    cases = [tuple(args.case)] if args.case else CASES
    assert all(0 < q <= kv for q, kv in cases)
    output = args.output or HERE / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), f"Preserve the previous result; choose a new --output: {output}"
    versions = {name: importlib.metadata.version(name) for name in ("torch", "triton", "flydsl")}
    result = dict(status="running", gpu=properties.name, arch=properties.gcnArchName,
        device=args.device, versions=versions, torch_hip_build=torch.version.hip,
        query_heads=H, kv_heads=HK, qk_dim=DQ, v_dim=DV, page_size=PAGE, dtype="bfloat16",
        flypa=dict(manual_schedule=False, workgroup_waves=8, cooperative_v=False, cta_multiplier=1),
        timing="Alternating warmed HIP graph replays; CUDA events; CK includes full KV gather on each replay",
        source_manifest_sha256=hashlib.sha256((HERE / "source-manifest.json").read_bytes()).hexdigest(),
        records=[])
    print(f"{properties.name}, {properties.gcnArchName}; {versions}", flush=True)
    print("Query tokens | KV tokens | FlyPA ms | CK incl. gather ms | Speedup", flush=True)
    for q, kv in cases:
        row = one_case(q, kv, args.samples)
        result["records"].append(row)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{q:12d} | {kv:9d} | {row['flypa_ms']:8.3f} | {row['ck_with_gather_ms']:18.3f} | {row['speedup']:.3f}x", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    result["status"] = "complete"
    result["loaded_rocm_libraries"] = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if path.startswith("/") and any(name in path for name in ("libamdhip64.so", "libhsa-runtime64.so", "libroctracer64.so")):
            result["loaded_rocm_libraries"][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
