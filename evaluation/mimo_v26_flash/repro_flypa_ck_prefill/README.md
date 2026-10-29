# Minimal FlyPA versus CK prefill reproducer

One GPU, no model weights, server, distributed initialization or SGLang import.
The bundle includes the qualified FlyPA/helper sources and the unchanged KV
gather functions; Torch, Triton, FlyDSL and AITER remain external dependencies.

In the existing workspace:

```bash
cd /root/workspace/rocm7.2.4/sglang
bash evaluation/mimo_v26_flash/repro_flypa_ck_prefill/run.sh
```

The wrapper selects the verified ROCm overlay and local AITER checkout. With
those dependencies already importable elsewhere, run `python benchmark.py`
directly with `AITER_USE_SYSTEM_TRITON=1 SPT_NOENV=1`. The tested environment is
Torch 2.11.0+rocm7.2, Triton 3.7.0+amd.rocm7.2.0.git89002410, FlyDSL **0.3.2**,
AITER `2b81433bbff27020ce98271b992a1aab93a5f3db`, and MI325X/gfx942 with the
ROCm 7.2.4 runtime overlay. Source origins/hashes are in
[source-manifest.json](source-manifest.json).

To run just the first shape, or collect more timing samples:

```bash
bash evaluation/mimo_v26_flash/repro_flypa_ck_prefill/run.sh --case 32768 32768 --samples 10
```

`--device` defaults to 0. Each invocation creates a fresh timestamped
`runs/.../results.json`; `--output PATH` selects an unused output filename.

The comparison preserves the original contract:

- BF16 Q/K/V, one sequence, 16 query heads, one KV head, QK192/V128, page64.
- Q is a view of fused QKV storage, with strides `(3392, 192, 1)`.
- Bottom-right causal mask: key position ≤ `KV - Q + query_position`.
- Tuned FlyPA uses eight waves, one CTA multiplier, `manual_schedule=False`
  and `cooperative_v=False` on the SHUFFLE-5D cache.
- CK gathers **all KV tokens on every timed call**, then invokes native
  `mha_varlen_fwd` with QK192/V128. There is no D256 padding or cached-gather bypass.
- `min_seqlen_q=0` is essential: it is a skip threshold, not the query length.

Five alternating, warmed graph replays per backend are timed with CUDA events.
Compilation, allocation setup and numerical checks are outside timing. CK's
gather kernels are inside its graph. Python cyclic GC is disabled during timing.
The historical table timed the backends in separate invocations; exact elapsed
times can differ with timing order and hardware state.

Fresh validation on 2026-10-29:

| Query tokens | KV tokens | FlyPA ms | CK including gather ms | Speedup |
|---:|---:|---:|---:|---:|
|32768|32768|13.187|9.014|1.463×|
|32768|65536|36.575|27.125|1.348×|
|32768|262144|170.265|132.404|1.286×|
|18432|378880|146.346|114.267|1.281×|

All eight full-output hashes—four shapes × two backends—match the historical
results exactly. Gather equality, finite full outputs, sampled FP32 references
at the unchanged `atol=0.02, rtol=0.02`, eager/graph equality and changed-query
replay checks pass. The reference checks eight query rows, not every output row.

[Fresh results](runs/validation_20261029/results.json),
[verification](runs/validation_20261029/verification.json),
[historical FlyPA results](reference-results/prefill-shape-matrix-manual0.json),
[historical CK results](reference-results/prefill-ck-varlen.json).

These are component timings; model throughput also includes projections,
other attention layers, MoE, communication and scheduling. For the existing
integration/unit/E2E workflow, see [the handoff](../flypa-optimization-handoff.md).
