# MiMo-V2.6-Pro MI355X Enablement Handoff

Date: 2026-09-22

## Scope and Reference

Enable and qualify `/models/MiMo-V2.6-Pro-RL` on MI355X with TP8/EP1 in two
modes:

1. Native MXFP4 target plus the bundled BF16 DFlash draft.
2. The same native MXFP4 target without speculative decoding.

Use `/models/MiMo-V2.5-Pro-FP4-DFlash` plus DFlash as the implementation and
performance reference. Do not use `/models/MiMo-V2.5-Pro` FP8+MTP as the
primary V2.6 reference.

## Working Trees

- SGLang: `/root/workspace/mimo-opt/sglang`
  - Branch: `mimo-opt-mxfp4-dflash`
  - Integration base: `90542dca9b25fcb38b752a51f00ca6c62df0ead7`
- AITER: `/root/workspace/mimo-opt/aiter-mimo-fp4-dflash`
  - Branch: `integration/mimo-fp4-dflash`
  - Integration base: `c93921971d06a0bc8d6c7214f3ff8e682e20ca78`
- Model: `/models/MiMo-V2.6-Pro-RL`
- Draft: `/models/MiMo-V2.6-Pro-RL/dflash`

Runtime versions:

- PyTorch: `2.9.1+rocm7.2.0.git7e1940d4`
- Transformers: `5.8.1`
- FlyDSL: `0.3.2`
- PyHIP import path: `/opt/venv/lib/python3.10/site-packages/pyhip`

The worktrees contain unrelated local changes. Do not reset, clean, or revert
them. The V2.6 work was committed as small signed commits; generated artifacts
remain untracked.

## Checkpoint Validation

The Pro checkpoint is complete and structurally valid:

- Indexed size: `566,027,990,784` bytes.
- Target shards: `130/130` present.
- Index metadata: `save_format=mxfp4`, `tp_size=8`.
- Architecture: `MiMoV2ForCausalLM`.
- Hidden size/layers: `6144/70`.
- Full attention: 128 Q heads, 8 KV heads, D192/V128.
- SWA: 128 Q heads, 8 KV heads, D192/V128, window 128.
- MoE: 384 routed experts, top-k 8, BF16 router weight and FP32 router output.
- DFlash: five layers, block/qlen 8, window 1024, mask token 151675.

Unlike the Flash checkpoint, the Pro checkpoint requires TP8 and needs no
local JSON repair.

## Integration Commits

SGLang commits after the integration base:

- `9062d50d5e` `fix(moe): align routed-only AITER MXFP4 execution`
- `3a9ed29a13` `fix(aiter): support qlen-8 Gluon verification fallback`
- `2f47de745a` `fix(mimo): support ROCm multimodal startup`

AITER commits after the integration base:

- `a722727a` `perf(moe): optimize routed-only MXFP4 execution`
- `8b8f6d3d` `fix(mega_moe): rebind shared layer weights safely`

The day-0 launchers and documents are in a separate SGLang evaluation commit.

## Remaining Local Changes

The committed V2.6 paths are no longer part of the dirty worktree. Remaining
changes such as the local `pyproject.toml` swap, generated logs, historical
evaluation material, GLM CSV edits, and scratch profiles are intentionally not
part of these commits.

The integration covered these paths:

```text
M  evaluation/launch_tp8_noep_aiter_mtp_accuracy.sh
M  evaluation/mimo_dflash_scripts/launch_tp8_noep_aiter_dflash_accuracy_baseline.sh
M  python/sglang/srt/layers/attention/aiter_utils.py
M  python/sglang/srt/multimodal/processors/mimo_v2.py
?? evaluation/mimo-v2.6-day0/
```

The Pro-specific changes are:

- `launch_mimo_v2.6_pro.sh` validates the target and draft checkpoints and
  invokes the established MiMo-V2.5-Pro MXFP4+DFlash TP8 launcher.
- The DFlash and general MiMo launchers now accept an optional
  `MM_ATTENTION_BACKEND` and pass `--mm-attention-backend` to SGLang.
- The V2.6 Pro wrapper defaults that value to `aiter_attn` on ROCm.
- `mimo_v2.py` reuses the optional `AudioDecoder` import from `mimo_audio.py`.
  Text and image model initialization therefore works without `torchcodec`;
  audio requests still require that package.

The changes in `aiter_utils.py` are for the separate TP4 Flash fallback and do
not affect the validated Pro TP8 path unless
`SGLANG_AITER_VEC5D_TARGET_VERIFY_QLEN1=1` is explicitly set.

## Resolved Startup Issue

The first TP8 Pro attempt failed before loading weights because the multimodal
tower selected `fa3`:

```text
Exception: VisionFlash3Attention is only available for cuda or musa
```

Passing `--mm-attention-backend aiter_attn` resolves the issue. The failed
attempt is retained at:

```text
evaluation/mimo-v2.6-day0/logs/pro_tp8_gsm8k_20260922T022138Z/
```

## Validated DFlash Server

Launch command:

```bash
cd /root/workspace/mimo-opt/sglang
echo 0 > /proc/sys/kernel/numa_balancing
RUN_ID=pro_tp8_dflash \
LOG_DIR="$PWD/evaluation/mimo-v2.6-day0/logs/pro_tp8_dflash" \
./evaluation/mimo-v2.6-day0/launch_mimo_v2.6_pro.sh
```

Validated behavior:

- TP8/EP1 target loaded successfully, approximately 67.83 GiB/GPU.
- BF16 DFlash draft loaded successfully, approximately 1.07 GiB/GPU.
- BF16 target and draft KV, page 64.
- Target full verification: FlyDSL qlen-8 D192/V128.
- Target SWA and draft SWA: FlyDSL/FlyPA.
- Decode graphs captured for batch sizes 1, 2, 4, 8, and 16.
- Trained DFlash mask embedding loaded from the draft checkpoint.
- Real acceptance was used; no acceptance simulation was enabled.

Successful run artifacts:

```text
evaluation/mimo-v2.6-day0/logs/pro_tp8_dflash_gsm8k_20260922T022417Z/
```

## GSM8K Results

Both evaluations used 64 questions, five-shot prompts, greedy decoding,
`max_tokens=2048`, and client concurrency 16.

| Mode | Correct | Score | Latency | Eval output throughput | Acceptance |
|---|---:|---:|---:|---:|---:|
| MXFP4 target + DFlash | 64/64 | 1.000000 | 16.312 s | 809.79 tok/s | 5.12-5.34 during the C14-C16 plateau |
| MXFP4 target only | 63/64 | 0.984375 | 28.168 s | 457.90 tok/s | N/A |

For this short sample, DFlash increased evaluator output throughput by 76.85%
and reduced total latency by 42.09%. The DFlash and target-only runs generated
13,209 and 12,898 completion tokens respectively. The one-question accuracy
difference is not enough to claim a model-quality difference; expand the
sample before making a promotion decision.

GSM8K command:

```bash
cd /root/workspace/mimo-opt/sglang
OPENAI_API_KEY=EMPTY PYTHONPATH="$PWD/python" \
python3 -u -m sglang.test.run_eval \
  --host 127.0.0.1 \
  --port 30000 \
  --model /models/MiMo-V2.6-Pro-RL \
  --eval-name gsm8k \
  --num-examples 64 \
  --num-threads 16 \
  --num-shots 5 \
  --max-tokens 2048
```

Result files:

```text
evaluation/mimo-v2.6-day0/logs/pro_tp8_dflash_gsm8k_20260922T022417Z/gsm8k_64_dflash.json
evaluation/mimo-v2.6-day0/logs/pro_tp8_dflash_gsm8k_20260922T022417Z/gsm8k_64_dflash.html
evaluation/mimo-v2.6-day0/logs/pro_tp8_target_only_gsm8k_20260922T022935Z/gsm8k_64_target_only.json
evaluation/mimo-v2.6-day0/logs/pro_tp8_target_only_gsm8k_20260922T022935Z/gsm8k_64_target_only.html
```

## Target-Only Prefill Sanity Results

Configuration:

- TP8/EP1, native MXFP4 target, no speculative decoding.
- BF16 KV, page 64, chunked prefill 32K.
- TBO off, radix cache off, mixed router off.
- Automatic NUMA balancing disabled during measurement.
- Four warmup requests and one measured request wave per point.

| ISL / OSL | Concurrency | Measured prompts | Completed | Input throughput | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 4K / 1 | 32 | 32 | 32/32 | 23,331.98 tok/s | 4,631.32 ms | 0.00 ms |
| 64K / 1 | 16 | 16 | 16/16 | 43,205.66 tok/s | 13,053.73 ms | 0.00 ms |
| 256K / 1 | 16 | 16 | 16/16 | 29,951.10 tok/s | 74,911.91 ms | 0.00 ms |
| 1,048,000 / 1 | 2 | 2 | 2/2 | 13,213.37 tok/s | 119,813.27 ms | 0.00 ms |

Artifacts:

```text
evaluation/mimo-v2.6-day0/logs/pro_tp8_target_only_prefill_20260922T024107Z/
```

These are sanity values, not full customer-matrix qualification results. The
4K point is particularly startup-sensitive because only 32 measured requests
were used instead of the standard 4,096. TPOT is `0.00 ms` by construction for
these OSL=1 prefill tests because there is no interval between generated
tokens; TTFT is the relevant per-request latency metric.

## Target-Only Launch Recipe

```bash
cd /root/workspace/mimo-opt/sglang
echo 0 > /proc/sys/kernel/numa_balancing

AITER_ROOT=/root/workspace/mimo-opt/aiter-mimo-fp4-dflash \
MODEL=/models/MiMo-V2.6-Pro-RL \
MIMO_TARGET_WEIGHT_FORMAT=mxfp4 \
SPECULATIVE_ALGORITHM=NONE \
MM_ATTENTION_BACKEND=aiter_attn \
HOST=0.0.0.0 PORT=30000 \
MAX_RUNNING_REQUESTS=16 \
MEM_FRACTION_STATIC=0.80 \
SWA_FULL_TOKENS_RATIO=0.01 \
CUDA_GRAPH_BS_DECODE="1 2 4 8 16" \
ENABLE_TWO_BATCH_OVERLAP=0 \
DISABLE_RADIX_CACHE=1 \
SGLANG_MIMO_MIXED_ROUTER=0 \
SGLANG_FLYDSL_MIMO_PREFILL=0 \
SGLANG_FLYPA_MIMO_PREFILL=1 \
./evaluation/launch_tp8_noep_aiter_mtp_accuracy.sh
```

## Pending 64K/20 Profile

Requested profile shape:

- Input/output: 65,536/20.
- Concurrency: 1.
- Measured prompts: 4.
- Warmup prompts: 4.
- Chunked prefill: 32,768.
- Target only, TP8/EP1, BF16 KV, TBO off.

Do not start while another container owns GPU memory. At handoff time all eight
GPUs held approximately 87-88% VRAM and reported active KFD processes whose
PIDs were outside this container's PID namespace. No profiling server or
client was started.

After the GPUs have remained idle and at zero VRAM use, start the target-only
server with the recipe above. Then run:

```bash
cd /root/workspace/mimo-opt/sglang
RUN_ID="mimo_v26_pro_target_only_64k20_c1_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$PWD/evaluation/mimo-v2.6-day0/logs/${RUN_ID}"
mkdir -p "$RUN_DIR/profile"

INPUT_TOKENS=65536 \
OUTPUT_TOKENS=20 \
NUM_PROMPTS=4 \
WARMUP_REQUESTS=4 \
MAX_CONCURRENCY=1 \
MODEL_PATH=/models/MiMo-V2.6-Pro-RL \
HOST=127.0.0.1 \
PORT=30000 \
PROFILE_OUTPUT_DIR="$RUN_DIR/profile" \
PROFILE_PREFIX=mimo_v2_6_pro_mxfp4_target_only_64k20_c1 \
./evaluation/run_sglang_profile.sh 2>&1 | tee "$RUN_DIR/client.log"
```

Verify that the profile directory contains eight `TP-*.trace.json.gz` files,
that no `_compile_impl` event occurs inside the measured range, and that the
server log has no scheduler exception, watchdog timeout, retraction, or GPU
memory fault.

## Known Tuning Follow-Ups

- The V2.6 Pro language shapes match V2.5 Pro MXFP4, so the existing MXFP4 MoE
  rows are reusable.
- Several small DFlash/graph GEMMs such as `M=8/16/32/64/128,
  N=3392,K=6144` and BF16 `N=6144,K=2048` reported no exact tuned row. They
  used fallback kernels. This does not block correctness but is a performance
  optimization opportunity.
- `AITER_CONFIG_FMOE` and both A8W8 GEMM config variables are passed as explicit
  repository paths. Preserve that behavior so stale `/tmp/aiter_configs`
  merges cannot select unrelated GLM rows.
- GSM8K passed, but ShareGPT, long-form accuracy, 64K/1K decode throughput,
  and the full customer prefill matrix remain unqualified for V2.6 Pro.

## Cleanup

After every run:

```bash
# Stop only the owned tmux/server session. Do not terminate foreign KFD PIDs.
echo 1 > /proc/sys/kernel/numa_balancing
rocm-smi --showuse --showmemuse --showpids
```

At handoff, no SGLang server or benchmark session owned by this work is
running. The GPU allocation visible through `rocm-smi` belongs to another
container/workload.
