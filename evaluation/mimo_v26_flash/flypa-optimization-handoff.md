# FlyPA optimization handoff

GPU validation was completed on 2026-10-29 on MI325X/gfx942. The current
machine needs GPU repair; the consolidation check is CPU-only. This
handoff covers tuned BF16 FlyPA and CK linear-KV prefill with gather included.
The integration and supporting files are recorded on branch
`feat/mimo-ck-linear-prefill`; base revision `a15045ff` alone does not contain
them. The AITER/CK dependency patches and file inventory are described in
[the code package](flypa_handoff_code/README.md).
The standalone
[reproducer](repro_flypa_ck_prefill/README.md) bundles the exact kernel sources.

Use the existing container: Torch `2.11.0+rocm7.2`, Triton
`3.7.0+amd.rocm7.2.0.git89002410`, FlyDSL **0.3.2**, and AITER
`2b81433bbff27020ce98271b992a1aab93a5f3db`. The overlay supplies the repaired
ROCm 7.2.4 libraries; `torch.version.hip` still describes the Torch build.
Set up both terminals below, keeping the same `MIMO_RUN` value:

```bash
export MIMO_ROOT="${MIMO_ROOT:-/root/workspace/rocm7.2.4}"
export MIMO_SRC="${MIMO_SRC:-$MIMO_ROOT/sglang-mimo-ck-prefill}"
export MIMO_MAIN="$MIMO_SRC"
export MIMO_ANALYSIS="$MIMO_MAIN/evaluation/mimo_v26_flash/rocm7.2.4-decode-analysis/mi325x_h200_optimization_20261028"
export MIMO_DATASET="${MIMO_DATASET:-$MIMO_ROOT/sglang/evaluation/data/ShareGPT_V3_unfiltered_cleaned_split.json}"
export MIMO_RUN="/tmp/flypa_handoff_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$MIMO_RUN"
export AITER_ROOT="$MIMO_ROOT/aiter"
export PYTHONPATH="$MIMO_ROOT/runtime724/python:$MIMO_SRC/python:$AITER_ROOT"
export SPT_NOENV=1 AITER_USE_SYSTEM_TRITON=1 SGLANG_USE_AITER=1
export SGLANG_DISABLE_CUDNN_CHECK=1 SGLANG_ROCM_DISABLE_LINEARQUANT=0
python "$MIMO_MAIN/evaluation/mimo_v26_flash/check_flypa_handoff_files.py" \
  --repo "$MIMO_SRC" --dependencies --dataset "$MIMO_DATASET"
```

For a relocated checkout, set `MIMO_ROOT` and `MIMO_SRC` before this block
(for example `/data/xisun/rocm7.2.4` and its `sglang` directory). Supporting
scripts, the noVS CSV and the reproducer now live in the consolidated checkout.
The model, ShareGPT dataset, AITER and runtime overlay are external inputs;
the CPU-only checker reports missing files without loading GPU libraries.

Run the standalone comparison and focused tests on an idle GPU:

```bash
bash "$MIMO_MAIN/evaluation/mimo_v26_flash/repro_flypa_ck_prefill/run.sh" \
  --output "$MIMO_RUN/component.json"
python -m pytest -q "$MIMO_SRC/test/manual/test_mimo_ck_linear_prefill.py" \
  --junitxml="$MIMO_RUN/ck-linear.xml"
python "$MIMO_ANALYSIS/test_prefill_candidate.py" \
  --source-root "$MIMO_SRC" --cooperative-v 0 --manual-schedule 0 \
  --cta-multiplier 1 --workgroup-waves 8 --output "$MIMO_RUN/flypa.json"
```

Latest result: four component cases pass, **5 CK integration tests pass with
no skips**, and **11 FlyPA cases pass**. Coverage includes ragged requests,
NaN tails, query padding, changed-input graph replay and stream isolation.
All component output hashes match the original comparison. Fresh speedups
are **1.46× / 1.35× / 1.29× / 1.28×**; the historical values were
1.55× / 1.37× / 1.28× / 1.28×. Keep the existing numerical tolerances.
[Unit artifacts](repro_flypa_ck_prefill/runs/unit_20261029/),
[component artifacts](repro_flypa_ck_prefill/runs/validation_20261029/).

For E2E, use eight GPUs and the existing DP-attention launcher. In terminal 1:

```bash
export MODEL_PATH=/models/MiMo-V2.6-Flash-MOPD/
export AITER_CONFIG_FMOE="$MIMO_MAIN/evaluation/mimo_v26_flash/configs/moe_novs_gfx942_rocm724.csv"
export ENABLE_DFLASH=1 ENABLE_DP_ENCODER=0
export MEM_FRACTION_STATIC=0.60 SWA_FULL_TOKENS_RATIO=0.03
export CHUNKED_PREFILL_SIZE=32768 MAX_RUNNING_REQUESTS=192 CUDA_GRAPH_MAX_BS_DECODE=96
export SGLANG_USE_ROCM700A=1 SGLANG_MIMO_FUSED_STABLE_ROUTER_WEIGHTS=1
export SGLANG_DFLASH_SWA_TOKEN_INTERVAL=1 SGLANG_FLYDSL_PA_TUNED_DECODE=1
export SGLANG_FLYPA_MIMO_BF16_MANUAL_SCHEDULE=0
export SGLANG_MIMO_SHARDED_FP8_PREFILL=0 SGLANG_MIMO_SHARDED_FP8_DECODE=0
export SGLANG_DP_FUSED_BUFFER_COPY=0 SGLANG_DP_HIERARCHICAL_COMM=0
export NCCL_MIN_NCHANNELS=112
unset NCCL_MAX_NCHANNELS

SGLANG_MIMO_CK_LINEAR_PREFILL=0 LOG_DIR="$MIMO_RUN/flypa_server" \
  bash "$MIMO_SRC/evaluation/mimo_v26_flash/launch_mimo_v26_flash_dp_attention.sh"
```

This is the tuned FlyPA control. After its tests, stop the foreground launcher
with Ctrl-C and start the same command with `SGLANG_MIMO_CK_LINEAR_PREFILL=1`
and `LOG_DIR="$MIMO_RUN/ck_server"`. Keep all other settings fixed. The CK log
must contain `Using experimental CK linear-KV BF16 full prefill on gfx942`.
The layout is global TP8, attention DP2×TP4, EP1, BF16 KV and real DFlash8.
Global chunk32768 resolves to16384/group. To reproduce the older E2E setting
with32768/group, use global65536 in **both** arms.

In terminal 2, use `MIMO_ARM=flypa` for the control and `MIMO_ARM=ck` for CK:

```bash
export MIMO_ARM=flypa
curl -fsS http://127.0.0.1:30001/health
curl -fsS http://127.0.0.1:30001/server_info > "$MIMO_RUN/${MIMO_ARM}_server-info.json"

MODEL_PATH=/models/MiMo-V2.6-Flash-MOPD/ \
BASE_URL=http://127.0.0.1:30001/v1 LIMIT=all FLUSH_CACHE=1 \
MIN_GSM8K_ACCURACY=0.97 MIN_ACCEPT_LENGTH=4.0 \
OUTPUT_DIR="$MIMO_RUN/${MIMO_ARM}_gsm8k" \
  python "$MIMO_MAIN/evaluation/run_evalscope_gsm8k_accuracy.py"

cd "$MIMO_MAIN/evaluation"
BENCHMARK_PRESET=mimo_v26_flash TOKEN_LIST_OVERRIDE='8192 65536' \
MODEL=/models/MiMo-V2.6-Flash-MOPD/ HOST=127.0.0.1 PORT=30001 \
WARMUP_REQUESTS_OVERRIDE=32 TOKENIZE_PROMPT=1 \
LOG_DIR="$MIMO_RUN/${MIMO_ARM}_prefill_r1" \
  bash run_benchmark_mimo_pro_prefill.sh \
  --dataset-path "$MIMO_DATASET" \
  --output-details --output-file "$MIMO_RUN/${MIMO_ARM}_prefill_r1.jsonl"
```

The client uses 8K/1 N1024 C32 and64K/1 N256 C32. Repeat with fresh `r2`
output names; confirm with reversed arm order. Require full1319-sample GSM8K
≥97%, zero API errors, worker acceptance≥4, exact input/output lengths and
complete requests. Compare **unprofiled input throughput**; component speedup
is not E2E speedup. Add `DRY_RUN=1` to either shell launcher/client command to
check arguments without starting a server or issuing benchmark requests.

E2E commands were checked in dry-run mode when this handoff was created;
the current consolidation checks do not rerun GPU tests. Prior CK results
are included as [the serving comparison](flypa_handoff_assets/ck-serving-comparison.json)
and [the reverse-order repeat](flypa_handoff_assets/ck-serving-repeat.json).
After relocation, rerun the unit and E2E gates on a healthy node.
