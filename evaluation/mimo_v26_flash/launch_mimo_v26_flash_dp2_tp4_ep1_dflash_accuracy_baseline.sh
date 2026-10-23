#!/usr/bin/env bash
set -euo pipefail

# MiMo-V2.6: convert checkpoint MXFP4 experts once at load time to FP8
# 128x128 blockscale weights. Activations remain dynamically quantized.
# Requires AITER's aiter.ops.flydsl.mxfp4_to_fp8_128block (PR #17),
# FlyDSL 0.3.2, and the CK MiMo prefill patch including gfx942 bounds fixes
# from AITER PR #18. Rebuild cached CK attention modules after patching CK.
# This launcher never applies patches or changes the installed packages.
#
# The Flash baseline uses TP4 x DP2 with DFlash block 8.
# Additional server options may be passed as positional arguments.
# DRY_RUN=1 prints the command
# AITER_ROOT may select a source checkout; otherwise use the installed AITER.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.6-Flash-MOPD/}"
export SPECULATIVE_DRAFT_MODEL_PATH="${SPECULATIVE_DRAFT_MODEL_PATH:-${MODEL_PATH%/}/dflash}"
export DFLASH_BLOCK_SIZE="${DFLASH_BLOCK_SIZE:-8}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-30001}"
if [[ -n "${AITER_ROOT:-}" ]]; then
  export PYTHONPATH="${AITER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

export SGLANG_MXFP4_TO_FP8_128BLOCK=1
export SGLANG_INT4_WEIGHT=0
# MiMo's encoder otherwise defaults to a CUDA-only attention implementation.
export MM_ATTENTION_BACKEND=aiter_attn
# gfx942 CK LINEAR prefill has no single-token generation-health kernel.
# Use /health for process readiness; validate generation with real prompts.
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_ENABLE_ASYNC_ASSERT=1

export SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1
export MC_GID_INDEX=3
export MC_TE_METRIC=1
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-112}"
export SGLANG_SPEC_NAN_DETECTION=1
export SGLANG_SPEC_OOB_DETECTION=1
export SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY="${SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY:-1}"
export SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE=1
# Baseline contract: never inherit the lossy Quick Reduce path or mixed-router
# optimization from the caller. Ablations must use a separate launcher.
unset ROCM_QUICK_REDUCE_QUANTIZATION
export SGLANG_MIMO_MIXED_ROUTER=0

# The local FlyPA prefill path is incompatible with FlyDSL 0.3.2.
export SGLANG_FLYPA_MIMO_PREFILL=0
export SGLANG_FLYDSL_MIMO_PREFILL=0

export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d
export SGLANG_AITER_UNIFIED_VERIFY=1
# Flash TP4 has two local draft KV heads. FlyDSL uses BF16 SHUFFLE-5D
# draft KV; the Gluon fallback retains the existing NHD draft layout.
export SGLANG_USE_AITER_UNIFIED_ATTN=1
# Set SGLANG_AITER_PA_DECODE_IMPL=gluon to restore the Gluon target and NHD draft.
export SGLANG_AITER_PA_DECODE_IMPL="${SGLANG_AITER_PA_DECODE_IMPL:-flydsl}"
export SGLANG_AITER_TARGET_VERIFY_SWA_IMPL="${SGLANG_AITER_TARGET_VERIFY_SWA_IMPL:-${SGLANG_AITER_PA_DECODE_IMPL}}"
export SGLANG_AITER_DFLASH_SWA_IMPL="${SGLANG_AITER_DFLASH_SWA_IMPL:-${SGLANG_AITER_PA_DECODE_IMPL}}"
# Retain qlen-1 expansion for the Gluon fallback. FlyDSL consumes Qlen8
# directly and uses native V128 for both Flash TP4 target KV pools.
export SGLANG_AITER_VEC5D_TARGET_VERIFY_QLEN1=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export SGLANG_FLYDSL_PA_NUM_PARTITIONS="${SGLANG_FLYDSL_PA_NUM_PARTITIONS:-16}"
export CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-full}"
export CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-}"
export REASONING_PARSER="${REASONING_PARSER:-mimo}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
export TOKENIZER_WORKER_NUM="${TOKENIZER_WORKER_NUM:-1}"
# Optional reproducibility control; unset preserves SGLang's seed selection.
export SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-}"
# Flash TP4's separate draft KV pool adds roughly 59% to target KV storage.
# Reserve space for both pools, graphs, and temporary activations on MI325X.
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.03}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export KV_CACHE_DTYPE=bf16
export DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
export ENABLE_TWO_BATCH_OVERLAP="${ENABLE_TWO_BATCH_OVERLAP:-0}"
export SGLANG_TBO_MIM_SEQ_LEN="${SGLANG_TBO_MIM_SEQ_LEN:-2000}"

if ! [[ "${SGLANG_TBO_MIM_SEQ_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SGLANG_TBO_MIM_SEQ_LEN must be a positive integer, observed '${SGLANG_TBO_MIM_SEQ_LEN}'" >&2
  exit 2
fi
if ! [[ "${TOKENIZER_WORKER_NUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TOKENIZER_WORKER_NUM must be a positive integer" >&2
  exit 2
fi
if ! [[ "${DFLASH_BLOCK_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DFLASH_BLOCK_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ -n "${SERVER_RANDOM_SEED}" ]] && ! [[ "${SERVER_RANDOM_SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SERVER_RANDOM_SEED must be a non-negative integer or unset" >&2
  exit 2
fi

export SGLANG_MIMO_FUSED_RMS_MOE_QUANT=0
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT=0
export SGLANG_AITER_MIMO_FRESH_BF16_ASM=0
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN=0
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN=0

# Real DFlash acceptance for accuracy validation.
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-${SGLANG_ROOT}/evaluation/logs/mimo_v2.6_static_fp8_${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-server_tp8_mimo_v2.6_static_fp8_accuracy.log}"

echo "MiMo-V2.6: static FP8 blockscale MoE weights converted from MXFP4 at load time"
echo "Attention: full target-verify=${SGLANG_AITER_PA_DECODE_IMPL}; SWA target-verify=${SGLANG_AITER_TARGET_VERIFY_SWA_IMPL}; BF16 draft=${SGLANG_AITER_DFLASH_SWA_IMPL}"

echo "Configuration: tp=4, dp=2, max-running=${MAX_RUNNING_REQUESTS}, page=64, chunked-prefill=${CHUNKED_PREFILL_SIZE}, ep=1, partitions=${SGLANG_FLYDSL_PA_NUM_PARTITIONS}, mem=${MEM_FRACTION_STATIC}, swa=${SWA_FULL_TOKENS_RATIO}, kv-cache-dtype=${KV_CACHE_DTYPE}, quick-ar=${ROCM_QUICK_REDUCE_QUANTIZATION:-unset}, nccl-min-channels=${NCCL_MIN_NCHANNELS}, mixed-router=${SGLANG_MIMO_MIXED_ROUTER}, fused-rms-moe=${SGLANG_MIMO_FUSED_RMS_MOE_QUANT}, fused-rms-qkv=${SGLANG_MIMO_FUSED_RMS_QKV_QUANT}, fresh-bf16-asm=${SGLANG_AITER_MIMO_FRESH_BF16_ASM}, fresh-bf16-varlen=${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN}, fresh-bf16-swa-varlen=${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN}, speculative=DFLASH, dflash-block=${DFLASH_BLOCK_SIZE}, aiter-ar-fusion=0, decode-graph=${CUDA_GRAPH_BACKEND_DECODE}, decode-graph-bs=${CUDA_GRAPH_BS_DECODE:-default}, reasoning-parser=${REASONING_PARSER}, tbo=${ENABLE_TWO_BATCH_OVERLAP}, tbo-min-isl=${SGLANG_TBO_MIM_SEQ_LEN}, overlap=enabled"
echo "Server log: ${LOG_DIR}/${LOG_FILE}"
echo "Tokenizer workers: ${TOKENIZER_WORKER_NUM}; server seed: ${SERVER_RANDOM_SEED:-auto}"
echo "Speculation: DFLASH, block=${DFLASH_BLOCK_SIZE}, draft=${SPECULATIVE_DRAFT_MODEL_PATH}, draft-attention=aiter/gluon, draft-kv=bf16, draft-window=1024, acceptance=real"


cuda_graph_args=(--cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}")
seed_args=()
if [[ -n "${SERVER_RANDOM_SEED}" ]]; then
  seed_args+=(--random-seed "${SERVER_RANDOM_SEED}")
fi
if [[ -n "${CUDA_GRAPH_BS_DECODE}" ]]; then
  read -r -a cuda_graph_bs <<< "${CUDA_GRAPH_BS_DECODE}"
  cuda_graph_args+=(--cuda-graph-bs-decode "${cuda_graph_bs[@]}")
fi

custom_all_reduce_args=()
custom_all_reduce_status=enabled
if [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]]; then
  custom_all_reduce_args+=(--disable-custom-all-reduce)
  custom_all_reduce_status=disabled
fi

echo "Custom all-reduce: ${custom_all_reduce_status}"

tbo_args=()
if [[ "${ENABLE_TWO_BATCH_OVERLAP}" == "1" ]]; then
  tbo_args+=(--enable-two-batch-overlap)
fi

kv_cache_args=()
if [[ -n "${KV_CACHE_DTYPE}" && "${KV_CACHE_DTYPE}" != "auto" ]]; then
  kv_cache_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

radix_cache_args=()
radix_cache_status=enabled
if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  radix_cache_args+=(--disable-radix-cache)
  radix_cache_status=disabled
fi

echo "Radix cache: ${radix_cache_status}"
echo "HIP non-greedy EAGLE verifier: ${SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY}"

mm_attention_args=()
if [[ -n "${MM_ATTENTION_BACKEND:-}" ]]; then
  mm_attention_args+=(--mm-attention-backend "${MM_ATTENTION_BACKEND}")
fi

server_cmd=("${PYTHON_BIN}" -u -m sglang.launch_server \
  "${mm_attention_args[@]}" \
  --model-path "${MODEL_PATH}" \
  --tp-size 4 \
  --dp-size 2 \
  --ep-size 1 \
  --moe-a2a-backend none \
  --moe-runner-backend aiter \
  --speculative-moe-a2a-backend none \
  --speculative-moe-runner-backend aiter \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "${SPECULATIVE_DRAFT_MODEL_PATH}" \
  --speculative-dflash-block-size "${DFLASH_BLOCK_SIZE}" \
  --speculative-draft-attention-backend aiter \
  --speculative-draft-kv-cache-dtype bf16 \
  --speculative-draft-window-size 1024 \
  --tokenizer-worker-num "${TOKENIZER_WORKER_NUM}" \
  "${seed_args[@]}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --trust-remote-code \
  --reasoning-parser "${REASONING_PARSER}" \
  --tool-call-parser mimo \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
  --context-length 1048576 \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --max-prefill-tokens 1048576 \
  --attention-backend aiter \
  "${kv_cache_args[@]}" \
  --page-size 64 \
  "${tbo_args[@]}" \
  "${radix_cache_args[@]}" \
  "${cuda_graph_args[@]}" \
  "${custom_all_reduce_args[@]}" \
  "$@" \
)

printf 'Command:'
printf ' %q' "${server_cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

mkdir -p -- "${LOG_DIR}"
"${server_cmd[@]}" 2>&1 | tee "${LOG_DIR}/${LOG_FILE}"
