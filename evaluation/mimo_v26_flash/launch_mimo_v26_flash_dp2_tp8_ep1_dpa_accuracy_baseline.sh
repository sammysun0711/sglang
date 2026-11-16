#!/usr/bin/env bash
set -euo pipefail

# Eight GPUs, global MoE TP8, two attention TP4 groups, FP8 experts by default.
# DFlash is enabled by default; ENABLE_DFLASH=0 selects target-only serving.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.6-Flash-MOPD/}"
export PYTHONPATH="${SGLANG_ROOT}/python:${AITER_ROOT:-/root/workspace/aiter}${PYTHONPATH:+:${PYTHONPATH}}"

export SGLANG_MXFP4_TO_FP8_128BLOCK="${SGLANG_MXFP4_TO_FP8_128BLOCK:-1}"
case "${SGLANG_MXFP4_TO_FP8_128BLOCK}" in
  0)
    moe_weight_description="native MXFP4 experts"
    # Native A16W4/A8W4 expects BF16 input, not fused per-block FP8 activations.
    export SGLANG_MIMO_FUSED_RMS_MOE_QUANT=0
    ;;
  1) moe_weight_description="converted FP8 experts" ;;
  *) echo "SGLANG_MXFP4_TO_FP8_128BLOCK must be 0 or 1" >&2; exit 2 ;;
esac
export SGLANG_INT4_WEIGHT=0
export SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-112}"
export SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE=1
unset ROCM_QUICK_REDUCE_QUANTIZATION
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD
export SGLANG_MIMO_MIXED_ROUTER=0
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_ENABLE_ASYNC_ASSERT=1
export SGLANG_FUSED_ASYNC_OOB_CHECKS=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
export SGLANG_ENABLE_MIMO_SWA_SP="${SGLANG_ENABLE_MIMO_SWA_SP:-1}"
export SGLANG_USE_PYHIP_FP8_BLOCKSCALE_GEMM="${SGLANG_USE_PYHIP_FP8_BLOCKSCALE_GEMM:-1}"
export SGLANG_USE_PYHIP_MIMO_PREFILL="${SGLANG_USE_PYHIP_MIMO_PREFILL:-1}"
export FLYDSL_RUNTIME_ENABLE_CACHE="${FLYDSL_RUNTIME_ENABLE_CACHE:-1}"
export FLYDSL_AUTOTUNE="${FLYDSL_AUTOTUNE:-0}"
export SGLANG_FLYPA_MIMO_PREFILL="${SGLANG_FLYPA_MIMO_PREFILL:-1}"
export SGLANG_FLYDSL_MIMO_PREFILL=0
export SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d
# Use paged metadata for the draft too; avoids its unused legacy PA workspace.
export SGLANG_USE_AITER_UNIFIED_ATTN="${SGLANG_USE_AITER_UNIFIED_ATTN:-1}"
export SGLANG_AITER_UNIFIED_VERIFY="${SGLANG_AITER_UNIFIED_VERIFY:-1}"
export SGLANG_AITER_PA_DECODE_IMPL="${SGLANG_AITER_PA_DECODE_IMPL:-flydsl}"
# Both attention families must select FlyDSL to allocate native V128 caches.
export SGLANG_AITER_TARGET_VERIFY_SWA_IMPL="${SGLANG_AITER_TARGET_VERIFY_SWA_IMPL:-${SGLANG_AITER_PA_DECODE_IMPL}}"
export SGLANG_FLYDSL_PA_NUM_PARTITIONS="${SGLANG_FLYDSL_PA_NUM_PARTITIONS:-16}"
export SGLANG_MIMO_FUSED_RMS_MOE_QUANT="${SGLANG_MIMO_FUSED_RMS_MOE_QUANT:-1}"
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT="${SGLANG_MIMO_FUSED_RMS_QKV_QUANT:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM=0
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN=0
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN=0

# DPA divides both limits by DP2: 16K prefill and 48 requests/group.
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"
CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-full}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-${SGLANG_ROOT}/evaluation/logs/mimo_dp_attention_${RUN_ID}}"
ENABLE_DFLASH="${ENABLE_DFLASH:-1}"
dflash_args=()
case "${ENABLE_DFLASH}" in
  0) ;;
  1)
    # Five draft KV layers mirror full target slots and are included in sizing.
    export SGLANG_AITER_DFLASH_SWA_IMPL="${SGLANG_AITER_DFLASH_SWA_IMPL:-flydsl}"
    export SGLANG_DFLASH_FUSED_COMPACT_KV="${SGLANG_DFLASH_FUSED_COMPACT_KV:-1}"
    export SGLANG_SPEC_NAN_DETECTION=1
    export SGLANG_SPEC_OOB_DETECTION=1
    dflash_args=(
      --speculative-algorithm DFLASH
      --speculative-draft-model-path "${MODEL_PATH%/}/dflash"
      --speculative-dflash-block-size 8
      --speculative-draft-attention-backend aiter
      --speculative-draft-kv-cache-dtype bf16
      --speculative-draft-window-size 1024
      --speculative-moe-a2a-backend none
      --speculative-moe-runner-backend aiter
    )
    ;;
  *) echo "ENABLE_DFLASH must be 0 or 1" >&2; exit 2 ;;
esac

prefill_args=()
case "${ENABLE_PREFILL_DELAYER:-1}" in
  1)
    prefill_args+=(
      --enable-prefill-delayer
      --prefill-delayer-max-delay-passes "${PREFILL_DELAYER_MAX_DELAY_PASSES:-100}"
    )
    ;;
  0) ;;
  *) echo "ENABLE_PREFILL_DELAYER must be 0 or 1" >&2; exit 2 ;;
esac
# MiMo vision stays within each attention TP4 replica.

server_cmd=("${PYTHON_BIN}" -u -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --skip-server-warmup
  --tp-size 8 --dp-size 2 --ep-size 1
  --enable-dp-attention --enable-dp-lm-head
  --moe-a2a-backend none --moe-runner-backend aiter
  --moe-dense-tp-size 1
  --tokenizer-worker-num "${TOKENIZER_WORKER_NUM:-8}"
  --tokenizer-backend "${TOKENIZER_BACKEND:-huggingface}"
  --disable-tokenizer-batch-decode
  --attention-backend aiter --mm-attention-backend aiter_attn
  --kv-cache-dtype bf16 --page-size 64
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --max-prefill-tokens 1048576 --context-length 1048576
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO:-0.03}"
  --disable-radix-cache
  --cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}"
  --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE:-96}"
  --random-seed "${SERVER_RANDOM_SEED:-12345}"
  --host "${SERVER_HOST:-127.0.0.1}" --port "${SERVER_PORT:-30001}"
  --trust-remote-code --reasoning-parser mimo --tool-call-parser mimo
  "${dflash_args[@]}"
  "${prefill_args[@]}"
  "$@"
)

echo "Experimental MiMo DP attention: global TP8 / attention DP2 x TP4 / EP1"
echo "${moe_weight_description}; DP LM head; DFlash=${ENABLE_DFLASH}; vision attention TP=4; TBO disabled"
echo "Requested chunk=${CHUNKED_PREFILL_SIZE}, requests=${MAX_RUNNING_REQUESTS}; each is divided by DP2"
printf 'Command:'
printf ' %q' "${server_cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
mkdir -p -- "${LOG_DIR}"
"${server_cmd[@]}" 2>&1 | tee "${LOG_DIR}/server.log"
