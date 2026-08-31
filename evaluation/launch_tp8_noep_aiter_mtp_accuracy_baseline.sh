#!/usr/bin/env bash
set -euo pipefail

export SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1
export SGLANG_USE_AITER_MOE_GU_ITLV=1
export SGLANG_SET_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1
export MC_GID_INDEX=3
export MC_TE_METRIC=1
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-112}"
export SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE=1

if [[ "${SGLANG_ABLATION_ENABLE_QUICK_REDUCE:-0}" == "1" ]]; then
  export ROCM_QUICK_REDUCE_QUANTIZATION="${ROCM_QUICK_REDUCE_QUANTIZATION:-INT8}"
else
  unset ROCM_QUICK_REDUCE_QUANTIZATION || true
fi

export SGLANG_MIMO_MIXED_ROUTER="${SGLANG_MIMO_MIXED_ROUTER:-0}"
export SGLANG_FLYPA_MIMO_PREFILL="${SGLANG_FLYPA_MIMO_PREFILL:-1}"
export SGLANG_FLYDSL_MIMO_PREFILL="${SGLANG_FLYDSL_MIMO_PREFILL:-1}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export SGLANG_AITER_PA_DECODE_IMPL="${SGLANG_AITER_PA_DECODE_IMPL:-flydsl}"
export SGLANG_FLYDSL_PA_NUM_PARTITIONS="${SGLANG_FLYDSL_PA_NUM_PARTITIONS:-16}"
export SGLANG_MIMO_FUSED_RMS_MOE_QUANT="${SGLANG_MIMO_FUSED_RMS_MOE_QUANT:-1}"
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT="${SGLANG_MIMO_FUSED_RMS_QKV_QUANT:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM="${SGLANG_AITER_MIMO_FRESH_BF16_ASM:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN:-1}"

unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD

export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.01}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
export PAGE_SIZE="${PAGE_SIZE:-1}"
export ENABLE_TWO_BATCH_OVERLAP="${ENABLE_TWO_BATCH_OVERLAP:-0}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-./logs/mimo_fp4_prefill_baseline_${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-server.log}"
mkdir -p "${LOG_DIR}"

if (( PAGE_SIZE % 8 == 0 )); then
  export SGLANG_AITER_KV_CACHE_LAYOUT="${SGLANG_AITER_KV_CACHE_LAYOUT:-vectorized_5d}"
else
  unset SGLANG_AITER_KV_CACHE_LAYOUT || true
fi

tbo_args=()
if [[ "${ENABLE_TWO_BATCH_OVERLAP}" == "1" ]]; then
  tbo_args+=(--enable-two-batch-overlap)
fi
kv_cache_args=()
if [[ -n "${KV_CACHE_DTYPE}" && "${KV_CACHE_DTYPE}" != "auto" ]]; then
  kv_cache_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

echo "Configuration: TP8/EP1, native MXFP4, chunk=${CHUNKED_PREFILL_SIZE}, TBO=${ENABLE_TWO_BATCH_OVERLAP}, attention=${ATTENTION_BACKEND}, page=${PAGE_SIZE}"
echo "Server log: ${LOG_DIR}/${LOG_FILE}"

python3 -u -m sglang.launch_server \
  --model-path /models/MiMo-V2.5-Pro-FP4-DFlash \
  --tp-size 8 \
  --ep-size 1 \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --host 0.0.0.0 \
  --port 30001 \
  --trust-remote-code \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
  --context-length 1048576 \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --max-prefill-tokens 1048576 \
  --attention-backend "${ATTENTION_BACKEND}" \
  "${kv_cache_args[@]}" \
  --page-size "${PAGE_SIZE}" \
  --quantization fp8 \
  --moe-runner-backend aiter \
  --aiter-mxfp4-stage2-output-dtype fp8 \
  "${tbo_args[@]}" \
  2>&1 | tee "${LOG_DIR}/${LOG_FILE}"
