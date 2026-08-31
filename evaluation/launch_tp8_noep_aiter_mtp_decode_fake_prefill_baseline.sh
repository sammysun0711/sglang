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
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=5000
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=5000
export SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=128
export SGLANG_SPEC_NAN_DETECTION=1
export SGLANG_SPEC_OOB_DETECTION=1
export SGLANG_SIMULATE_ACC_LEN="${SGLANG_SIMULATE_ACC_LEN:-3}"
export SGLANG_SIMULATE_ACC_METHOD="${SGLANG_SIMULATE_ACC_METHOD:-match-expected}"
export ROCM_QUICK_REDUCE_QUANTIZATION="${ROCM_QUICK_REDUCE_QUANTIZATION:-INT8}"

export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.8}"
export DISABLE_HYBRID_SWA_MEMORY="${DISABLE_HYBRID_SWA_MEMORY:-1}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-./logs/mimo_fp4_dflash_fake_prefill_${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-server.log}"
mkdir -p "${LOG_DIR}"

hybrid_swa_args=()
if [[ "${DISABLE_HYBRID_SWA_MEMORY}" == "1" ]]; then
  hybrid_swa_args+=(--disable-hybrid-swa-memory)
fi

echo "Configuration: TP8/EP1, native MXFP4, DFlash, fake prefill, simulated accept=${SGLANG_SIMULATE_ACC_LEN}"
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
  "${hybrid_swa_args[@]}" \
  --context-length 1048576 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 1048576 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend fake \
  --attention-backend triton \
  --page-size 1 \
  --quantization fp8 \
  --moe-runner-backend aiter \
  --aiter-mxfp4-stage2-output-dtype fp8 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /models/MiMo-V2.5-Pro-FP4-DFlash/dflash \
  --speculative-num-draft-tokens 8 \
  --disable-overlap-schedule \
  --disable-chunked-prefix-cache \
  2>&1 | tee "${LOG_DIR}/${LOG_FILE}"
