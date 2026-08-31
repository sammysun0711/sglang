#!/usr/bin/env bash
set -euo pipefail

export SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1
export SGLANG_USE_AITER_MOE_GU_ITLV=1
export SGLANG_MORI_DISPATCH_DTYPE=mxfp8
export SGLANG_MORI_COMBINE_DTYPE=bf16
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384
export MORI_SHMEM_MODE=ISOLATION
export MORI_ENABLE_SDMA=0
export HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=1
export AITER_LOG_MORE="${AITER_LOG_MORE:-0}"

export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-./logs/mimo_fp4_dflash_mori_mxfp8_${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-server_tp8_ep8_mori_mxfp8.log}"

mkdir -p "${LOG_DIR}"

echo "Configuration: TP8/EP8, MORI MXFP8 dispatch, BF16 combine, chunked-prefill=${CHUNKED_PREFILL_SIZE}, max-running=${MAX_RUNNING_REQUESTS}, mem=${MEM_FRACTION_STATIC}"
echo "Server log: ${LOG_DIR}/${LOG_FILE}"

python3 -u -m sglang.launch_server \
  --model-path /models/MiMo-V2.5-Pro-FP4-DFlash \
  --trust-remote-code \
  --quantization fp8 \
  --tensor-parallel-size 8 \
  --ep-size 8 \
  --moe-a2a-backend mori \
  --moe-runner-backend aiter \
  --aiter-mxfp4-stage2-output-dtype bf16 \
  --deepep-mode normal \
  --attention-backend triton \
  --page-size 1 \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --max-prefill-tokens "${CHUNKED_PREFILL_SIZE}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /models/MiMo-V2.5-Pro-FP4-DFlash/dflash \
  --speculative-num-draft-tokens 8 \
  --disable-overlap-schedule \
  --disable-chunked-prefix-cache \
  --host 0.0.0.0 \
  --port 30001 \
  2>&1 | tee "${LOG_DIR}/${LOG_FILE}"
