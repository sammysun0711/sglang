#!/usr/bin/env bash
set -euo pipefail

ROOT="${MIMO_ROOT:-/root/workspace/mimo-opt}"
SGLANG_ROOT="${SGLANG_ROOT:-${ROOT}/sglang-fp4-dflash-upstream-latest}"
MODEL_ROOT="${MODEL_ROOT:-/models/MiMo-V2.5-Pro-FP4-DFlash}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30046}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-${SGLANG_ROOT}/evaluation/logs/mimo_fp4_dflash_ep8_standard_${RUN_ID}}"

mkdir -p "${LOG_DIR}"

exec env \
  HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  PYTHONPATH="${SGLANG_ROOT}/python:${ROOT}/aiter-w4a8-readiness:${ROOT}/.flydsl-runtime-0.3.2${PYTHONPATH:+:${PYTHONPATH}}" \
  SGLANG_USE_AITER=1 \
  SGLANG_MOE_PADDING=1 \
  SGLANG_USE_AITER_MOE_GU_ITLV=1 \
  HSA_NO_SCRATCH_RECLAIM=1 \
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=1 \
  AITER_LOG_MORE=1 \
  python3 -u -m sglang.launch_server \
    --model-path "${MODEL_ROOT}" \
    --trust-remote-code \
    --quantization fp8 \
    --tensor-parallel-size 8 \
    --ep-size 8 \
    --moe-runner-backend aiter \
    --aiter-mxfp4-stage2-output-dtype bf16 \
    --attention-backend triton \
    --page-size 1 \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --max-prefill-tokens "${CHUNKED_PREFILL_SIZE}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --mem-fraction-static 0.8 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "${MODEL_ROOT}/dflash" \
    --speculative-num-draft-tokens 8 \
    --disable-overlap-schedule \
    --disable-chunked-prefix-cache \
    --host "${HOST}" \
    --port "${PORT}" \
    2>&1 | tee "${LOG_DIR}/server.log"
