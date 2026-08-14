#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-/models/MiMo-V2.5-Pro}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30001}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-1.0}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.01}"
export PAGE_SIZE="${PAGE_SIZE:-64}"
export FLYDSL_PA_NUM_PARTITIONS="${FLYDSL_PA_NUM_PARTITIONS:-16}"

export SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1
export MC_GID_INDEX=3
export MC_TE_METRIC=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=5000
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=5000
export SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=128
export SGLANG_SPEC_NAN_DETECTION=1
export SGLANG_SPEC_OOB_DETECTION=1

export SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d
export SGLANG_FLYDSL_MIMO_PREFILL=1
export SGLANG_AITER_PA_DECODE_IMPL=flydsl
export SGLANG_FLYDSL_PA_NUM_PARTITIONS="${FLYDSL_PA_NUM_PARTITIONS}"

export SGLANG_SIMULATE_ACC_LEN=3
export SGLANG_SIMULATE_ACC_METHOD=match-expected

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/peak_output_server_${RUN_ID}}"
export SERVER_LOG_FILE="${SERVER_LOG_FILE:-decode_server_tp8_flydsl_fake_prefill.log}"
if [[ "${SERVER_LOG_FILE}" = /* ]]; then
  SERVER_LOG_PATH="${SERVER_LOG_FILE}"
else
  SERVER_LOG_PATH="${LOG_DIR}/${SERVER_LOG_FILE}"
fi
mkdir -p "${LOG_DIR}" "$(dirname -- "${SERVER_LOG_PATH}")"

echo "FlyDSL hybrid: full target-verify=FlyDSL, SWA/sink and decode=AITER"
echo "Configuration: max-running=${MAX_RUNNING_REQUESTS}, page=${PAGE_SIZE}, partitions=${FLYDSL_PA_NUM_PARTITIONS}, mem=${MEM_FRACTION_STATIC}, swa=${SWA_FULL_TOKENS_RATIO}, overlap=enabled"
echo "MTP acceptance: simulated length 3, match-expected"
echo "Server log: ${SERVER_LOG_PATH}"

python3 -u -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp-size 8 \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
  --context-length 1048576 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 1048576 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend fake \
  --attention-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --page-size "${PAGE_SIZE}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-multi-layer-eagle \
  2>&1 | tee "${SERVER_LOG_PATH}"

