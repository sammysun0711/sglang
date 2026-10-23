#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="${SGLANG_ROOT:-$(dirname -- "${SCRIPT_DIR}")}"
AITER_ROOT="${AITER_ROOT:-/root/workspace/mimo-opt/aiter-mimo-fp4-dflash}"
export PYTHONPATH="${SGLANG_ROOT}/python:${AITER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE="${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE:-${AITER_ROOT}/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_mimo_v2_5_pro.csv}"
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE="${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE:-${AITER_ROOT}/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_mimo_v2_5_pro.csv}"

export MODEL="${MODEL:-/models/MiMo-V2.5-Pro}"
MIMO_TARGET_WEIGHT_FORMAT="${MIMO_TARGET_WEIGHT_FORMAT:-auto}"
if [[ "${MIMO_TARGET_WEIGHT_FORMAT}" == "auto" ]]; then
  if [[ -f "${MODEL%/}/config.json" ]]; then
    MIMO_TARGET_WEIGHT_FORMAT="$(python3 -c 'import json, sys; print(str(json.load(open(sys.argv[1])).get("quantization_config", {}).get("store_dtype", "fp8")).lower())' "${MODEL%/}/config.json")"
  else
    MIMO_TARGET_WEIGHT_FORMAT=fp8
  fi
fi
case "${MIMO_TARGET_WEIGHT_FORMAT}" in
  fp8)
    default_fmoe_config="${AITER_ROOT}/aiter/configs/model_configs/mimo_v2_5_pro_b16_tuned_fmoe.csv"
    ;;
  mxfp4)
    default_fmoe_config="${AITER_ROOT}/aiter/configs/model_configs/mimo_v2_5_pro_tuned_fmoe.csv"
    ;;
  *)
    echo "MIMO_TARGET_WEIGHT_FORMAT must be auto, fp8, or mxfp4; observed '${MIMO_TARGET_WEIGHT_FORMAT}'" >&2
    exit 2
    ;;
esac
export MIMO_TARGET_WEIGHT_FORMAT
export AITER_CONFIG_FMOE="${AITER_CONFIG_FMOE:-${default_fmoe_config}}"
export AITER_MXFP4_STAGE2_OUTPUT_DTYPE="${AITER_MXFP4_STAGE2_OUTPUT_DTYPE:-fp8}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30001}"
export SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-EAGLE}"
SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM^^}"
export SPECULATIVE_DRAFT_MODEL="${SPECULATIVE_DRAFT_MODEL:-/models/MiMo-V2.5-Pro-FP4-DFlash/dflash}"
export SPECULATIVE_DRAFT_ATTENTION_BACKEND="${SPECULATIVE_DRAFT_ATTENTION_BACKEND:-aiter}"
export SPECULATIVE_DRAFT_KV_CACHE_DTYPE="${SPECULATIVE_DRAFT_KV_CACHE_DTYPE:-bf16}"
export SPECULATIVE_DRAFT_WINDOW_SIZE="${SPECULATIVE_DRAFT_WINDOW_SIZE:-1024}"
export MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-aiter}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
export TOKENIZER_WORKER_NUM="${TOKENIZER_WORKER_NUM:-1}"
export DECODE_LOG_INTERVAL="${DECODE_LOG_INTERVAL:-1}"
export SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-}"
if [[ -z "${MEM_FRACTION_STATIC:-}" ]]; then
  if [[ "${SPECULATIVE_ALGORITHM}" == "DFLASH" ]]; then
    MEM_FRACTION_STATIC=0.80
  else
    MEM_FRACTION_STATIC=1.0
  fi
fi
export MEM_FRACTION_STATIC
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.01}"
export PAGE_SIZE="${PAGE_SIZE:-64}"
export FLYDSL_PA_NUM_PARTITIONS="${FLYDSL_PA_NUM_PARTITIONS:-16}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bf16}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-full}"
export CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-}"

export SGLANG_FLYPA_MIMO_PREFILL="${SGLANG_FLYPA_MIMO_PREFILL:-1}"
export SGLANG_FLYDSL_MIMO_PREFILL="${SGLANG_FLYDSL_MIMO_PREFILL:-1}"

export SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1
export MC_GID_INDEX=3
export MC_TE_METRIC=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=5000
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=5000
export SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS="${SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS:-96}"
export SGLANG_SPEC_NAN_DETECTION=1
export SGLANG_SPEC_OOB_DETECTION=1

export SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE=1
# QuickReduce is reserved for prefill experiments. Decode uses the standard
# custom-all-reduce path so the accuracy and performance protocols stay aligned.
unset ROCM_QUICK_REDUCE_QUANTIZATION
export SGLANG_MIMO_MIXED_ROUTER="${SGLANG_MIMO_MIXED_ROUTER:-1}"
export SGLANG_MIMO_FUSED_RMS_MOE_QUANT="${SGLANG_MIMO_FUSED_RMS_MOE_QUANT:-1}"
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT="${SGLANG_MIMO_FUSED_RMS_QKV_QUANT:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM="${SGLANG_AITER_MIMO_FRESH_BF16_ASM:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN:-1}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d
export SGLANG_FLYDSL_MIMO_PREFILL="${SGLANG_FLYDSL_MIMO_PREFILL:-1}"
export SGLANG_AITER_PA_DECODE_IMPL="${SGLANG_AITER_PA_DECODE_IMPL:-flydsl}"
export SGLANG_USE_AITER_UNIFIED_ATTN="${SGLANG_USE_AITER_UNIFIED_ATTN:-1}"
if [[ "${SPECULATIVE_ALGORITHM}" == "DFLASH" ]]; then
  default_draft_swa_impl=flydsl
  default_overlap_plan_stream=1
  if [[ "${KV_CACHE_DTYPE}" == "fp8_e4m3" ]]; then
    default_target_swa_impl=gluon
  else
    default_target_swa_impl=flydsl
  fi
  default_simulated_acceptance=4
else
  default_draft_swa_impl=gluon
  default_target_swa_impl=gluon
  default_overlap_plan_stream=0
  default_simulated_acceptance=3
fi
export SGLANG_AITER_DFLASH_SWA_IMPL="${SGLANG_AITER_DFLASH_SWA_IMPL:-${default_draft_swa_impl}}"
export SGLANG_AITER_TARGET_VERIFY_SWA_IMPL="${SGLANG_AITER_TARGET_VERIFY_SWA_IMPL:-${default_target_swa_impl}}"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM="${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-${default_overlap_plan_stream}}"
export SGLANG_FLYDSL_PA_NUM_PARTITIONS="${FLYDSL_PA_NUM_PARTITIONS}"

export SGLANG_SIMULATE_ACC_LEN="${SGLANG_SIMULATE_ACC_LEN:-${default_simulated_acceptance}}"
export SGLANG_SIMULATE_ACC_METHOD="${SGLANG_SIMULATE_ACC_METHOD:-match-expected}"

if ! [[ "${MAX_RUNNING_REQUESTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RUNNING_REQUESTS must be a positive integer" >&2
  exit 2
fi
if ! [[ "${TOKENIZER_WORKER_NUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TOKENIZER_WORKER_NUM must be a positive integer" >&2
  exit 2
fi
if ! [[ "${DECODE_LOG_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DECODE_LOG_INTERVAL must be a positive integer" >&2
  exit 2
fi
if ! [[ "${SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS must be a positive integer" >&2
  exit 2
fi
if [[ -n "${SERVER_RANDOM_SEED}" ]] && ! [[ "${SERVER_RANDOM_SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SERVER_RANDOM_SEED must be a non-negative integer or unset" >&2
  exit 2
fi

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/peak_output_server_${RUN_ID}}"
export SERVER_LOG_FILE="${SERVER_LOG_FILE:-decode_server_tp8_flydsl_fake_prefill.log}"
if [[ "${SERVER_LOG_FILE}" = /* ]]; then
  SERVER_LOG_PATH="${SERVER_LOG_FILE}"
else
  SERVER_LOG_PATH="${LOG_DIR}/${SERVER_LOG_FILE}"
fi
mkdir -p "${LOG_DIR}" "$(dirname -- "${SERVER_LOG_PATH}")"

speculative_args=()
case "${SPECULATIVE_ALGORITHM}" in
  EAGLE)
    SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
    speculative_args+=(
      --speculative-algorithm EAGLE
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
      --enable-multi-layer-eagle
    )
    ;;
  DFLASH)
    SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-8}"
    if [[ "${PAGE_SIZE}" != "64" ]]; then
      echo "DFLASH FlyPA SWA requires PAGE_SIZE=64" >&2
      exit 2
    fi
    if [[ "${SPECULATIVE_DRAFT_ATTENTION_BACKEND}" != "aiter" ]]; then
      echo "DFLASH FlyPA SWA requires SPECULATIVE_DRAFT_ATTENTION_BACKEND=aiter" >&2
      exit 2
    fi
    if [[ "${SPECULATIVE_DRAFT_KV_CACHE_DTYPE}" != "bf16" ]]; then
      echo "DFLASH FlyPA SWA requires SPECULATIVE_DRAFT_KV_CACHE_DTYPE=bf16" >&2
      exit 2
    fi
    speculative_args+=(
      --speculative-algorithm DFLASH
      --speculative-draft-model-path "${SPECULATIVE_DRAFT_MODEL}"
      --speculative-draft-attention-backend "${SPECULATIVE_DRAFT_ATTENTION_BACKEND}"
      --speculative-draft-kv-cache-dtype "${SPECULATIVE_DRAFT_KV_CACHE_DTYPE}"
      --speculative-draft-window-size "${SPECULATIVE_DRAFT_WINDOW_SIZE}"
      --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
    )
    ;;
  *)
    echo "SPECULATIVE_ALGORITHM must be EAGLE or DFLASH" >&2
    exit 2
    ;;
esac

echo "Attention hybrid: prefill-flydsl=${SGLANG_FLYDSL_MIMO_PREFILL}, full-target-verify=${SGLANG_AITER_PA_DECODE_IMPL}, target-swa=${SGLANG_AITER_TARGET_VERIFY_SWA_IMPL}, draft-swa=${SGLANG_AITER_DFLASH_SWA_IMPL}"
echo "Configuration: max-running=${MAX_RUNNING_REQUESTS}, preallocate-reqs=${SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS}, decode-log-interval=${DECODE_LOG_INTERVAL}, page=${PAGE_SIZE}, chunked-prefill=${CHUNKED_PREFILL_SIZE}, moe-runner=${MOE_RUNNER_BACKEND}, partitions=${FLYDSL_PA_NUM_PARTITIONS}, mem=${MEM_FRACTION_STATIC}, swa=${SWA_FULL_TOKENS_RATIO}, kv-cache-dtype=${KV_CACHE_DTYPE}, quick-ar=disabled, decode-graph=${CUDA_GRAPH_BACKEND_DECODE}, decode-graph-bs=${CUDA_GRAPH_BS_DECODE:-default}, overlap=enabled, overlap-plan-stream=${SGLANG_ENABLE_OVERLAP_PLAN_STREAM}"
echo "AITER root/configs: ${AITER_ROOT}; fmoe=$(basename -- "${AITER_CONFIG_FMOE}"); gemm=$(basename -- "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE}"); bpreshuffle=$(basename -- "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE}")"
echo "Simulated acceptance: length=${SGLANG_SIMULATE_ACC_LEN}, method=${SGLANG_SIMULATE_ACC_METHOD}"
echo "Speculative decoding: algorithm=${SPECULATIVE_ALGORITHM}, draft-tokens=${SPECULATIVE_NUM_DRAFT_TOKENS}, draft-model=${SPECULATIVE_DRAFT_MODEL}, draft-attention=${SPECULATIVE_DRAFT_ATTENTION_BACKEND}, draft-kv=${SPECULATIVE_DRAFT_KV_CACHE_DTYPE}"
echo "Server log: ${SERVER_LOG_PATH}"
echo "Tokenizer workers: ${TOKENIZER_WORKER_NUM}; server seed: ${SERVER_RANDOM_SEED:-auto}"

seed_args=()
if [[ -n "${SERVER_RANDOM_SEED}" ]]; then
  seed_args+=(--random-seed "${SERVER_RANDOM_SEED}")
fi

cuda_graph_args=(--cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}")
if [[ -n "${CUDA_GRAPH_BS_DECODE}" ]]; then
  read -r -a cuda_graph_bs <<<"${CUDA_GRAPH_BS_DECODE}"
  cuda_graph_args+=(--cuda-graph-bs-decode "${cuda_graph_bs[@]}")
fi

kv_cache_args=()
if [[ -n "${KV_CACHE_DTYPE}" && "${KV_CACHE_DTYPE}" != "auto" ]]; then
  kv_cache_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

capacity_args=()
if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
  capacity_args+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi

python3 -u -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp-size 8 \
  --tokenizer-worker-num "${TOKENIZER_WORKER_NUM}" \
  "${seed_args[@]}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  "${capacity_args[@]}" \
  --decode-log-interval "${DECODE_LOG_INTERVAL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
  --context-length 1048576 \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --max-prefill-tokens 1048576 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend fake \
  --attention-backend aiter \
  --moe-runner-backend "${MOE_RUNNER_BACKEND}" \
  --aiter-mxfp4-stage2-output-dtype "${AITER_MXFP4_STAGE2_OUTPUT_DTYPE}" \
  "${kv_cache_args[@]}" \
  --page-size "${PAGE_SIZE}" \
  "${speculative_args[@]}" \
  "${cuda_graph_args[@]}" \
  2>&1 | tee "${SERVER_LOG_PATH}"
