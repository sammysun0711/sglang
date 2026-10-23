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
export SPECULATIVE_ATTENTION_MODE="${SPECULATIVE_ATTENTION_MODE:-prefill}"
export DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-}"
export SPECULATIVE_DRAFT_MODEL="${SPECULATIVE_DRAFT_MODEL:-/models/MiMo-V2.5-Pro-FP4-DFlash/dflash}"
export SPECULATIVE_DRAFT_ATTENTION_BACKEND="${SPECULATIVE_DRAFT_ATTENTION_BACKEND:-aiter}"
export SPECULATIVE_DRAFT_KV_CACHE_DTYPE="${SPECULATIVE_DRAFT_KV_CACHE_DTYPE:-bf16}"
export SPECULATIVE_DRAFT_WINDOW_SIZE="${SPECULATIVE_DRAFT_WINDOW_SIZE:-1024}"
export PAGE_SIZE="${PAGE_SIZE:-64}"
export MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-aiter}"
export MM_ATTENTION_BACKEND="${MM_ATTENTION_BACKEND:-}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bf16}"
export ENABLE_PREFILL_QUICK_REDUCE="${ENABLE_PREFILL_QUICK_REDUCE:-0}"

if [[ "${ENABLE_PREFILL_QUICK_REDUCE}" != "0" && "${ENABLE_PREFILL_QUICK_REDUCE}" != "1" ]]; then
  echo "ENABLE_PREFILL_QUICK_REDUCE must be 0 or 1, observed '${ENABLE_PREFILL_QUICK_REDUCE}'" >&2
  exit 2
fi

export SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1
export MC_GID_INDEX=3
export MC_TE_METRIC=1
export SGLANG_SPEC_NAN_DETECTION=1
export SGLANG_SPEC_OOB_DETECTION=1
export SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY="${SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY:-1}"
export SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE=1
# QuickReduce is opt-in for prefill performance only. Accuracy and decode runs
# must not inherit its lossy INT8 communication mode from the caller.
if [[ "${ENABLE_PREFILL_QUICK_REDUCE}" == "1" ]]; then
  export ROCM_QUICK_REDUCE_QUANTIZATION=INT8
else
  unset ROCM_QUICK_REDUCE_QUANTIZATION
fi
export SGLANG_MIMO_MIXED_ROUTER="${SGLANG_MIMO_MIXED_ROUTER:-1}"

export SGLANG_FLYPA_MIMO_PREFILL="${SGLANG_FLYPA_MIMO_PREFILL:-1}"
export SGLANG_FLYDSL_MIMO_PREFILL="${SGLANG_FLYDSL_MIMO_PREFILL:-1}"

export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d
export SGLANG_AITER_PA_DECODE_IMPL="${SGLANG_AITER_PA_DECODE_IMPL:-flydsl}"
if [[ "${SPECULATIVE_ALGORITHM}" == "DFLASH" ]]; then
  default_draft_swa_impl=flydsl
  default_overlap_plan_stream=1
  if [[ "${KV_CACHE_DTYPE}" == "fp8_e4m3" ]]; then
    default_target_swa_impl=gluon
  else
    default_target_swa_impl=flydsl
  fi
else
  default_draft_swa_impl=gluon
  default_target_swa_impl=gluon
  default_overlap_plan_stream=0
fi
export SGLANG_AITER_DFLASH_SWA_IMPL="${SGLANG_AITER_DFLASH_SWA_IMPL:-${default_draft_swa_impl}}"
export SGLANG_AITER_TARGET_VERIFY_SWA_IMPL="${SGLANG_AITER_TARGET_VERIFY_SWA_IMPL:-${default_target_swa_impl}}"
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM="${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-${default_overlap_plan_stream}}"
export SGLANG_FLYDSL_PA_NUM_PARTITIONS="${SGLANG_FLYDSL_PA_NUM_PARTITIONS:-16}"
export CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-full}"
export CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-}"
export REASONING_PARSER="${REASONING_PARSER:-mimo}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
export TOKENIZER_WORKER_NUM="${TOKENIZER_WORKER_NUM:-1}"
export SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.01}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
export DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
export ENABLE_TWO_BATCH_OVERLAP="${ENABLE_TWO_BATCH_OVERLAP:-0}"
export SGLANG_TBO_MIM_SEQ_LEN="${SGLANG_TBO_MIM_SEQ_LEN:-2000}"

if ! [[ "${SGLANG_TBO_MIM_SEQ_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SGLANG_TBO_MIM_SEQ_LEN must be a positive integer, observed '${SGLANG_TBO_MIM_SEQ_LEN}'" >&2
  exit 2
fi
if ! [[ "${TOKENIZER_WORKER_NUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TOKENIZER_WORKER_NUM must be a positive integer, observed '${TOKENIZER_WORKER_NUM}'" >&2
  exit 2
fi
if [[ -n "${SERVER_RANDOM_SEED}" ]] && ! [[ "${SERVER_RANDOM_SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SERVER_RANDOM_SEED must be a non-negative integer or unset" >&2
  exit 2
fi
export SGLANG_MIMO_FUSED_RMS_MOE_QUANT="${SGLANG_MIMO_FUSED_RMS_MOE_QUANT:-1}"
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT="${SGLANG_MIMO_FUSED_RMS_QKV_QUANT:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM="${SGLANG_AITER_MIMO_FRESH_BF16_ASM:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN:-1}"

# Real MTP acceptance for accuracy validation.
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-./logs/accuracy_${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-server_tp8_flydsl_accuracy.log}"

speculative_args=()
speculative_summary=none
case "${SPECULATIVE_ALGORITHM}" in
  NONE)
    ;;
  EAGLE)
    SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
    speculative_summary="draft-tokens=${SPECULATIVE_NUM_DRAFT_TOKENS}"
    speculative_args+=(
      --speculative-algorithm EAGLE
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
      --speculative-attention-mode "${SPECULATIVE_ATTENTION_MODE}"
      --enable-multi-layer-eagle
    )
    ;;
  DFLASH)
    SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-8}"
    speculative_summary="draft-tokens=${SPECULATIVE_NUM_DRAFT_TOKENS}, draft-model=${SPECULATIVE_DRAFT_MODEL}, draft-attention=${SPECULATIVE_DRAFT_ATTENTION_BACKEND}, draft-kv=${SPECULATIVE_DRAFT_KV_CACHE_DTYPE}"
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
    echo "SPECULATIVE_ALGORITHM must be NONE, EAGLE, or DFLASH" >&2
    exit 2
    ;;
esac

kv_cache_args=()
if [[ -n "${KV_CACHE_DTYPE}" && "${KV_CACHE_DTYPE}" != "auto" ]]; then
  kv_cache_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

decode_attention_args=()
if [[ -n "${DECODE_ATTENTION_BACKEND}" ]]; then
  decode_attention_args+=(--decode-attention-backend "${DECODE_ATTENTION_BACKEND}")
fi

mm_attention_args=()
if [[ -n "${MM_ATTENTION_BACKEND}" ]]; then
  mm_attention_args+=(--mm-attention-backend "${MM_ATTENTION_BACKEND}")
fi

echo "Attention hybrid: prefill-flydsl=${SGLANG_FLYDSL_MIMO_PREFILL}, full-target-verify=${SGLANG_AITER_PA_DECODE_IMPL}, target-swa=${SGLANG_AITER_TARGET_VERIFY_SWA_IMPL}, draft-swa=${SGLANG_AITER_DFLASH_SWA_IMPL}"
echo "Configuration: max-running=${MAX_RUNNING_REQUESTS}, page=${PAGE_SIZE}, chunked-prefill=${CHUNKED_PREFILL_SIZE}, ep=1, moe-runner=${MOE_RUNNER_BACKEND}, partitions=${SGLANG_FLYDSL_PA_NUM_PARTITIONS}, mem=${MEM_FRACTION_STATIC}, swa=${SWA_FULL_TOKENS_RATIO}, kv-cache-dtype=${KV_CACHE_DTYPE}, quick-ar=${ROCM_QUICK_REDUCE_QUANTIZATION:-disabled}, mixed-router=${SGLANG_MIMO_MIXED_ROUTER}, fused-rms-moe=${SGLANG_MIMO_FUSED_RMS_MOE_QUANT}, fused-rms-qkv=${SGLANG_MIMO_FUSED_RMS_QKV_QUANT}, fresh-bf16-asm=${SGLANG_AITER_MIMO_FRESH_BF16_ASM}, fresh-bf16-varlen=${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN}, fresh-bf16-swa-varlen=${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN}, aiter-ar-fusion=0, decode-graph=${CUDA_GRAPH_BACKEND_DECODE}, decode-graph-bs=${CUDA_GRAPH_BS_DECODE:-default}, reasoning-parser=${REASONING_PARSER}, tbo=${ENABLE_TWO_BATCH_OVERLAP}, tbo-min-isl=${SGLANG_TBO_MIM_SEQ_LEN}, overlap=enabled, overlap-plan-stream=${SGLANG_ENABLE_OVERLAP_PLAN_STREAM}"
echo "AITER root/configs: ${AITER_ROOT}; fmoe=$(basename -- "${AITER_CONFIG_FMOE}"); gemm=$(basename -- "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE}"); bpreshuffle=$(basename -- "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE}")"
echo "Multimodal attention: ${MM_ATTENTION_BACKEND:-auto}"
echo "Server log: ${LOG_DIR}/${LOG_FILE}"
echo "Speculative decoding: algorithm=${SPECULATIVE_ALGORITHM}, attention-mode=${SPECULATIVE_ATTENTION_MODE}, ${speculative_summary}"
echo "Tokenizer workers: ${TOKENIZER_WORKER_NUM}; server seed: ${SERVER_RANDOM_SEED:-auto}"

mkdir -p ${LOG_DIR}

cuda_graph_args=(--cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}")
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

radix_cache_args=()
radix_cache_status=enabled
if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  radix_cache_args+=(--disable-radix-cache)
  radix_cache_status=disabled
fi

echo "Radix cache: ${radix_cache_status}"
echo "HIP non-greedy EAGLE verifier: ${SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY}"

seed_args=()
if [[ -n "${SERVER_RANDOM_SEED}" ]]; then
  seed_args+=(--random-seed "${SERVER_RANDOM_SEED}")
fi

python3 -u -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp-size 8 \
  --tokenizer-worker-num "${TOKENIZER_WORKER_NUM}" \
  "${seed_args[@]}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --reasoning-parser "${REASONING_PARSER}" \
  --tool-call-parser mimo \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
  --context-length 1048576 \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --max-prefill-tokens 1048576 \
  --attention-backend aiter \
  "${decode_attention_args[@]}" \
  "${mm_attention_args[@]}" \
  --moe-runner-backend "${MOE_RUNNER_BACKEND}" \
  --aiter-mxfp4-stage2-output-dtype "${AITER_MXFP4_STAGE2_OUTPUT_DTYPE}" \
  "${kv_cache_args[@]}" \
  --page-size "${PAGE_SIZE}" \
  "${speculative_args[@]}" \
  "${tbo_args[@]}" \
  "${radix_cache_args[@]}" \
  "${cuda_graph_args[@]}" \
  "${custom_all_reduce_args[@]}" \
  2>&1 | tee "${LOG_DIR}/${LOG_FILE}"
