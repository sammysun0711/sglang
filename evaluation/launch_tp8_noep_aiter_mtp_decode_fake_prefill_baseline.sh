#!/usr/bin/env bash
set -euo pipefail

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
export SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY="${SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY:-1}"
export SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE=1
# Accuracy baseline keeps quick-reduce disabled by default. Ablation runners can
# opt in explicitly without changing the default accuracy-safe behavior.
if [[ "${SGLANG_ABLATION_ENABLE_QUICK_REDUCE:-0}" == "1" ]]; then
  export ROCM_QUICK_REDUCE_QUANTIZATION="${ROCM_QUICK_REDUCE_QUANTIZATION:-INT8}"
  echo "ROCM_QUICK_REDUCE_QUANTIZATION enabled for ablation: ${ROCM_QUICK_REDUCE_QUANTIZATION}"
elif [[ -v ROCM_QUICK_REDUCE_QUANTIZATION ]]; then
  echo "ROCM_QUICK_REDUCE_QUANTIZATION was set to '${ROCM_QUICK_REDUCE_QUANTIZATION}', unsetting for accuracy"
  unset ROCM_QUICK_REDUCE_QUANTIZATION
else
  echo "ROCM_QUICK_REDUCE_QUANTIZATION is unset"
fi
# Accuracy baseline keeps mixed router and FlyDSL prefill disabled by default;
# ablation runners may override these with explicit environment settings.
export SGLANG_MIMO_MIXED_ROUTER="${SGLANG_MIMO_MIXED_ROUTER:-0}"

export SGLANG_FLYPA_MIMO_PREFILL="${SGLANG_FLYPA_MIMO_PREFILL:-1}"
export SGLANG_FLYDSL_MIMO_PREFILL="${SGLANG_FLYDSL_MIMO_PREFILL:-1}"

export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d
export SGLANG_AITER_PA_DECODE_IMPL="${SGLANG_AITER_PA_DECODE_IMPL:-flydsl}"
export SGLANG_FLYDSL_PA_NUM_PARTITIONS="${SGLANG_FLYDSL_PA_NUM_PARTITIONS:-16}"
export CUDA_GRAPH_BACKEND_DECODE="${CUDA_GRAPH_BACKEND_DECODE:-full}"
export CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-}"
export REASONING_PARSER="${REASONING_PARSER:-mimo}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-1.0}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.01}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
export DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-0}"

export SGLANG_MIMO_FUSED_RMS_MOE_QUANT="${SGLANG_MIMO_FUSED_RMS_MOE_QUANT:-1}"
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT="${SGLANG_MIMO_FUSED_RMS_QKV_QUANT:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM="${SGLANG_AITER_MIMO_FRESH_BF16_ASM:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN:-1}"

# Fake-prefill decode benchmarking uses deterministic simulated MTP acceptance.
export SGLANG_SIMULATE_ACC_LEN="${SGLANG_SIMULATE_ACC_LEN:-3}"
export SGLANG_SIMULATE_ACC_METHOD="${SGLANG_SIMULATE_ACC_METHOD:-match-expected}"

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-./logs/decode_fake_prefill_baseline_${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-decode_fake_prefill_baseline.log}"

echo "Attention hybrid: prefill-flydsl=${SGLANG_FLYDSL_MIMO_PREFILL}, target-verify=${SGLANG_AITER_PA_DECODE_IMPL}, SWA/sink and ordinary decode=AITER/Gluon"

echo "Configuration: max-running=${MAX_RUNNING_REQUESTS}, page=64, chunked-prefill=${CHUNKED_PREFILL_SIZE}, ep=1, partitions=${SGLANG_FLYDSL_PA_NUM_PARTITIONS}, mem=${MEM_FRACTION_STATIC}, swa=${SWA_FULL_TOKENS_RATIO}, kv-cache-dtype=${KV_CACHE_DTYPE}, quick-ar=${ROCM_QUICK_REDUCE_QUANTIZATION:-unset}, mixed-router=${SGLANG_MIMO_MIXED_ROUTER}, fused-rms-moe=${SGLANG_MIMO_FUSED_RMS_MOE_QUANT}, fused-rms-qkv=${SGLANG_MIMO_FUSED_RMS_QKV_QUANT}, fresh-bf16-asm=${SGLANG_AITER_MIMO_FRESH_BF16_ASM}, fresh-bf16-varlen=${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN}, fresh-bf16-swa-varlen=${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN}, mtp=fake:${SGLANG_SIMULATE_ACC_LEN}/${SGLANG_SIMULATE_ACC_METHOD}, aiter-ar-fusion=0, decode-graph=${CUDA_GRAPH_BACKEND_DECODE}, decode-graph-bs=${CUDA_GRAPH_BS_DECODE:-default}, reasoning-parser=${REASONING_PARSER}, overlap=enabled"
echo "Server log: ${LOG_DIR}/${LOG_FILE}"
echo "MTP: EAGLE, steps=3, top-k=1, draft-tokens=4, multi-layer=enabled"
echo "Fake prefill: disaggregation-mode=decode, transfer-backend=fake"

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

python3 -u -m sglang.launch_server \
  --model-path /models/MiMo-V2.5-Pro/ \
  --tp-size 8 \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --host 0.0.0.0 \
  --port 30001 \
  --trust-remote-code \
  --reasoning-parser "${REASONING_PARSER}" \
  --tool-call-parser mimo \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
  --context-length 1048576 \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --max-prefill-tokens 1048576 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend fake \
  --attention-backend aiter \
  "${kv_cache_args[@]}" \
  --page-size 64 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-multi-layer-eagle \
  "${radix_cache_args[@]}" \
  "${cuda_graph_args[@]}" \
  "${custom_all_reduce_args[@]}" \
  2>&1 | tee "${LOG_DIR}/${LOG_FILE}"
