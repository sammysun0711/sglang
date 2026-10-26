#!/usr/bin/env bash
set -euo pipefail

export NODE_RANK="${NODE_RANK:?Set NODE_RANK to 0 or 1}"
export DIST_INIT_ADDR="${DIST_INIT_ADDR:?Set DIST_INIT_ADDR to <node0-ip>:<port>}"
case "${NODE_RANK}" in
  0 | 1) ;;
  *) echo "NODE_RANK must be 0 or 1, got: ${NODE_RANK}" >&2; exit 2 ;;
esac
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-${HOST_IP:?Set SGLANG_HOST_IP (or HOST_IP) to the local node IP}}"
export HOST_IP="${HOST_IP:-${SGLANG_HOST_IP}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp193s0f1np1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=${GLOO_SOCKET_IFNAME}}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-=rocep121s0,rocep9s0,rocep105s0,rocep25s0,rocep249s0,rocep137s0,rocep233s0,rocep153s0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-112}"
# Cross-node MoE TP16 cannot use node-local AITER custom all-reduce.
# Force blocking wait for c10d fallbacks to avoid the observed
# ProcessGroupNCCL watchdog/HIP Graph capture conflict (hipErrorCapturedEvent).
export TORCH_NCCL_BLOCKING_WAIT=1
export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"
# enable it only for rocm 7.0.0 alpha
# default is 1 for rocm docker image
# export SGLANG_USE_ROCM700A=0


export SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=5000
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=5000
export SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS="${SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS:-128}"
export SGLANG_SPEC_NAN_DETECTION=1
export SGLANG_SPEC_OOB_DETECTION=1
export SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY="${SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY:-1}"
export SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE=1
if [[ "${SGLANG_ABLATION_ENABLE_QUICK_REDUCE:-0}" == "1" ]]; then
  export ROCM_QUICK_REDUCE_QUANTIZATION="${ROCM_QUICK_REDUCE_QUANTIZATION:-INT8}"
else
  unset ROCM_QUICK_REDUCE_QUANTIZATION
fi
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
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-226}"
export TOKENIZER_WORKER_NUM="${TOKENIZER_WORKER_NUM:-1}"
export DECODE_LOG_INTERVAL="${DECODE_LOG_INTERVAL:-1}"
export SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-1234}"
for name in MAX_RUNNING_REQUESTS TOKENIZER_WORKER_NUM DECODE_LOG_INTERVAL SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS; do
  if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer" >&2
    exit 2
  fi
done
max_running_requests_per_dp=$((MAX_RUNNING_REQUESTS / 2))
if (( max_running_requests_per_dp < 1 )); then
  echo "DP2 requires MAX_RUNNING_REQUESTS >= 2" >&2
  exit 2
fi
if [[ -n "${SERVER_RANDOM_SEED}" ]] && ! [[ "${SERVER_RANDOM_SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SERVER_RANDOM_SEED must be a non-negative integer or unset" >&2
  exit 2
fi
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-1.0}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.01}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
export DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
# require dp_attn_size == 1
export SGLANG_MIMO_FUSED_RMS_MOE_QUANT="${SGLANG_MIMO_FUSED_RMS_MOE_QUANT:-0}"
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT="${SGLANG_MIMO_FUSED_RMS_QKV_QUANT:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM="${SGLANG_AITER_MIMO_FRESH_BF16_ASM:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN:-1}"
export SGLANG_SIMULATE_ACC_LEN="${SGLANG_SIMULATE_ACC_LEN:-3}"
export SGLANG_SIMULATE_ACC_METHOD="${SGLANG_SIMULATE_ACC_METHOD:-match-expected}"
export AITER_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950=0
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-./logs/decode_tp16_noep_fake_prefill_${RUN_ID}/node${NODE_RANK}}"
export LOG_FILE="${LOG_FILE:-decode_tp16_noep_fake_prefill_node${NODE_RANK}.log}"
mkdir -p "${LOG_DIR}"
LOG_DIR="$(cd -- "${LOG_DIR}" && pwd)"
export SERVER_WORK_DIR="${SERVER_WORK_DIR:-/tmp/sglang-mimo-tp16-noep/${RUN_ID}/node${NODE_RANK}}"
mkdir -p "${SERVER_WORK_DIR}"
cd -- "${SERVER_WORK_DIR}"
{
  echo "Topology: nodes=2, tp=16, ep=1, moe-dp=1, moe-tp=16, dp=2, attention-tp=8, dense-tp=1, dp-lm-head=enabled"
  echo "MoE A2A backend: none; MORI is not used by this TP-only path"
  echo "Network: node-rank=${NODE_RANK}, dist-init=${DIST_INIT_ADDR}, gloo=${GLOO_SOCKET_IFNAME}, nccl=${NCCL_SOCKET_IFNAME}"
  echo "RCCL: hca=${NCCL_IB_HCA}, gid=${NCCL_IB_GID_INDEX}, channels=${NCCL_MIN_NCHANNELS}, blocking-wait=${TORCH_NCCL_BLOCKING_WAIT:-unset}"
  echo "Configuration: max-running-global=${MAX_RUNNING_REQUESTS}, max-running-per-dp=${max_running_requests_per_dp}, mem=${MEM_FRACTION_STATIC}, swa=${SWA_FULL_TOKENS_RATIO}, page=64, kv=${KV_CACHE_DTYPE}, partitions=${SGLANG_FLYDSL_PA_NUM_PARTITIONS}"
  echo "Fake prefill: enabled; MTP: ${SGLANG_SIMULATE_ACC_LEN}/${SGLANG_SIMULATE_ACC_METHOD}; routing=normal (non-EP path)"
  echo "Server working directory: ${PWD}"
  echo "Server log: ${LOG_DIR}/${LOG_FILE}"
} | tee "${LOG_DIR}/launcher_network.log"

cuda_graph_args=(--cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}")
if [[ -n "${CUDA_GRAPH_BS_DECODE}" ]]; then
  read -r -a cuda_graph_bs <<< "${CUDA_GRAPH_BS_DECODE}"
  cuda_graph_args+=(--cuda-graph-bs-decode "${cuda_graph_bs[@]}")
fi
seed_args=()
if [[ -n "${SERVER_RANDOM_SEED}" ]]; then
  seed_args+=(--random-seed "${SERVER_RANDOM_SEED}")
fi
custom_all_reduce_args=()
if [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]]; then
  custom_all_reduce_args+=(--disable-custom-all-reduce)
fi
kv_cache_args=()
if [[ -n "${KV_CACHE_DTYPE}" && "${KV_CACHE_DTYPE}" != "auto" ]]; then
  kv_cache_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
radix_cache_args=()
if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  radix_cache_args+=(--disable-radix-cache)
fi

python3 -u -m sglang.launch_server \
  --model-path /models/MiMo-V2.5-Pro/ \
  --tp-size 16 \
  --ep-size 1 \
  --moe-dp-size 1 \
  --dp-size 2 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --nnodes 2 \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --moe-a2a-backend none \
  --moe-runner-backend aiter \
  --moe-dense-tp-size 1 \
  --tokenizer-worker-num "${TOKENIZER_WORKER_NUM}" \
  "${seed_args[@]}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --decode-log-interval "${DECODE_LOG_INTERVAL}" \
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