#!/usr/bin/env bash
set -euo pipefail

# Two nodes x 8 GPUs: global TP16/EP16, DP2, attention TP8 per node.
# Run the same script on both nodes, with each node's own IP and TCP interface:
# NODE_RANK=0|1 DIST_INIT_ADDR=<node0-ip>:50000 SGLANG_HOST_IP=<this-node-ip> \
#   GLOO_SOCKET_IFNAME=<tcp-interface> ./<script>
export NODE_RANK="${NODE_RANK:?Set NODE_RANK to 0 or 1}"
export DIST_INIT_ADDR="${DIST_INIT_ADDR:?Set DIST_INIT_ADDR to <node0-ip>:<port>}"
case "${NODE_RANK}" in
  0 | 1) ;;
  *) echo "NODE_RANK must be 0 or 1, got: ${NODE_RANK}" >&2; exit 2 ;;
esac
# TCP bootstrap/control traffic. Interface names come from `ip -br addr`,
# not the RDMA device list below. RCCL defaults to the same interface as Gloo.
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-${HOST_IP:?Set SGLANG_HOST_IP (or HOST_IP) to the local node IP}}"
export HOST_IP="${HOST_IP:-${SGLANG_HOST_IP}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp193s0f1np1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=${GLOO_SOCKET_IFNAME}}"
export MORI_SOCKET_IFNAME="${MORI_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}"

# Ionic/AINIC RDMA. The container needs the host's ionic verbs provider and
# libionic.so. Override these device names/GID for each node's actual fabric.
# GID 1 matches the reference cluster; it is not a universal RoCE default.
export MORI_DEVICE_NIC="${MORI_DEVICE_NIC:-ionic}"
export MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-rocep121s0,rocep9s0,rocep105s0,rocep25s0,rocep249s0,rocep137s0,rocep233s0,rocep153s0}"
export MORI_IB_GID_INDEX="${MORI_IB_GID_INDEX:-1}"
# RCCL uses separate envs. '=' means exact matching for this explicit NIC list;
# MORI_RDMA_DEVICES uses plain device names. Mooncake MC_* envs do not apply.
export NCCL_IB_HCA="${NCCL_IB_HCA:-=${MORI_RDMA_DEVICES}}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-${MORI_IB_GID_INDEX}}"
# export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

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
# Global DP2 limit: 2 * 185 preserves the source launcher's 185 requests per DP.
# For 96 active requests + one spare per DP, set MAX_RUNNING_REQUESTS=194.
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-370}"
export TOKENIZER_WORKER_NUM="${TOKENIZER_WORKER_NUM:-1}"
export DECODE_LOG_INTERVAL="${DECODE_LOG_INTERVAL:-1}"
# Reproducible EP baseline; use the same seed for the TP comparison.
export SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-1234}"
if ! [[ "${MAX_RUNNING_REQUESTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RUNNING_REQUESTS must be a positive integer" >&2
  exit 2
fi
max_running_requests_per_dp=$(( MAX_RUNNING_REQUESTS / 2 ))
if (( max_running_requests_per_dp < 1 )); then
  echo "DP2 requires MAX_RUNNING_REQUESTS >= 2" >&2
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
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-1.0}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.01}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
export DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"

# Pre-MoE RMS/quant fusion is no-EP only; QKV fusion below also enables FP8 AG.
export SGLANG_MIMO_FUSED_RMS_MOE_QUANT=0
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT="${SGLANG_MIMO_FUSED_RMS_QKV_QUANT:-0}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM="${SGLANG_AITER_MIMO_FRESH_BF16_ASM:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN:-1}"

# Fake-prefill decode benchmarking uses deterministic simulated MTP acceptance.
export SGLANG_SIMULATE_ACC_LEN="${SGLANG_SIMULATE_ACC_LEN:-3}"
export SGLANG_SIMULATE_ACC_METHOD="${SGLANG_SIMULATE_ACC_METHOD:-match-expected}"

# optimized decode fp8 moe kernel; set 0 to use native AITER MoE selection.
export AITER_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950="${AITER_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950:-1}"
# EP16 + MORI. Keep TBO off for the decode mode comparison.
# normal + SDMA=0 selects InterNodeV1LL at capacity <= 256 (default switch
# threshold), otherwise InterNodeV1. low_latency or SDMA=1 selects AsyncLL.
export MORI_MODE="${MORI_MODE:-normal}"
export MORI_EP_LAUNCH_CONFIG_MODE="${MORI_EP_LAUNCH_CONFIG_MODE:-MANUAL}"
export SGLANG_MORI_DISPATCH_DTYPE="${SGLANG_MORI_DISPATCH_DTYPE:-auto}"
export SGLANG_MORI_COMBINE_DTYPE="${SGLANG_MORI_COMBINE_DTYPE:-auto}"
export MORI_SHMEM_MODE="${MORI_SHMEM_MODE:-ISOLATION}"
# The EP16 baseline uses the inter-node kernel without SDMA.
export MORI_ENABLE_SDMA="${MORI_ENABLE_SDMA:-0}"
unset MORI_DISABLE_P2P

# Each DP has attention TP8; bound padded verify tokens from its local batch.
mori_dispatch_default=128
if (( max_running_requests_per_dp > 256 )); then
  mori_dispatch_default=256
fi
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-${mori_dispatch_default}}"
mori_dispatch_required=$(( ((max_running_requests_per_dp + 7) / 8) * 4 ))
if ! [[ "${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}" =~ ^[1-9][0-9]*$ ]] || \
   (( SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK < mori_dispatch_required )); then
  echo "MORI dispatch capacity must be at least ${mori_dispatch_required} for max-running-per-dp=${max_running_requests_per_dp} with padded attention-TP8 verify graphs" >&2
  exit 2
fi
export ENABLE_FAKE_EP_DISPATCH="${ENABLE_FAKE_EP_DISPATCH:-1}"

# MANUAL ignores tuning JSONs. After decode tuning, select AUTO and explicitly
# export MORI_EP_TUNING_CONFIG=/path/to/the/matching/ep16-decode-config.json.

ep_args=()
if [[ "${ENABLE_FAKE_EP_DISPATCH}" == "1" ]]; then
  ep_args+=(--ep-dispatch-algorithm fake)
fi

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-./logs/decode_ep16_fake_prefill_baseline_${RUN_ID}/node${NODE_RANK}}"
export LOG_FILE="${LOG_FILE:-decode_ep16_fake_prefill_baseline_node${NODE_RANK}.log}"

echo "Attention hybrid: prefill-flydsl=${SGLANG_FLYDSL_MIMO_PREFILL}, target-verify=${SGLANG_AITER_PA_DECODE_IMPL}, SWA/sink and ordinary decode=AITER/Gluon"
echo "Topology: nodes=2, node-rank=${NODE_RANK}, dist-init=${DIST_INIT_ADDR}, tp=16, ep=16, dp=2, attention-tp=8"
echo "DP scheduling: max-running-global=${MAX_RUNNING_REQUESTS}, max-running-per-dp=${max_running_requests_per_dp}"

echo "Configuration: max-running=${MAX_RUNNING_REQUESTS}, preallocate-reqs=${SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS}, decode-log-interval=${DECODE_LOG_INTERVAL}, page=64, chunked-prefill=${CHUNKED_PREFILL_SIZE}, tp=16, ep=16, dp=2, attention-tp=8, partitions=${SGLANG_FLYDSL_PA_NUM_PARTITIONS}, mem=${MEM_FRACTION_STATIC}, swa=${SWA_FULL_TOKENS_RATIO}, kv-cache-dtype=${KV_CACHE_DTYPE}, quick-ar=${ROCM_QUICK_REDUCE_QUANTIZATION:-unset}, mixed-router=${SGLANG_MIMO_MIXED_ROUTER}, fused-rms-moe=${SGLANG_MIMO_FUSED_RMS_MOE_QUANT}, fused-rms-qkv=${SGLANG_MIMO_FUSED_RMS_QKV_QUANT}, fresh-bf16-asm=${SGLANG_AITER_MIMO_FRESH_BF16_ASM}, fresh-bf16-varlen=${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN}, fresh-bf16-swa-varlen=${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN}, mtp=fake:${SGLANG_SIMULATE_ACC_LEN}/${SGLANG_SIMULATE_ACC_METHOD}, aiter-ar-fusion=0, decode-graph=${CUDA_GRAPH_BACKEND_DECODE}, decode-graph-bs=${CUDA_GRAPH_BS_DECODE:-default}, reasoning-parser=${REASONING_PARSER}, tbo=off"
echo "MORI: mode=${MORI_MODE}, sdma=${MORI_ENABLE_SDMA}, launch-config=${MORI_EP_LAUNCH_CONFIG_MODE}, tuning-config=${MORI_EP_TUNING_CONFIG:-package lookup/fallback} (AUTO only), dispatch=${SGLANG_MORI_DISPATCH_DTYPE}, combine=${SGLANG_MORI_COMBINE_DTYPE}, input-cap/rank=${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}, fake-ep=${ENABLE_FAKE_EP_DISPATCH}"
echo "FlyDSL MoE opt-in (EP decode, FP8 Block128, gfx950): ${AITER_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950}"
echo "Server log: ${LOG_DIR}/${LOG_FILE}"
echo "Tokenizer workers: ${TOKENIZER_WORKER_NUM}; server seed: ${SERVER_RANDOM_SEED:-auto}"
echo "MTP: EAGLE, steps=3, top-k=1, draft-tokens=4, multi-layer=enabled"
echo "Fake prefill: disaggregation-mode=decode, transfer-backend=fake"

mkdir -p "${LOG_DIR}"

{
  echo "Network: node-rank=${NODE_RANK}, dist-init=${DIST_INIT_ADDR}, sglang-host-ip=${SGLANG_HOST_IP}, host-ip=${HOST_IP}, gloo-ifname=${GLOO_SOCKET_IFNAME}, mori-socket-ifname=${MORI_SOCKET_IFNAME}, nccl-socket-ifname=${NCCL_SOCKET_IFNAME}"
  echo "MORI RDMA: nic=${MORI_DEVICE_NIC}, devices=${MORI_RDMA_DEVICES}, gid-index=${MORI_IB_GID_INDEX}"
  echo "RCCL RDMA: hca=${NCCL_IB_HCA}, gid-index=${NCCL_IB_GID_INDEX}, torch-blocking-wait=${TORCH_NCCL_BLOCKING_WAIT:-unset}"
} | tee "${LOG_DIR}/launcher_network.log"

cuda_graph_args=(--cuda-graph-backend-decode "${CUDA_GRAPH_BACKEND_DECODE}")
seed_args=()
if [[ -n "${SERVER_RANDOM_SEED}" ]]; then
  seed_args+=(--random-seed "${SERVER_RANDOM_SEED}")
fi
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
  --tp-size 16 \
  --ep-size 16 \
  --dp-size 2 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --nnodes 2 \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --moe-a2a-backend mori \
  --moe-runner-backend aiter \
  --deepep-mode "${MORI_MODE}" \
  "${ep_args[@]}" \
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
