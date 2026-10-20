#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export EP_SIZE="${EP_SIZE:-8}"
case "${EP_SIZE}" in
  8) dp_size=1; nnodes=1 ;;
  16) dp_size=2; nnodes=2 ;;
  *) echo "EP_SIZE must be 8 or 16" >&2; exit 2 ;;
esac
export NODE_RANK="${NODE_RANK:-0}"
if ! [[ "${NODE_RANK}" =~ ^[01]$ ]] || (( NODE_RANK >= nnodes )); then
  echo "NODE_RANK must be in [0, $((nnodes - 1))] for EP_SIZE=${EP_SIZE}" >&2
  exit 2
fi
parallel_args=(--tp-size "${EP_SIZE}" --ep-size "${EP_SIZE}")
if [[ "${EP_SIZE}" == "16" ]]; then
  export DIST_INIT_ADDR="${DIST_INIT_ADDR:?Set DIST_INIT_ADDR to <node0-ip>:<port>}"
  export SGLANG_HOST_IP="${SGLANG_HOST_IP:-${HOST_IP:?Set SGLANG_HOST_IP or HOST_IP}}"
  export HOST_IP="${HOST_IP:-${SGLANG_HOST_IP}}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp193s0f1np1}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=${GLOO_SOCKET_IFNAME}}"
  export MORI_SOCKET_IFNAME="${MORI_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}"
  export MORI_DEVICE_NIC="${MORI_DEVICE_NIC:-ionic}"
  export MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-rocep121s0,rocep9s0,rocep105s0,rocep25s0,rocep249s0,rocep137s0,rocep233s0,rocep153s0}"
  export MORI_IB_GID_INDEX="${MORI_IB_GID_INDEX:-1}"
  export NCCL_IB_HCA="${NCCL_IB_HCA:-=${MORI_RDMA_DEVICES}}"
  export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-${MORI_IB_GID_INDEX}}"
  export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
  parallel_args+=(--dp-size "${dp_size}" --enable-dp-attention
    --enable-dp-lm-head --moe-dense-tp-size 1 --nnodes "${nnodes}"
    --node-rank "${NODE_RANK}" --dist-init-addr "${DIST_INIT_ADDR}")
fi

# Temporary GSM8K setup: new MoE with real prefill, routing and MTP.
export MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.5-Pro/}"
export SGLANG_USE_AITER=1
export AITER_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950="${AITER_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950:-1}"
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export HSA_NO_SCRATCH_RECLAIM=1
export MC_GID_INDEX=3
export MC_TE_METRIC=1
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-112}"
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
# Accuracy baseline keeps mixed router disabled by default; ablation runners
# may override it with an explicit environment setting.
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
# Use the same reproducible server seed as the TP8 launcher by default.
export SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-1234}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-$((96 * dp_size))}"
# use these below settings for mem_fraction and swa_full_tokens_ratio under 1m input prefill tbo test case
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.02}"
# Keep a 1K effective prefill chunk per attention-TP8 group.
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-$((1024 * dp_size))}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
export DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"

if ! [[ "${SERVER_RANDOM_SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SERVER_RANDOM_SEED must be a non-negative integer" >&2
  exit 2
fi

export SGLANG_AITER_MIMO_FRESH_BF16_ASM="${SGLANG_AITER_MIMO_FRESH_BF16_ASM:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN:-1}"
export SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN="${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN:-1}"
# Cached-SWA compacts prefix-tail + extend KV for the BF16 varlen kernel.
export SGLANG_AITER_MIMO_CACHED_BF16_SWA_VARLEN="${SGLANG_AITER_MIMO_CACHED_BF16_SWA_VARLEN:-0}"

# EP + MORI + TBO settings.
# QKV fusion also enables gathering prequantized FP8 activations (as uint8)
# with FP32 scales and scale-layout restoration. Set to 0 for the combined
# fused-RMSNorm/quant + FP8-all-gather ablation; there is no separate AG toggle.
export SGLANG_MIMO_FUSED_RMS_QKV_QUANT="${SGLANG_MIMO_FUSED_RMS_QKV_QUANT:-0}"
# Fused pre-MoE quantization is restricted to no-EP execution in MiMoV2 and is
# disabled explicitly for this EP8/A2A launcher.
export SGLANG_MIMO_FUSED_RMS_MOE_QUANT=0
export SGLANG_MORI_DISPATCH_DTYPE="${SGLANG_MORI_DISPATCH_DTYPE:-auto}"
export SGLANG_MORI_COMBINE_DTYPE="${SGLANG_MORI_COMBINE_DTYPE:-bf16}"
export MORI_SHMEM_MODE="${MORI_SHMEM_MODE:-ISOLATION}"
export MORI_ENABLE_SDMA=0
unset MORI_DISABLE_P2P
# MORI launch-config workflow:
# - MANUAL keeps SGLang's constructor defaults (IntraNode: 80 blocks, 16 waves/block).
# - AUTO loads the bundled MiMo EP8 dispatch/combine tuning JSON.
#   Reuse the no-TBO-tuned communication config for both ENABLE_TBO=0 and 1.
#   Set MORI_EP_TUNING_CONFIG=/path/to/config.json to select another bundle.
#   The file is inactive in MANUAL mode; setting a path does not switch modes.
# - MORI_TUNING_SCOPE only controls the offline tuner and is intentionally not set here.
export MORI_EP_LAUNCH_CONFIG_MODE="${MORI_EP_LAUNCH_CONFIG_MODE:-MANUAL}"
if [[ "${EP_SIZE}" == "8" ]]; then
  export MORI_EP_TUNING_CONFIG="${MORI_EP_TUNING_CONFIG:-${script_dir}/mimo_v25_mori_ep8_comm.json}"
else
  export MORI_EP_TUNING_CONFIG="${MORI_EP_TUNING_CONFIG:-}"
fi
case "${MORI_EP_LAUNCH_CONFIG_MODE}" in
  MANUAL | AUTO) ;;
  *)
    echo "MORI_EP_LAUNCH_CONFIG_MODE must be MANUAL or AUTO, got: ${MORI_EP_LAUNCH_CONFIG_MODE}" >&2
    exit 2
    ;;
esac
for name in MAX_RUNNING_REQUESTS CHUNKED_PREFILL_SIZE; do
  if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be positive for this bounded-buffer setup" >&2
    exit 2
  fi
done
max_running_per_dp=$((MAX_RUNNING_REQUESTS / dp_size))
chunked_prefill_per_dp=$((CHUNKED_PREFILL_SIZE / dp_size))
if (( max_running_per_dp < 1 || chunked_prefill_per_dp < 1 )); then
  echo "Request and prefill limits must allow at least one item per DP" >&2
  exit 2
fi
# SGLang checks dispatch capacity against the effective per-DP chunk.
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-1024}"
# Keep MORI's full transport allocation: IntraNode combine uses send-capacity
# strides even when a smaller receive cap is configured.
export SGLANG_MORI_PREALLOC_MAX_RECV_TOKENS="${SGLANG_MORI_PREALLOC_MAX_RECV_TOKENS:-0}"
mori_decode_required=$(( ((max_running_per_dp + 7) / 8) * 4 ))
mori_dispatch_required=${chunked_prefill_per_dp}
if (( mori_decode_required > mori_dispatch_required )); then
  mori_dispatch_required=${mori_decode_required}
fi
if ! [[ "${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}" =~ ^[1-9][0-9]*$ ]] || \
   (( SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK < mori_dispatch_required )); then
  echo "MORI dispatch capacity must be at least ${mori_dispatch_required} for chunk-per-dp=${chunked_prefill_per_dp} and max-running-per-dp=${max_running_per_dp}" >&2
  exit 2
fi
mori_receive_required=$(( mori_decode_required * EP_SIZE ))
mori_prefill_required=$(( ((chunked_prefill_per_dp + 7) / 8) * 8 * dp_size ))
if (( mori_prefill_required > mori_receive_required )); then
  mori_receive_required=${mori_prefill_required}
fi
# Limit only the MoE tensor view, covering every possible global input token.
# Larger prefill chunks automatically exceed FlyDSL's guard and fall back.
export SGLANG_MORI_MOE_MAX_INPUT_TOKENS="${SGLANG_MORI_MOE_MAX_INPUT_TOKENS:-${mori_receive_required}}"
if ! [[ "${SGLANG_MORI_MOE_MAX_INPUT_TOKENS}" =~ ^(0|[1-9][0-9]*)$ ]] || \
   (( SGLANG_MORI_MOE_MAX_INPUT_TOKENS > 0 && SGLANG_MORI_MOE_MAX_INPUT_TOKENS < mori_receive_required )); then
  echo "MORI MoE input view must cover ${mori_receive_required} tokens; 0 disables the view limit" >&2
  exit 2
fi
if ! [[ "${SGLANG_MORI_PREALLOC_MAX_RECV_TOKENS}" =~ ^(0|[1-9][0-9]*)$ ]] || \
  (( SGLANG_MORI_PREALLOC_MAX_RECV_TOKENS > 0 && ((SGLANG_MORI_PREALLOC_MAX_RECV_TOKENS + EP_SIZE - 1) / EP_SIZE) * EP_SIZE < mori_receive_required )); then
  echo "MORI receive capacity must cover ${mori_receive_required} tokens for chunk=${CHUNKED_PREFILL_SIZE} and max-running=${MAX_RUNNING_REQUESTS}; 0 disables the receive cap" >&2
  exit 2
fi
export SGLANG_ENABLE_WAR_BARRIER="${SGLANG_ENABLE_WAR_BARRIER:-0}"
export ENABLE_TBO="${ENABLE_TBO:-0}"
# Benchmark-only: replace learned routing with random logits when enabled.
# turn it off when evaluating accuracy
export ENABLE_FAKE_EP_DISPATCH="${ENABLE_FAKE_EP_DISPATCH:-0}"
export TBO_MIN_EXTEND_TOKENS="${TBO_MIN_EXTEND_TOKENS:-2048}"
export TBO_TOKEN_DISTRIBUTION_THRESHOLD="${TBO_TOKEN_DISTRIBUTION_THRESHOLD:-0.48}"

case "${ENABLE_TBO}" in
  0 | 1) ;;
  *)
    echo "ENABLE_TBO must be 0 or 1, got: ${ENABLE_TBO}" >&2
    exit 2
    ;;
esac

tbo_attn_comm_override="${SGLANG_MIMO_TBO_ATTN_COMM:-}"
case "${tbo_attn_comm_override}" in
  "" | 0 | 1) ;;
  *)
    echo "SGLANG_MIMO_TBO_ATTN_COMM must be 0 or 1, got: ${tbo_attn_comm_override}" >&2
    exit 2
    ;;
esac

# Attention-communication overlap is a TBO subfeature. Default it to enabled
# with TBO, allow an explicit opt-out, and force it off whenever TBO is off.
if [[ "${ENABLE_TBO}" == "1" ]]; then
  export SGLANG_MIMO_TBO_ATTN_COMM="${tbo_attn_comm_override:-1}"
else
  if [[ "${tbo_attn_comm_override}" == "1" ]]; then
    echo "ENABLE_TBO=0; forcing SGLANG_MIMO_TBO_ATTN_COMM=0"
  fi
  export SGLANG_MIMO_TBO_ATTN_COMM=0
fi

for toggle in SGLANG_MIMO_FUSED_RMS_QKV_QUANT SGLANG_AITER_MIMO_CACHED_BF16_SWA_VARLEN ENABLE_FAKE_EP_DISPATCH AITER_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950; do
  case "${!toggle}" in
    0 | 1) ;;
    *)
      echo "${toggle} must be 0 or 1, got: ${!toggle}" >&2
      exit 2
      ;;
  esac
done
if [[ "${ENABLE_TBO}" == "1" ]] &&
  ! [[ "${TBO_MIN_EXTEND_TOKENS}" =~ ^[0-9]+$ && "${TBO_MIN_EXTEND_TOKENS}" =~ [1-9] ]]; then
  echo "TBO_MIN_EXTEND_TOKENS must be a positive integer, got: ${TBO_MIN_EXTEND_TOKENS}" >&2
  exit 2
fi

# AITER loads package/model tuned CSVs by default and handles custom CSV lists.
# Untuned ablation: AITER_BYPASS_TUNE_CONFIG=1 AITER_ONLINE_TUNE=0 before launch.
if [[ "${MORI_EP_LAUNCH_CONFIG_MODE}" == "AUTO" &&
  ( ! -f "${MORI_EP_TUNING_CONFIG}" || ! -r "${MORI_EP_TUNING_CONFIG}" ) ]]; then
  echo "MORI_EP_TUNING_CONFIG requires a readable file, got: ${MORI_EP_TUNING_CONFIG}" >&2
  exit 2
fi

# Real MTP acceptance for accuracy validation.
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD

export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_DIR="${LOG_DIR:-./logs/accuracy_${RUN_ID}}"
export LOG_FILE="${LOG_FILE:-server_tp${EP_SIZE}_ep${EP_SIZE}_aiter_mori_tbo_mtp_accuracy.log}"

mkdir -p "${LOG_DIR}"

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
#  --kv-cache-dtype fp8_e4m3 \

ep_dispatch_args=()
if [[ "${ENABLE_FAKE_EP_DISPATCH}" == "1" ]]; then
  ep_dispatch_args+=(--ep-dispatch-algorithm fake)
fi

tbo_args=()
tbo_status=disabled
if [[ "${ENABLE_TBO}" == "1" ]]; then
  tbo_args+=(
    --enable-two-batch-overlap
    --tbo-min-extend-tokens "${TBO_MIN_EXTEND_TOKENS}"
    --tbo-token-distribution-threshold "${TBO_TOKEN_DISTRIBUTION_THRESHOLD}"
  )
  tbo_status=enabled
fi

# MiMo TP8/EP8 TBO decode/target-verify graphs are conservatively aligned to
# 2 TBO children * attention TP8 = 16 requests. In
# validation, a smaller request pool reproducibly produced non-finite target-
# verification logits. Keep at least one complete graph bucket unless decode
# graphs are disabled; the exact undersized-pool failure site remains broader
# than this launcher-side guard.
if [[ "${ENABLE_TBO}" == "1" && "${CUDA_GRAPH_BACKEND_DECODE}" != "disabled" ]]; then
  if (( max_running_per_dp < 16 )); then
    echo "Attention-TP8 TBO decode graph requires at least 16 requests per DP, got: ${max_running_per_dp}" >&2
    exit 2
  fi
fi

{
  echo "Working directory: ${PWD}"
  echo "Topology: tp=${EP_SIZE}, ep=${EP_SIZE}, dp=${dp_size}, attention-tp=8, nodes=${nnodes}, node-rank=${NODE_RANK}, dist-init=${DIST_INIT_ADDR:-local}"
  echo "Per-DP limits: max-running=${max_running_per_dp}, chunked-prefill=${chunked_prefill_per_dp}"
  echo "Network: gloo=${GLOO_SOCKET_IFNAME:-auto}, mori=${MORI_SOCKET_IFNAME:-auto}, nccl=${NCCL_SOCKET_IFNAME:-auto}, rdma=${MORI_RDMA_DEVICES:-auto}, gid=${MORI_IB_GID_INDEX:-auto}, torch-blocking-wait=${TORCH_NCCL_BLOCKING_WAIT:-unset}"
  echo "Attention hybrid: prefill-flydsl=${SGLANG_FLYDSL_MIMO_PREFILL}, target-verify=${SGLANG_AITER_PA_DECODE_IMPL}, SWA/sink and ordinary decode=AITER/Gluon"
  echo "Custom all-reduce: ${custom_all_reduce_status}"
  echo "Radix cache: ${radix_cache_status}"
  echo "HIP non-greedy EAGLE verifier: ${SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY}"
  echo "Configuration: model=${MODEL_PATH}, max-running=${MAX_RUNNING_REQUESTS}, page=64, chunked-prefill=${CHUNKED_PREFILL_SIZE}, tp=${EP_SIZE}, ep=${EP_SIZE}, partitions=${SGLANG_FLYDSL_PA_NUM_PARTITIONS}, mem=${MEM_FRACTION_STATIC}, swa=${SWA_FULL_TOKENS_RATIO}, kv-cache-dtype=${KV_CACHE_DTYPE}, quick-ar=${ROCM_QUICK_REDUCE_QUANTIZATION:-unset}, mixed-router=${SGLANG_MIMO_MIXED_ROUTER}, fused-rms-moe=${SGLANG_MIMO_FUSED_RMS_MOE_QUANT}, fused-rms-qkv=${SGLANG_MIMO_FUSED_RMS_QKV_QUANT}, fresh-bf16-asm=${SGLANG_AITER_MIMO_FRESH_BF16_ASM}, fresh-bf16-varlen=${SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN}, fresh-bf16-swa-varlen=${SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN}, mtp=1, mori-mode=normal, mori-launch-config=${MORI_EP_LAUNCH_CONFIG_MODE}, tbo=${tbo_status}, attn-comm-tbo=${SGLANG_MIMO_TBO_ATTN_COMM}, war-barrier=${SGLANG_ENABLE_WAR_BARRIER}, decode-graph=${CUDA_GRAPH_BACKEND_DECODE}, decode-graph-bs=${CUDA_GRAPH_BS_DECODE:-default}, seed=${SERVER_RANDOM_SEED}, reasoning-parser=${REASONING_PARSER}, overlap=enabled"
  echo "Server log: ${LOG_DIR}/${LOG_FILE}"
  echo "TBO minimum EXTEND tokens: ${TBO_MIN_EXTEND_TOKENS}"
  echo "TBO token distribution threshold: ${TBO_TOKEN_DISTRIBUTION_THRESHOLD}"
  echo "Cached SWA fast path: ${SGLANG_AITER_MIMO_CACHED_BF16_SWA_VARLEN}"
  echo "MORI transport: dispatch=${SGLANG_MORI_DISPATCH_DTYPE}, combine=${SGLANG_MORI_COMBINE_DTYPE}, dispatch-capacity-per-rank=${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK}, receive-cap=${SGLANG_MORI_PREALLOC_MAX_RECV_TOKENS}, moe-input-view=${SGLANG_MORI_MOE_MAX_INPUT_TOKENS}, sdma=${MORI_ENABLE_SDMA}"
  echo "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS}"
  echo "Fake EP dispatch: ${ENABLE_FAKE_EP_DISPATCH}"
  echo "FlyDSL MoE opt-in (EP, FP8 Block128, gfx950): ${AITER_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950}"
  if [[ "${MORI_EP_LAUNCH_CONFIG_MODE}" == "AUTO" ]]; then
    echo "MORI tuning config: ${MORI_EP_TUNING_CONFIG}"
  else
    echo "MORI tuning config: inactive (MANUAL)"
  fi
  echo "AITER FMoE: config=${AITER_CONFIG_FMOE:-package defaults}, bypass=${AITER_BYPASS_TUNE_CONFIG:-0}, online-tune=${AITER_ONLINE_TUNE:-0}"
  echo "MTP: EAGLE, steps=3, top-k=1, draft-tokens=4, multi-layer=enabled"
} | tee "${LOG_DIR}/launcher_config.log"

python3 -u -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  "${parallel_args[@]}" \
  --moe-a2a-backend mori \
  --moe-runner-backend aiter \
  --deepep-mode normal \
  "${ep_dispatch_args[@]}" \
  "${tbo_args[@]}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --host 0.0.0.0 \
  --port 30001 \
  --random-seed "${SERVER_RANDOM_SEED}" \
  --trust-remote-code \
  --reasoning-parser "${REASONING_PARSER}" \
  --tool-call-parser mimo \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
  --context-length 1048576 \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --max-prefill-tokens "${CHUNKED_PREFILL_SIZE}" \
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
