#!/usr/bin/env bash
# Validated native-MXFP4/BF16 DPA decode profile for fake-prefill evaluation.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_MXFP4_TO_FP8_128BLOCK="${SGLANG_MXFP4_TO_FP8_128BLOCK:-0}"
export SGLANG_USE_ROCM700A="${SGLANG_USE_ROCM700A:-0}"
export SGLANG_ENABLE_DPA_SP="${SGLANG_ENABLE_DPA_SP:-1}"
export SGLANG_MIMO_GRAPH_VERIFY_SP="${SGLANG_MIMO_GRAPH_VERIFY_SP:-1}"
export SGLANG_MIMO_GRAPH_SP_PACKED="${SGLANG_MIMO_GRAPH_SP_PACKED:-1}"
export SGLANG_MIMO_FUSED_SP_PACKET="${SGLANG_MIMO_FUSED_SP_PACKET:-1}"
export SGLANG_DFLASH_SWA_TOKEN_INTERVAL="${SGLANG_DFLASH_SWA_TOKEN_INTERVAL:-1}"
export SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER="${SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER:-1.0}"
export SGLANG_DFLASH_BATCH_SWA_EVICTION="${SGLANG_DFLASH_BATCH_SWA_EVICTION:-1}"
export SGLANG_MIMO_GROUPED_PA_SWA="${SGLANG_MIMO_GROUPED_PA_SWA:-1}"
export SGLANG_MIMO_SP_PACKED_OPROJ="${SGLANG_MIMO_SP_PACKED_OPROJ:-1}"
export SGLANG_DFLASH_FAST_CPU_STAGING="${SGLANG_DFLASH_FAST_CPU_STAGING:-1}"
export SGLANG_MIMO_FUSED_STABLE_ROUTER_WEIGHTS="${SGLANG_MIMO_FUSED_STABLE_ROUTER_WEIGHTS:-1}"
export AITER_FLYDSL_PA_BF16_BUFFER_CACHE="${AITER_FLYDSL_PA_BF16_BUFFER_CACHE:-2}"
export AITER_PACKED_ALLGATHER_SMALL_GRID="${AITER_PACKED_ALLGATHER_SMALL_GRID:-1}"
export SGLANG_DFLASH_GREEDY_CHUNK_SIZE="${SGLANG_DFLASH_GREEDY_CHUNK_SIZE:-2048}"
export SGLANG_DFLASH_REQUEST_OWNER_VERIFY="${SGLANG_DFLASH_REQUEST_OWNER_VERIFY:-1}"
export SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY="${SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY:-1}"
export SGLANG_DFLASH_PROB_SORT32="${SGLANG_DFLASH_PROB_SORT32:-1}"
export SGLANG_DFLASH_FUSED_CHAIN_VERIFY="${SGLANG_DFLASH_FUSED_CHAIN_VERIFY:-1}"
export SGLANG_DFLASH_PREFILL_REFILL_TARGET="${SGLANG_DFLASH_PREFILL_REFILL_TARGET:-1}"
export SGLANG_FAKE_PREFILL_ZERO_KV_ON_FLUSH="${SGLANG_FAKE_PREFILL_ZERO_KV_ON_FLUSH:-1}"
export TOKENIZER_WORKER_NUM="${TOKENIZER_WORKER_NUM:-4}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-480}"
export CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-240}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9411764705882354}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.03}"
default_args=()
has_dense_tp=0
has_tokenizer_backend=0
has_batch_decode=0
has_warmup=0
for arg in "$@"; do
  case "$arg" in
    --moe-dense-tp-size|--moe-dense-tp-size=*) has_dense_tp=1 ;;
    --tokenizer-backend|--tokenizer-backend=*) has_tokenizer_backend=1 ;;
    --disable-tokenizer-batch-decode) has_batch_decode=1 ;;
    --skip-server-warmup) has_warmup=1 ;;
  esac
done
if [[ "$has_dense_tp" == 0 ]]; then default_args+=(--moe-dense-tp-size 1); fi
if [[ "$has_tokenizer_backend" == 0 ]]; then default_args+=(--tokenizer-backend huggingface); fi
if [[ "$has_batch_decode" == 0 ]]; then default_args+=(--disable-tokenizer-batch-decode); fi
if [[ "$has_warmup" == 0 ]]; then default_args+=(--skip-server-warmup); fi
exec bash "${script_dir}/launch_mimo_v26_flash_dp2_tp8_ep1_dpa_decode_fake_prefill_baseline.sh" "${default_args[@]}" "$@"
