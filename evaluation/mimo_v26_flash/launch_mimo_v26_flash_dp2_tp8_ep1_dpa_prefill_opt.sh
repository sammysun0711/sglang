#!/usr/bin/env bash
set -euo pipefail

# MI300X prefill: PR44 TP SP + PR40 FP8 MoE, BF16 target/draft KV.
# The underlying launcher converts MXFP4 experts to static FP8 block scales.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SGLANG_ENABLE_DPA_SP="${SGLANG_ENABLE_DPA_SP:-1}"
export SGLANG_ENABLE_MIMO_SWA_SP="${SGLANG_ENABLE_MIMO_SWA_SP:-0}"
export SGLANG_MIMO_TP_SP_FULL_OPROJ="${SGLANG_MIMO_TP_SP_FULL_OPROJ:-1}"
export SGLANG_MIMO_PREFILL_PYHIP_MOE="${SGLANG_MIMO_PREFILL_PYHIP_MOE:-1}"
export SGLANG_USE_PYHIP_FP8_BLOCKSCALE_GEMM="${SGLANG_USE_PYHIP_FP8_BLOCKSCALE_GEMM:-1}"
export SGLANG_USE_PYHIP_MIMO_PREFILL="${SGLANG_USE_PYHIP_MIMO_PREFILL:-1}"
export SGLANG_MXFP4_TO_FP8_128BLOCK="${SGLANG_MXFP4_TO_FP8_128BLOCK:-1}"
export FLYDSL_RUNTIME_ENABLE_CACHE="${FLYDSL_RUNTIME_ENABLE_CACHE:-1}"
export FLYDSL_AUTOTUNE="${FLYDSL_AUTOTUNE:-0}"

export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
export SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.03}"
export ENABLE_DFLASH="${ENABLE_DFLASH:-1}"
export SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-12345}"

# Optional isolated package installation; otherwise use the active environment.
fastokens_site="${FASTOKENS_SITE:-${FASTOKENS_EXPERIMENT_SITE:-}}"
if [[ -n "${fastokens_site}" ]]; then
  if [[ ! -d "${fastokens_site}" ]]; then
    echo "Fastokens directory does not exist: ${fastokens_site}" >&2
    exit 2
  fi
  export PYTHONPATH="${fastokens_site}${PYTHONPATH:+:${PYTHONPATH}}"
fi

prefill_args=(
  --moe-dense-tp-size 1
  --tokenizer-worker-num "${TOKENIZER_WORKER_NUM:-8}"
  --tokenizer-backend "${TOKENIZER_BACKEND:-fastokens}"
  --disable-tokenizer-batch-decode
)
case "${ENABLE_PREFILL_DELAYER:-1}" in
  1)
    prefill_args+=(
      --enable-prefill-delayer
      --prefill-delayer-max-delay-passes "${PREFILL_DELAYER_MAX_DELAY_PASSES:-100}"
    )
    ;;
  0) ;;
  *) echo "ENABLE_PREFILL_DELAYER must be 0 or 1" >&2; exit 2 ;;
esac

exec bash "${SCRIPT_DIR}/launch_mimo_v26_flash_dp2_tp8_ep1_dpa_accuracy_baseline.sh" \
  "${prefill_args[@]}" "$@"
