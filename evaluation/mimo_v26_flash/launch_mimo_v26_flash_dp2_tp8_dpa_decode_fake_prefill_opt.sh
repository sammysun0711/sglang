#!/usr/bin/env bash
# Validated native-MXFP4/BF16 DPA decode profile for fake-prefill evaluation.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SGLANG_MXFP4_TO_FP8_128BLOCK="${SGLANG_MXFP4_TO_FP8_128BLOCK:-0}"
export SGLANG_USE_ROCM700A="${SGLANG_USE_ROCM700A:-0}"
export SGLANG_ENABLE_DPA_SP="${SGLANG_ENABLE_DPA_SP:-1}"
export SGLANG_MIMO_GRAPH_VERIFY_SP="${SGLANG_MIMO_GRAPH_VERIFY_SP:-1}"
exec bash "${script_dir}/launch_mimo_v26_flash_dp2_tp8_ep1_dpa_decode_fake_prefill_baseline.sh" "$@"
