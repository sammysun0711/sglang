#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVALUATION_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SGLANG_ROOT="${SGLANG_ROOT:-$(dirname -- "${EVALUATION_DIR}")}"
MIMO_ROOT="${MIMO_ROOT:-$(dirname -- "${SGLANG_ROOT}")}"

AITER_ROOT="${AITER_ROOT:-${MIMO_ROOT}/aiter-mimo-fp4-dflash}"
MODEL="${MODEL:-/models/MiMo-V2.6-Pro-RL}"
SPECULATIVE_DRAFT_MODEL="${SPECULATIVE_DRAFT_MODEL:-${MODEL}/dflash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.01}"
MM_ATTENTION_BACKEND="${MM_ATTENTION_BACKEND:-aiter_attn}"
CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE:-1 2 4 8 16}"
ENABLE_TWO_BATCH_OVERLAP="${ENABLE_TWO_BATCH_OVERLAP:-0}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/mimo_v2_6_pro_day0_${RUN_ID}}"
LOG_FILE="${LOG_FILE:-server.log}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "${AITER_ROOT}/aiter" ]]; then
  echo "AITER_ROOT does not contain an aiter package: ${AITER_ROOT}" >&2
  exit 2
fi
for flag in ENABLE_TWO_BATCH_OVERLAP DISABLE_RADIX_CACHE DRY_RUN; do
  value="${!flag}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${flag} must be 0 or 1; got ${value}" >&2
    exit 2
  fi
done

python3 - "${MODEL}" "${SPECULATIVE_DRAFT_MODEL}" <<'PY'
import json
import os
import sys

model_root, draft_root = sys.argv[1:]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_index(root, index_name="model.safetensors.index.json"):
    index_path = os.path.join(root, index_name)
    index = load_json(index_path)
    missing = sorted(
        filename
        for filename in set(index["weight_map"].values())
        if not os.path.isfile(os.path.join(root, filename))
    )
    if missing:
        raise RuntimeError(
            f"{index_path} references missing checkpoint files: {missing}"
        )
    return index


target = load_json(os.path.join(model_root, "config.json"))
draft = load_json(os.path.join(draft_root, "config.json"))
target_index = check_index(model_root)
check_index(draft_root)

quant = target.get("quantization_config") or {}
expected_target = {
    "architectures": ["MiMoV2ForCausalLM"],
    "hidden_size": 6144,
    "num_hidden_layers": 70,
    "num_attention_heads": 128,
    "num_key_value_heads": 8,
    "swa_num_attention_heads": 128,
    "swa_num_key_value_heads": 8,
    "head_dim": 192,
    "v_head_dim": 128,
    "n_routed_experts": 384,
    "num_experts_per_tok": 8,
    "moe_router_dtype": "bfloat16",
}
for key, expected in expected_target.items():
    actual = target.get(key)
    if actual != expected:
        raise RuntimeError(
            f"Unexpected MiMo-V2.6-Pro target config {key}: "
            f"expected {expected!r}, got {actual!r}"
        )
if quant.get("quant_method") != "fp8" or quant.get("store_dtype") != "mxfp4":
    raise RuntimeError(f"Unexpected target quantization_config: {quant!r}")
if target_index.get("metadata", {}).get("tp_size") != 8:
    raise RuntimeError(
        "MiMo-V2.6-Pro target index must declare tp_size=8 for fused QKV"
    )

draft_cfg = draft.get("dflash_config") or {}
expected_draft = {
    "architectures": ["DFlashDraftModel"],
    "hidden_size": 6144,
    "num_hidden_layers": 5,
    "num_attention_heads": 128,
    "num_key_value_heads": 8,
    "block_size": 8,
    "sliding_window": 1024,
    "num_target_layers": 70,
    "target_hidden_size": 6144,
    "vocab_size": 152576,
}
for key, expected in expected_draft.items():
    actual = draft.get(key)
    if actual != expected:
        raise RuntimeError(
            f"Unexpected MiMo-V2.6-Pro draft config {key}: "
            f"expected {expected!r}, got {actual!r}"
        )
if draft_cfg.get("target_layer_ids") != [0, 15, 31, 47, 69]:
    raise RuntimeError(f"Unexpected DFlash target layers: {draft_cfg!r}")
if draft_cfg.get("mask_token_id") != 151675:
    raise RuntimeError(f"Unexpected DFlash mask token: {draft_cfg!r}")

print("MiMo-V2.6-Pro checkpoint preflight: PASS")
PY

baseline_launcher="${EVALUATION_DIR}/mimo_dflash_scripts/launch_tp8_noep_aiter_dflash_accuracy_baseline.sh"
if [[ ! -x "${baseline_launcher}" && ! -f "${baseline_launcher}" ]]; then
  echo "Validated baseline launcher is missing: ${baseline_launcher}" >&2
  exit 2
fi

echo "MiMo-V2.6-Pro MI355X day-0 configuration:"
echo "  target=${MODEL}"
echo "  draft=${SPECULATIVE_DRAFT_MODEL}"
echo "  tp=8, ep=1, page=64"
echo "  kv=bf16, draft-kv=bf16"
echo "  mem=${MEM_FRACTION_STATIC}, swa-ratio=${SWA_FULL_TOKENS_RATIO}"
echo "  baseline-launcher=${baseline_launcher}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Command: env AITER_ROOT=%q TARGET_MODEL_VARIANT=mxfp4 MODEL=%q SPECULATIVE_DRAFT_MODEL=%q HOST=%q PORT=%q MAX_RUNNING_REQUESTS=%q MEM_FRACTION_STATIC=%q SWA_FULL_TOKENS_RATIO=%q MM_ATTENTION_BACKEND=%q CUDA_GRAPH_BS_DECODE=%q ENABLE_TWO_BATCH_OVERLAP=%q DISABLE_RADIX_CACHE=%q bash %q\n' \
    "${AITER_ROOT}" "${MODEL}" "${SPECULATIVE_DRAFT_MODEL}" "${HOST}" "${PORT}" \
    "${MAX_RUNNING_REQUESTS}" "${MEM_FRACTION_STATIC}" "${SWA_FULL_TOKENS_RATIO}" "${MM_ATTENTION_BACKEND}" \
    "${CUDA_GRAPH_BS_DECODE}" "${ENABLE_TWO_BATCH_OVERLAP}" \
    "${DISABLE_RADIX_CACHE}" "${baseline_launcher}"
  exit 0
fi

exec env \
  AITER_ROOT="${AITER_ROOT}" \
  TARGET_MODEL_VARIANT=mxfp4 \
  MODEL="${MODEL}" \
  SPECULATIVE_DRAFT_MODEL="${SPECULATIVE_DRAFT_MODEL}" \
  HOST="${HOST}" \
  PORT="${PORT}" \
  MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
  SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO}" \
  MM_ATTENTION_BACKEND="${MM_ATTENTION_BACKEND}" \
  CUDA_GRAPH_BS_DECODE="${CUDA_GRAPH_BS_DECODE}" \
  ENABLE_TWO_BATCH_OVERLAP="${ENABLE_TWO_BATCH_OVERLAP}" \
  DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE}" \
  RUN_ID="${RUN_ID}" \
  LOG_DIR="${LOG_DIR}" \
  LOG_FILE="${LOG_FILE}" \
  bash "${baseline_launcher}"
