#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="${SGLANG_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
MIMO_ROOT="${MIMO_ROOT:-$(dirname -- "${SGLANG_ROOT}")}"
AITER_ROOT="${AITER_ROOT:-${MIMO_ROOT}/aiter-mimo-fp4-dflash}"

MODEL_ROOT="${MODEL_ROOT:-/models/MiMo-V2.6-Flash-RL}"
DRAFT_MODEL_ROOT="${DRAFT_MODEL_ROOT:-${MODEL_ROOT}/dflash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mimo-v2.6-flash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TOKENIZER_WORKER_NUM="${TOKENIZER_WORKER_NUM:-1}"
SERVER_RANDOM_SEED="${SERVER_RANDOM_SEED:-}"

# The fused QKV checkpoint is TP4-interleaved. TP8/DP1 is rejected by SGLang.
TP_SIZE="${TP_SIZE:-4}"
EP_SIZE="${EP_SIZE:-1}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.60}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.03}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-65536}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"
PAGE_SIZE="${PAGE_SIZE:-64}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bf16}"
SPECULATIVE_DRAFT_KV_CACHE_DTYPE="${SPECULATIVE_DRAFT_KV_CACHE_DTYPE:-bf16}"
SPECULATIVE_DRAFT_WINDOW_SIZE="${SPECULATIVE_DRAFT_WINDOW_SIZE:-1024}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-8}"
AITER_MXFP4_STAGE2_OUTPUT_DTYPE="${AITER_MXFP4_STAGE2_OUTPUT_DTYPE:-fp8}"
ENABLE_CUDA_GRAPH="${ENABLE_CUDA_GRAPH:-0}"
ENABLE_TWO_BATCH_OVERLAP="${ENABLE_TWO_BATCH_OVERLAP:-0}"
DISABLE_OVERLAP_SCHEDULE="${DISABLE_OVERLAP_SCHEDULE:-1}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/mimo_v2_6_flash_day0_${RUN_ID}}"

if [[ "${TP_SIZE}" != "4" ]]; then
  echo "MiMo-V2.6-Flash fused QKV requires TP_SIZE=4; got ${TP_SIZE}" >&2
  exit 2
fi
if [[ "${EP_SIZE}" != "1" ]]; then
  echo "This MI355X day-0 launcher currently supports EP_SIZE=1 only" >&2
  exit 2
fi
if [[ "${PAGE_SIZE}" != "64" ]]; then
  echo "This MI355X day-0 launcher requires PAGE_SIZE=64; got ${PAGE_SIZE}" >&2
  exit 2
fi
if [[ "${KV_CACHE_DTYPE}" != "bf16" || "${SPECULATIVE_DRAFT_KV_CACHE_DTYPE}" != "bf16" ]]; then
  echo "The initial MiMo-V2.6-Flash qualification requires BF16 target and draft KV caches" >&2
  exit 2
fi
if [[ "${SPECULATIVE_NUM_DRAFT_TOKENS}" != "8" ]]; then
  echo "MiMo-V2.6-Flash DFlash requires SPECULATIVE_NUM_DRAFT_TOKENS=8" >&2
  exit 2
fi
if [[ "${SPECULATIVE_DRAFT_WINDOW_SIZE}" != "1024" ]]; then
  echo "MiMo-V2.6-Flash DFlash requires SPECULATIVE_DRAFT_WINDOW_SIZE=1024" >&2
  exit 2
fi
for flag in ENABLE_CUDA_GRAPH ENABLE_TWO_BATCH_OVERLAP DISABLE_OVERLAP_SCHEDULE DISABLE_RADIX_CACHE DISABLE_CUSTOM_ALL_REDUCE DRY_RUN; do
  value="${!flag}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${flag} must be 0 or 1; got ${value}" >&2
    exit 2
  fi
done
if [[ ! -d "${AITER_ROOT}/aiter" ]]; then
  echo "AITER_ROOT does not contain an aiter package: ${AITER_ROOT}" >&2
  exit 2
fi
if ! [[ "${MAX_RUNNING_REQUESTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RUNNING_REQUESTS must be a positive integer" >&2
  exit 2
fi
if ! [[ "${TOKENIZER_WORKER_NUM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TOKENIZER_WORKER_NUM must be a positive integer" >&2
  exit 2
fi
if [[ -n "${SERVER_RANDOM_SEED}" ]] && ! [[ "${SERVER_RANDOM_SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "SERVER_RANDOM_SEED must be a non-negative integer or unset" >&2
  exit 2
fi

# Fail before allocating GPUs when a checkpoint download or DFlash config is
# incomplete. This also verifies the model-specific TP and DFlash contract.
python3 - "${MODEL_ROOT}" "${DRAFT_MODEL_ROOT}" <<'PY'
import json
import os
import sys

model_root, draft_root = sys.argv[1:]


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in {path}: {exc}. For the published Flash DFlash "
            "config, apply Hugging Face discussion #2."
        ) from exc


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
    "hidden_size": 4096,
    "num_hidden_layers": 48,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "swa_num_attention_heads": 64,
    "swa_num_key_value_heads": 8,
    "head_dim": 192,
    "v_head_dim": 128,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "moe_router_dtype": "bfloat16",
}
for key, expected in expected_target.items():
    actual = target.get(key)
    if actual != expected:
        raise RuntimeError(
            f"Unexpected MiMo-V2.6-Flash target config {key}: "
            f"expected {expected!r}, got {actual!r}"
        )
if quant.get("quant_method") != "fp8" or quant.get("store_dtype") != "mxfp4":
    raise RuntimeError(f"Unexpected target quantization_config: {quant!r}")
if target_index.get("metadata", {}).get("tp_size") != 4:
    raise RuntimeError(
        "MiMo-V2.6-Flash target index must declare tp_size=4 for fused QKV"
    )

draft_cfg = draft.get("dflash_config") or {}
expected_draft = {
    "architectures": ["DFlashDraftModel"],
    "hidden_size": 4096,
    "num_hidden_layers": 5,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "block_size": 8,
    "sliding_window": 1024,
    "num_target_layers": 48,
    "target_hidden_size": 4096,
    "vocab_size": 152576,
}
for key, expected in expected_draft.items():
    actual = draft.get(key)
    if actual != expected:
        raise RuntimeError(
            f"Unexpected MiMo-V2.6-Flash draft config {key}: "
            f"expected {expected!r}, got {actual!r}"
        )
if draft_cfg.get("target_layer_ids") != [0, 11, 23, 35, 47]:
    raise RuntimeError(f"Unexpected DFlash target layers: {draft_cfg!r}")
if draft_cfg.get("mask_token_id") != 151675:
    raise RuntimeError(f"Unexpected DFlash mask token: {draft_cfg!r}")

print("MiMo-V2.6-Flash checkpoint preflight: PASS")
PY

mkdir -p "${LOG_DIR}"

cuda_graph_args=()
if [[ "${ENABLE_CUDA_GRAPH}" != "1" ]]; then
  cuda_graph_args+=(--disable-cuda-graph)
else
  cuda_graph_args+=(--cuda-graph-max-bs-decode "${MAX_RUNNING_REQUESTS}")
fi

seed_args=()
if [[ -n "${SERVER_RANDOM_SEED}" ]]; then
  seed_args+=(--random-seed "${SERVER_RANDOM_SEED}")
fi

capacity_args=()
if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
  capacity_args+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi

tbo_args=()
if [[ "${ENABLE_TWO_BATCH_OVERLAP}" == "1" ]]; then
  tbo_args+=(--enable-two-batch-overlap)
fi

overlap_args=()
if [[ "${DISABLE_OVERLAP_SCHEDULE}" == "1" ]]; then
  overlap_args+=(--disable-overlap-schedule)
fi

radix_args=()
if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  radix_args+=(--disable-radix-cache)
fi

custom_all_reduce_args=()
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
  custom_all_reduce_args+=(--disable-custom-all-reduce)
fi

server_cmd=(
  python3 -u -m sglang.launch_server
  --model-path "${MODEL_ROOT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP_SIZE}"
  --ep-size "${EP_SIZE}"
  --tokenizer-worker-num "${TOKENIZER_WORKER_NUM}"
  "${seed_args[@]}"
  --trust-remote-code
  --quantization fp8
  --moe-runner-backend aiter
  --aiter-mxfp4-stage2-output-dtype "${AITER_MXFP4_STAGE2_OUTPUT_DTYPE}"
  --attention-backend aiter
  --speculative-draft-attention-backend aiter
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --speculative-draft-kv-cache-dtype "${SPECULATIVE_DRAFT_KV_CACHE_DTYPE}"
  --page-size "${PAGE_SIZE}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  "${capacity_args[@]}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --max-prefill-tokens "${MAX_PREFILL_TOKENS}"
  --context-length "${CONTEXT_LENGTH}"
  --cuda-graph-backend-prefill disabled
  --speculative-algorithm DFLASH
  --speculative-draft-model-path "${DRAFT_MODEL_ROOT}"
  --speculative-draft-window-size "${SPECULATIVE_DRAFT_WINDOW_SIZE}"
  --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
  --reasoning-parser mimo
  --tool-call-parser mimo
  --mm-enable-dp-encoder
  --mm-attention-backend aiter_attn
  --host "${HOST}"
  --port "${PORT}"
  "${cuda_graph_args[@]}"
  "${tbo_args[@]}"
  "${overlap_args[@]}"
  "${radix_args[@]}"
  "${custom_all_reduce_args[@]}"
  "$@"
)

echo "MiMo-V2.6-Flash MI355X day-0 configuration:"
echo "  target=${MODEL_ROOT}"
echo "  draft=${DRAFT_MODEL_ROOT}"
echo "  tp=${TP_SIZE}, ep=${EP_SIZE}, page=${PAGE_SIZE}"
echo "  kv=${KV_CACHE_DTYPE}, draft-kv=${SPECULATIVE_DRAFT_KV_CACHE_DTYPE}"
echo "  mem=${MEM_FRACTION_STATIC}, swa-ratio=${SWA_FULL_TOKENS_RATIO}"
echo "  attention=Gluon qlen-1 target verify, cuda-graph=${ENABLE_CUDA_GRAPH}, tbo=${ENABLE_TWO_BATCH_OVERLAP}"
echo "  tokenizer-workers=${TOKENIZER_WORKER_NUM}, server-seed=${SERVER_RANDOM_SEED:-auto}"
printf 'Command:'
printf ' %q' "${server_cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

unset ROCM_QUICK_REDUCE_QUANTIZATION

exec env \
  HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}" \
  PYTHONPATH="${SGLANG_ROOT}/python:${AITER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  SGLANG_USE_AITER=1 \
  MIMO_TARGET_WEIGHT_FORMAT=mxfp4 \
  SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d \
  SGLANG_USE_AITER_UNIFIED_ATTN=1 \
  SGLANG_FLYDSL_MIMO_PREFILL=0 \
  SGLANG_FLYPA_MIMO_PREFILL=0 \
  SGLANG_AITER_PA_DECODE_IMPL=gluon \
  SGLANG_AITER_TARGET_VERIFY_SWA_IMPL=gluon \
  SGLANG_AITER_DFLASH_SWA_IMPL=gluon \
  SGLANG_AITER_VEC5D_TARGET_VERIFY_QLEN1=1 \
  SGLANG_AITER_MIMO_FRESH_BF16_ASM=1 \
  SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN=1 \
  SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN=0 \
  SGLANG_MIMO_FUSED_RMS_QKV_QUANT=1 \
  SGLANG_MIMO_FUSED_RMS_MOE_QUANT=1 \
  SGLANG_MIMO_MIXED_ROUTER=0 \
  SGLANG_MOE_PADDING=1 \
  SGLANG_SET_CPU_AFFINITY=1 \
  SGLANG_USE_AITER_MOE_GU_ITLV=1 \
  SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0 \
  SGLANG_SPEC_NAN_DETECTION=1 \
  SGLANG_SPEC_OOB_DETECTION=1 \
  SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY=1 \
  SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE=1 \
  AITER_LOG_TUNED_CONFIG=0 \
  HSA_NO_SCRATCH_RECLAIM=1 \
  MC_GID_INDEX=3 \
  MC_TE_METRIC=1 \
  NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-112}" \
  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
  "${server_cmd[@]}" 2>&1 | tee "${LOG_DIR}/server.log"
