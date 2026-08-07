#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_SHAREGPT_DATASET="${SCRIPT_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_DATASET_URL="${SHAREGPT_DATASET_URL:-https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json}"
SHAREGPT_DATASET="${SHAREGPT_DATASET:-${DEFAULT_SHAREGPT_DATASET}}"

if [[ "${SHAREGPT_DATASET}" == "${DEFAULT_SHAREGPT_DATASET}" && ! -f "${SHAREGPT_DATASET}" ]]; then
  tmp_dataset="$(mktemp "${SCRIPT_DIR}/.ShareGPT_V3_unfiltered_cleaned_split.json.tmp.XXXXXX")"
  echo "ShareGPT dataset not found in ${SCRIPT_DIR}; downloading to ${SHAREGPT_DATASET}"
  curl -L --fail --retry 3 --connect-timeout 10 \
    --output "${tmp_dataset}" \
    "${SHAREGPT_DATASET_URL}"
  mv "${tmp_dataset}" "${SHAREGPT_DATASET}"
fi

if [[ ! -f "${SHAREGPT_DATASET}" ]]; then
  echo "ShareGPT dataset not found: ${SHAREGPT_DATASET}" >&2
  echo "Set SHAREGPT_DATASET to an existing file, or allow download to ${DEFAULT_SHAREGPT_DATASET}." >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/mtp_sharegpt_${RUN_ID}}"
RESULTS_JSONL="${RESULTS_JSONL:-${LOG_DIR}/results.jsonl}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-4}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
MIN_ACCEPT_LENGTH="${MIN_ACCEPT_LENGTH:-3.0}"
MIN_ACCEPT_RATE="${MIN_ACCEPT_RATE:-0.67}"
MODEL_PATH="${MODEL_PATH:-/models/MiMo-V2.5-Pro/}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-30001}"
SPECULATIVE_STEPS="${SPECULATIVE_STEPS:-3}"
mkdir -p "${LOG_DIR}"

unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD
curl --max-time 10 -fsS "http://${SERVER_HOST}:${SERVER_PORT}/v1/models" >/dev/null

{
  echo "run_id=${RUN_ID}"
  echo "dataset=${SHAREGPT_DATASET}"
  echo "num_prompts=${NUM_PROMPTS}"
  echo "warmup_requests=${WARMUP_REQUESTS}"
  echo "max_concurrency=${MAX_CONCURRENCY}"
  echo "min_accept_length=${MIN_ACCEPT_LENGTH}"
  echo "min_accept_rate=${MIN_ACCEPT_RATE}"
  echo "SGLANG_SIMULATE_ACC_LEN=${SGLANG_SIMULATE_ACC_LEN:-unset}"
  echo "SGLANG_SIMULATE_ACC_METHOD=${SGLANG_SIMULATE_ACC_METHOD:-unset}"
} > "${LOG_DIR}/benchmark_config.txt"

python3 -m sglang.bench_serving \
  --backend sglang \
  --model "${MODEL_PATH}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --dataset-name sharegpt \
  --dataset-path "${SHAREGPT_DATASET}" \
  --flush-cache \
  --seed 12345 \
  --num-prompts "${NUM_PROMPTS}" \
  --warmup-requests "${WARMUP_REQUESTS}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --temperature 0 \
  --output-file "${RESULTS_JSONL}" \
  --output-details \
  --tag "${RUN_ID}_real_mtp_sharegpt" \
  2>&1 | tee "${LOG_DIR}/benchmark_sharegpt_con${MAX_CONCURRENCY}.log"

python3 - "${RESULTS_JSONL}" "${LOG_DIR}/gate_summary.json" \
  "${NUM_PROMPTS}" "${SPECULATIVE_STEPS}" "${MIN_ACCEPT_LENGTH}" "${MIN_ACCEPT_RATE}" <<'PY'
import json
import sys
from pathlib import Path

results_path, summary_path = map(Path, sys.argv[1:3])
expected_requests = int(sys.argv[3])
speculative_steps = int(sys.argv[4])
min_accept_length = float(sys.argv[5])
min_accept_rate = float(sys.argv[6])

lines = [line for line in results_path.read_text().splitlines() if line.strip()]
if not lines:
    raise SystemExit("ShareGPT result JSONL is empty")
result = json.loads(lines[-1])
completed = int(result.get("completed", 0))
accept_length = result.get("accept_length")
if accept_length is None:
    raise SystemExit("Server did not report avg_spec_accept_length")
accept_length = float(accept_length)
accept_rate = (accept_length - 1.0) / speculative_steps
errors = [error for error in result.get("errors", []) if error]

summary = {
    "completed": completed,
    "expected_requests": expected_requests,
    "accept_length": accept_length,
    "accept_rate": accept_rate,
    "min_accept_length": min_accept_length,
    "min_accept_rate": min_accept_rate,
    "error_count": len(errors),
    "output_throughput": result.get("output_throughput"),
    "mean_tpot_ms": result.get("mean_tpot_ms"),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

if completed != expected_requests:
    raise SystemExit(f"completed {completed}/{expected_requests} requests")
if errors:
    raise SystemExit(f"ShareGPT returned {len(errors)} request errors")
if accept_length < min_accept_length:
    raise SystemExit(
        f"accept length {accept_length:.6f} is below {min_accept_length:.6f}"
    )
if accept_rate < min_accept_rate:
    raise SystemExit(f"accept rate {accept_rate:.6f} is below {min_accept_rate:.6f}")
PY
