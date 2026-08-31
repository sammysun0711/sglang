#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/models/MiMo-V2.5-Pro-FP4-DFlash}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30001}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-./logs/mimo_fp4_dflash_decode_matrix_${RUN_ID}}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-1024}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-32}"
PROMPT_WAVES="${PROMPT_WAVES:-4}"
NUM_PROMPTS="${NUM_PROMPTS:-}"
mkdir -p "${LOG_DIR}"
curl -fsS --max-time 5 "http://${HOST}:${PORT}/health" >/dev/null

dataset_args=()
if [[ -n "${DATASET_PATH:-}" ]]; then
  dataset_args+=(--dataset-path "${DATASET_PATH}")
fi

if [[ -n "${CASE_SPECS:-}" ]]; then
  mapfile -t cases <<<"${CASE_SPECS}"
else
  cases=(
    "65532|65536|64k|16 32 48 64 80 88 96 112 114 116 118 120 124 126 128"
    "262140|262144|256k|16 32 36 38 39 40"
    "524284|524288|512k|16 18 19 20"
    "786428|786428|768k|12 13 14 16 17"
    "1047548|1047552|1m|8 9 10"
  )
fi

test_count=0
for case_spec in "${cases[@]}"; do
  IFS='|' read -r supplied_ids actual_input label concurrency_spec <<<"${case_spec}"
  read -r -a concurrencies <<<"${concurrency_spec}"
  for concurrency in "${concurrencies[@]}"; do
    if [[ -n "${NUM_PROMPTS}" ]]; then
      num_prompts="${NUM_PROMPTS}"
    else
      num_prompts="$((PROMPT_WAVES * concurrency))"
    fi
    log_file="${LOG_DIR}/benchmark_${label}_con${concurrency}.log"
    echo "Running ${label}: input=${actual_input}, output=${OUTPUT_TOKENS}, concurrency=${concurrency}, prompts=${num_prompts}"
    python3 -m sglang.bench_serving \
      --backend sglang --model "${MODEL}" --host "${HOST}" --port "${PORT}" \
      --dataset-name random --random-input-len "${supplied_ids}" \
      --random-output-len "${OUTPUT_TOKENS}" --random-range-ratio 1.0 \
      "${dataset_args[@]}" --flush-cache --seed 12345 \
      --num-prompts "${num_prompts}" --warmup-requests "${WARMUP_REQUESTS}" \
      --max-concurrency "${concurrency}" --tokenize-prompt --fake-prefill \
      2>&1 | tee "${log_file}"
    test_count=$((test_count + 1))
  done
done
echo "Completed ${test_count} decoding-server benchmark tests with fake prefill"
