#!/usr/bin/env bash
set -euo pipefail

read -r -a TOKEN_LIST <<< "${TOKEN_LIST_OVERRIDE:-4096 8192 16384 32768 65536 131068 262144 524284 786428 1047548}"
output_tokens=1
small_input_concurrency_list="${SMALL_INPUT_CONCURRENCY_LIST_OVERRIDE:-${SHORT_CONCURRENCY_LIST_OVERRIDE:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32}}"
short_concurrency_list="${SHORT_CONCURRENCY_LIST_OVERRIDE:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16}"
long_concurrency_list="${LONG_CONCURRENCY_LIST_OVERRIDE:-1}"
warmup_requests="${WARMUP_REQUESTS_OVERRIDE:-4}"
small_input_num_prompts="${SMALL_INPUT_NUM_PROMPTS_OVERRIDE:-64}"
prompt_waves="${PROMPT_WAVES:-4}"
min_num_prompts="${MIN_NUM_PROMPTS:-32}"
MODEL="${MODEL:-/models/MiMo-V2.5-Pro-FP4-DFlash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30001}"
LOG_DIR="${LOG_DIR:-./logs/benchmark_tp8_prefill}"
mkdir -p "${LOG_DIR}"

concurrency_spec_for_input() {
  local input_tokens="$1"
  if [[ -n "${CONCURRENCY_LIST_OVERRIDE:-}" ]]; then
    echo "${CONCURRENCY_LIST_OVERRIDE}"
  elif (( input_tokens <= 8192 )); then
    echo "${small_input_concurrency_list}"
  elif (( input_tokens <= 65536 )); then
    echo "${short_concurrency_list}"
  else
    echo "${long_concurrency_list}"
  fi
}

for input_tokens in "${TOKEN_LIST[@]}"; do
  read -r -a concurrency_list <<< "$(concurrency_spec_for_input "${input_tokens}")"
  for concurrency in "${concurrency_list[@]}"; do
    if [[ -n "${NUM_PROMPTS_OVERRIDE:-}" ]]; then
      num_prompts="${NUM_PROMPTS_OVERRIDE}"
    elif (( input_tokens <= 8192 )); then
      num_prompts="${small_input_num_prompts}"
    else
      num_prompts=$((prompt_waves * concurrency))
      if (( num_prompts < min_num_prompts )); then num_prompts="${min_num_prompts}"; fi
    fi
    echo "Testing input=${input_tokens}, concurrency=${concurrency}, prompts=${num_prompts}"
    python3 -m sglang.bench_serving \
      --backend sglang --model "${MODEL}" --host "${HOST}" --port "${PORT}" \
      --dataset-name random --random-input-len "${input_tokens}" \
      --random-output-len "${output_tokens}" --random-range-ratio 1.0 \
      --flush-cache --seed 12345 --num-prompts "${num_prompts}" \
      --warmup-requests "${warmup_requests}" --max-concurrency "${concurrency}" \
      2>&1 | tee "${LOG_DIR}/benchmark_${input_tokens}_con${concurrency}.log"
  done
done
