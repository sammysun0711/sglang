#!/usr/bin/env bash
set -euo pipefail

# 1P1D prefill benchmark
read -r -a TOKEN_LIST <<< "${TOKEN_LIST_OVERRIDE:-4096 8192 16384 32768 65536 131068 262144 524284 786428 1047548}"
output_tokens=1
small_input_concurrency_list="${SMALL_INPUT_CONCURRENCY_LIST_OVERRIDE:-${SHORT_CONCURRENCY_LIST_OVERRIDE:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32}}"
short_concurrency_list="${SHORT_CONCURRENCY_LIST_OVERRIDE:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16}"
long_concurrency_list="${LONG_CONCURRENCY_LIST_OVERRIDE:-1}"
warmup_requests="${WARMUP_REQUESTS_OVERRIDE:-4}"
small_input_num_prompts="${SMALL_INPUT_NUM_PROMPTS_OVERRIDE:-64}"
prompt_waves="${PROMPT_WAVES:-4}"
min_num_prompts="${MIN_NUM_PROMPTS:-32}"
LOG_DIR="${LOG_DIR:-./logs/benchmark_tp8_prefill}"
mkdir -p "$LOG_DIR"

if ! [[ "${small_input_num_prompts}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SMALL_INPUT_NUM_PROMPTS_OVERRIDE must be a positive integer, observed '${small_input_num_prompts}'" >&2
  exit 2
fi

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

# Loop execution: run each length multiple times (1 time), each run with a different concurrency
for input_tokens in "${TOKEN_LIST[@]}"; do
  read -r -a concurrency_list <<< "$(concurrency_spec_for_input "${input_tokens}")"
  for concurrency in "${concurrency_list[@]}"; do # iterate over the concurrency list
    # Run 1 round of test for each length and concurrency combination
    run=1 # only run one round
    if [[ -n "${NUM_PROMPTS_OVERRIDE:-}" ]]; then
      num_prompts="${NUM_PROMPTS_OVERRIDE}"
    elif (( input_tokens <= 8192 )); then
      num_prompts="${small_input_num_prompts}"
    else
      num_prompts=$((prompt_waves * concurrency))
      if (( num_prompts < min_num_prompts )); then
        num_prompts="${min_num_prompts}"
      fi
    fi
    echo -e "\n============================================================"
    echo "Testing: Input Token = ${input_tokens}, Concurrency = ${concurrency} | Run ${run}"
    echo "Measured prompts = ${num_prompts}, warmups = ${warmup_requests}"
    echo "Log file: benchmark_${input_tokens}_con${concurrency}.log"
    echo "============================================================"

    # Run benchmark: display in terminal + write to a separate log
    python3 -m sglang.bench_serving \
        --backend sglang \
        --model /models/MiMo-V2.5-Pro/ \
        --host 0.0.0.0 \
        --port 30001 \
        --dataset-name random \
        --random-input-len "${input_tokens}" \
        --random-output-len "${output_tokens}" \
        --random-range-ratio 1.0 \
        --flush-cache \
        --seed 12345 \
        --num-prompts "${num_prompts}" \
        --warmup-requests "${warmup_requests}" \
        --max-concurrency "${concurrency}" \
        ${TOKENIZE_PROMPT:+--tokenize-prompt} \
        2>&1 | tee "$LOG_DIR/benchmark_${input_tokens}_con${concurrency}.log"
    echo -e "============================================================\n"
  done
done

echo "All lengths and concurrency tests completed!"
