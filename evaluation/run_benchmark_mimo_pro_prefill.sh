#!/usr/bin/env bash
set -euo pipefail

# Prefill-only client benchmark.
# The legacy customer preset's server reference is CHUNKED_PREFILL_SIZE=65536. Run the
# same client matrix against separately launched TBO/radix-cache configurations
# and record those server-side settings with each result.
# --flush-cache below does not disable the server's radix cache.
#
# DRY_RUN=1 prints commands without contacting the server or creating logs.
# BENCHMARK_PRESET=sweep restores the previous matrix and prompt-count policy.
# BENCHMARK_PRESET=simple uses 64 measured prompts per input, OSL=1, and
# concurrency 16/16/12/3/2/2/2/2/2/2 for 4K/8K/16K/32K/64K/128K/256K/512K/768K/1M.
# The 1M case uses 1048000 input tokens, as in the customer preset.
# BENCHMARK_PRESET=mimo_v26_flash uses the client matrix from
# mimo_v26_flash/h200_reference/MiMo-v2.6-Flash-H200-prefill-data.csv (K=1024).
# Select rows with TOKEN_LIST_OVERRIDE, e.g. 65536 for 64K/1, 256 prompts, C32.
# Server topology, chunk size, quantization and backends are configured separately.
# NUM_PROMPTS_OVERRIDE overrides all counts; SMALL_INPUT_NUM_PROMPTS_OVERRIDE
# overrides counts for inputs <=8K. PROMPT_WAVES/MIN_NUM_PROMPTS are sweep-only.
benchmark_preset="${BENCHMARK_PRESET:-simple}"
dry_run="${DRY_RUN:-0}"
model="${MODEL:-/models/MiMo-V2.6-Flash-MOPD}"
host="${HOST:-0.0.0.0}"
port="${PORT:-30001}"
case "${benchmark_preset}" in
  customer)
    default_tokens="4096 8192 16384 32768 65536 131068 262144 524284 786428 1047548"
    default_small_concurrency="32"
    default_short_concurrency="16"
    default_long_concurrency="2"
    short_input_max_tokens=262144
    default_small_num_prompts=4096
    ;;
  sweep)
    default_tokens="4096 8192 16384 32768 65536 131068 262144 524284 786428 1047548"
    default_small_concurrency="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32"
    default_short_concurrency="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
    default_long_concurrency="1"
    short_input_max_tokens=65536
    default_small_num_prompts=64
    ;;
  simple)
    default_tokens="4096 8192 16384 32768 65536 131072 262144 524288 786432 1048000"
    default_small_concurrency="16"
    default_short_concurrency="12"
    default_long_concurrency="2"
    short_input_max_tokens=65536
    default_small_num_prompts=64
    ;;
  mimo_v26_flash)
    short_input_max_tokens=65536
    default_tokens=""
    declare -A flash_output_tokens flash_num_prompts flash_concurrency
    # ISL     OSL  measured prompts  global concurrency
    while read -r flash_isl flash_osl flash_prompts flash_c; do
      default_tokens+="${default_tokens:+ }${flash_isl}"
      flash_output_tokens["${flash_isl}"]="${flash_osl}"
      flash_num_prompts["${flash_isl}"]="${flash_prompts}"
      flash_concurrency["${flash_isl}"]="${flash_c}"
    done <<'MIMO_V26_FLASH_MATRIX'
8192      1    1024    32
16384     1    1024    32
32768     1    512     32
65536     1    256     32
131072    1    128     16
262144    1    64      8
524288    1    32      4
983040    1    16      4
MIMO_V26_FLASH_MATRIX
    default_small_concurrency="${flash_concurrency[8192]}"
    default_short_concurrency="${flash_concurrency[16384]}"
    default_long_concurrency="${flash_concurrency[983040]}"
    default_small_num_prompts="${flash_num_prompts[8192]}"
    ;;
  *)
    echo "BENCHMARK_PRESET must be customer, sweep, simple, or mimo_v26_flash, observed '${benchmark_preset}'" >&2
    exit 2
    ;;
esac

read -r -a TOKEN_LIST <<< "${TOKEN_LIST_OVERRIDE:-${default_tokens}}"
output_tokens=1
small_input_concurrency_list="${SMALL_INPUT_CONCURRENCY_LIST_OVERRIDE:-${SHORT_CONCURRENCY_LIST_OVERRIDE:-${default_small_concurrency}}}"
short_concurrency_list="${SHORT_CONCURRENCY_LIST_OVERRIDE:-${default_short_concurrency}}"
long_concurrency_list="${LONG_CONCURRENCY_LIST_OVERRIDE:-${default_long_concurrency}}"
warmup_requests="${WARMUP_REQUESTS_OVERRIDE:-4}"
tokenize_prompt="${TOKENIZE_PROMPT:-0}"
small_input_num_prompts="${SMALL_INPUT_NUM_PROMPTS_OVERRIDE:-${default_small_num_prompts}}"
prompt_waves="${PROMPT_WAVES:-4}"
min_num_prompts="${MIN_NUM_PROMPTS:-32}"
LOG_DIR="${LOG_DIR:-./logs/benchmark_tp8_prefill}"

require_positive_integer() {
  if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
    echo "$1 must be a positive integer, observed '$2'" >&2
    exit 2
  fi
}

if [[ "${dry_run}" != "0" && "${dry_run}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1, observed '${dry_run}'" >&2
  exit 2
fi
if ! [[ "${warmup_requests}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "WARMUP_REQUESTS_OVERRIDE must be a non-negative integer, observed '${warmup_requests}'" >&2
  exit 2
fi
if [[ "${tokenize_prompt}" != "0" && "${tokenize_prompt}" != "1" ]]; then
  echo "TOKENIZE_PROMPT must be 0 or 1, observed '${tokenize_prompt}'" >&2
  exit 2
fi
require_positive_integer SMALL_INPUT_NUM_PROMPTS_OVERRIDE "${small_input_num_prompts}"
if [[ -n "${NUM_PROMPTS_OVERRIDE:-}" ]]; then
  require_positive_integer NUM_PROMPTS_OVERRIDE "${NUM_PROMPTS_OVERRIDE}"
fi
if [[ "${benchmark_preset}" == "sweep" ]]; then
  require_positive_integer PROMPT_WAVES "${prompt_waves}"
  require_positive_integer MIN_NUM_PROMPTS "${min_num_prompts}"
fi

concurrency_spec_for_input() {
  local input_tokens="$1"
  if [[ -n "${CONCURRENCY_LIST_OVERRIDE:-}" ]]; then
    echo "${CONCURRENCY_LIST_OVERRIDE}"
  elif (( input_tokens <= 8192 )); then
    echo "${small_input_concurrency_list}"
  elif [[ "${benchmark_preset}" == "mimo_v26_flash" ]]; then
    if (( input_tokens <= short_input_max_tokens )); then
      echo "${SHORT_CONCURRENCY_LIST_OVERRIDE:-${flash_concurrency[${input_tokens}]}}"
    else
      echo "${LONG_CONCURRENCY_LIST_OVERRIDE:-${flash_concurrency[${input_tokens}]}}"
    fi
  elif [[ "${benchmark_preset}" == "simple" ]] && (( input_tokens <= 65536 )); then
    if (( input_tokens <= 16384 )); then
      echo "${short_concurrency_list}"
    elif (( input_tokens <= 32768 )); then
      echo "${SHORT_CONCURRENCY_LIST_OVERRIDE:-3}"
    else
      echo "${SHORT_CONCURRENCY_LIST_OVERRIDE:-2}"
    fi
  elif (( input_tokens <= short_input_max_tokens )); then
    echo "${short_concurrency_list}"
  else
    echo "${long_concurrency_list}"
  fi
}

num_prompts_for_input() {
  local input_tokens="$1" concurrency="$2" num_prompts
  if [[ -n "${NUM_PROMPTS_OVERRIDE:-}" ]]; then
    echo "${NUM_PROMPTS_OVERRIDE}"
  elif (( input_tokens <= 8192 )); then
    echo "${small_input_num_prompts}"
  elif [[ "${benchmark_preset}" == "mimo_v26_flash" ]]; then
    echo "${flash_num_prompts[${input_tokens}]}"
  elif [[ "${benchmark_preset}" == "simple" ]]; then
    echo 64
  elif [[ "${benchmark_preset}" == "customer" ]]; then
    if (( input_tokens <= 16384 )); then
      echo 3000
    elif (( input_tokens <= 65536 )); then
      echo 1024
    elif (( input_tokens <= 131072 )); then
      echo 512
    else
      echo 32
    fi
  else
    num_prompts=$((prompt_waves * concurrency))
    if (( num_prompts < min_num_prompts )); then
      num_prompts="${min_num_prompts}"
    fi
    echo "${num_prompts}"
  fi
}

# Validate the whole matrix before starting any requests or creating logs.
if (( ${#TOKEN_LIST[@]} == 0 )); then
  echo "TOKEN_LIST_OVERRIDE must contain at least one input length" >&2
  exit 2
fi
for input_tokens in "${TOKEN_LIST[@]}"; do
  require_positive_integer "Input length" "${input_tokens}"
  if [[ "${benchmark_preset}" == "mimo_v26_flash" ]]; then
    if [[ -z "${flash_num_prompts[${input_tokens}]+set}" ]]; then
      echo "mimo_v26_flash has no row for input ${input_tokens}; select from: ${default_tokens}" >&2
      exit 2
    fi
    require_positive_integer "Output length" "${flash_output_tokens[${input_tokens}]}"
    require_positive_integer "Measured prompts" "${flash_num_prompts[${input_tokens}]}"
  fi
  read -r -a concurrency_list <<< "$(concurrency_spec_for_input "${input_tokens}")"
  if (( ${#concurrency_list[@]} == 0 )); then
    echo "Concurrency list for input ${input_tokens} must not be empty" >&2
    exit 2
  fi
  for concurrency in "${concurrency_list[@]}"; do
    require_positive_integer Concurrency "${concurrency}"
  done
done

echo "Benchmark preset: ${benchmark_preset}; dry run: ${dry_run}"
if [[ "${benchmark_preset}" == "mimo_v26_flash" ]]; then
  echo "Client matrix: MiMo-v2.6-Flash-H200-prefill-data.csv; input K=1024 tokens."
  echo "Record actual server topology, effective chunk size, TBO, quantization, and backends with results."
fi
if [[ "${benchmark_preset}" == "customer" ]]; then
  echo "Required customer server setting: chunked_prefill_size=65536"
  echo "Record TBO and radix-cache modes from the server log for this run."
  echo "Run separately for TBO off/on; record SGLANG_TBO_MIM_SEQ_LEN (current launcher default 2000; 8000 excludes 4K requests)."
  if [[ -n "${PROMPT_WAVES:-}${MIN_NUM_PROMPTS:-}" ]]; then
    echo "PROMPT_WAVES/MIN_NUM_PROMPTS apply only to BENCHMARK_PRESET=sweep; using customer prompt counts."
  fi
fi
if [[ "${dry_run}" == "0" ]]; then
  mkdir -p "$LOG_DIR"
fi

for input_tokens in "${TOKEN_LIST[@]}"; do
  if [[ "${benchmark_preset}" == "mimo_v26_flash" ]]; then
    output_tokens="${flash_output_tokens[${input_tokens}]}"
  fi
  read -r -a concurrency_list <<< "$(concurrency_spec_for_input "${input_tokens}")"
  for concurrency in "${concurrency_list[@]}"; do
    num_prompts="$(num_prompts_for_input "${input_tokens}" "${concurrency}")"
    echo -e "\n============================================================"
    echo "Testing: Input Token = ${input_tokens}, Output Token = ${output_tokens}, Concurrency = ${concurrency} | Run 1"
    echo "Measured prompts = ${num_prompts}, warmups = ${warmup_requests}"
    echo "Log file: benchmark_${input_tokens}_con${concurrency}.log"
    echo "============================================================"

    benchmark_cmd=(python3 -m sglang.bench_serving \
        --backend sglang \
        --model "${model}" \
        --host "${host}" \
        --port "${port}" \
        --dataset-name random \
        --random-input-len "${input_tokens}" \
        --random-output-len "${output_tokens}" \
        --random-range-ratio 1.0 \
        --flush-cache \
        --seed 12345 \
        --num-prompts "${num_prompts}" \
        --warmup-requests "${warmup_requests}" \
        --max-concurrency "${concurrency}")
    if [[ "${tokenize_prompt}" == "1" ]]; then
      benchmark_cmd+=(--tokenize-prompt)
    fi
    benchmark_cmd+=("$@")
    printf 'Command:'
    printf ' %q' "${benchmark_cmd[@]}"
    printf '\n'
    if [[ "${dry_run}" == "0" ]]; then
      "${benchmark_cmd[@]}" 2>&1 | tee "$LOG_DIR/benchmark_${input_tokens}_con${concurrency}.log"
    fi
    echo -e "============================================================\n"
  done
done

if [[ "${dry_run}" == "1" ]]; then
  echo "Dry run complete; no benchmark requests sent."
else
  echo "All lengths and concurrency tests completed!"
fi
