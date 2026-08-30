#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
input_tokens="${INPUT_TOKENS:-65536}"
output_tokens="${OUTPUT_TOKENS:-1}"
num_prompts="${NUM_PROMPTS:-4}"
warmup_requests="${WARMUP_REQUESTS:-4}"
max_concurrency="${MAX_CONCURRENCY:-1}"
model_path="${MODEL_PATH:-/models/MiMo-V2.5-Pro/}"
port="${PORT:-30001}"
dataset_path="${DATASET_PATH:-${script_dir}/ShareGPT_V3_unfiltered_cleaned_split.json}"
profile_output_dir="${PROFILE_OUTPUT_DIR:-${script_dir}/profiles/mimo_pro_2_5_aiter_prefill_${input_tokens}_con${max_concurrency}}"
profile_prefix="${PROFILE_PREFIX:-mimo_${input_tokens}_con${max_concurrency}}"

export SGLANG_TORCH_PROFILER_DIR="${profile_output_dir}"

echo "Profiling input=${input_tokens}, output=${output_tokens}, concurrency=${max_concurrency}, prompts=${num_prompts}, warmups=${warmup_requests}"
echo "Profile output: ${profile_output_dir}"
dataset_args=()
if [[ -n "${dataset_path}" && -s "${dataset_path}" ]]; then
  dataset_args+=(--dataset-path "${dataset_path}")
fi
#--profile-activities GPU \
python3 -m sglang.bench_serving \
    --backend sglang \
    --model "${model_path}" \
    --host 0.0.0.0 \
    --port "${port}" \
    --dataset-name random \
    --random-input-len ${input_tokens} \
    --random-output-len ${output_tokens} \
    --random-range-ratio 1.0 \
    "${dataset_args[@]}" \
    --flush-cache \
    --seed 12345 \
    --num-prompts "${num_prompts}" \
    --warmup-requests "${warmup_requests}" \
    --max-concurrency "${max_concurrency}" \
    ${TOKENIZE_PROMPT:+--tokenize-prompt} \
    --profile \
    --profile-output-dir "${profile_output_dir}" \
    --profile-prefix "${profile_prefix}"
