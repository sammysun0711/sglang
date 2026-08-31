#!/usr/bin/env bash
set -euo pipefail

input_tokens="${INPUT_TOKENS:-16384}"
output_tokens="${OUTPUT_TOKENS:-1}"
num_prompts="${NUM_PROMPTS:-1}"
warmup_requests="${WARMUP_REQUESTS:-1}"
max_concurrency="${MAX_CONCURRENCY:-1}"
model_path="${MODEL_PATH:-/models/MiMo-V2.5-Pro-FP4-DFlash}"
host="${HOST:-127.0.0.1}"
port="${PORT:-30001}"
profile_output_dir="${PROFILE_OUTPUT_DIR:-./profiles/mimo_fp4_prefill_${input_tokens}_con${max_concurrency}}"
profile_prefix="${PROFILE_PREFIX:-mimo_fp4_${input_tokens}_con${max_concurrency}}"
profile_start_step="${PROFILE_START_STEP:-}"
profile_steps="${PROFILE_STEPS:-}"
profile_num_steps="${PROFILE_NUM_STEPS:-}"
profile_by_stage="${PROFILE_BY_STAGE:-0}"
profile_activities="${PROFILE_ACTIVITIES:-CPU GPU}"
tokenize_prompt="${TOKENIZE_PROMPT:-0}"
fake_prefill="${FAKE_PREFILL:-0}"

export SGLANG_TORCH_PROFILER_DIR="${profile_output_dir}"

echo "Profiling input=${input_tokens}, output=${output_tokens}, concurrency=${max_concurrency}, prompts=${num_prompts}, warmups=${warmup_requests}, steps=${profile_steps:-all}, by_stage=${profile_by_stage}, tokenize_prompt=${tokenize_prompt}"
echo "Profile output: ${profile_output_dir}"

read -r -a profile_activity_args <<< "${profile_activities}"
profile_args=(
    --profile
    --profile-activities "${profile_activity_args[@]}"
    --profile-output-dir "${profile_output_dir}"
    --profile-prefix "${profile_prefix}"
)
if [[ -n "${profile_start_step}" ]]; then
    profile_args+=(--profile-start-step "${profile_start_step}")
fi
if [[ -n "${profile_steps}" ]]; then
    profile_args+=(--profile-steps "${profile_steps}")
fi
if [[ -n "${profile_num_steps}" ]]; then
    profile_args+=(--profile-num-steps "${profile_num_steps}")
fi
if [[ "${profile_by_stage}" == "1" ]]; then
    profile_args+=(--profile-by-stage)
fi
if [[ "${tokenize_prompt}" == "1" ]]; then
    profile_args+=(--tokenize-prompt)
fi
if [[ "${fake_prefill}" == "1" ]]; then
    profile_args+=(--fake-prefill)
fi

python3 -m sglang.bench_serving \
    --backend sglang \
    --model "${model_path}" \
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
    --max-concurrency "${max_concurrency}" \
    --disable-tqdm \
    "${profile_args[@]}"
