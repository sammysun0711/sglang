#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
input_tokens="${INPUT_TOKENS:-65536}"
output_tokens="${OUTPUT_TOKENS:-1}"
num_prompts="${NUM_PROMPTS:-4}"
warmup_requests="${WARMUP_REQUESTS:-4}"
max_concurrency="${MAX_CONCURRENCY:-1}"
model_path="${MODEL_PATH:-/models/MiMo-V2.5-Pro/}"
served_model_name="${SERVED_MODEL_NAME:-}"
host="${HOST:-0.0.0.0}"
port="${PORT:-30001}"
dataset_path="${DATASET_PATH:-${script_dir}/ShareGPT_V3_unfiltered_cleaned_split.json}"
profile_output_dir="${PROFILE_OUTPUT_DIR:-${script_dir}/profiles/mimo_pro_2_5_aiter_chunk_prefill_32k_${input_tokens}_con${max_concurrency}}-check-moe"
profile_prefix="${PROFILE_PREFIX:-mimo_chunk_prefill_32k_${input_tokens}_con${max_concurrency}}-check-moe-new"
profile_activities="${PROFILE_ACTIVITIES:-CPU GPU}"
profile_num_steps="${PROFILE_NUM_STEPS:-}"
profile_by_stage="${PROFILE_BY_STAGE:-0}"
profile_stages="${PROFILE_STAGES:-}"
tokenize_prompt="${TOKENIZE_PROMPT:-0}"
fake_prefill="${FAKE_PREFILL:-0}"

export SGLANG_TORCH_PROFILER_DIR="${profile_output_dir}"

echo "Profiling input=${input_tokens}, output=${output_tokens}, concurrency=${max_concurrency}, prompts=${num_prompts}, warmups=${warmup_requests}, by_stage=${profile_by_stage}, fake_prefill=${fake_prefill}"
echo "Profile output: ${profile_output_dir}"

profile_args=(
    --profile
    --profile-output-dir "${profile_output_dir}"
    --profile-prefix "${profile_prefix}"
)
read -r -a profile_activity_args <<<"${profile_activities}"
profile_args+=(--profile-activities "${profile_activity_args[@]}")
if [[ -n "${profile_num_steps}" ]]; then
    profile_args+=(--profile-num-steps "${profile_num_steps}")
fi
if [[ "${profile_by_stage}" == "1" ]]; then
    profile_args+=(--profile-by-stage)
fi
if [[ -n "${profile_stages}" ]]; then
    read -r -a profile_stage_args <<<"${profile_stages}"
    profile_args+=(--profile-stages "${profile_stage_args[@]}")
fi
if [[ "${tokenize_prompt}" == "1" ]]; then
    profile_args+=(--tokenize-prompt)
fi
if [[ "${fake_prefill}" == "1" ]]; then
    profile_args+=(--fake-prefill)
fi

model_args=(--model "${model_path}")
if [[ -n "${served_model_name}" ]]; then
    model_args+=(--served-model-name "${served_model_name}")
fi

python3 -m sglang.bench_serving \
    --backend sglang \
    "${model_args[@]}" \
    --host "${host}" \
    --port "${port}" \
    --dataset-name random \
    --random-input-len ${input_tokens} \
    --random-output-len ${output_tokens} \
    --random-range-ratio 1.0 \
    --dataset-path "${dataset_path}" \
    --flush-cache \
    --seed 12345 \
    --num-prompts "${num_prompts}" \
    --warmup-requests "${warmup_requests}" \
    --max-concurrency "${max_concurrency}" \
    "${profile_args[@]}"
