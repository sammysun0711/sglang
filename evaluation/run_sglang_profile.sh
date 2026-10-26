#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
input_tokens="${INPUT_TOKENS:-8192}"
output_tokens="${OUTPUT_TOKENS:-20}"
num_prompts="${NUM_PROMPTS:-4}"
warmup_requests="${WARMUP_REQUESTS:-4}"
max_concurrency="${MAX_CONCURRENCY:-1}"
model_path="${MODEL_PATH:-/models/MiMo-V2.6-Flash-MOPD}"
served_model_name="${SERVED_MODEL_NAME:-}"
host="${HOST:-0.0.0.0}"
port="${PORT:-30001}"
dataset_path="${DATASET_PATH:-${script_dir}/ShareGPT_V3_unfiltered_cleaned_split.json}"
profile_output_dir="${PROFILE_OUTPUT_DIR:-${script_dir}/profiles/mimo_v2_6_flash_aiter_chunk_prefill_16k_isl${input_tokens}_osl${output_tokens}_con${max_concurrency}-}"
profile_prefix="${PROFILE_PREFIX:-mimo_v2_6_flash_aiter_chunk_prefill_16k_isl${input_tokens}_osl${output_tokens}_con${max_concurrency}}"
tokenize_prompt="${TOKENIZE_PROMPT:-0}"
fake_prefill="${FAKE_PREFILL:-0}"
profile_by_stage="${PROFILE_BY_STAGE:-0}"
profile_stages="${PROFILE_STAGES:-prefill}"
profile_num_steps="${PROFILE_NUM_STEPS:-}"

export SGLANG_TORCH_PROFILER_DIR="${profile_output_dir}"
model_args=(--model "${model_path}")
if [[ -n "${served_model_name}" ]]; then
    model_args+=(--served-model-name "${served_model_name}")
fi

profile_args=(--profile-prefix "${profile_prefix}")
if [[ "${profile_by_stage}" == "1" ]]; then
    if ! [[ "${profile_num_steps}" =~ ^[1-9][0-9]*$ ]]; then
        echo "PROFILE_BY_STAGE=1 requires a positive PROFILE_NUM_STEPS" >&2
        exit 2
    fi
    read -r -a selected_stages <<< "${profile_stages}"
    profile_args+=(--profile-by-stage --profile-stages "${selected_stages[@]}")
elif [[ "${profile_by_stage}" != "0" ]]; then
    echo "PROFILE_BY_STAGE must be 0 or 1" >&2
    exit 2
fi
if [[ -n "${profile_num_steps}" ]]; then
    if ! [[ "${profile_num_steps}" =~ ^[1-9][0-9]*$ ]]; then
        echo "PROFILE_NUM_STEPS must be a positive integer" >&2
        exit 2
    fi
    profile_args+=(--profile-num-steps "${profile_num_steps}")
fi
if [[ "${tokenize_prompt}" == "1" ]]; then
    profile_args+=(--tokenize-prompt)
fi
if [[ "${fake_prefill}" == "1" ]]; then
    profile_args+=(--fake-prefill)
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
    --profile \
    "${profile_args[@]}"
