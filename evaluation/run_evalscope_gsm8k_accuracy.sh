#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# pip install evalscope==1.11.0
base_url="${BASE_URL:-http://127.0.0.1:30001/v1}"
model_path="${MODEL_PATH:-/models/MiMo-V2.5/}"
dataset_dir="${DATASET_DIR:-${script_dir}/evalscope_cache}"
limit="${LIMIT:-all}"
eval_batch_size="${EVAL_BATCH_SIZE:-32}"
max_tokens="${MAX_TOKENS:-1024}"
seed="${SEED:-12345}"
flush_cache="${FLUSH_CACHE:-1}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${OUTPUT_DIR:-${script_dir}/logs/gsm8k_${run_id}}"
model_id="${MODEL_ID:-MiMo-V2.5-tp4-ep4-mori-tbo-mtp}"
result_scope="full"
if [[ "${limit}" != "all" ]]; then
  result_scope="debug-only"
fi

server_root="${base_url%/}"
server_root="${server_root%/v1}"

mkdir -p "${output_dir}"

curl --max-time 10 --fail --silent --show-error \
  "${server_root}/server_info" > "${output_dir}/server_info.json"

if [[ "${flush_cache}" == "1" ]]; then
  curl --max-time 60 --fail --silent --show-error --request POST \
    "${server_root}/flush_cache?timeout=60" > "${output_dir}/flush_cache.txt"
fi

{
  echo "run_id=${run_id}"
  echo "base_url=${base_url}"
  echo "model=${model_path}"
  echo "dataset=gsm8k"
  echo "result_scope=${result_scope}"
  echo "limit=${limit}"
  echo "eval_batch_size=${eval_batch_size}"
  echo "max_tokens=${max_tokens}"
  echo "seed=${seed}"
  echo "flush_cache=${flush_cache}"
  printf 'evalscope_version='
  evalscope --version
} > "${output_dir}/eval_metadata.txt"

limit_args=()
if [[ "${limit}" != "all" ]]; then
  echo "WARNING: LIMIT=${limit} is a debug-only run, not a reportable final accuracy result." >&2
  limit_args+=(--limit "${limit}")
fi

echo "Running EvalScope GSM8K: model=${model_path}, limit=${limit}, batch=${eval_batch_size}"
echo "Results: ${output_dir}"

evalscope eval \
  --model "${model_path}" \
  --model-id "${model_id}" \
  --eval-type openai_api \
  --api-url "${base_url}" \
  --api-key "${API_KEY:-EMPTY}" \
  --datasets gsm8k \
  --dataset-dir "${dataset_dir}" \
  "${limit_args[@]}" \
  --eval-batch-size "${eval_batch_size}" \
  --seed "${seed}" \
  --generation-config "temperature=0.0,top_p=1.0,max_tokens=${max_tokens},seed=${seed},stream=False,timeout=1800,retries=1" \
  --work-dir "${output_dir}/evalscope" \
  --no-timestamp \
  2>&1 | tee "${output_dir}/evalscope_console.log"

echo "EvalScope report: ${output_dir}/evalscope/reports/report.html"
