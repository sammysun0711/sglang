#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname -- "${SCRIPT_DIR}")"
AITER_ROOT="${AITER_ROOT:-/root/workspace/mimo-opt/aiter-mimo-fp4-dflash}"
TARGET_MODEL_VARIANT="${TARGET_MODEL_VARIANT:-fp8}"
PHASE="${PHASE:-all}"
PORT="${PORT:-30001}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_GPU_TIMEOUT_SEC="${WAIT_GPU_TIMEOUT_SEC:-3600}"
GPU_POLL_INTERVAL_SEC="${GPU_POLL_INTERVAL_SEC:-1800}"
GPU_IDLE_CONFIRM_SAMPLES="${GPU_IDLE_CONFIRM_SAMPLES:-6}"
GPU_IDLE_SAMPLE_INTERVAL_SEC="${GPU_IDLE_SAMPLE_INTERVAL_SEC:-10}"
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-1}"
GPU_VRAM_THRESHOLD="${GPU_VRAM_THRESHOLD:-1}"
STEADY_MONITOR_INTERVAL_SEC="${STEADY_MONITOR_INTERVAL_SEC:-600}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${LOG_DIR:-${SCRIPT_DIR}/logs/dflash_customer_baseline_${TARGET_MODEL_VARIANT}_${RUN_ID}}"

case "${TARGET_MODEL_VARIANT}" in
  fp8) MODEL=/models/MiMo-V2.5-Pro ;;
  mxfp4) MODEL=/models/MiMo-V2.5-Pro-FP4-DFlash ;;
  *) echo "TARGET_MODEL_VARIANT must be fp8 or mxfp4" >&2; exit 2 ;;
esac
case "${PHASE}" in
  all|prefill|decode) ;;
  *) echo "PHASE must be all, prefill, or decode" >&2; exit 2 ;;
esac

server_pid=""
server_pgid=""
monitor_pid=""
orig_numa="$(cat /proc/sys/kernel/numa_balancing)"

stop_monitor() {
  if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  monitor_pid=""
}

cleanup_server() {
  stop_monitor
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM -- "-${server_pgid}" 2>/dev/null || true
    local deadline=$((SECONDS + 90))
    while kill -0 "${server_pid}" 2>/dev/null && (( SECONDS < deadline )); do sleep 2; done
    if kill -0 "${server_pid}" 2>/dev/null; then kill -KILL -- "-${server_pgid}" 2>/dev/null || true; fi
    wait "${server_pid}" 2>/dev/null || true
  fi
  server_pid=""
  server_pgid=""
}
cleanup() {
  cleanup_server
  printf '%s\n' "${orig_numa}" > /proc/sys/kernel/numa_balancing
}
trap cleanup EXIT INT TERM

wait_for_gpu_free() {
  local deadline=$((SECONDS + WAIT_GPU_TIMEOUT_SEC))
  while (( SECONDS < deadline )); do
    local confirmed=1
    for ((sample = 1; sample <= GPU_IDLE_CONFIRM_SAMPLES; sample++)); do
      local snapshot max_util max_vram
      snapshot="$(rocm-smi --showuse --showmemuse --csv 2>/dev/null)"
      max_util="$(awk -F, '$1 ~ /^card/ {gsub(/[^0-9.]/,"",$2); if (($2+0)>m) m=$2+0} END{print m+0}' <<<"${snapshot}")"
      max_vram="$(awk -F, '$1 ~ /^card/ {gsub(/[^0-9.]/,"",$4); if (($4+0)>m) m=$4+0} END{print m+0}' <<<"${snapshot}")"
      printf '%s sample=%s/%s max_gpu_util=%s max_vram_pct=%s\n' \
        "$(date -u +%FT%TZ)" "${sample}" "${GPU_IDLE_CONFIRM_SAMPLES}" \
        "${max_util}" "${max_vram}" >>"${RUN_ROOT}/gpu_wait.log"
      if awk -v u="${max_util}" -v v="${max_vram}" \
        -v ut="${GPU_UTIL_THRESHOLD}" -v vt="${GPU_VRAM_THRESHOLD}" \
        'BEGIN{exit !((u < ut) && (v < vt))}'; then
        :
      else
        confirmed=0
        break
      fi
      if (( sample < GPU_IDLE_CONFIRM_SAMPLES )); then
        sleep "${GPU_IDLE_SAMPLE_INTERVAL_SEC}"
      fi
    done
    (( confirmed == 1 )) && return 0
    (( SECONDS < deadline )) || return 1
    sleep "${GPU_POLL_INTERVAL_SEC}"
  done
  return 1
}
wait_for_server() {
  local deadline=$((SECONDS + 1200))
  while ! curl --max-time 3 -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    kill -0 "${server_pid}" 2>/dev/null || return 1
    (( SECONDS < deadline )) || return 1
    sleep 5
  done
}

start_monitor() {
  local case_dir="$1"
  local monitor_log="${case_dir}/interference_monitor.log"
  (
    while kill -0 "${server_pid}" 2>/dev/null; do
      {
        echo "timestamp=$(date -u +%FT%TZ)"
        rocm-smi --showpids 2>/dev/null
        rocm-smi --showuse --showmemuse --csv 2>/dev/null
      } >>"${monitor_log}"
      sleep "${STEADY_MONITOR_INTERVAL_SEC}"
    done
  ) &
  monitor_pid=$!
}

print_plan() {
  cat <<EOF
target_variant=${TARGET_MODEL_VARIANT}
model=${MODEL}
phase=${PHASE}
label=baseline prompt compute + qualified FlyDSL verification
common=BF16 target/draft KV, DFlash block 8, full verify FlyDSL, SWA FlyPA, TBO off, NUMA balancing off
prefill_client=run_benchmark_mimo_pro_prefill.sh (default customer matrix)
decode_client=run_benchmark_mimo_pro_decode_fake_prefill_matrix.sh (default customer cases)
decode_servers=fresh server per case, MEM_FRACTION_STATIC=1.0, MAX_RUNNING_REQUESTS=C+1
decode_capacity=max_total_num_tokens / nominal_input_tokens
capacity_limited=excluded from final performance; optional highest-resident diagnostic only
gpu_wait=30-minute polling; six 10-second samples below 1% utilization and 1% VRAM
steady_monitor=read-only KFD/utilization snapshot every 10 minutes
EOF
}

if [[ "${DRY_RUN}" == "1" ]]; then
  print_plan
  if [[ "${PHASE}" == "all" || "${PHASE}" == "prefill" ]]; then
    echo
    echo '--- prefill client ---'
    MODEL="${MODEL}" DRY_RUN=1 bash "${SCRIPT_DIR}/run_benchmark_mimo_pro_prefill.sh"
  fi
  if [[ "${PHASE}" == "all" || "${PHASE}" == "decode" ]]; then
    echo
    echo '--- decode client ---'
    MODEL="${MODEL}" DRY_RUN=1 bash "${SCRIPT_DIR}/run_benchmark_mimo_pro_decode_fake_prefill_matrix.sh"
  fi
  exit 0
fi

mkdir -p "${RUN_ROOT}"
print_plan | tee "${RUN_ROOT}/plan.txt"
printf 'status=waiting_for_gpu\n' >"${RUN_ROOT}/status.env"
wait_for_gpu_free || { echo 'status=failed reason=gpus_not_free' >>"${RUN_ROOT}/status.env"; exit 1; }
printf '0\n' > /proc/sys/kernel/numa_balancing
{
  echo "sglang=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "aiter=$(git -C "${AITER_ROOT}" rev-parse HEAD)"
  echo "flydsl=$(python3 -c 'import flydsl; print(flydsl.__version__)')"
  echo "numa_balancing=$(cat /proc/sys/kernel/numa_balancing)"
} >"${RUN_ROOT}/versions.env"

if [[ "${PHASE}" == "all" || "${PHASE}" == "prefill" ]]; then
  case_dir="${RUN_ROOT}/prefill"
  mkdir -p "${case_dir}/server" "${case_dir}/client"
  setsid env TARGET_MODEL_VARIANT="${TARGET_MODEL_VARIANT}" MODEL="${MODEL}" \
    PORT="${PORT}" RUN_ID="${RUN_ID}_prefill" LOG_DIR="${case_dir}/server" \
    LOG_FILE=server.log bash "${SCRIPT_DIR}/launch_tp8_noep_aiter_dflash_accuracy_baseline.sh" \
    >"${case_dir}/server_driver.log" 2>&1 &
  server_pid=$!
  server_pgid="$(ps -o pgid= -p "${server_pid}" | tr -d ' ')"
  wait_for_server || { echo 'status=failed reason=prefill_server' >>"${RUN_ROOT}/status.env"; exit 1; }
  start_monitor "${case_dir}"
  curl -fsS --max-time 30 "http://127.0.0.1:${PORT}/server_info" >"${case_dir}/server_info_before.json"
  env -u NUM_PROMPTS_OVERRIDE -u CONCURRENCY_LIST_OVERRIDE -u PROMPT_WAVES -u MIN_NUM_PROMPTS \
    MODEL="${MODEL}" HOST=127.0.0.1 PORT="${PORT}" BENCHMARK_PRESET=customer \
    LOG_DIR="${case_dir}/client" bash "${SCRIPT_DIR}/run_benchmark_mimo_pro_prefill.sh" \
    >"${case_dir}/client_driver.log" 2>&1
  cleanup_server
  wait_for_gpu_free
fi

if [[ "${PHASE}" == "all" || "${PHASE}" == "decode" ]]; then
  mkdir -p "${RUN_ROOT}/decode"
  printf 'case\tnominal_input\trequested_concurrency\tmax_running_requests\tmax_total_num_tokens\testimated_max_concurrency\tstatus\n' \
    >"${RUN_ROOT}/decode/capacity.tsv"
  for spec in \
    'D64K204|65532|65536|64k|204' \
    'D256K63|262140|262144|256k|63' \
    'D64K196|65532|65536|64k|196'; do
    IFS='|' read -r case_name supplied_ids nominal_input label requested_c <<<"${spec}"
    max_running=$((requested_c + 1))
    case_dir="${RUN_ROOT}/decode/${case_name}"
    mkdir -p "${case_dir}/server" "${case_dir}/client" "${case_dir}/analysis"
    setsid env TARGET_MODEL_VARIANT="${TARGET_MODEL_VARIANT}" MODEL="${MODEL}" \
      PORT="${PORT}" RUN_ID="${RUN_ID}_${case_name}" LOG_DIR="${case_dir}/server" \
      SERVER_LOG_FILE=server.log MEM_FRACTION_STATIC=1.0 \
      MAX_RUNNING_REQUESTS="${max_running}" \
      bash "${SCRIPT_DIR}/launch_tp8_noep_aiter_dflash_decode_fake_prefill_baseline.sh" \
      >"${case_dir}/server_driver.log" 2>&1 &
    server_pid=$!
    server_pgid="$(ps -o pgid= -p "${server_pid}" | tr -d ' ')"
    if ! wait_for_server; then
      printf '%s\t%s\t%s\t%s\t-\t-\tstartup_failed\n' \
        "${case_name}" "${nominal_input}" "${requested_c}" "${max_running}" \
        >>"${RUN_ROOT}/decode/capacity.tsv"
      cleanup_server
      wait_for_gpu_free
      continue
    fi

    start_monitor "${case_dir}"
    curl -fsS --max-time 30 "http://127.0.0.1:${PORT}/server_info" >"${case_dir}/server_info_before.json"
    max_total_num_tokens="$(jq -r '.max_total_num_tokens' "${case_dir}/server_info_before.json")"
    estimated_max_c=$((max_total_num_tokens / nominal_input))
    if (( estimated_max_c < requested_c )); then
      printf '%s\t%s\t%s\t%s\t%s\t%s\tcapacity_limited\n' \
        "${case_name}" "${nominal_input}" "${requested_c}" "${max_running}" \
        "${max_total_num_tokens}" "${estimated_max_c}" \
        >>"${RUN_ROOT}/decode/capacity.tsv"
      if (( estimated_max_c > 0 )); then
        diagnostic_dir="${case_dir}/capacity_diagnostic_c${estimated_max_c}"
        mkdir -p "${diagnostic_dir}"
        env MODEL="${MODEL}" HOST=127.0.0.1 PORT="${PORT}" \
          CASE_SPECS="${supplied_ids}|${nominal_input}|${label}_capacity_diagnostic|${estimated_max_c}" \
          OUTPUT_TOKENS=1024 WARMUP_REQUESTS=32 NUM_PROMPTS="$((4 * estimated_max_c))" \
          LOG_DIR="${diagnostic_dir}" \
          bash "${SCRIPT_DIR}/run_benchmark_mimo_pro_decode_fake_prefill_matrix.sh" \
          >"${diagnostic_dir}/client_driver.log" 2>&1 || true
        python3 "${SCRIPT_DIR}/analyze_server_output_throughput.py" \
          "${case_dir}/server/server.log" --target-bs "${estimated_max_c}" --output json \
          >"${diagnostic_dir}/server_plateau_analysis.json" || true
      fi
      cleanup_server
      wait_for_gpu_free
      continue
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\tofficial_requested\n' \
      "${case_name}" "${nominal_input}" "${requested_c}" "${max_running}" \
      "${max_total_num_tokens}" "${estimated_max_c}" \
      >>"${RUN_ROOT}/decode/capacity.tsv"
    env MODEL="${MODEL}" HOST=127.0.0.1 PORT="${PORT}" \
      CASE_SPECS="${supplied_ids}|${nominal_input}|${label}|${requested_c}" \
      OUTPUT_TOKENS=1024 WARMUP_REQUESTS=32 NUM_PROMPTS="$((4 * requested_c))" \
      LOG_DIR="${case_dir}/client" \
      bash "${SCRIPT_DIR}/run_benchmark_mimo_pro_decode_fake_prefill_matrix.sh" \
      >"${case_dir}/client_driver.log" 2>&1
    python3 "${SCRIPT_DIR}/analyze_server_output_throughput.py" \
      "${case_dir}/server/server.log" --target-bs "${requested_c}" --output json \
      >"${case_dir}/analysis/official.json"
    python3 "${SCRIPT_DIR}/analyze_server_output_throughput.py" \
      "${case_dir}/server/server.log" --target-bs "${requested_c}" \
      >"${case_dir}/analysis/official.txt"
    cleanup_server
    wait_for_gpu_free
  done
fi

echo 'status=complete' >>"${RUN_ROOT}/status.env"
