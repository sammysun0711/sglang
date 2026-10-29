#!/usr/bin/env bash
set -euo pipefail
repro_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="${MIMO_WORKSPACE:-$(cd -- "${repro_dir}/../../../.." && pwd)}"
export AITER_ROOT="${AITER_ROOT:-${workspace_dir}/aiter}"
runtime_python="${RUNTIME724_PYTHON_ROOT:-${workspace_dir}/runtime724/python}"
export PYTHONPATH="${runtime_python}:${AITER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export AITER_USE_SYSTEM_TRITON=1
export SPT_NOENV=1
exec "${PYTHON_BIN:-python}" "${repro_dir}/benchmark.py" "$@"
