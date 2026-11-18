#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
[[ $# == 1 && ! -e "$1" ]] || { echo 'Usage: run_bounded_profile.sh NEW_OUTPUT_DIR' >&2; exit 2; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
# Only this child process can be terminated, never any shared GPU owner's process.
exec timeout --signal=TERM --kill-after=10s 1450s /mnt/workspace/dengjianfeng/venvs/e29/bin/python -m e82.profile --config configs/piper_v1.json --output "$1"
