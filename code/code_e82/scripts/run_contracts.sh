#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 E82_TEST_DINO=1
exec /mnt/workspace/dengjianfeng/venvs/e29/bin/python -m pytest --disable-warnings "$@"
