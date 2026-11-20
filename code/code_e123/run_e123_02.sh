#!/usr/bin/env bash
set -euo pipefail
T=/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu
PY=/mnt/workspace/dengjianfeng/venvs/e29/bin/python
OUT=$T/results/e123_action_sens_20261118_02
test ! -e "$OUT"
export CUDA_VISIBLE_DEVICES=2
export E102_GROUP=A2
"$PY" -I -B -u "$T/code_e123/e123_extract_02.py"
"$PY" -I -B -u "$T/code_e123/e123_analyze_02.py"
touch "$OUT/COMPLETE"
