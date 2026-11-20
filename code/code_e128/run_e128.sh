#!/usr/bin/env bash
set -euo pipefail
T=/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu
PY=/mnt/workspace/dengjianfeng/venvs/e29/bin/python
OUT=$T/results/e128_ablation_20261119_01
test ! -e "$OUT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export E102_GROUP=A2
"$PY" -I -B -u "$T/code_e128/e128_ablation_a.py"
"$PY" -I -B -u "$T/code_e128/e128_pure_vis_b.py"
touch "$OUT/COMPLETE"
