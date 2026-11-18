#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
PY=/mnt/workspace/dengjianfeng/venvs/e29/bin/python
RUN=$1
[[ ! -e "$RUN" ]] || exit 2
mkdir "$RUN"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$PY" -m e82.prepare s0 --fixture --output "$RUN/S0.pt"
"$PY" -m e82.train --run "$RUN/P1" --stage P1 --arm P --fixture --steps 1 --init "$RUN/S0.pt" --threads 2
"$PY" -m e82.train --run "$RUN/P3_warm" --stage P3_warm --arm P --fixture --steps 1 --init "$RUN/P1/stage_end.pt" --threads 2
"$PY" -m e82.train --run "$RUN/P3_joint" --stage P3_joint --arm P --fixture --steps 1 --init "$RUN/P3_warm/stage_end.pt" --threads 2
"$PY" -m e82.train --run "$RUN/P3_joint" --stage P3_joint --arm P --fixture --steps 2 --init "$RUN/P3_warm/stage_end.pt" --resume --threads 2
"$PY" -c 'from e82.runtime import atomic_json,brt;import sys;atomic_json(sys.argv[1],dict(status="complete",brt=brt(),formal_training=False,GPUh=0,stages=["P1","P3_warm","P3_joint","resume"]))' "$RUN/status.json"
