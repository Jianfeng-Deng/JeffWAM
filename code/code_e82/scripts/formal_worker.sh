#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
RUN=$1; CONFIG=$2; STATS=$3; S0=$4; GPU=$5
CODE=$(cd "$(dirname "$0")/.." && pwd)
PY=/mnt/workspace/dengjianfeng/venvs/e29/bin/python
export CUDA_VISIBLE_DEVICES="$GPU"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$CODE"
status() {
  "$PY" -c 'import sys; from e82.runtime import atomic_json,brt; atomic_json(sys.argv[1],dict(status=sys.argv[2],stage=sys.argv[3],brt=brt()))' "$RUN/chain_status.json" "$1" "${2:-none}"
}
trap 'status stopped "${CURRENT:-initialization}"' ERR INT TERM
stage() {
  local ARM=$1 PHASE=$2 INIT=$3
  CURRENT="${ARM}_${PHASE}"
  status running "$CURRENT"
  "$PY" -m e82.train --config "$CONFIG" --run "$RUN/$CURRENT" --stage "$PHASE" --arm "$ARM" --stats "$STATS" --init "$INIT" --device cuda:0 --ledger "$RUN/paired_budget.jsonl"
}
# A single shared S0 and stat contract. P3 streams have no arm key.
stage P P1 "$S0"
stage P P3_warm "$RUN/P_P1/stage_end.pt"
stage P P3_joint "$RUN/P_P3_warm/stage_end.pt"
stage R P3_warm "$S0"
stage R P3_joint "$RUN/R_P3_warm/stage_end.pt"
status complete all
