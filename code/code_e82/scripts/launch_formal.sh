#!/usr/bin/env bash
# Generated only in E82-M. DO NOT execute during the implementation/profile round.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
if [[ $# != 6 ]]; then
  echo "Usage: $0 NEW_RUN_DIR CONFIG_JSON STAT_JSON S0_PT REAL_PROFILE_JSON IDLE_GPU_INDEX" >&2
  exit 2
fi
RUN=$1; CONFIG=$(realpath "$2"); STATS=$(realpath "$3"); S0=$(realpath "$4"); PROFILE=$(realpath "$5"); GPU=$6
CODE=$(cd "$(dirname "$0")/.." && pwd)
PY=/mnt/workspace/dengjianfeng/venvs/e29/bin/python
[[ ! -e "$RUN" ]] || { echo "Run already exists: $RUN" >&2; exit 2; }
cd "$CODE"
"$PY" -m e82.preflight --config "$CONFIG" --stats "$STATS" --s0 "$S0" --profile "$PROFILE" >/dev/null
read -r MEM UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
(( MEM < 512 && UTIL < 5 )) || { echo "GPU $GPU is not idle; no processes touched" >&2; exit 75; }
mkdir "$RUN"
RUN=$(realpath "$RUN")
"$PY" -m e82.preflight --config "$CONFIG" --stats "$STATS" --s0 "$S0" --profile "$PROFILE" > "$RUN/input_lock.json"
setsid nohup bash "$CODE/scripts/formal_worker.sh" "$RUN" "$CONFIG" "$STATS" "$S0" "$GPU" > "$RUN/chain.log" 2>&1 < /dev/null &
PID=$!
"$PY" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"status":"launched","pid":int(sys.argv[2]),"gpu":sys.argv[3]}))' "$RUN/chain_status.json" "$PID" "$GPU"
echo "Started paired chain PID=$PID; status=$RUN/chain_status.json"
