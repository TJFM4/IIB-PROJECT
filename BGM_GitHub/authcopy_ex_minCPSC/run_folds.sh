#!/bin/bash
# Train and test the _authcopy_ex_minCPSC configs (Dec settings, no CPSC validation, last epoch).
# Usage: bash authcopy_ex_minCPSC/run_folds.sh [folds...]    e.g. bash authcopy_ex_minCPSC/run_folds.sh 4 5
# Default is folds 4 and 5. Outputs go to new_runs/{runs,checkpoint,results}; terminal output to new_runs/logs.

cd "$(dirname "$0")/.." || exit 1          # BGM_GitHub: configs use ./new_runs/... paths
source ../.venv/bin/activate

FOLDS="${*:-4 5}"
LOG="new_runs/logs/authcopy_ex_minCPSC_folds${FOLDS// /}.log"

FOLDS="$FOLDS" caffeinate -i bash -c '
for f in $FOLDS; do
  cfg=authcopy_ex_minCPSC/bgm_afdb_${f}_authcopy_ex_minCPSC.yaml
  python train.py --config $cfg && python test.py --config $cfg
done' 2>&1 | tee "$LOG"
