#!/bin/bash
# Train and test the _authcopy_exact configs (author's 17-18 Dec runs).
# Usage: bash authcopy_exact/run_folds.sh [folds...]    e.g. bash authcopy_exact/run_folds.sh 4 5
# Default is all five folds. Outputs go to new_runs/{runs,checkpoint,results}; terminal output to new_runs/logs.

cd "$(dirname "$0")/.." || exit 1          # BGM_GitHub: configs use ./new_runs/... paths
source ../.venv/bin/activate

FOLDS="${*:-1 2 3 4 5}"
LOG="new_runs/logs/authcopy_exact_folds${FOLDS// /}.log"

FOLDS="$FOLDS" caffeinate -i bash -c '
for f in $FOLDS; do
  cfg=authcopy_exact/bgm_afdb_${f}_authcopy_exact.yaml
  python train.py --config $cfg && python test.py --config $cfg
done' 2>&1 | tee "$LOG"
