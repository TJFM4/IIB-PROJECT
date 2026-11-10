#!/bin/bash
# Train and test the _authcopy_ex_afdbval configs (Dec settings, validated on held-out AFDB training records).
# Usage: bash replication_trials/authcopy_ex_afdbval/run_folds.sh [folds...]    e.g. bash replication_trials/authcopy_ex_afdbval/run_folds.sh 4 5
# Default is folds 2-5. Outputs go to new_runs/{runs,checkpoint,results}; terminal output to new_runs/logs.

cd "$(dirname "$0")/../.." || exit 1       # BGM_GitHub: configs use ./new_runs/... paths
source ../.venv/bin/activate

FOLDS="${*:-2 3 4 5}"
LOG="new_runs/logs/authcopy_ex_afdbval_folds${FOLDS// /}.log"

FOLDS="$FOLDS" caffeinate -i bash -c '
for f in $FOLDS; do
  cfg=replication_trials/authcopy_ex_afdbval/bgm_afdb_${f}_authcopy_ex_afdbval.yaml
  python train.py --config $cfg && python test.py --config $cfg
done' 2>&1 | tee "$LOG"
