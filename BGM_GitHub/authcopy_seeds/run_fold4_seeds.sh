#!/bin/bash
# Train and test _authcopy fold 4 with seeds 2 and 3.
# Outputs go to new_runs/{runs,checkpoint,results}; terminal output to new_runs/logs.
# Usage (from anywhere): bash "BGM_GitHub/authcopy_seeds/run_fold4_seeds.sh"

cd "$(dirname "$0")/.." || exit 1          # BGM_GitHub: configs use ./new_runs/... paths
source ../.venv/bin/activate

LOG=new_runs/logs/authcopy_fold4_seeds23.log

caffeinate -i bash -c '
for s in 2 3; do
  cfg=authcopy_seeds/bgm_afdb_4_authcopy_seed$s.yaml
  python train.py --config $cfg && python test.py --config $cfg
done' 2>&1 | tee "$LOG"
