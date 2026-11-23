#!/bin/bash
# Train and test CPSC-trained _authcopy folds 4 and 5 with random seeds (fold 4: 963198, fold 5: 320601,
# drawn with secrets.randbelow on 22 Nov 2026). Each title includes its seed, so earlier runs are not overwritten.
# Outputs go to new_runs/{runs,checkpoint,results}; terminal output to new_runs/logs.
# Usage (from anywhere): bash "BGM_GitHub/replication_trials/authcopy_seeds/run_cpsc_folds45_random.sh"

cd "$(dirname "$0")/../.." || exit 1       # BGM_GitHub
PY=~/venvs/iib/bin/python                   # venv outside iCloud (see runs.md)
LOG=new_runs/logs/authcopy_cpsc_folds45_random.log

caffeinate -i bash -c "
for cfg in replication_trials/authcopy_seeds/bgm_cpsc_4_authcopy_seed963198.yaml \
           replication_trials/authcopy_seeds/bgm_cpsc_5_authcopy_seed320601.yaml; do
  echo \"=== \$cfg  \$(date)\"
  $PY -u train.py --config \$cfg && $PY -u test.py --config \$cfg
done" 2>&1 | tee "$LOG"
