# Fold 4 seed check

Tests whether the gap between `bgm_afdb_4_authcopy` (acc 0.9161) and the author's fold 4 (0.9748) can come from run-to-run variation alone.

The configs are the `_authcopy` fold 4 config from before commit e6ff752 (CPSC2021 validation, best-val checkpoint, early stopping, vertical flip on). Only the title and seed changed:

| Config | Seed |
|---|---|
| `bgm_afdb_4_authcopy` (original, in git history) | 3407 |
| `bgm_afdb_4_authcopy_seed2.yaml` | 2 |
| `bgm_afdb_4_authcopy_seed3.yaml` | 3 |

Run: `bash authcopy_seeds/run_fold4_seeds.sh` from `BGM_GitHub/`. Takes about 2–2.5 h per seed on the Mac.

Outputs go to the usual places: `new_runs/checkpoint/`, `new_runs/results/`, `new_runs/runs/`, and the log `new_runs/logs/authcopy_fold4_seeds23.log`. Add both runs to `new_runs/runs.md` when done.
