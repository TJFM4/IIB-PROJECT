# Fold 4 seed check

Tests whether the gap between `bgm_afdb_4_authcopy` (acc 0.9161) and the author's fold 4 (0.9748) can come from run-to-run variation alone.

The configs are the `_authcopy` fold 4 config from before commit e6ff752 (CPSC2021 validation, best-val checkpoint, early stopping, vertical flip on). Only the title and seed changed:

| Config | Seed |
|---|---|
| `bgm_afdb_4_authcopy` (original, in git history) | 3407 |
| `bgm_afdb_4_authcopy_seed2.yaml` | 2 |
| `bgm_afdb_4_authcopy_seed3.yaml` | 3 |

Run: `bash replication_trials/authcopy_seeds/run_fold4_seeds.sh` from `BGM_GitHub/`. Takes about 2–2.5 h per seed on the Mac.

Outputs go to the usual places: `new_runs/checkpoint/`, `new_runs/results/`, `new_runs/runs/`, and the log `new_runs/logs/authcopy_fold4_seeds23.log`. Add both runs to `new_runs/runs.md` when done.

# CPSC-trained folds 4 and 5, other seeds

Copies of `replication_trials/authcopy_cpsc/bgm_cpsc_{4,5}.yaml` with only the title and seed changed. The configs have
Windows (CRLF) line endings, so edit them in Python or an editor: a `sed` pattern ending in `$` does not match.

| Config | Title | Seed |
|---|---|---|
| `authcopy_cpsc/bgm_cpsc_4.yaml` (original) | `bgm_cpsc_4_authcopy` | 3407 |
| `bgm_cpsc_4_authcopy_seed2.yaml` | `bgm_cpsc_4_authcopy` (title not changed, so it overwrote the seed-3407 checkpoint and results on 22 Nov) | 2 |
| `bgm_cpsc_4_authcopy_seed963198.yaml` | `bgm_cpsc_4_authcopy_seed963198` | 963198 (random, `secrets.randbelow(10**6)`, 22 Nov) |
| `bgm_cpsc_5_authcopy_seed320601.yaml` | `bgm_cpsc_5_authcopy_seed320601` | 320601 (random, as above) |

Run both: `bash replication_trials/authcopy_seeds/run_cpsc_folds45_random.sh` (log `new_runs/logs/authcopy_cpsc_folds45_random.log`,
about 8–9 h for both). Add both runs to `new_runs/runs.md` when done.
