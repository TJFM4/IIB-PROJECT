# Author's December settings, folds 2–5, validated on held-out AFDB records

Same as `authcopy_exact` folds 2–5 (lr 1e-3 for folds 2–3 and 1e-4 for folds 4–5, InverseTimeLR, SCL λ = 0.01, seed 3407, vertical flip on, 25 epochs, patience 10, best-val checkpoint), except that the validation set is three AFDB training records instead of CPSC2021.

For folds 4 and 5, together with `authcopy_exact` (best CPSC epoch) and `authcopy_ex_minCPSC` (last epoch), this gives three ways of choosing the checkpoint for the same training settings.

## Held-out records

Each fold holds out three mixed-rhythm records (both N and AF), like the hardest test records. None is a test record in its fold. The base set is `06995`, `07879`, `08219`. Where one of those is a test record, it is replaced by the closest-matching mixed record still in training.

| Fold | Validation records | Train (N / AF) | Validation (N / AF) | Test (N / AF), unchanged |
|---|---|---|---|---|
| 2 | 07879, 08219, **08455** (replaces 06995) | 22,751 / 12,625 | 5,038 / 4,141 | 8,929 / 5,287 |
| 3 | 06995, 07879, **04908** (replaces 08219) | 22,154 / 13,388 | 6,718 / 3,737 | 7,846 / 4,928 |
| 4 | 06995, 07879, 08219 | 26,296 / 15,312 | 5,952 / 3,916 | 4,470 / 2,825 |
| 5 | 06995, 07879, 08219 | 22,899 / 13,704 | 5,952 / 3,916 | 7,867 / 4,433 |

The split files are in `data_split/`. `inner_val` is the validation records; `val` is the test records, as in `preprocess/data_split/bihaf`, which the test block still reads.

The validation augmentation uses `original_fs: 250`. The CPSC configs had 200 Hz, which would have resampled the AFDB windows.

## Caveat

Training uses about 20% fewer windows than the other runs, so a lower test score could come from less training data rather than the selection method.

## Run

From `BGM_GitHub/`, plugged in:

```bash
bash replication_trials/authcopy_ex_afdbval/run_folds.sh          # folds 2-5, about 10 h
bash replication_trials/authcopy_ex_afdbval/run_folds.sh 4 5      # a subset
```

Log: `new_runs/logs/authcopy_ex_afdbval_folds2345.log`. Add every run to `new_runs/runs.md`.
