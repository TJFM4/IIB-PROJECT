# Author's December runs, per fold

Configs that replicate, as far as the TensorBoard logs allow, the author's run that most likely produced each reported fold result. The checkpoint `train.py` saves is overwritten by every rerun of a fold, so the reported result is taken to come from the last logged run of each fold. Fold 1 is confirmed by the shipped checkpoint; folds 2–5 are inferred.

| Fold | Author run (`BGM_GitHub/runs/bgm_afdb_N/`) | lr | Author best CPSC val | Author test acc |
|---|---|---|---|---|
| 1 | 17 Dec 03:27, `model` host (confirmed) | 1e-3 | 93.05@12, stopped at 22 | 0.9989 |
| 2 | 17 Dec 00:06 | 1e-3 | 94.38@18, 25 epochs | 0.9817 |
| 3 | 17 Dec 01:01 | 1e-3 | 90.22@13, stopped at 23 | 0.9858 |
| 4 | 18 Dec 00:07 | 1e-4 | 88.10@20, 25 epochs | 0.9748 |
| 5 | 17 Dec 09:28 | 1e-4 | 95.43@21, 25 epochs | 0.9980 |

## What is matched

Each config is the `_authcopy` config (before commit e6ff752) with only the title changed, and `lr: 0.0001` for folds 4 and 5.

- **Read from the logs:** learning rate, constant `InverseTimeLR` schedule, SCL λ = 0.01 (final training loss about 0.15–0.16; cross-entropy alone would end near 0.03), 25 epochs with patience 10 (early stops fall exactly 10 epochs after the best epoch).
- **Taken from the author's shipped configs, not visible in the logs:** seed 3407, batch 128, Adam with weight decay 1e-4, augmentation settings, 128×128 spectrograms, CPSC2021 validation with best-val checkpoint.
- **Validation data:** corrected CPSC2021 set (neurokit2 0.2.12), which matches the author's `label_counts.json` per case.

## What cannot be matched

- **Unlogged settings:** the author's lr-1e-4 fold-4 runs on 17 Dec 04:32, 06:06 and 18 Dec 00:07 differ from each other, although two other runs (16 Dec 06:59, 17 Dec 08:50) are identical, so the author's runs were deterministic. Something the logs do not record (seed, data or code) changed between them.
- **Hardware:** MPS vs the author's CUDA, so curves will not match epoch for epoch even with identical settings.

## Overlap with earlier runs

Folds 1–3 are the same configs as `bgm_afdb_{1,2,3}_authcopy`, so they mainly check run-to-run repeatability on MPS. Folds 4 and 5 are the new test: `_authcopy` used lr 1e-3 there.

## Run

From `BGM_GitHub/`:

```bash
bash authcopy_exact/run_folds.sh          # all five folds, about 12 h
bash authcopy_exact/run_folds.sh 4 5      # only the new configs, about 5 h
```

Plug the laptop in first: on battery, MPS training was about 3× slower and the 4 Nov run died. Log: `new_runs/logs/authcopy_exact_folds<folds>.log`. Add every run to `new_runs/runs.md`.
