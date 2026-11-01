# Run log

Every training run, test and evaluation I have done. All future runs write here as well.

The author's downloaded outputs (`bgm_afdb_N`, `bgm_cpsc_*`, ablations, TensorBoard logs from `dsserver` and `model`) stay in `BGM_GitHub/runs`, `BGM_GitHub/checkpoint` and `BGM_GitHub/results` and are not listed here.

## Layout

| Folder | Contents |
|---|---|
| `runs/<title>/` | TensorBoard event files (`tensorboard --logdir new_runs/runs`) |
| `checkpoint/<title>.pth` | Best-validation-accuracy weights saved by `train.py` |
| `results/<title>/` | `test.py` output: `metrics.json`, `classification_report.txt`, `test_predictions.csv`, `predictions.npz` |
| `logs/` | Terminal output of batch runs |

Configs write here via `logging` paths: `./new_runs/...` in `BGM_GitHub/config/bgm/bgm_afdb_N.yaml` (run from `BGM_GitHub/`), and absolute `.../BGM_GitHub/new_runs/...` in `MGCNet-Research/config/`.

## Conventions

- **Title suffixes:** `_authcopy` = author's January code and config (only data paths, device and SCL weight edited). `_oldcpsc` = validated on the neurokit2 0.2.13 CPSC2021 set. `_authexact` = author's December 2025 code and configs. No suffix = my MGCNet-Research pipeline ("Mine").
- **Metrics:** class 1 = AF. Spec (N) = recall on N = true specificity with AF positive. `test.py`'s "specificity_avg" is balanced accuracy and is not reported here.
- **Best val** = best CPSC2021 validation accuracy (%) @ epoch; this is the checkpoint that is tested.
- **Hardware:** all runs on MacBook Pro, MPS, seed 3407, batch 128, 25 epochs max, early stopping patience 10.

## Data versions

| Data | Version | Used by |
|---|---|---|
| AFDB (`BGM_GitHub/afdb_dataset/data`) | Author's segments (download, Dec 2025). 9 records have ECG1/ECG2 swapped. | All runs |
| CPSC2021 old (`cpsc2021_dataset/data_nk0213`) | Author's `segment_CPSC2021.py` under neurokit2 0.2.13: 140,312 segments | Runs before 29 Oct 22:48 |
| CPSC2021 corrected (`cpsc2021_dataset/data`) | Same script under neurokit2 0.2.12 + numpy 1.26: 137,010 segments, matches author's `label_counts.json` per case | Runs after 29 Oct 22:48 |

Details: `MGCNet-Research/MDs/AFDB_data_audit.md`.

## Training runs (AFDB five-fold, tested on the AFDB fold)

| Title | Finished | Code / config | CPSC | lr | SCL λ | Epochs | Best val | Acc | AUC | F1 | Prec | Recall | Spec (N) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bgm_afdb_1` | 24 Oct | Mine | old | 1e-3 | 0.01 | 19 | 91.86@9 | 0.9866 | 0.9990 | 0.9821 | 0.9891 | 0.9751 | 0.9936 |
| `bgm_afdb_2` | 24 Oct | Mine | old | 1e-3 | 0.01 | 25 | 93.72@25 | 0.8644 | 0.8956 | 0.8443 | 0.7367 | 0.9887 | 0.7908 |
| `bgm_afdb_2_ce` | 25 Oct | Mine | old | 1e-3 | 0 | 20 | 86.07@10 | 0.8590 | 0.9411 | 0.8323 | 0.7462 | 0.9408 | 0.8105 |
| `bgm_afdb_3` | 24 Oct | Mine | old | 1e-3 | 0.01 | 25 | 91.45@21 | 0.8662 | 0.9811 | 0.7941 | 0.9775 | 0.6686 | 0.9903 |
| `bgm_afdb_4` | 26 Oct | Mine | old | 1e-4 | 0.01 | 25 | 89.61@18 | 0.8748 | 0.9740 | 0.8156 | 0.9497 | 0.7147 | 0.9761 |
| `bgm_afdb_5` | 26 Oct | Mine | old | 1e-4 | 0.01 | 25 | 94.72@19 | 0.9359 | 0.9897 | 0.9083 | 0.9369 | 0.8813 | 0.9666 |
| `bgm_afdb_1_authcopy_oldcpsc` | 28 Oct | authcopy | old | 1e-3 | 0.01 | 19 | 91.86@9 | 0.9866 | 0.9990 | 0.9821 | 0.9891 | 0.9751 | 0.9936 |
| `bgm_afdb_2_authcopy_oldcpsc` | 29 Oct | authcopy | old | 1e-3 | 0.01 | 25 | 93.72@25 | 0.8644 | 0.8956 | 0.8443 | 0.7367 | 0.9887 | 0.7908 |
| `bgm_afdb_3_authcopy_oldcpsc` | 29 Oct | authcopy | old | 1e-3 | 0.01 | 25 | 91.45@21 | 0.8662 | 0.9811 | 0.7941 | 0.9775 | 0.6686 | 0.9903 |
| `bgm_afdb_4_authcopy_oldcpsc` | 29 Oct | authcopy | old | 1e-3 | 0.01 | 17 | 86.93@7 | 0.9335 | 0.9862 | 0.9099 | 0.9577 | 0.8665 | 0.9758 |
| `bgm_afdb_5_authcopy_oldcpsc` | 29 Oct | authcopy | old | 1e-3 | 0.01 | 20 | 94.86@10 | 0.8737 | 0.9988 | 0.7875 | 1.0000 | 0.6494 | 1.0000 |
| `bgm_afdb_2_authcopy` | 30 Oct | authcopy | corrected | 1e-3 | 0.01 | 25 | 91.56@19 | 0.8153 | 0.8890 | 0.7623 | 0.7309 | 0.7965 | 0.8264 |
| `bgm_afdb_3_authcopy` | 30 Oct | authcopy | corrected | 1e-3 | 0.01 | 21 | 87.48@11 | 0.8278 | 0.9704 | 0.7161 | 0.9833 | 0.5631 | 0.9940 |
| `bgm_afdb_4_authcopy` | 30 Oct | authcopy | corrected | 1e-3 | 0.01 | 25 | 88.27@21 | 0.9161 | 0.9805 | 0.8825 | 0.9639 | 0.8138 | 0.9808 |
| `bgm_afdb_1_authcopy` | in progress | authcopy | corrected | 1e-3 | 0.01 | | | | | | | | |
| `bgm_afdb_5_authcopy` | in progress | authcopy | corrected | 1e-3 | 0.01 | | | | | | | | |

Author's saved results, for reference (`BGM_GitHub/results/bgm_afdb_N`): accuracy 0.9989, 0.9817, 0.9858, 0.9748, 0.9980.

Notes:
- Mine and authcopy are bit-identical where the settings match (folds 1–3, old CPSC): MPS training is deterministic and the code paths are equivalent.
- Folds 4–5 "Mine" used lr 1e-4, the value in the author's last logged fold-4/5 runs. Authcopy uses 1e-3 for all folds, as in the paper.
- `_authcopy_oldcpsc` folds 2–4 checkpoints were overwritten by the corrected-CPSC reruns. Only their results and TensorBoard logs survive.
- Old fold 1 and 5 authcopy checkpoints are kept as `bgm_afdb_{1,5}_authcopy_oldcpsc.pth`.

## Evaluations without training

| Results folder | Date | Checkpoint | Test set | Acc | AUC | F1 | Prec | Recall | Spec (N) | Purpose |
|---|---|---|---|---|---|---|---|---|---|---|
| `author1_on_physionet/bgm_afdb_1` | 25 Oct | Author's `bgm_afdb_1.pth` | PhysioNet 2017 N + A, 10 s windows (18,244: 15,851 N / 2,393 AF) | 0.8670 | 0.9411 | 0.6339 | 0.4960 | 0.8780 | 0.8653 | Generalisation to an unseen dataset |
| `check_author1_on_afdb1` | 31 Oct | Author's `bgm_afdb_1.pth` | AFDB fold 1 (12,186) | 0.9989 | 1.0000 | 0.9985 | 0.9978 | 0.9991 | 0.9987 | Local inference reproduces the author's saved fold-1 result exactly |
| `check_author1_on_cpsc1` | 31 Oct | Author's `bgm_afdb_1.pth` | Corrected CPSC2021 fold-1 val (21,320) | 0.9305 | 0.9819 | 0.9070 | 0.8987 | 0.9153 | 0.9394 | Identify the run that made the checkpoint |

`check_author1_on_cpsc1` equals the author's 17 Dec 03:25 log at its best epoch (12, 93.049%). The checkpoint therefore came from the January pipeline (CPSC validation, InverseTimeLR, lr 1e-3, SCL 0.01, 128×128 spectrograms).

## Aborted runs

| Title | Date | Reason |
|---|---|---|
| `bgm_afdb_1_authexact` | 30–31 Oct | Author's December 2025 code and configs (validation = AFDB test fold, CosineAnnealing, 256×256 spectrograms). Needs about 18 GB per step at batch 128, more than the Mac's 16 GB, so it swapped at about 108 s per batch. Stopped partway through epoch 1. No checkpoint or results; `runs/` holds an empty event file. |

## Logs

| File | Contents |
|---|---|
| `logs/authcopy_nk0212_folds234.log` | Authcopy folds 2–4, corrected CPSC (30 Oct) |
| `logs/authexact_folds12345.log` | Aborted authexact fold 1 (30–31 Oct) |
| `logs/authcopy_nk0212_folds15.log` | Authcopy folds 1 and 5, corrected CPSC (started 1 Nov 22:43) |
