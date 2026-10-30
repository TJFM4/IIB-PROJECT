# AFDB and CPSC2021 Data Audit

Comparison of the author's preprocessed segments (`BGM_GitHub/afdb_dataset/data`) with my own preprocessing (`MGCNet-Research/afdb_dataset/data`) and with the author's preprocessing script (`BGM_GitHub/preprocess/segment_BIHAF.py`, `segment_CPSC2021.py`) re-run locally (neurokit2 0.2.13). Dates: 29 Oct 2026.

## 1. How the author's segments are produced

`segment_BIHAF.py` does the following per record:

1. `nk.ecg_clean(signal, sampling_rate=fs)` on **both** leads.
2. Discard everything before the first rhythm annotation, `ann.sample[0]`.
3. Cut non-overlapping 10 s windows (2,500 samples) from there.
4. Quality check on channel 0: `nk.ecg_process` for R-peaks, then `nk.ecg_quality(..., method="zhao2018")`. Drop only `Unacceptable` (keep `Barely acceptable` and `Excellent`). Windows that raise an exception are dropped.
5. Keep a window only if it is 100% `N` or 100% `AFIB`.
6. Save `segments` (n, 2500, 2) float64 and `labels`.

`segment_CPSC2021.py` is the same, except that it fills empty `aux_note` entries with the previous rhythm and does not resample (200 Hz, 2,000-sample windows; resampling to 250 Hz happens at load time in `ecg_augment`).

## 2. Differences between my preprocessing and the author's

| Step | Mine (`raw_to_npz_converter.ipynb`) | Author (`segment_BIHAF.py`) |
|---|---|---|
| Filter | Butterworth 0.5–40 Hz, order 4, ECG1 only | `nk.ecg_clean`, both leads |
| Grid start | Sample 0 | `ann.sample[0]` |
| Quality | Keep only `Excellent` | Drop only `Unacceptable` |
| Output | (n, 2500, 1) float32 | (n, 2500, 2) float64 |
| Labels | Single-rhythm N/AFIB windows | Same |

The quality threshold is the largest difference. For example, record 04908 gives 7 segments with my pipeline and 3,561 in the author's data.

## 3. Lead order in the author's AFDB data

In 9 of 23 records the author's two channels are exactly swapped (channel 0 = ECG2): 04746, 04908, 05091, 07859, 07879, 07910, 08215, 08219, 08434. Verified bit-exact against `nk.ecg_clean` of the raw records.

`segment_BIHAF.py` cannot produce this, so the author's raw input files must have had the leads in the other order. `data/dataset.py` uses `seg[:, 0]` only, so the author's model was trained on ECG1 for 14 records and ECG2 for 9.

## 4. Re-running the author's script

### AFDB

Every author segment is reproduced bit-exactly (waveform, grid and label), but the set of segments kept is very different:

| Record | Re-run (nk 0.2.13) | Author's data | Author segments found in re-run |
|---|---|---|---|
| 04043 | 3,407 | 314 | 314 / 314 |
| 06426 | 1,854 | 99 | 99 / 99 |

Re-grading with zhao2018 today, the windows the author kept and the label-eligible windows the author dropped have the same grade distribution (almost all `Barely acceptable`). The author's selection is therefore not reproduced by the published script in this environment.

### CPSC2021

My CPSC2021 data (`BGM_GitHub/cpsc2021_dataset/data`) was generated with the author's `segment_CPSC2021.py`. Re-running it reproduces those files bit-exactly (checked on data_44_7, data_44_6, data_26_3).

| | N | AFIB | Total |
|---|---|---|---|
| Label-eligible windows (no quality check) | 108,718 | 58,561 | 167,279 |
| Author's `label_counts.json` | 89,570 | 47,440 | 137,010 |
| Author's script, re-run | 90,400 | 49,912 | 140,312 |

- Same 105 cases and same 1,425 files.
- No case has more segments than its label-eligible count, in either set. Labelling and segmentation are identical; all differences come from the quality step.
- Both keep a median of about 92% of eligible windows per case, but 103 of 105 cases differ, in both directions. Largest differences:

| Case | Eligible N / AF | Author N / AF | Re-run N / AF |
|---|---|---|---|
| 44 | 6,523 / 0 | 1,652 / 0 | 3,309 / 0 |
| 38 | 0 / 4,330 | 0 / 624 | 0 / 2,078 |
| 26 | 2,220 / 0 | 1,895 / 0 | 1,166 / 0 |
| 32 | 1,273 / 1,326 | 438 / 37 | 719 / 248 |
| 8 | 0 / 3,281 | 0 / 2,910 | 0 / 2,436 |

## 5. Implications

- The author's training data cannot be regenerated from the published code in the current environment. Segment waveforms match exactly; segment selection does not.
- The CPSC2021 count gap (slide 14) is not caused by my preprocessing: the author's own script gives the same result.
- The CPSC2021 validation set (used to select checkpoints) contains different segments from the author's, which may contribute to the fold-level performance gaps.
- The author's model was trained on mixed leads (section 3).

## 6. Cause: neurokit2 version

Each neurokit2 release was installed in a separate environment (Python 3.11) and the author's clean → `ecg_process` → zhao2018 selection was re-run on the first 800 windows of 04043 and 06426. In that range the author kept 22 and 30 windows.

| neurokit2 | numpy | 04043 kept (author 22) | 06426 kept (author 30) |
|---|---|---|---|
| 0.2.3, 0.2.4 | 1.26 | 0 | 0 |
| 0.2.5, 0.2.6, 0.2.7, 0.2.9, 0.2.10, 0.2.12 | 1.26 | 22, all the author's | 30, all the author's |
| 0.2.11 | 1.26 | not run (missing `pywt`) | |
| 0.2.12 | 2.4 | 0 | 0 |
| 0.2.13 | 2.4 | 755 (21 of the author's) | 562 (27 of the author's) |

`ecg_clean` reproduces the author's waveforms bit-exactly in every version.

Confirmation on CPSC2021: running `segment_CPSC2021.py` under neurokit2 0.2.12 + numpy 1.26 reproduces the author's `label_counts.json` exactly for the cases with the largest discrepancies:

| Case | neurokit2 0.2.12 N / AF | Author N / AF | neurokit2 0.2.13 N / AF |
|---|---|---|---|
| 38 | 0 / 624 | 0 / 624 | 0 / 2,078 |
| 26 | 1,895 / 0 | 1,895 / 0 | 1,166 / 0 |
| 32 | 438 / 37 | 438 / 37 | 719 / 248 |
| 86 | 0 / 550 | 0 / 550 | 0 / 991 |

Conclusions:

- The discrepancies in sections 4 and 5 are caused by the neurokit2 version. The zhao2018 quality grading changed in 0.2.13 (March 2026), after the author generated their data. The author used a version between 0.2.5 and 0.2.12 with numpy < 2.
- "0 kept" rows are versions where `ecg_process`/`ecg_quality` raise on every window. The author's `try/except: continue` silently drops them, so a broken environment produces empty output without any error. Any re-run should check that the counts are non-zero.
- The paper's preprocessing is only reproducible with the neurokit2 version pinned. Neither the paper nor the repository states it.

### Full CPSC2021 regeneration

All 1,425 CPSC2021 files were re-run through the unchanged `segment_CPSC2021.py` under neurokit2 0.2.12 + numpy 1.26.4 + wfdb 4.1.2. Output now lives in `BGM_GitHub/cpsc2021_dataset/data/` (per-file counts in `_counts.json`); the previous nk 0.2.13 data was moved to `data_nk0213/`.

| | N | AFIB | Total |
|---|---|---|---|
| Author's `label_counts.json` | 89,570 | 47,440 | 137,010 |
| Regenerated (nk 0.2.12) | 89,570 | 47,440 | 137,010 |
| Previous data (nk 0.2.13) | 90,400 | 49,912 | 140,312 |

All 105 cases match the author's N and AFIB counts exactly. 71 files produce no segments, but per-case totals still match, so these are real rejections and not a broken environment.

## 7. Next steps

- ~~Regenerate CPSC2021 with neurokit2 0.2.12 + numpy < 2.~~ Done: exact match (above).
- The regenerated set has replaced `cpsc2021_dataset/data/`, so existing configs use it. Retrain at least one fold to measure the effect of the corrected validation set.
- Standardise my own preprocessing (AFDB, PhysioNet 2017) on the same environment.
