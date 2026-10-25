# Summary 1: Test Results

Class 0 = N, class 1 = AF. "Specificity (avg)" is the value `test.py` reports, which is balanced accuracy. Specificity (N) is the true specificity.

## 1. Overview

| Trial | Model | Test set | Segments (N / AF) | Accuracy | AUC | F1 (AF) | Precision (AF) | Recall / Sensitivity (AF) | Specificity (N) | Specificity (avg) | Loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | Mine (CE + 0.01·SCL) | AFDB fold 1 | 12,186 (7,606 / 4,580) | 0.9866 | 0.9990 | 0.9821 | 0.9891 | 0.9751 | 0.9936 | 0.9843 | 0.0353 |
| Fold 2 | Mine (CE + 0.01·SCL) | AFDB fold 2 | 14,216 (8,929 / 5,287) | 0.8644 | 0.8956 | 0.8443 | 0.7367 | 0.9887 | 0.7908 | 0.8897 | 0.6178 |
| Fold 2 (CE only) | Mine (CE, λ = 0) | AFDB fold 2 | 14,216 (8,929 / 5,287) | 0.8590 | 0.9411 | 0.8323 | 0.7462 | 0.9408 | 0.8105 | 0.8757 | 0.4421 |
| Fold 3 | Mine (CE + 0.01·SCL) | AFDB fold 3 | 12,774 (7,846 / 4,928) | 0.8662 | 0.9811 | 0.7941 | 0.9775 | 0.6686 | 0.9903 | 0.8295 | 0.3909 |
| PhysioNet 2017 | Author's fold-1 checkpoint | PhysioNet 2017 N + A, all 10 s windows (5,834 records) | 18,244 (15,851 / 2,393) | 0.8670 | 0.9411 | 0.6339 | 0.4960 | 0.8780 | 0.8653 | 0.8716 | 0.5093 |

Author's saved AFDB accuracies for comparison: fold 1 = 0.9989, fold 2 = 0.9817, fold 3 = 0.9858.

## 2. Per-class reports

| Trial | Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| Fold 1 | N | 0.9851 | 0.9936 | 0.9893 | 7,606 |
| | AF | 0.9891 | 0.9751 | 0.9821 | 4,580 |
| | Macro avg | 0.9871 | 0.9843 | 0.9857 | 12,186 |
| | Weighted avg | 0.9866 | 0.9866 | 0.9866 | 12,186 |
| Fold 2 | N | 0.9916 | 0.7908 | 0.8799 | 8,929 |
| | AF | 0.7367 | 0.9887 | 0.8443 | 5,287 |
| | Macro avg | 0.8641 | 0.8897 | 0.8621 | 14,216 |
| | Weighted avg | 0.8968 | 0.8644 | 0.8666 | 14,216 |
| Fold 2 (CE only) | N | 0.9585 | 0.8105 | 0.8783 | 8,929 |
| | AF | 0.7462 | 0.9408 | 0.8323 | 5,287 |
| | Macro avg | 0.8524 | 0.8757 | 0.8553 | 14,216 |
| | Weighted avg | 0.8796 | 0.8590 | 0.8612 | 14,216 |
| Fold 3 | N | 0.8263 | 0.9903 | 0.9009 | 7,846 |
| | AF | 0.9775 | 0.6686 | 0.7941 | 4,928 |
| | Macro avg | 0.9019 | 0.8295 | 0.8475 | 12,774 |
| | Weighted avg | 0.8846 | 0.8662 | 0.8597 | 12,774 |
| PhysioNet 2017 | N | 0.9792 | 0.8653 | 0.9187 | 15,851 |
| | AF | 0.4960 | 0.8780 | 0.6339 | 2,393 |
| | Macro avg | 0.7376 | 0.8716 | 0.7763 | 18,244 |
| | Weighted avg | 0.9158 | 0.8670 | 0.8814 | 18,244 |

## 3. Fold 2 per record: original vs CE only

FP = N segment predicted as AF; FN = AF segment predicted as N.

| Record | Segments | AF segments | Original FP | Original FN | CE-only FP | CE-only FN |
|---|---|---|---|---|---|---|
| 04015 | 3,382 | 0 | 58 | 0 | 226 | 0 |
| 05121 | 72 | 28 | 5 | 11 | 33 | 7 |
| 05261 | 3,592 | 31 | 13 | 0 | 99 | 0 |
| 06995 | 3,664 | 1,722 | 1,792 | 4 | 1,334 | 253 |
| 07859 | 3,506 | 3,506 | 0 | 45 | 0 | 53 |
| **Total** | 14,216 | 5,287 | 1,868 | 60 | 1,692 | 313 |

Accuracy without record 06995: original 0.9875, CE only 0.9604. On 06995, the author's model makes 25 FP and 25 FN.

## 4. Notes

- Fold 2's errors are mostly from record 06995 (N segments predicted as AF). Fold 3's are mostly from record 08215 (1,581 missed AF segments). Without 08215, fold 3 accuracy is about 0.986.
- PhysioNet 2017 used all windows, with no quality filtering. The data is single-lead, 300 Hz, resampled to 250 Hz, and the leftover tail of each record is dropped. 11 records shorter than 10 s gave no windows.
