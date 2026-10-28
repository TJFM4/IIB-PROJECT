# Summary 1: Test Results

Class 0 = N, class 1 = AF. "Specificity (avg)" is the value `test.py` reports, which is balanced accuracy. "Specificity (AF +ve)" is the true specificity with AF as the positive class: TN / (TN + FP), i.e. recall on N.

## 1. Overview

| Trial | Model | Test set | Segments (N / AF) | Accuracy | AUC | F1 (AF) | Precision (AF) | Recall / Sensitivity (AF) | Specificity (AF +ve) | Specificity (avg) | Loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | Mine (CE + 0.01·SCL) | AFDB fold 1 | 12,186 (7,606 / 4,580) | 0.9866 | 0.9990 | 0.9821 | 0.9891 | 0.9751 | 0.9936 | 0.9843 | 0.0353 |
| Fold 1 (author) | Author's saved results | AFDB fold 1 | 12,186 (7,606 / 4,580) | 0.9989 | 1.0000 | 0.9985 | 0.9978 | 0.9991 | 0.9987 | 0.9989 | 0.0035 |
| Fold 2 | Mine (CE + 0.01·SCL) | AFDB fold 2 | 14,216 (8,929 / 5,287) | 0.8644 | 0.8956 | 0.8443 | 0.7367 | 0.9887 | 0.7908 | 0.8897 | 0.6178 |
| Fold 2 (author) | Author's saved results | AFDB fold 2 | 14,216 (8,929 / 5,287) | 0.9817 | 0.9991 | 0.9750 | 0.9935 | 0.9571 | 0.9963 | 0.9767 | 0.0587 |
| Fold 2 (CE only) | Mine (CE, λ = 0) | AFDB fold 2 | 14,216 (8,929 / 5,287) | 0.8590 | 0.9411 | 0.8323 | 0.7462 | 0.9408 | 0.8105 | 0.8757 | 0.4421 |
| Fold 3 | Mine (CE + 0.01·SCL) | AFDB fold 3 | 12,774 (7,846 / 4,928) | 0.8662 | 0.9811 | 0.7941 | 0.9775 | 0.6686 | 0.9903 | 0.8295 | 0.3909 |
| Fold 3 (author) | Author's saved results | AFDB fold 3 | 12,774 (7,846 / 4,928) | 0.9858 | 0.9978 | 0.9814 | 0.9948 | 0.9683 | 0.9968 | 0.9826 | 0.0538 |
| Fold 4 | Mine (CE + 0.01·SCL, lr 1e-4) | AFDB fold 4 | 7,295 (4,470 / 2,825) | 0.8748 | 0.9740 | 0.8156 | 0.9497 | 0.7147 | 0.9761 | 0.8454 | 0.3188 |
| Fold 4 (author) | Author's saved results | AFDB fold 4 | 7,295 (4,470 / 2,825) | 0.9748 | 0.9966 | 0.9683 | 0.9421 | 0.9961 | 0.9613 | 0.9787 | 0.0947 |
| Fold 5 | Mine (CE + 0.01·SCL, lr 1e-4) | AFDB fold 5 | 12,300 (7,867 / 4,433) | 0.9359 | 0.9897 | 0.9083 | 0.9369 | 0.8813 | 0.9666 | 0.9240 | 0.1482 |
| Fold 5 (author) | Author's saved results | AFDB fold 5 | 12,300 (7,867 / 4,433) | 0.9980 | 1.0000 | 0.9973 | 1.0000 | 0.9946 | 1.0000 | 0.9973 | 0.0362 |
| PhysioNet 2017 | Author's fold-1 checkpoint | PhysioNet 2017 N + A, all 10 s windows (5,834 records) | 18,244 (15,851 / 2,393) | 0.8670 | 0.9411 | 0.6339 | 0.4960 | 0.8780 | 0.8653 | 0.8716 | 0.5093 |

"(author)" rows are the author's saved test results from `BGM_GitHub/results/bgm_afdb_<fold>/`.

Folds 1–3 use lr 1e-3. Folds 4–5 use lr 1e-4, matching the author's `bgm_afdb_4.yaml` and `bgm_afdb_5.yaml` (confirmed from the TensorBoard learning-rate log).

## 2. Per-class reports

| Trial | Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| Fold 1 | N | 0.9851 | 0.9936 | 0.9893 | 7,606 |
| | AF | 0.9891 | 0.9751 | 0.9821 | 4,580 |
| | Macro avg | 0.9871 | 0.9843 | 0.9857 | 12,186 |
| | Weighted avg | 0.9866 | 0.9866 | 0.9866 | 12,186 |
| Fold 1 (author) | N | 0.9995 | 0.9987 | 0.9991 | 7,606 |
| | AF | 0.9978 | 0.9991 | 0.9985 | 4,580 |
| | Macro avg | 0.9986 | 0.9989 | 0.9988 | 12,186 |
| | Weighted avg | 0.9989 | 0.9989 | 0.9989 | 12,186 |
| Fold 2 | N | 0.9916 | 0.7908 | 0.8799 | 8,929 |
| | AF | 0.7367 | 0.9887 | 0.8443 | 5,287 |
| | Macro avg | 0.8641 | 0.8897 | 0.8621 | 14,216 |
| | Weighted avg | 0.8968 | 0.8644 | 0.8666 | 14,216 |
| Fold 2 (author) | N | 0.9751 | 0.9963 | 0.9856 | 8,929 |
| | AF | 0.9935 | 0.9571 | 0.9750 | 5,287 |
| | Macro avg | 0.9843 | 0.9767 | 0.9803 | 14,216 |
| | Weighted avg | 0.9820 | 0.9817 | 0.9816 | 14,216 |
| Fold 2 (CE only) | N | 0.9585 | 0.8105 | 0.8783 | 8,929 |
| | AF | 0.7462 | 0.9408 | 0.8323 | 5,287 |
| | Macro avg | 0.8524 | 0.8757 | 0.8553 | 14,216 |
| | Weighted avg | 0.8796 | 0.8590 | 0.8612 | 14,216 |
| Fold 3 | N | 0.8263 | 0.9903 | 0.9009 | 7,846 |
| | AF | 0.9775 | 0.6686 | 0.7941 | 4,928 |
| | Macro avg | 0.9019 | 0.8295 | 0.8475 | 12,774 |
| | Weighted avg | 0.8846 | 0.8662 | 0.8597 | 12,774 |
| Fold 3 (author) | N | 0.9804 | 0.9968 | 0.9886 | 7,846 |
| | AF | 0.9948 | 0.9683 | 0.9814 | 4,928 |
| | Macro avg | 0.9876 | 0.9826 | 0.9850 | 12,774 |
| | Weighted avg | 0.9860 | 0.9858 | 0.9858 | 12,774 |
| Fold 4 | N | 0.8441 | 0.9761 | 0.9053 | 4,470 |
| | AF | 0.9497 | 0.7147 | 0.8156 | 2,825 |
| | Macro avg | 0.8969 | 0.8454 | 0.8604 | 7,295 |
| | Weighted avg | 0.8850 | 0.8748 | 0.8705 | 7,295 |
| Fold 4 (author) | N | 0.9974 | 0.9613 | 0.9790 | 4,470 |
| | AF | 0.9421 | 0.9961 | 0.9683 | 2,825 |
| | Macro avg | 0.9698 | 0.9787 | 0.9737 | 7,295 |
| | Weighted avg | 0.9760 | 0.9748 | 0.9749 | 7,295 |
| Fold 5 | N | 0.9353 | 0.9666 | 0.9507 | 7,867 |
| | AF | 0.9369 | 0.8813 | 0.9083 | 4,433 |
| | Macro avg | 0.9361 | 0.9240 | 0.9295 | 12,300 |
| | Weighted avg | 0.9359 | 0.9359 | 0.9354 | 12,300 |
| Fold 5 (author) | N | 0.9970 | 1.0000 | 0.9985 | 7,867 |
| | AF | 1.0000 | 0.9946 | 0.9973 | 4,433 |
| | Macro avg | 0.9985 | 0.9973 | 0.9979 | 12,300 |
| | Weighted avg | 0.9981 | 0.9980 | 0.9980 | 12,300 |
| PhysioNet 2017 | N | 0.9792 | 0.8653 | 0.9187 | 15,851 |
| | AF | 0.4960 | 0.8780 | 0.6339 | 2,393 |
| | Macro avg | 0.7376 | 0.8716 | 0.7763 | 18,244 |
| | Weighted avg | 0.9158 | 0.8670 | 0.8814 | 18,244 |

## 3. Fold 2 per record: original vs CE only

FP = N segment predicted as AF; FN = AF segment predicted as N. Spec = specificity with AF as the positive class, (N segments − FP) / N segments; — where the record has no N segments.

| Record | Segments | AF segments | Original FP | Original FN | Original Spec | CE-only FP | CE-only FN | CE-only Spec |
|---|---|---|---|---|---|---|---|---|
| 04015 | 3,382 | 0 | 58 | 0 | 0.9829 | 226 | 0 | 0.9332 |
| 05121 | 72 | 28 | 5 | 11 | 0.8864 | 33 | 7 | 0.2500 |
| 05261 | 3,592 | 31 | 13 | 0 | 0.9963 | 99 | 0 | 0.9722 |
| 06995 | 3,664 | 1,722 | 1,792 | 4 | 0.0772 | 1,334 | 253 | 0.3131 |
| 07859 | 3,506 | 3,506 | 0 | 45 | — | 0 | 53 | — |
| **Total** | 14,216 | 5,287 | 1,868 | 60 | 0.7908 | 1,692 | 313 | 0.8105 |

Accuracy without record 06995: original 0.9875, CE only 0.9604. On 06995, the author's model makes 25 FP and 25 FN.

## 4. Folds 4 and 5 per record: mine vs author

Spec is defined as in section 3.

| Fold | Record | Segments | AF segments | My FP | My FN | My Spec | Author FP | Author FN | Author Spec |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 04043 | 314 | 219 | 12 | 3 | 0.8737 | 12 | 2 | 0.8737 |
| 4 | 04048 | 267 | 2 | 5 | 1 | 0.9811 | 15 | 1 | 0.9434 |
| 4 | 04126 | 3,498 | 41 | 84 | 0 | 0.9757 | 140 | 0 | 0.9595 |
| 4 | 04936 | 3,216 | 2,563 | 6 | 802 | 0.9908 | 6 | 8 | 0.9908 |
| 4 | **Total** | 7,295 | 2,825 | 107 | 806 | 0.9761 | 173 | 11 | 0.9613 |
| 5 | 04908 | 3,561 | 229 | 262 | 3 | 0.9214 | 0 | 2 | 1.0000 |
| 5 | 05091 | 3,508 | 1 | 0 | 0 | 1.0000 | 0 | 0 | 1.0000 |
| 5 | 07162 | 2,256 | 2,256 | 0 | 512 | — | 0 | 3 | — |
| 5 | 08455 | 2,975 | 1,947 | 1 | 11 | 0.9990 | 0 | 19 | 1.0000 |
| 5 | **Total** | 12,300 | 4,433 | 263 | 526 | 0.9666 | 0 | 24 | 1.0000 |

Accuracy without record 04936 (fold 4): mine 0.9743, author 0.9583. Accuracy without record 07162 (fold 5): mine 0.9724, author 0.9979.

## 5. Notes

- Fold 2's errors are mostly from record 06995 (N segments predicted as AF). Fold 3's are mostly from record 08215 (1,581 missed AF segments). Without 08215, fold 3 accuracy is about 0.986.
- Fold 4's errors are mostly from record 04936 (802 missed AF segments; the author's model misses 8). On the other three fold-4 records, my model makes fewer errors than the author's (105 vs 170). Fold 5's errors come from 07162 (512 missed AF segments) and 04908 (262 N segments predicted as AF).
- Folds 4 and 5 ran all 25 epochs without early stopping. The saved checkpoints are from epoch 18 (fold 4, CPSC validation accuracy 89.6%) and epoch 19 (fold 5, 94.7%). Fold 4's validation accuracy ranged from 71% to 90% across epochs.
- PhysioNet 2017 used all windows, with no quality filtering. The data is single-lead, 300 Hz, resampled to 250 Hz, and the leftover tail of each record is dropped. 11 records shorter than 10 s gave no windows.
