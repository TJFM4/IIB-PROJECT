## zeroshot
This was an evaluation of the best performing MGCNet folds (specified in documentation.md) trained on the MIT-BIH-AFDB dataset with tests on Icentia11k 97 patient sample (for an initial trial).

There were no real concerns here, apart from those already associated with MIT-BIH-AFDB (e.g. outliers, class imbalance). 

Each fold from AFDB ran on all 97 patients in the sample group. 

## icentia_100
Training and testing MGCNet on 97 patient records from the Icentia11k dataset. This is a prototype of the full-scale run that might follow. 

Over the 97 patients, 8 have AF segments. Three patients (p00060, p00065 and p00075) account for 39k of the 50,214 AF windows. Therefore, we expect fold-by-fold performance differences. 

N segments: 59,859
AF segments: 50,214
AFIB ratio: 45.619%

However, for seed 78 used in constructing the folds, the maximum deviation of the val-fold from the overall AFIB ratio is 23.431%. 

These metadata can be verified using: (from trials)

python preprocess/split_dataset.py --split_dir preprocess/data_split/icentia_100

#### icentia_100 - fold 1
The held-out 20 patients include 2 AF patients, p00016 (207 AF windows) and p00091 (6,654 AF windows). There are also 10,806 N windows in the held-out set. Recall will thus mostly reflect p00091. 
