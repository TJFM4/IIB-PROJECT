### results/zeroshot_pilot100
Zeroshot Icentia11k tests of the best models trained on AFDB (metric: specificity). This includes:
- Fold 1: bgm_afdb_1_afdb_only
- Fold 2: bgm_afdb_2_afdb_only
- Fold 3: bgm_afdb_3_authcopy
- Fold 4: bgm_afdb_4_authcopy
- Fold 5: bgm_afdb_5_afdb_only 

### configs/zeroshot
Contains the config files (no training or validation set) for the zeroshot tests of the best AFDB folds on the Icentia11k 97-patient sample. 

### preprocess/data_split/icentia_zeroshot
Contains the data split for the 5 folds for the zeroshot tests. 

### preprocess/data_split/icentia_100
Contains the data split for the 5 folds for the training of MGCNet on the Icentia11k 97-patient sample. 

label_counts.json produced using preprocess/statstic.py
fold_i.json generated using preprocess/split_dataset.py