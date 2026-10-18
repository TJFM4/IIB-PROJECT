# run_experiments.py
import os

# This runs a train and test run, both using the same config file in ./config/bgm/bgm_afdb_1.yaml. 
# test.py only starts after train.py is done since os.system(cmd) runs these as if they were typed in the terminal. 
# The config has some interesting conditions: 
### The training set is fold 1 'train' of AFDB. The validation set is fold 1 'val' of CPSC2021. The test set is fold 1 'val' of AFDB. 
### Only fold 1 runs. Folds 2-5 could be added to commands to run all 5. 
### The data_dir paths point to the original author's machine. These need to be updated. 
### Settings: 25 epochs, early stopping patience of 10, batch size 128, Adam with learning rate of 1e-3, and z-score normalization. 


commands = [
    "python train.py --config ./config/bgm/bgm_afdb_1.yaml",
    "python test.py --config ./config/bgm/bgm_afdb_1.yaml",
]

for cmd in commands:
    print(f"Running: {cmd}")

    os.system(cmd)