This file contains documentation and summaries of the changes made to the original MGCNet repository during this research project. 

**The preprocess sub-folder is rendered useless by the arrival of the BGM_GitHub download which includes the author's original folds.**

#### preprocess/raw_to_npz_converter.ipynb
split_dataset.py and stastic.py are the author's original files in the preprocess folder. They operate on 10 second .npz windows from a Baidu zip file. Hence, I created raw_to_npz_converter.py to turn the 23 10-hour segments from datasets/MIT-BIH-AFDB into the  .npz format used in the author's code. 

These .npz files were saved to afdb_dataset/data. 

#### preprocess/stastic.py
The only change required for this file are the input (.npz data) and output (label_counts.json) paths. 

After running stastic.py with this change, I compared my label_counts.json with the author's original. The values differed quite significantly for certain recordings. 

From the MGCNet paper, they describe using the method of Zhao et al. to eliminate segments which are too noisy from the training data. This must be implemented into raw_to_npz_converter.ipynb. 

I was unable to replicate the exact segment counts in the author's original repository using the approach described in their paper. However, I was able to find their real training data on a Baidu download, which is now in BGM_GitHub/afdb_dataset. 

#### preprocess/split_dataset.py
The only changes required for this file are the input (label_counts.json) and output (fold_1.json - fold_5.json) paths. 

### BGM_GitHub/cpsc2021_dataset
Using Claude, I used the **BGM_GitHub/preprocess** files to generate the CPSC2021's .npz files for the validation set. The original CPSC2021 data is located in **datasets/CPSC2021** and the .npz files are in **BGM_GitHub/cpsc2021_dataset/data**. 

### MGCNet-Research/config/bgm/bgm_afdb_1.yaml
I updated the training, validation, and test paths to reflect changes to the repository. I also changed the device used to "mcs" to run the training locally on my computer. 

### MGCNet-Research/train.py
I updated the code here to adopt a few changes: 
- Measure the time required for the training run
- Using the **max_batches** parameter, break the loop for a smaller training run (e.g. a single batch smoke test). 
- Update the training and validation ratios to adapt to the number of batches used. 

