This file contains documentation and summaries of the changes made to the original MGCNet repository during this research project. 

### preprocess/raw_to_npz_converter.ipynb
split_dataset.py and stastic.py are the author's original files in the preprocess folder. They operate on 10 second .npz windows from a Baidu zip file. Hence, I created raw_to_npz_converter.py to turn the 23 10-hour segments from datasets/MIT-BIH-AFDB into the  .npz format used in the author's code. 

These .npz files were saved to afdb_dataset/data. 

### preprocess/stastic.py
The only change required for this file are the input (.npz data) and output (label_counts.json) paths. 

After running stastic.py with this change, I compared my label_counts.json with the author's original. The values differed quite significantly for certain recordings. 

From the MGCNet paper, they describe using the method of Zhao et al. to eliminate segments which are too noisy from the training data. This must be implemented into raw_to_npz_converter.ipynb. 