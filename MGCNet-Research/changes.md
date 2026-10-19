This file contains documentation and summaries of the changes made to the original MGCNet repository during this research project. 

### preprocess/raw_to_npz_converter.ipynb
split_dataset.py and stastic.py are the author's original files in the preprocess folder. They operate on 10 second .npz windows from a Baidu zip file. Hence, I created raw_to_npz_converter.py to turn the 23 10-hour segments from datasets/MIT-BIH-AFDB into the  .npz format used in the author's code. 

These .npz files were saved to afdb_dataset/data. 

