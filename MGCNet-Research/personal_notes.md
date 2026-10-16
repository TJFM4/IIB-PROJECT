# Notes on the MGCNet Repository

## afdb_dataset, checkpoint, results
These folders contain a placeholder link to a Baidu .zip file called BGM_Github2.zip. 

The AFDB's online address is: https://physionet.org/content/afdb/1.0.0/

**checkpoint** is also the destination for TensorBoard checkpoints output from experiments. 

**results** is the destination for TensorBoard results output from experiments. 

## config/bgm
bgm_afdb_1.yaml is the training configuration for the bgm_afdb_1 experiments. The experiment uses the AFDB dataset for training and the CPSC2021 dataset for validation. The data paths point to the original author's machine. **Need to change these for my own experiment.**

**Augmentation:** 250 Hz signals, augmentation applied with p = 0.5. This includes Gaussian noise, time warp, time masking, amplitude scaling, zoom in/out, or z-score normalization (as dicussed in the paper). 

The validation dataset is resampled from 200 to 250 Hz. No pre-processing is applied apart from the z-score normalization. 

**Model:** bgm, a binary classifier (AF vs. non-AF) with single-lead ECG input. 

**Training:** 25 epochs, early stopping with patience 10, batch size of 128, shuffled, seed 3407, runs on CUDA. 

**Optimizer and Learning Rate:** Adam with learning rate of 1e-3 and weight decay of 1e-4. The scheduler is InverseTimeLR with decay rate 1e-4. 

**Outputs:** TensorBoard checkpoints go to ./checkpoint, and results to ./results. 

## data
**dataset.py:** This file defines the AFDataset class, a PyTorch dataset. Each sample comes back in a pair including the original ECG segment and the 128 x 128 spectrogram image. These feed the dual-branch MGCNet model as inputs. 

**ecg_augment.py:** Introduces ECGAugment, a transform that AFDataset.__getitem__ calls on each sample before splitting it into the waveform and spectrogram views. The YAML **augmentation:** block is passed directly as its constructor arguments, so the code's defaults are always overriden. 

## model
**bgm.py:** Contains the network itself (the GMCNet class) and supervised contrastive losses that **train.py** adds to the standard cross-entropy loss. 

The naming in this file is a bit off. BGMs are one of the primary building blocks of the network, but they aren't the model itself. Additionally, the model class is known as the GMCNet whereas the model is actually called MGCNet. 

## preprocess
Under afdb_data/cloud_address.txt, there is a link to BGM_Github2.zip which contains the MIT-BIH AF records pre-segmented into 10 second windows at 250 Hz. Everything in **preprocess** starts from those files. 

**statistic.py:** Runs through each .npz in the dataset, prints the shape of its segments, and counts the windows labelled N and AFIB. Other labels such as AFL are ignored. The results are written to label_counts.json. 

**split_dataset.py:** Reads the label counts and creates 5-fold cross-validation splits by patient, so one person's segments never end up in both train and validation sets. This is done to avoid correlation issues related to a single patient's long ECG. The goal is to learn the condition's general signs, not the patient's heart rhythm. StratifiedKFold spreads the buckets of patients with low, medium or high fractions of AF across all five folds. 

The validation set is pointed at ./preprocess/data_split/cpsc2021 but this folder does not exist in the repository. As such, I will need to download CPSC2021 and create the file splits myself. 

## train.py, test.py, run_experiments.py
These three scripts make up the experiment pipeline. **train.py** fits the model and saves the best weights, **test.py** reloads those weighhts and scores them, and **run_experiments.py** runs the two in order. 

## changes.md
Documentation of the changes made to MGCNet-Research relative to the original MGCNet repository. 