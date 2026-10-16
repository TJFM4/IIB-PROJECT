# IIB Project: Investigating the Applicability of State-of-the-Art Deep Learning ECG Classification Models to the Case of Single-Lead Handheld Device Screening of Atrial Fibrillation
While MGCNet and ECG-Mamba-V2 are promising models for the automated classification of ECG data, their performance on lower quality data has yet to be evaluted. Screening for Atrial Fibrillation (AF) remains an ongoing challenge as the arrhythmia's prevalence continues to rise amongst today's aging population. Using single-lead handheld devices like AliveCor's KardiaMobile, the hope is that large populations of at-risk individuals could be screened for AF using only a few single-lead ECGs. To this end, this research project investigates how state-of-the-art classification models can be improved in order to handle the noisy, artefact-ridden data provided by cheap, reusable devices for ECG data collection.

## Overviw
### papers
This folder contains the MGCNet, ECG-Mamba, and ECG-Mamba-V2 papers which characterize two state-of-the-art models for ECG classification in 2026. 

ECG-Mamba-V2 was trained on 12-lead data while MGCNet was trained on 2-lead data collected from a Holter monitor. MGCNet's implementation of Cross-Modal Supervised Contrastive Loss (CSCL) was designed to sharpen the model's classification boundaries in order to improve robustness across datasets. While it exhibited strong performance across two datasets (trained on one, tested on the other), both were comprised of high quality data collected in a clinical setting. Once again, this data is likely more relevant to the diagostic use case than the screening application which could preventatively alleviate hospital loads due to unexpected stroke and heart palpitations.

### MGCNet
This folder contains the original MGCNet repository copied from https://github.com/Ou-Young-1999/MGCNet. 

### MGCNet-Research
This folder contains a modified version of the MGCNet repository. Adjustments include changes to the data management, storage, and calling pipeline, the preprocessing techniques, the data augmentation techniques, the model's training, and the architecture. These changes are documented in MGCNet-Research/changes. 

### physionet2017
This folder contains the dataset used in the 2017 PhysioNet Computing in Cardiology (CinC) Challenge. 

This is a collection of 8,528 single-lead ECG signals obtained using AliveCor's KardiaMobile device. Samples range from 9 s to just over 60 s. There are four classes: Normal Sinus Rhythm, AF, Other Rhythm, and Noisy. 

### misc
This folder contains miscellaneous documents required for this project (e.g. planning and safety forms). 

### tools
This folder contains software developed for the purpose of manually evaluating ECG recordings from the PhysioNet 2017 dataset. It uses the WFDB library to plot, segment, filter, resample, and annotate individual ECG recordings depending on certain criteria. 
