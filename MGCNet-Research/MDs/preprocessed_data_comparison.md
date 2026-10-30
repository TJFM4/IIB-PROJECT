## Comparing the Preprocessed Data Results
After implementing preprocessing aligned with the author's paper, there are discrepancies in the final segments achieved. 

While the AFDB 10s segments used by the author are available, the CPSC ones are not. In order to apply appropriate preprocessing to this dataset (and others in the future), I need to understand where the discrepancies arise. 

### Lead 1 vs. Lead 2
The author's channel 0 is ECG2, not ECG1, in 9 of 23 records: 04746, 04908, 05091, 07859, 07879, 07910, 08215, 08219, 08434. dataset.py only uses seg[:, 0], so the author's model was trained on a mix of ECG1 and ECG2 depending on the record. 