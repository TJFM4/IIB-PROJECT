import os
import numpy as np
import json
from collections import Counter
import argparse
from tqdm import tqdm

def analyze_npz_labels(folder_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    results = {}
    errors = {}

    count_sum = Counter()

    files = sorted(f for f in os.listdir(folder_path) if f.endswith('.npz'))
    for filename in tqdm(files):
        file_path = os.path.join(folder_path, filename)
        pid = os.path.splitext(filename)[0]

        try:
            with np.load(file_path) as data:
                labels = np.array(data['labels']).flatten()
        except Exception as e:
            errors[pid] = str(e)
            continue

        count_N = int(np.sum(labels == 'N'))
        count_AFIB = int(np.sum(labels == 'AFIB'))
        results[pid] = {'N': count_N, 'AFIB': count_AFIB}
        count_sum.update({'N': count_N, 'AFIB': count_AFIB})

    n_af_patients = sum(1 for d in results.values() if d['AFIB'] > 0)
    print(f'Patients: {len(results)} of which {n_af_patients} with AFIB windows')
    print(f'Windows: {dict(count_sum)}')
    if errors: 
        print(f'Failed to read {len(errors)} files: {errors}')

    output_file = os.path.join(output_folder, 'label_counts.json')
    with open(output_file, 'w', encoding = 'utf-8') as f:
        json.dump(results, f, indent = 4, ensure_ascii = False)

    print(f'Saved to: {output_file}')
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required = True, help = 'Folder of per-patient .npz files from segment_icentia.py')
    parser.add_argument('--out', required=True, help='Split folder, e.g. preprocess/data_split/icentia')
    args = parser.parse_args()

    analyze_npz_labels(args.data, args.out)