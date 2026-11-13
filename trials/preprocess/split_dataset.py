import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import argparse


def load_patient_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def afib_ratio(patient_data, pids):
    n = sum(patient_data[pid]['N'] for pid in pids)
    afib = sum(patient_data[pid]['AFIB'] for pid in pids)
    return afib / (n + afib) if n + afib > 0 else 0.0

def print_split_stats(patient_data, pids, title):
    n = sum(patient_data[pid]['N'] for pid in pids)
    afib = sum(patient_data[pid]['AFIB'] for pid in pids)
    n_af = sum(1 for pid in pids if patient_data[pid]['AFIB'] > 0)
    print(f'{title:<6} {len(pids):>6} patients (n{n_af:>4,} with AF) |'
          f'N: {n:>9,} | AFIB: {afib:>8,} | AFIB ratio: {afib_ratio(patient_data, pids):.3%}')

def save_fold_to_json(train_pids, val_pids, fold_idx, output_dir): # as in BGM_GitHub's split_dataset_2.py
    fold_data = {
        "train": sorted(train_pids), 
        "val": sorted(val_pids)
    }
    filename = os.path.join(output_dir, f'fold_{fold_idx}.json')
    with open(filename, 'w', encoding = 'utf-8') as f:
        json.dump(fold_data, f, indent = 4)
    print(f'Saved: {filename}')

def best_seed(patient_data, pids, strata, n_splits, max_seeds):
    overall = afib_ratio(patient_data, pids)
    best_dev, best = np.inf, None
    for seed in range(max_seeds):
        skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = seed)
        dev = max(abs(afib_ratio(patient_data, [pids[i] for i in val_idx]) - overall)
                  for _, val_idx in skf.split(pids, strata))
        if dev < best_dev:
            best_dev, best = dev, seed
    return best, best_dev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_dir', required=True, help='Folder containing label_counts.json; fold JSONs are written here')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--max_seeds', type=int, default=200, help='Shuffles tried when balancing the AFIB ratio')
    parser.add_argument('--all_val', action='store_true', help='Write one fold_1.json with every patient in val (zero-shot testing)')
    args = parser.parse_args()  

    patient_data = load_patient_data(os.path.join(args.split_dir, 'label_counts.json'))
    pids = sorted(patient_data)
    print(f'{len(pids)} patients')
    print_split_stats(patient_data, pids, 'All')

    if args.all_val:
        save_fold_to_json([], pids, 1, args.split_dir)
        return

    strata = [1 if patient_data[pid]['AFIB'] > 0 else 0 for pid in pids]
    if sum(strata) < args.n_splits:
        raise ValueError(f'Only {sum(strata)} patients with AF: need at least n_splits = {args.n_splits}')
    seed, dev = best_seed(patient_data, pids, strata, args.n_splits, args.max_seeds)
    print(f'Seed {seed}: largest val-fold deviation from overall AFIB ratio = {dev:.3%}')

    skf = StratifiedKFold(n_splits = args.n_splits, shuffle = True, random_state = seed)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(pids, strata), 1):
        train_pids = [pids[i] for i in train_idx]
        val_pids = [pids[i] for i in val_idx]
        print(f'Fold {fold_idx}')
        print_split_stats(patient_data, train_pids, 'Train')
        print_split_stats(patient_data, val_pids, 'Val')
        save_fold_to_json(train_pids, val_pids, fold_idx, args.split_dir)

if __name__ == "__main__":
    main()