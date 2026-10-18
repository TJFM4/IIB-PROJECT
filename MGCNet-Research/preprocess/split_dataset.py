import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
import os

# Loads the JSON into a dictionary mapping each patient ID to its counts
def load_patient_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# For a list of patient IDs, it adds up their N and AFIB counts and prints them
def print_split_stats(patient_data, pids, title="Split"):
    """打印指定患者列表的标签统计"""
    if not pids:
        print(f"{title}: 0 人")
        return 0.0

    N_total = sum(patient_data[pid]['N'] for pid in pids)
    AFIB_total = sum(patient_data[pid]['AFIB'] for pid in pids)
    total_labels = N_total + AFIB_total
    afib_ratio = AFIB_total / total_labels if total_labels > 0 else 0

    print(f"{title}")
    print(f"  N: {N_total:>6,} | AFIB: {AFIB_total:>6,} | 总标签数: {total_labels:>7,} | AFIB占比: {afib_ratio:.3%}")

    return afib_ratio

# Writes a fold as {"train": [...] and "val": [...]}, with the IDs sorted, to fold_{k}.json (for output_dir = data_split/bihaf)
def save_fold_to_json(patient_data, train_pids, val_pids, fold_idx, output_dir):
    """保存每一折的划分"""
    os.makedirs(output_dir, exist_ok=True)
    fold_data = {
        "train": sorted(train_pids),
        "val": sorted(val_pids)
    }
    filename = os.path.join(output_dir, f"fold_{fold_idx}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(fold_data, f, indent=4)
    print(f"已保存: {filename}")


def main():
    input_path = './data_split/bihaf/label_counts.json'
    output_dir = "./data_split/bihaf"
    
    n_splits = 5  # 5 folds
    random_state = 42  # seed?

    # Load the patient data
    patient_data = load_patient_data(input_path)
    pids = list(patient_data.keys())
    print(f"Length of Patient IDs: {len(pids)}")

    # Assigns a bin label to the group of segments - 0 if less than 30% AFIB, 1 if 30-70% AFIB, 2 otherwise
    afib_ratios = []
    for pid in pids:
        n = patient_data[pid]['N']
        afib = patient_data[pid]['AFIB']
        total = n + afib
        ratio = afib / total if total > 0 else 0
        # 分桶：0=低AFIB, 1=中, 2=高
        if ratio < 0.3:
            bin_label = 0
        elif ratio < 0.7:
            bin_label = 1
        else:
            bin_label = 2
        afib_ratios.append(bin_label)

    # Stratified K Fold: Splits the patients so each fold has roughly the same mix of low, mid, and high AF patients. This is done at the patient level so one patient's segments never end up in both train and val sets. 
    random_state = 0
    while True:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        flag = 0
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(pids, afib_ratios), 1):
            print(f"\n--- Fold {fold_idx} ---")

            train_pids = [pids[i] for i in train_idx]
            val_pids = [pids[i] for i in val_idx]

            # Check the split ratios for the sets
            all_ratio = print_split_stats(patient_data, pids, title="总数据集")
            train_ratio = print_split_stats(patient_data, train_pids, title="训练集")
            val_ratio = print_split_stats(patient_data, val_pids, title="验证集")

            # Adds a flag if the set ratios are relatively similar
            if abs(all_ratio - train_ratio) < 0.02 and abs(all_ratio - val_ratio) < 0.02:
                flag += 1
            else:
                continue
        if flag == n_splits: # n_splits = 5
            print("Accepted Seed " + str(random_state))
            fold_idx = 1
            for train_idx, val_idx in skf.split(pids, afib_ratios):
                train_pids = [pids[i] for i in train_idx]
                val_pids = [pids[i] for i in val_idx]
                save_fold_to_json(patient_data, train_pids, val_pids, fold_idx, output_dir)
                fold_idx += 1
            break
        else:
            print("Failed Seed " + str(random_state))
            random_state += 1 # Changes the random state
        # except Exception:
        #     print("Error Seed" + str(random_state))
        #     random_state += 1
        #     continue

        # save_fold_to_json(patient_data, train_pids, val_pids, fold_idx, output_dir)

    print(f"\n{'=' * 60}")
    print("✅ 5 Folds")
    print(f"Output Location: {output_dir}")
    print(f"Files: fold_1.json ~ fold_{n_splits}.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()