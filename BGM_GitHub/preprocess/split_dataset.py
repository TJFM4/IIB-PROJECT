import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, train_test_split
import os


def load_patient_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def create_stratified_test_split(patient_data, test_size=0.2, random_state=42):
    """
    基于患者标签分布，分层划分测试集（20%），保持标签比例
    """
    pids = list(patient_data.keys())
    features = np.array([[d['N'], d['AFIB']] for d in patient_data.values()])

    # 使用 KMeans 聚类患者，模拟“标签分布类型”
    n_clusters = min(5, len(pids) // 2)  # 防止聚类数太多
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(features)

    # 在聚类上进行分层划分
    train_val_idx, test_idx = train_test_split(
        np.arange(len(pids)),
        test_size=test_size,
        stratify=clusters,
        random_state=random_state
    )

    test_pids = [pids[i] for i in test_idx]
    train_val_pids = [pids[i] for i in train_val_idx]

    return train_val_pids, test_pids

def print_split_stats(patient_data, test_pids, title="Split"):
    """打印总数据集、测试集、训练验证集的标签统计"""
    print(f"{title}")

    test_N = sum(patient_data[pid]['N'] for pid in test_pids)
    test_AFIB = sum(patient_data[pid]['AFIB'] for pid in test_pids)
    total_test = test_N + test_AFIB
    ratio_test = test_AFIB / total_test if total_test > 0 else 0

    print(f"  N: {test_N:>6,} | AFIB: {test_AFIB:>6,} | 总标签数: {total_test:>7,} | AFIB占比: {ratio_test:.3%}")

    return ratio_test

def save_fold_to_json(patient_data, train_pids, val_pids, test_pids, fold_idx, output_dir):
    """保存每一折的划分"""
    os.makedirs(output_dir, exist_ok=True)
    print_split_stats(patient_data, train_pids, title="Fold"+str(fold_idx)+"Train")
    print_split_stats(patient_data, val_pids, title="Fold" + str(fold_idx) + "Valid")
    print_split_stats(patient_data, test_pids, title="Fold" + str(fold_idx) + "Test")
    fold_data = {
        "train": sorted(train_pids),
        "val": sorted(val_pids),
        "test": sorted(test_pids)
    }
    filename = os.path.join(output_dir, f"fold_{fold_idx}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(fold_data, f, indent=4)
    print(f"已保存: {filename}")


def main():
    # === 输入配置 ===
    # input_path = './data_split/bihaf/label_counts.json'
    # output_dir = "./data_split/bihaf"  # 输出文件夹

    # input_path = './data_split/ltaf/label_counts.json'
    # output_dir = "./data_split/ltaf"  # 输出文件夹

    input_path = './data_split/cpsc2021/label_counts.json'
    output_dir = "./data_split/cpsc2021"  # 输出文件夹

    n_splits = 4  # 4折

    # === 1. 加载数据 ===
    patient_data = load_patient_data(input_path)
    print(f"共 {len(patient_data)} 名患者")

    random_state = 0
    while True:
        try:
            # === 2. 划分测试集（20%），保持标签分布一致 ===
            train_val_pids, test_pids = create_stratified_test_split(
                patient_data, test_size=0.2, random_state=random_state
            )
            print(f"训练和验证集 ({len(train_val_pids)} 人): {sorted(train_val_pids)}")
            print(f"测试集 ({len(test_pids)} 人): {sorted(test_pids)}")

            # === 3. 输出统计 ===
            all_ratio = print_split_stats(patient_data, train_val_pids+test_pids, title="总集划分统计")
            train_ratio = print_split_stats(patient_data, train_val_pids, title="训练和验证集划分统计")
            test_ratio = print_split_stats(patient_data, test_pids, title="测试集划分统计")

            if abs(all_ratio-train_ratio) < 0.01 and abs(all_ratio-test_ratio) < 0.01:
                print("随机种子选择："+str(random_state))
                break
            else:
                print("随机种子比例不适配：" + str(random_state))
                random_state += 1
        except Exception:
            print("随机种子错误不适配：" + str(random_state))
            random_state += 1
            continue  # 跳过

    # === 4. 对剩下的 80% 做 4 折交叉验证 ===
    # 构造标签用于分层：使用 AFIB / (N + AFIB) 的比例分桶
    afib_ratios = []
    for pid in train_val_pids:
        n = patient_data[pid]['N']
        afib = patient_data[pid]['AFIB']
        total = n + afib
        ratio = afib / total if total > 0 else 0
        if ratio < 0.3:
            bin_label = 0
        elif ratio < 0.7:
            bin_label = 1
        else:
            bin_label = 2
        afib_ratios.append(bin_label)

    random_state = 0
    while True:
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            print(f"\n{'=' * 50}")
            print("训练集，验证集，测试集四折比例")
            flag = 0
            for train_idx, val_idx in skf.split(train_val_pids, afib_ratios):
                train_pids = [train_val_pids[i] for i in train_idx]
                val_pids = [train_val_pids[i] for i in val_idx]
                all_ratio = print_split_stats(patient_data, train_val_pids + test_pids, title="总集划分统计")
                train_ratio = print_split_stats(patient_data, train_pids, title="训练集划分统计")
                valid_ratio = print_split_stats(patient_data, val_pids, title="验证集划分统计")
                test_ratio = print_split_stats(patient_data, test_pids, title="测试集划分统计")
                if abs(all_ratio - train_ratio) < 0.01 and abs(all_ratio - valid_ratio) < 0.01 and abs(all_ratio - test_ratio) < 0.01:
                    flag += 1
                else:
                    continue
            if flag == 4:
                print("随机种子选择：" + str(random_state))
                fold_idx = 1
                for train_idx, val_idx in skf.split(train_val_pids, afib_ratios):
                    train_pids = [train_val_pids[i] for i in train_idx]
                    val_pids = [train_val_pids[i] for i in val_idx]
                    save_fold_to_json(patient_data, train_pids, val_pids, test_pids, fold_idx, output_dir)
                    fold_idx += 1
                break
            else:
                print("随机种子比例不适配：" + str(random_state))
                random_state += 1
        except Exception:
            print("随机种子错误不适配：" + str(random_state))
            random_state += 1
            continue  # 跳过

    print("所有划分已保存完成！")


if __name__ == "__main__":
    # 确保训练集，验证集，测试集中不同标签的比例近似
    main()