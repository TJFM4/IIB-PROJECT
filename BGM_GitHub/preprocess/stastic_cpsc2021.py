import os
import numpy as np
import json
from collections import Counter


def analyze_npz_labels_cpsc(folder_path, output_folder):
    """
    分析 CPSC2021 风格的 .npz 文件（data_case_sub.npz），
    按 case ID 聚合 labels 中 'N' 和 'AFIB' 的数量。
    """
    os.makedirs(output_folder, exist_ok=True)

    # 存储按 case ID 的汇总结果
    results_by_case = {}
    count_sum = Counter()  # 总计

    # 用于记录每个 case 的子文件是否都处理成功
    case_files = {}

    # 遍历文件夹中的所有 .npz 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.npz'):
            file_path = os.path.join(folder_path, filename)
            try:
                # 解析文件名：data_case_sub.npz -> case_id = case
                # 示例：data_0_1.npz -> case_id = '0'
                name_part = filename[:-4]  # 去掉 .npz
                parts = name_part.split('_')
                if len(parts) < 3:
                    print(f"跳过无效文件名格式: {filename}")
                    continue
                try:
                    case_id = parts[1]  # data_0_1.npz -> parts[1] = '0'
                except:
                    print(f"无法解析样例号: {filename}")
                    continue

                # 加载 .npz 文件
                data = np.load(file_path, allow_pickle=True)

                if 'labels' in data:
                    labels = data['labels']
                    labels = np.array(labels).flatten()

                    count_N = int(np.sum(labels == 'N'))
                    count_AFIB = int(np.sum(labels == 'AFIB'))

                    # 初始化该 case 的统计
                    if case_id not in results_by_case:
                        results_by_case[case_id] = Counter()
                    results_by_case[case_id].update({'N': count_N, 'AFIB': count_AFIB})

                    # 记录该 case 包含哪些文件（可选，用于调试）
                    if case_id not in case_files:
                        case_files[case_id] = []
                    case_files[case_id].append(filename)
                else:
                    print(f"警告: {filename} 中没有 'labels' 键")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    # 将 Counter 转为普通字典并汇总总计
    final_results = {}
    for case_id, counter in results_by_case.items():
        count_dict = dict(counter)
        final_results[case_id] = {
            'N': count_dict.get('N', 0),
            'AFIB': count_dict.get('AFIB', 0)
        }
        # 累加到总计
        count_sum.update(count_dict)

    # 保存结果
    print(f'总计：{dict(count_sum)}')
    output_file = os.path.join(output_folder, 'label_counts.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"统计完成，结果已按样例号聚合，保存至: {output_file}")

    # （可选）保存每个 case 对应的文件列表
    case_file_mapping = os.path.join(output_folder, 'case_to_files.json')
    with open(case_file_mapping, 'w', encoding='utf-8') as f:
        json.dump(case_files, f, indent=4, ensure_ascii=False)
    print(f"样例与文件映射已保存至: {case_file_mapping}")


if __name__ == "__main__":
    # 根据你的数据集切换路径
    folder_path = '/home/oyxq/PythonProjects/dataset/CPSC2021/data'  # CPSC2021 数据路径
    output_folder = './data_split/cpsc2021'  # 输出目录

    analyze_npz_labels_cpsc(folder_path, output_folder)