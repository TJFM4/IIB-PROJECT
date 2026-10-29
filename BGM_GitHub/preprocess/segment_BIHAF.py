import wfdb
import numpy as np
import neurokit2 as nk
import os
from collections import Counter
from tqdm import tqdm


def get_files(path):
    # 获取数据集中不同样例的名字
    file_names = []
    for f in os.listdir(path):
        if f.endswith('.dat') and os.path.isfile(os.path.join(path, f)):
            file_names.append(os.path.splitext(f)[0])

    return file_names

def read_data(base_path):
    # 获取数据和标签
    record = wfdb.rdrecord(base_path)
    annotation = wfdb.rdann(base_path, 'atr')
    ecg_raw = record.p_signal
    fs = record.fs
    return ecg_raw, annotation, fs

def ecg_denoise(ecg_raw, fs):
    # ECG去噪
    ecg_filtered = np.zeros_like(ecg_raw)
    for i in range(ecg_raw.shape[1]):
        ecg_filtered[:, i] = nk.ecg_clean(ecg_raw[:, i], sampling_rate=fs)
    return ecg_filtered

def segment_signal(signal, annotation, seg_len=2500):
    # 切片分段并给标签
    rhythm_labels = np.empty(len(signal), dtype=object)
    for i in range(len(annotation.aux_note)-1):
        start = annotation.sample[i]
        rhythm = annotation.aux_note[i].replace('(', '')
        end = annotation.sample[i + 1]
        rhythm_labels[start:end] = rhythm
    first_rhythm = annotation.aux_note[0].replace('(', '')
    rhythm_labels[0:annotation.sample[0]] = first_rhythm
    last_rhythm = annotation.aux_note[-1].replace('(', '')
    rhythm_labels[annotation.sample[-1]:] = last_rhythm

    # 去除首段
    new_signal = signal[annotation.sample[0]:]
    new_rhythm_labels = rhythm_labels[annotation.sample[0]:]

    segments, seg_labels = [], []
    num_segments = len(new_signal) // seg_len
    for i in tqdm(range(num_segments)):
        start = i * seg_len
        end = start + seg_len
        segment = new_signal[start:end, :]
        label_segment = new_rhythm_labels[start:end]
        try:
            # 使用 SQI 检测信号质量
            # 使用 MLII 导联（通常是第 0 列）
            _, info = nk.ecg_process(segment[:, 0], sampling_rate=fs)

            quality = nk.ecg_quality(segment[:, 0],
                                     rpeaks=info["ECG_R_Peaks"],
                                     sampling_rate=fs,
                                     method="zhao2018")  # 返回 (``Unacceptable``, ``Barely acceptable`` or ``Excellent``)

            if quality == 'Unacceptable':
                continue  # 低质量，跳过
        except Exception as e:
            continue  # 安静跳过

        # 取该片段的主要标签作为整体标签
        # unique, counts = np.unique(label_segment, return_counts=True)
        # major_label = unique[np.argmax(counts)]

        afib_ratio = np.sum(label_segment == 'AFIB') / len(label_segment)
        n_ratio = np.sum(label_segment == 'N') / len(label_segment)
        # 取该片段房颤出现则为房颤
        if afib_ratio == 1.0:
            major_label = 'AFIB'
            segments.append(segment)
            seg_labels.append(major_label)
        elif n_ratio == 1.0:
            major_label = 'N'
            segments.append(segment)
            seg_labels.append(major_label)
        else:
            continue

    return np.array(segments), np.array(seg_labels)

if __name__ == "__main__":
    path = "C:/Users/Administrator/PythonProjects/dataset/MIT-BIH-AF/files"
    npz_path = "C:/Users/Administrator/PythonProjects/dataset/MIT-BIH-AF/data"
    file_names = get_files(path)
    print(f'所有样例共：{len(file_names)}')
    print(f'这些文件分别是：{file_names}')
    total_segments = 0
    label_counts = Counter()
    for i, file_name in enumerate(file_names):
        base_path = os.path.join(path,file_name)
        print(f"{i}：处理{base_path}中")
        ecg_raw, annotation, fs = read_data(base_path)
        ecg_filtered = ecg_denoise(ecg_raw, fs)
        seg_len = fs * 10
        segments, seg_labels = segment_signal(ecg_filtered, annotation, seg_len)
        output_path = os.path.join(npz_path,file_name+'.npz')
        print(f"写入{output_path}中")
        np.savez(output_path, segments=segments, labels=seg_labels)
        total_segments += len(segments)
        sub_label_counts = Counter(seg_labels)
        label_counts.update(seg_labels)
        print(f"分段数：{len(segments)}")
        print(f"标签数：{sub_label_counts}")
    print(f"总分段数：{total_segments}")
    print(f"标签数：{label_counts}")