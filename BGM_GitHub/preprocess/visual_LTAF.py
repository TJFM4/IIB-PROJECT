from collections import Counter
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import os

# =========================
# Step 1. 读取原始数据
# =========================
base_path = 'F:\\LTAF\\files\\13'

record = wfdb.rdrecord(base_path)
annotation = wfdb.rdann(base_path, 'atr')

fs = record.fs  # 采样率
# 信号 shape: (样本点数, 通道数)
ecg_raw = record.p_signal
channels = record.sig_name
total_len = len(ecg_raw)

# print("采样率：", fs)
# print("信号shape:", ecg_raw.shape)
# print("通道:", channels)
# print("总采样点数:", total_len)
#
# print("总标注数量：", len(annotation.sample))
# print("标注采样点：", annotation.sample[:10])
# print("总标注符号：", len(annotation.symbol))
#
# print("标注符号（symbol）：", annotation.symbol[:10])  # 标准心搏类型
# # non_empty_values = [x for x in annotation.symbol if x]
# # print(non_empty_values)
#
# print("总标注注释：", len(annotation.aux_note))
# print("辅助注释（aux_note）：", annotation.aux_note[:10])  # 额外说明，如 [AFIB]
# # non_empty_values = [x for x in annotation.aux_note if x]
# # print(non_empty_values)
#
# unique_notes = set(annotation.symbol)
# print("所有标注符号种类：", unique_notes, Counter(annotation.symbol))
# unique_notes = set(annotation.aux_note)
# print("所有辅助注释种类：", unique_notes, Counter(annotation.aux_note))


# =========================
# Step 2. 滤波（去噪）
# =========================
ecg_filtered = np.zeros_like(ecg_raw)
for i in range(ecg_raw.shape[1]):
    ecg_filtered[:, i] = nk.ecg_clean(ecg_raw[:, i], sampling_rate=fs)


# =========================
# Step 3. 归一化
# =========================
# def z_score_normalize(signal):
#     return (signal - np.mean(signal)) / np.std(signal)
#
#
# ecg_norm = np.zeros_like(ecg_filtered)
# for i in range(ecg_filtered.shape[1]):
#     ecg_norm[:, i] = z_score_normalize(ecg_filtered[:, i])

# 不做归一化更适合
ecg_norm = ecg_filtered

# =========================
# Step 4. 10s 切片（每片2500点）
# =========================
segment_length = 10 * fs  # 10秒=2500点

def segment_signal(signal, annotation, seg_len=2500):
    """
    根据节律标注进行切片，并为每段分配标签
    """
    # 扩展节律标注到逐点标签
    temp_aux_note = annotation.aux_note
    aux_note = []
    aux_note.append(temp_aux_note[0])
    for i in range(1,len(temp_aux_note)):
        if temp_aux_note[i] == '':
            aux_note.append(aux_note[i-1])
        else:
            aux_note.append(temp_aux_note[i])

    rhythm_labels = np.empty(len(signal), dtype=object)
    for i in range(len(aux_note)-1):
        start = annotation.sample[i]
        rhythm = aux_note[i].replace('(', '')
        end = annotation.sample[i + 1]
        rhythm_labels[start:end] = rhythm
    first_rhythm = aux_note[0].replace('(', '')
    rhythm_labels[0:annotation.sample[0]] = first_rhythm
    last_rhythm = aux_note[-1].replace('(', '')
    rhythm_labels[annotation.sample[-1]:] = last_rhythm

    segments, seg_labels = [], []
    num_segments = len(signal) // seg_len
    for i in range(num_segments):
        start = i * seg_len
        end = start + seg_len
        segment = signal[start:end, :]
        label_segment = rhythm_labels[start:end]

        # 取该片段的**主要标签**作为整体标签
        # unique, counts = np.unique(label_segment, return_counts=True)
        # major_label = unique[np.argmax(counts)]
        if 'AFIB' in label_segment:
            major_label = 'AFIB'
        else:
            unique, counts = np.unique(label_segment, return_counts=True)
            major_label = unique[np.argmax(counts)]
        segments.append(segment)
        seg_labels.append(major_label)
    return np.array(segments), np.array(seg_labels)


segments, seg_labels = segment_signal(ecg_norm, annotation, segment_length)
print("切片后数据 shape:", segments.shape)  # (num_segments, 2500, 通道数)
print("切片后标签 shape:", seg_labels.shape)
print("示例标签分布:", dict(zip(*np.unique(seg_labels, return_counts=True))))

# =========================
# Step 5. 可视化
# =========================

# 5.1 滤波前 vs 滤波后
def plot_filter_comparison(raw_signal, filtered_signal, fs, start=0, duration=5, channel_idx=0):
    """
    可视化滤波前后对比
    """
    end = start + duration * fs
    t = np.arange(start, end) / fs

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, raw_signal[start:end], color='blue')
    plt.title(f"Raw ECG - {channels[channel_idx]}")
    plt.ylabel("mV")

    plt.subplot(2, 1, 2)
    plt.plot(t, filtered_signal[start:end], color='green')
    plt.title(f"Filtered ECG - {channels[channel_idx]}{channel_idx}")
    plt.xlabel("Time (s)")
    plt.ylabel("mV")
    plt.savefig("./visual/ltaf/"+str(channels[channel_idx])+str(channel_idx)+"_RawFiltered.png")

    plt.tight_layout()
    plt.show()


# 展示0~5秒的滤波效果
plot_filter_comparison(ecg_raw[:, 0], ecg_filtered[:, 0], fs, start=0, duration=5, channel_idx=0)
plot_filter_comparison(ecg_raw[:, 1], ecg_filtered[:, 1], fs, start=0, duration=5, channel_idx=1)


## 5.2 不同标签的片段可视化（两个通道分子图）
def plot_segment_by_label(segments, seg_labels, target_label, fs):
    """
    从切好的片段中找到指定标签，绘制其两个通道信号
    """
    idx = np.where(seg_labels == target_label)[0]
    if len(idx) == 0:
        print(f"没有找到标签 {target_label} 的片段")
        return

    seg = segments[idx[0]]  # 取第一个
    t = np.arange(seg.shape[0]) / fs

    plt.figure(figsize=(12, 6))

    # 通道1
    plt.subplot(2, 1, 1)
    plt.plot(t, seg[:, 0], color='purple')
    plt.title(f"Label: {target_label} - Channel: {channels[0]}")
    plt.ylabel("Normalized Amplitude")

    # 通道2
    plt.subplot(2, 1, 2)
    plt.plot(t, seg[:, 1], color='orange')
    plt.title(f"Label: {target_label} - Channel: {channels[1]}")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Amplitude")
    plt.savefig("./visual/ltaf/" + str(target_label) + ".png")

    plt.tight_layout()
    plt.show()


# 示例：可视化AFIB和N
plot_segment_by_label(segments, seg_labels, "AFIB", fs)
plot_segment_by_label(segments, seg_labels, "N", fs)
