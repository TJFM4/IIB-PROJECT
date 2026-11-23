import argparse
import wfdb
import numpy as np
import neurokit2 as nk
import os
from collections import Counter
from tqdm import tqdm

# Same pipeline as segment_CPSC2021.py (ecg_clean on the whole file, 10 s grid at the native 200 Hz starting
# at the first annotation, zhao2018 quality on channel 0, drop only 'Unacceptable'), but keeps the windows
# that are 100% atrial flutter. They go to their own folder, so the existing N/AF files, splits and results
# are untouched. Resampling to 250 Hz happens later in ECGAugment, as for the existing CPSC data.
#
# Run with ~/venvs/nk0212/bin/python (neurokit2 0.2.12 + numpy 1.26): other versions keep different windows.
#   python segment_CPSC2021_afl.py --check data_25_13   # re-make existing N/AF windows and compare
#   python segment_CPSC2021_afl.py                      # write cpsc2021_dataset/data_afl/<file>.npz

BGM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_ROOT = os.path.join(os.path.dirname(BGM), 'datasets', 'CPSC2021')
RAW_DIRS = [os.path.join(RAW_ROOT, 'Training_set_I'), os.path.join(RAW_ROOT, 'Training_set_II')]
DATA_DIR = os.path.join(BGM, 'cpsc2021_dataset', 'data')    # existing N/AF windows (used by --check)
OUT_DIR = os.path.join(BGM, 'cpsc2021_dataset', 'data_afl')


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

def segment_signal(signal, annotation, seg_len, fs, keep=('AFL',)):
    """Windows that are 100% one of the rhythms in `keep`, after the quality check.

    keep=('AFIB', 'N') gives exactly the windows of the original segment_CPSC2021.py.
    Returns segments [n, seg_len, 2], labels [n], starts [n] (start sample in the raw file, at 200 Hz).
    """
    # 扩展节律标注到逐点标签
    temp_aux_note = annotation.aux_note
    aux_note = []
    first_aux = temp_aux_note[0]
    if first_aux != '':
        aux_note.append(first_aux)
    else:
        aux_note.append(annotation.symbol[0])
    for i in range(1, len(temp_aux_note)):
        if temp_aux_note[i] == '' or temp_aux_note[i] == 'None':
            aux_note.append(aux_note[i - 1])
        else:
            aux_note.append(temp_aux_note[i])
    # 切片分段并给标签
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

    # 去除首段
    offset = annotation.sample[0]
    new_signal = signal[offset:]
    new_rhythm_labels = rhythm_labels[offset:]

    segments, seg_labels, starts = [], [], []
    num_segments = len(new_signal) // seg_len
    for i in tqdm(range(num_segments)):
        start = i * seg_len
        end = start + seg_len
        label_segment = new_rhythm_labels[start:end]

        # Label check first (the original checks quality first). Each window's quality is judged on its own,
        # so the kept windows are identical, but ecg_process only runs on windows that could be kept.
        ratios = {label: np.sum(label_segment == label) / len(label_segment) for label in keep}
        major_label = next((label for label, ratio in ratios.items() if ratio == 1.0), None)
        if major_label is None:
            continue

        segment = new_signal[start:end, :]
        try:
            # 使用 SQI 检测信号质量（第 0 列）
            _, info = nk.ecg_process(segment[:, 0], sampling_rate=fs)

            quality = nk.ecg_quality(segment[:, 0],
                                     rpeaks=info["ECG_R_Peaks"],
                                     sampling_rate=fs,
                                     method="zhao2018")  # 返回 (``Unacceptable``, ``Barely acceptable`` or ``Excellent``)

            if quality == 'Unacceptable':
                continue  # 低质量，跳过
        except Exception as e:
            continue  # 安静跳过 (a broken neurokit2 install lands here for every window)

        segments.append(segment)
        seg_labels.append(major_label)
        starts.append(offset + start)

    return np.array(segments), np.array(seg_labels), np.array(starts, dtype=np.int64)

def raw_path(name):
    for d in RAW_DIRS:
        if os.path.exists(os.path.join(d, name + '.dat')):
            return os.path.join(d, name)
    raise FileNotFoundError(f'{name}.dat is not in {RAW_DIRS}')

def process(name, keep):
    ecg_raw, annotation, fs = read_data(raw_path(name))
    ecg_filtered = ecg_denoise(ecg_raw, fs)
    return segment_signal(ecg_filtered, annotation, fs * 10, fs, keep)

def check(name):
    """Re-make the N/AF windows of one file and compare them with the existing .npz."""
    segments, labels, _ = process(name, keep=('AFIB', 'N'))
    old = np.load(os.path.join(DATA_DIR, name + '.npz'))
    same_windows = segments.shape == old['segments'].shape and np.array_equal(labels, old['labels'])
    diff = np.abs(segments - old['segments']).max() if same_windows and len(segments) else 0.0
    # 'MATCH' allows 1e-9 for floating-point rounding between scipy/numpy builds (the CPSC files were
    # regenerated in this environment, so they normally match exactly). A different count means some
    # quality decisions differ.
    ok = same_windows and diff < 1e-9
    print(f"{name}: remade {dict(Counter(labels))}, existing {dict(Counter(old['labels']))} -> "
          f"{'MATCH' if ok else 'DIFFERENT'}" + (f' (max |diff| {diff:.0e})' if same_windows else ''))
    return ok

def check_environment():
    if nk.__version__ != '0.2.12' or not np.__version__.startswith('1.'):
        raise SystemExit(f'Run this with ~/venvs/nk0212/bin/python: found neurokit2 {nk.__version__}, numpy {np.__version__} '
                         f'(the author\'s windows only reproduce with neurokit2 0.2.12 and numpy < 2)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cut 100% atrial flutter windows from CPSC2021')
    parser.add_argument('--check', nargs='+', metavar='FILE',
                        help='only re-make the N/AF windows of these files (e.g. data_25_13) and compare with the existing .npz')
    args = parser.parse_args()
    check_environment()

    if args.check:
        results = [check(name) for name in args.check]
        raise SystemExit(0 if all(results) else 1)

    os.makedirs(OUT_DIR, exist_ok=True)
    file_names = sorted(f for d in RAW_DIRS for f in get_files(d)
                        if '(AFL' in wfdb.rdann(os.path.join(d, f), 'atr').aux_note)
    print(f'Files with flutter: {len(file_names)} {file_names}')
    total_segments = 0
    label_counts = Counter()
    for i, file_name in enumerate(file_names):
        print(f"{i}: {file_name}")
        segments, seg_labels, starts = process(file_name, keep=('AFL',))
        output_path = os.path.join(OUT_DIR, file_name + '.npz')
        np.savez(output_path, segments=segments, labels=seg_labels, starts=starts)
        total_segments += len(segments)
        label_counts.update(seg_labels)
        print(f"  {len(segments)} windows -> {output_path}")
    print(f"Total windows: {total_segments} {dict(label_counts)}")
    if total_segments == 0:
        print('WARNING: no windows kept. Check the environment: a broken neurokit2 drops every window silently.')
