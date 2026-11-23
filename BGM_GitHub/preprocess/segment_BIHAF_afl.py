import argparse
import wfdb
import numpy as np
import neurokit2 as nk
import os
from collections import Counter
from tqdm import tqdm

# Same pipeline as segment_BIHAF.py (ecg_clean on the whole record, 10 s grid starting at the first
# annotation, zhao2018 quality on channel 0, drop only 'Unacceptable'), but keeps the windows that are
# 100% atrial flutter. They go to their own folder, so the existing N/AF files, splits and results are untouched.
#
# Run with ~/venvs/nk0212/bin/python (neurokit2 0.2.12 + numpy 1.26): other versions keep different windows.
#   python segment_BIHAF_afl.py --check 04043 07910   # re-make existing N/AF windows and compare
#   python segment_BIHAF_afl.py                       # write afdb_dataset/data_afl/<record>.npz

BGM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(os.path.dirname(BGM), 'datasets', 'MIT-BIH-AFDB')
DATA_DIR = os.path.join(BGM, 'afdb_dataset', 'data')        # existing N/AF windows (used by --check)
OUT_DIR = os.path.join(BGM, 'afdb_dataset', 'data_afl')

# In these records the existing .npz files have the leads swapped (channel 0 = ECG2), so the author's quality
# check and the model's input both used ECG2. Swapping before segmenting reproduces that for the flutter windows.
# Set to False to use ECG1 as channel 0 in every record instead.
MATCH_AUTHOR_LEADS = True
SWAPPED_LEADS = {'04746', '04908', '05091', '07859', '07879', '07910', '08215', '08219', '08434'}


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

    keep=('AFIB', 'N') gives exactly the windows of the original segment_BIHAF.py.
    Returns segments [n, seg_len, 2], labels [n], starts [n] (start sample in the raw record).
    """
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

def process(name, keep):
    ecg_raw, annotation, fs = read_data(os.path.join(RAW_DIR, name))
    if MATCH_AUTHOR_LEADS and name in SWAPPED_LEADS:
        ecg_raw = np.ascontiguousarray(ecg_raw[:, ::-1])
    ecg_filtered = ecg_denoise(ecg_raw, fs)
    return segment_signal(ecg_filtered, annotation, fs * 10, fs, keep)

def check(name):
    """Re-make the N/AF windows of one record and compare them with the existing .npz."""
    segments, labels, _ = process(name, keep=('AFIB', 'N'))
    old = np.load(os.path.join(DATA_DIR, name + '.npz'))
    same_windows = segments.shape == old['segments'].shape and np.array_equal(labels, old['labels'])
    diff = np.abs(segments - old['segments']).max() if same_windows and len(segments) else 0.0
    # The author's files differ from a re-run by ~1e-13 (floating-point rounding of a different scipy/numpy
    # build), so 'MATCH' allows 1e-9. A different window count means some quality decisions differ.
    ok = same_windows and diff < 1e-9
    print(f"{name}: remade {dict(Counter(labels))}, existing {dict(Counter(old['labels']))} -> "
          f"{'MATCH' if ok else 'DIFFERENT'}" + (f' (max |diff| {diff:.0e})' if same_windows else ''))
    return ok

def check_environment():
    if nk.__version__ != '0.2.12' or not np.__version__.startswith('1.'):
        raise SystemExit(f'Run this with ~/venvs/nk0212/bin/python: found neurokit2 {nk.__version__}, numpy {np.__version__} '
                         f'(the author\'s windows only reproduce with neurokit2 0.2.12 and numpy < 2)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cut 100% atrial flutter windows from MIT-BIH AFDB')
    parser.add_argument('--check', nargs='+', metavar='RECORD',
                        help='only re-make the N/AF windows of these records and compare with the existing .npz')
    args = parser.parse_args()
    check_environment()

    if args.check:
        results = [check(name) for name in args.check]
        raise SystemExit(0 if all(results) else 1)

    os.makedirs(OUT_DIR, exist_ok=True)
    file_names = sorted(f for f in get_files(RAW_DIR)
                        if '(AFL' in wfdb.rdann(os.path.join(RAW_DIR, f), 'atr').aux_note)
    print(f'Records with flutter: {len(file_names)} {file_names}')
    total_segments = 0
    label_counts = Counter()
    for i, file_name in enumerate(file_names):
        print(f"{i}: {file_name}" + (' (leads swapped to match the existing files)'
                                     if MATCH_AUTHOR_LEADS and file_name in SWAPPED_LEADS else ''))
        segments, seg_labels, starts = process(file_name, keep=('AFL',))
        output_path = os.path.join(OUT_DIR, file_name + '.npz')
        np.savez(output_path, segments=segments, labels=seg_labels, starts=starts)
        total_segments += len(segments)
        label_counts.update(seg_labels)
        print(f"  {len(segments)} windows -> {output_path}")
    print(f"Total windows: {total_segments} {dict(label_counts)}")
    if total_segments == 0:
        print('WARNING: no windows kept. Check the environment: a broken neurokit2 drops every window silently.')
