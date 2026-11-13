import wfdb
import numpy as np
import neurokit2 as nk
import argparse
import os
import glob
from multiprocessing import Pool
from collections import Counter
from tqdm import tqdm

# Convert raw Icentia11k WFDB records into one .npz file (10 s windows) for each patient. 


def get_patients(root): # Patients are nested as root/pXX/pXXXXX/
    return sorted(glob.glob(os.path.join(root, 'p[0-9][0-9]', 'p[0-9]*')))

def get_segments(patient_dir): # Each patient has up to 50 ~70 min segments: pXXXXX_s00 ... pXXXXX_s49
    return sorted(os.path.splitext(f)[0] for f in glob.glob(os.path.join(patient_dir, '*.hea')))

def read_data(base_path):
    record = wfdb.rdrecord(base_path)
    annotation = wfdb.rdann(base_path, 'atr')
    ecg_raw = record.p_signal
    fs = record.fs
    return ecg_raw, annotation, fs

def ecg_denoise(ecg_raw, fs):
    ecg_filtered = np.zeros_like(ecg_raw)
    for i in range(ecg_raw.shape[1]):
        ecg_filtered[:, i] = nk.ecg_clean(ecg_raw[:, i], sampling_rate=fs)
    return ecg_filtered

def build_rhythm_labels(annotation, sig_len):
    rhythm_labels = np.full(sig_len, '', dtype='<U4')
    open_label, open_start = None, None
    for sample, symbol, note in zip(annotation.sample, annotation.symbol, annotation.aux_note):
        if symbol != '+':
            continue
        if note.startswith('(') and not note.endswith(')'):
            open_label, open_start = note[1:], sample
        elif note == ')' and open_label is not None: 
            rhythm_labels[open_start:sample] = open_label
            open_label = None
    if open_label is not None: 
        rhythm_labels[open_start:] = open_label
    return rhythm_labels

def quality_ok(window, fs): # Drop only the "Unacceptable" segments. 
    try: 
        _, info = nk.ecg_process(window[:, 0], sampling_rate=fs)
        quality = nk.ecg_quality(window[:, 0], rpeaks=info["ECG_R_Peaks"],
                                 sampling_rate=fs, method="zhao2018")
    except Exception:
        return False
    return quality != 'Unacceptable'

def segment_signal(signal, rhythm_labels, fs, seg_len, n_frac, rng):
    starts, labels = [], []
    for start in range(0, len(signal) - seg_len + 1, seg_len):
        label_segment = rhythm_labels[start: start + seg_len]
        if np.all(label_segment == 'AFIB'):
            starts.append(start)
            labels.append('AFIB')
        elif np.all(label_segment == 'N'):
            starts.append(start)
            labels.append('N')
    
    starts = np.array(starts, dtype = np.int64)
    labels = np.array(labels, dtype = '<U4')

    keep = (labels == 'AFIB') | (rng.random(len(labels)) < n_frac)
    starts, labels = starts[keep], labels[keep]

    good = np.array([quality_ok(signal[s:s + seg_len], fs) for s in starts], dtype = bool)
    starts, labels = starts[good], labels[good]
    segments = np.stack([signal[s:s + seg_len] for s in starts]) if len(starts) else np.empty((0, seg_len, signal.shape[1]))
    return segments.astype(np.float32), labels, starts

def process_patient(job):
    patient_dir, out_dir, n_frac, seed = job
    pid = os.path.basename(patient_dir)
    out_path = os.path.join(out_dir, pid + '.npz')

    if os.path.exists(out_path):
        return pid, None

    rng = np.random.default_rng([seed, int(pid[1:])])
    all_X, all_y, all_seg, all_start = [], [], [], []

    segments = get_segments(patient_dir)
    if not segments:
        raise FileNotFoundError(f'No segment files found in {patient_dir}')
    for base_path in segments:
        if not os.path.exists(base_path + '.atr'):
            continue
        ecg_raw, annotation, fs = read_data(base_path)
        rhythm_labels = build_rhythm_labels(annotation, len(ecg_raw))
        ecg_filtered = ecg_denoise(ecg_raw, fs)
        X, y, starts = segment_signal(ecg_filtered, rhythm_labels, fs, fs*10, n_frac, rng)
        all_X.append(X)
        all_y.append(y)
        all_seg.append(np.full(len(y), int(base_path[-2:]), dtype = np.int16))
        all_start.append(starts)

    labels = np.concatenate(all_y) if all_y else np.array([])
    if len(labels) == 0:
        return pid, Counter()
    np.savez(out_path, 
             segments = np.concatenate(all_X), 
             labels = labels, 
             seg_idx = np.concatenate(all_seg),
             start_sample = np.concatenate(all_start))
    return pid, Counter(labels.tolist())
                       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Extracted Icentia11k folder containing p00/, p01/, ...')
    parser.add_argument('--out', required=True, help='Folder for the per-patient .npz files')
    parser.add_argument('--n_frac', type=float, default=0.05, help='Fraction of pure-N windows to keep')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    patients = get_patients(args.root)
    print(f'Patients found: {len(patients)}')
    jobs = [(p, args.out, args.n_frac, args.seed) for p in patients]

    label_counts = Counter()
    skipped = 0
    with Pool(args.workers) as pool:
        for pid, counts in tqdm(pool.imap_unordered(process_patient, jobs), total=len(jobs)):
            if counts is None:
                skipped += 1
            else:
                label_counts.update(counts)
    print(f'Already done (skipped): {skipped}')
    print(f'Windows kept this run: {dict(label_counts)}')