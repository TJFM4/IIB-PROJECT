import sys
import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

PROJECT = Path('/Users/tristanmartin/Desktop/IIB PROJECT')

# Reuse the exact preprocessing and model from training/testing, so the plots show what the model saw
# (trials/data and trials/model are identical to the BGM_GitHub copies)
sys.path.insert(0, str(PROJECT / 'trials'))
from data.dataset import AFDataset      # noqa: E402
from data.ecg_augment import ECGAugment  # noqa: E402
from model.bgm import GMCNet             # noqa: E402

FS = 250
CLASSES = {0: 'N', 1: 'AF'}
CAM_CMAP = 'jet'    # as in the paper: red = high attention, blue = low

# Best AFDB checkpoint per fold (BGM_GitHub/best), chosen on AFDB test results
BEST_AFDB = {
    'acc':  {1: 'bgm_afdb_1_afdb_only', 2: 'bgm_afdb_2_afdb_only', 3: 'bgm_afdb_3_afdb_only',
             4: 'bgm_afdb_4_authcopy_seed2', 5: 'bgm_afdb_5'},
    'spec': {1: 'bgm_afdb_1_afdb_only', 2: 'bgm_afdb_2_afdb_only', 3: 'bgm_afdb_3_authcopy',
             4: 'bgm_afdb_4_authcopy', 5: 'bgm_afdb_5_authcopy_oldcpsc'},
}

_AFDB_SPLIT = PROJECT / 'BGM_GitHub/preprocess/data_split/bihaf'
_AFDB_DATA = PROJECT / 'BGM_GitHub/afdb_dataset/data'
_CPSC_ZS_SPLIT = PROJECT / 'BGM_GitHub/preprocess/data_split/cpsc2021_zeroshot'
_CPSC_DATA = PROJECT / 'BGM_GitHub/cpsc2021_dataset/data'
_CPSC_ZS_RESULTS = PROJECT / 'BGM_GitHub/new_runs/results/afdb_zeroshot_cpsc'
_ICENTIA_ZS_SPLIT = PROJECT / 'trials/preprocess/data_split/icentia_zeroshot'
_ICENTIA_DATA = Path('/Volumes/seagate_hard/icentia_100_npz')       # needs the Seagate mounted
_ICENTIA_ZS_RESULTS = PROJECT / 'trials/icentia_runs/results/zeroshot_pilot100'

# split_dir / data_dir: split jsons and npz files.
# results_dir / checkpoint_dir: hold <run>/test_predictions.csv and <run>.pth.
# runs: {fold: run name}; the results folder and checkpoint share this name. Override with run=... in the functions.
# split_fold: fold_k.json to read (None = same as the fold argument). Zero-shot sets have one split
#             (fold_1 = every patient), so there 'fold' only picks the AFDB checkpoint.
# original_fs: sampling rate of the npz segments (CPSC2021 is 200 Hz; test.py resamples it to 250 Hz).
DATASETS = {
    # AFDB five-fold, tested on each fold's held-out records
    'afdb_acc': {
        'split_dir': _AFDB_SPLIT, 'data_dir': _AFDB_DATA, 'split_fold': None, 'original_fs': 250,
        'results_dir':    PROJECT / 'BGM_GitHub/best/afdb_acc',
        'checkpoint_dir': PROJECT / 'BGM_GitHub/best/afdb_acc',
        'runs': BEST_AFDB['acc'],
    },
    'afdb_spec': {
        'split_dir': _AFDB_SPLIT, 'data_dir': _AFDB_DATA, 'split_fold': None, 'original_fs': 250,
        'results_dir':    PROJECT / 'BGM_GitHub/best/afdb_spec',
        'checkpoint_dir': PROJECT / 'BGM_GitHub/best/afdb_spec',
        'runs': BEST_AFDB['spec'],
    },
    # Zero-shot AFDB -> CPSC2021 (all 105 cases). Patients are case IDs as strings ('0' ... '104'),
    # each spread over several npz files listed in case_to_files.json.
    'afdb_zeroshot_cpsc_acc': {
        'split_dir': _CPSC_ZS_SPLIT, 'data_dir': _CPSC_DATA, 'split_fold': 1, 'original_fs': 200,
        'results_dir':    _CPSC_ZS_RESULTS,
        'checkpoint_dir': PROJECT / 'BGM_GitHub/best/afdb_acc',
        'runs': BEST_AFDB['acc'],
    },
    'afdb_zeroshot_cpsc_spec': {
        'split_dir': _CPSC_ZS_SPLIT, 'data_dir': _CPSC_DATA, 'split_fold': 1, 'original_fs': 200,
        'results_dir':    _CPSC_ZS_RESULTS,
        'checkpoint_dir': PROJECT / 'BGM_GitHub/best/afdb_spec',
        'runs': BEST_AFDB['spec'],
    },
    # CPSC2021 five-fold, trained on CPSC (validated on AFDB), tested on each fold's held-out cases.
    # Patients are case IDs as strings, e.g. '100' (fold 4).
    'cpsc': {
        'split_dir': PROJECT / 'BGM_GitHub/preprocess/data_split/cpsc2021', 'data_dir': _CPSC_DATA,
        'split_fold': None, 'original_fs': 200,
        'results_dir':    PROJECT / 'BGM_GitHub/new_runs/results',
        'checkpoint_dir': PROJECT / 'BGM_GitHub/new_runs/checkpoint',
        'runs': {f: f'bgm_cpsc_{f}_authcopy' for f in range(1, 6)},
    },
    # Icentia11k 100-patient sample, trained and tested on it
    'icentia_100': {
        'split_dir': PROJECT / 'trials/preprocess/data_split/icentia_100', 'data_dir': _ICENTIA_DATA,
        'split_fold': None, 'original_fs': 250,
        'results_dir':    PROJECT / 'trials/icentia_runs/results/icentia_100',
        'checkpoint_dir': PROJECT / 'trials/icentia_runs/checkpoints/icentia_100',
        'runs': {f: f'bgm_icentia_100_{f}' for f in range(1, 6)},
    },
    # Zero-shot AFDB -> Icentia11k (all 97 patients). Only some best checkpoints have been tested here:
    # acc has results for folds 1-2, spec for folds 1-4 (zeroshot_pilot100 also has bgm_afdb_5_afdb_only).
    'icentia_zeroshot_acc': {
        'split_dir': _ICENTIA_ZS_SPLIT, 'data_dir': _ICENTIA_DATA, 'split_fold': 1, 'original_fs': 250,
        'results_dir':    _ICENTIA_ZS_RESULTS,
        'checkpoint_dir': PROJECT / 'BGM_GitHub/best/afdb_acc',
        'runs': BEST_AFDB['acc'],
    },
    'icentia_zeroshot_spec': {
        'split_dir': _ICENTIA_ZS_SPLIT, 'data_dir': _ICENTIA_DATA, 'split_fold': 1, 'original_fs': 250,
        'results_dir':    _ICENTIA_ZS_RESULTS,
        'checkpoint_dir': PROJECT / 'BGM_GitHub/best/afdb_spec',
        'runs': BEST_AFDB['spec'],
    },
}


# ---------------------------------------------------------------------------
# Data: one test patient's windows, lined up with test_predictions.csv
# ---------------------------------------------------------------------------

def _test_patients(cfg, fold):
    split_fold = cfg['split_fold'] or fold
    with open(cfg['split_dir'] / f'fold_{split_fold}.json') as f:
        return json.load(f)['val']


@lru_cache(maxsize=None)
def _case_to_files(split_dir):
    path = split_dir / 'case_to_files.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _patient_files(cfg, pid):
    """npz files holding one patient's windows, in the order AFDataset reads them."""
    case_to_files = _case_to_files(cfg['split_dir'])
    if case_to_files is not None:           # CPSC2021: a case is spread over several npz files
        return [cfg['data_dir'] / name for name in case_to_files[pid]]
    return [cfg['data_dir'] / f'{pid}.npz']


def _normaliser(cfg):
    """Resample to 250 Hz and z-score, exactly as test.py does (prob=0: no augmentation)."""
    return ECGAugment(target_fs=FS, original_fs=cfg['original_fs'], prob=0.0, norm_type='zscore')


def checkpoint_path(dataset, fold, run=None, checkpoint=None):
    """Path to the .pth that produced a run's test_predictions.csv.

    By default <checkpoint_dir>/<run>.pth, where run is DATASETS[dataset]['runs'][fold] (or the run argument).
    checkpoint: explicit path to use instead, e.g. a copy kept somewhere else.
    """
    cfg = DATASETS[dataset]
    if checkpoint is not None:
        path = Path(checkpoint)
    else:
        path = cfg['checkpoint_dir'] / f"{run or cfg['runs'][fold]}.pth"
    if not path.exists():
        raise FileNotFoundError(f'No checkpoint at {path}. Pass checkpoint=... to use a copy elsewhere.')
    return path


def load_patient(dataset, fold, patient, run=None):
    """Return (segments, table) for one test patient.

    segments are the raw npz windows (CPSC2021: 200 Hz, 2000 samples; resampled when normalised).
    table has one row per window: true label, prediction, P(AF), correct, plus
    seg_idx/start_sample when the npz has them (Icentia11k) and the source npz file (CPSC2021).
    """
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset '{dataset}'. Choose from: {', '.join(DATASETS)}")
    patient = str(patient)          # CPSC case IDs are strings in the splits ('100', not 100)
    cfg = DATASETS[dataset]
    run = run or cfg['runs'][fold]
    preds = pd.read_csv(cfg['results_dir'] / run / 'test_predictions.csv')
    patients = _test_patients(cfg, fold)
    if patient not in patients:
        raise ValueError(f'{patient} is not in the test set of {dataset} fold {fold}')

    # test.py doesn't shuffle: rows follow the val list, patient by patient and file by file.
    # AFDataset skips npz files with no windows; those have no labels, so they add 0 rows here too.
    offset = 0
    for pid in patients:
        if pid != patient:
            for path in _patient_files(cfg, pid):
                with np.load(path) as d:
                    offset += len(d['labels'])
            continue

        parts = []
        for path in _patient_files(cfg, pid):
            with np.load(path) as d:
                if len(d['labels']) == 0:
                    continue
                part = {k: d[k] for k in ('segments', 'labels', 'seg_idx', 'start_sample') if k in d}
                part['file'] = np.full(len(d['labels']), path.name)
                parts.append(part)
        break

    if not parts:
        raise ValueError(f'{patient} has no windows')
    segments = np.concatenate([p['segments'] for p in parts])
    labels = (np.concatenate([p['labels'] for p in parts]) == 'AFIB').astype(int)
    extra = {k: np.concatenate([p[k] for p in parts]) for k in ('seg_idx', 'start_sample') if k in parts[0]}
    if len(parts) > 1:
        extra['file'] = np.concatenate([p['file'] for p in parts])
    n = len(labels)

    rows = preds.iloc[offset:offset + n].reset_index(drop=True)
    if len(rows) != n or not np.array_equal(rows['label'].to_numpy(), labels):
        raise RuntimeError(f'Predictions in {run} do not line up with {patient}: '
                           'the npz files or split differ from the ones used by test.py')

    table = pd.DataFrame({
        'true': rows['label'].map(CLASSES),
        'pred': rows['prediction'].map(CLASSES),
        'p_AF': rows['prob_class_1'],
        'correct': rows['label'] == rows['prediction'],
        **extra,
    })
    return segments, table


def patient_summary(dataset, fold, patient, run=None):
    """Confusion counts for one patient: rows = true class, columns = predicted class."""
    _, table = load_patient(dataset, fold, patient, run)
    order = ['N', 'AF']
    cm = (pd.crosstab(table['true'], table['pred'])
          .reindex(index=order, columns=order, fill_value=0))
    cm.index = [f'True {c}' for c in order]
    cm.columns = [f'Predicted {c}' for c in order]
    cm['Total'] = cm.sum(axis=1)
    cm.loc['Total'] = cm.sum()
    cm['Correct'] = [cm.iloc[i, i] for i in range(2)] + [cm.iloc[0, 0] + cm.iloc[1, 1]]
    cm['Accuracy'] = (cm['Correct'] / cm['Total']).round(4)
    return cm


def _choose_windows(table, show, n, windows, seed, label=None):
    """Explicit window indices, or a seeded random sample of 'incorrect' / 'correct' / 'all' windows,
    optionally only those whose true class is label ('AF' or 'N')."""
    if windows is not None:
        return list(windows)
    keep = {'incorrect': ~table['correct'], 'correct': table['correct'], 'all': table['correct'] | True}[show]
    if label is not None:
        keep = keep & (table['true'] == label)
    pool = table.index[keep]
    rng = np.random.default_rng(seed)
    return list(np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False)))


def _window_title(i, row):
    """Title and colour for one window: green if correctly classified, red if not."""
    if 'seg_idx' in row:
        where = f" | seg {row['seg_idx']} @ {row['start_sample'] / FS / 60:.1f} min"
    elif 'file' in row:
        where = f" | {row['file']}"
    else:
        where = ''
    title = (f"window {i}{where} | true {row['true']} → pred {row['pred']} "
             f"(P(AF)={row['p_AF']:.3f}) {'✓' if row['correct'] else '✗'}")
    return title, 'tab:green' if row['correct'] else 'tab:red'


def plot_windows(dataset, fold, patient, show='incorrect', n=6, windows=None, run=None, seed=0, label=None):
    """Plot ECG windows next to the spectrograms the model received.

    show: 'incorrect', 'correct' or 'all' (ignored if windows is given)
    n: how many windows to draw (random sample, fixed by seed)
    windows: explicit list of window indices (the row numbers in load_patient's table)
    label: only windows whose true class is 'AF' or 'N' (None = both)
    """
    segments, table = load_patient(dataset, fold, patient, run)
    windows = _choose_windows(table, show, n, windows, seed, label)
    if not windows:
        print(f"No {show} {label or ''} windows for {patient}")
        return table.iloc[:0]
    normalise = _normaliser(DATASETS[dataset])

    fig, axes = plt.subplots(len(windows), 2, figsize=(14, 2.4 * len(windows)),
                             gridspec_kw={'width_ratios': [3, 1]}, squeeze=False)
    for (ax_ecg, ax_spec), i in zip(axes, windows):
        x = normalise(segments[i])                     # 250 Hz, z-scored, as in test.py
        img = AFDataset._generate_image(None, x)       # 128x128 RGB STFT, 0-40 Hz
        t = np.arange(len(x)) / FS
        title, colour = _window_title(i, table.loc[i])

        ax_ecg.plot(t, x[:, 0], lw=0.6, color='k')
        ax_ecg.set_xlim(0, t[-1])
        ax_ecg.set_ylabel('z-score')
        ax_ecg.set_title(title, color=colour, fontsize=10, loc='left')
        for spine in ax_ecg.spines.values():
            spine.set_edgecolor(colour)

        ax_spec.imshow(img, origin='lower', aspect='auto', extent=[0, t[-1], 0, 40])
        ax_spec.set_ylabel('Hz')

    axes[-1, 0].set_xlabel('time (s)')
    axes[-1, 1].set_xlabel('time (s)')
    n_wrong = (~table['correct']).sum()
    fig.suptitle(f'{dataset} fold {fold} | {patient}: {n_wrong}/{len(table)} windows misclassified',
                 fontsize=12)
    fig.tight_layout()
    plt.show()
    return table.loc[windows]


# ---------------------------------------------------------------------------
# Windows discarded by the quality check during preprocessing
# ---------------------------------------------------------------------------
# A window is a candidate when it lies entirely inside one rhythm (100% AFIB or 100% N) on the 10 s grid
# the segmentation scripts use. Discarded = candidates that are not in the npz files, found by comparing
# with the saved windows, so this is exact whatever neurokit2 version made the original decision.
# The grade shown is computed with the neurokit2 version that made the files (0.2.12 for AFDB/CPSC, via
# ~/venvs/nk0212), so it shows why each window was dropped.

# 9 AFDB records whose saved windows have the leads swapped (channel 0 = ECG2), as in segment_BIHAF_afl.py
_AFDB_SWAPPED = {'04746', '04908', '05091', '07859', '07879', '07910', '08215', '08219', '08434'}
_ICENTIA_RAW = Path('/Volumes/seagate_hard/icentia11k')     # raw WFDB root with p00/, p01/, ... (edit if elsewhere)

RAW_SOURCES = {
    'afdb':    {'raw': [PROJECT / 'datasets/MIT-BIH-AFDB'], 'npz': _AFDB_DATA},
    'cpsc':    {'raw': [PROJECT / 'datasets/CPSC2021/Training_set_I', PROJECT / 'datasets/CPSC2021/Training_set_II'],
                'npz': _CPSC_DATA},
    'icentia': {'raw': [_ICENTIA_RAW], 'npz': _ICENTIA_DATA},
}


def _raw_path(source, name):
    for d in RAW_SOURCES[source]['raw']:
        if (d / f'{name}.hea').exists():
            return d / name
    raise FileNotFoundError(f'{name}.hea not found in {RAW_SOURCES[source]["raw"]}')


def _rhythm_labels(source, ann, n):
    """Per-sample rhythm label, built exactly as the dataset's segmentation script does."""
    if source == 'icentia':                      # segment_icentia.build_rhythm_labels: '(X' opens, ')' closes
        labels = np.full(n, '', dtype='<U4')
        open_label, open_start = None, None
        for s, sym, note in zip(ann.sample, ann.symbol, ann.aux_note):
            if sym != '+':
                continue
            if note.startswith('(') and not note.endswith(')'):
                open_label, open_start = note[1:], s
            elif note == ')' and open_label is not None:
                labels[open_start:s] = open_label
                open_label = None
        if open_label is not None:
            labels[open_start:] = open_label
        return labels
    aux = list(ann.aux_note)
    if source == 'cpsc':                         # segment_CPSC2021: empty notes carry the previous rhythm forward
        aux = [aux[0] if aux[0] != '' else ann.symbol[0]]
        for note in ann.aux_note[1:]:
            aux.append(aux[-1] if note in ('', 'None') else note)
    labels = np.empty(n, dtype=object)
    for i in range(len(aux) - 1):
        labels[ann.sample[i]:ann.sample[i + 1]] = aux[i].replace('(', '')
    labels[:ann.sample[0]] = aux[0].replace('(', '')
    labels[ann.sample[-1]:] = aux[-1].replace('(', '')
    return labels


def _candidates(labels, fs, start0):
    """(start, label) of 10 s grid windows that are 100% AFIB or 100% N."""
    L = fs * 10
    out = []
    for s in range(start0, len(labels) - L + 1, L):
        w = labels[s:s + L]
        for lab in ('AFIB', 'N'):
            if np.all(w == lab):
                out.append((s, lab))
                break
    return out


@lru_cache(maxsize=2)
def _clean_record(path, swap):
    """Raw record filtered with nk.ecg_clean on every lead, as the segmentation scripts do."""
    import wfdb
    import neurokit2 as nk
    rec = wfdb.rdrecord(path)
    sig = rec.p_signal[:, ::-1] if swap else rec.p_signal
    clean = np.stack([nk.ecg_clean(np.ascontiguousarray(sig[:, c]), sampling_rate=rec.fs)
                      for c in range(sig.shape[1])], axis=1)
    return clean, rec.fs


def _key(window):
    return tuple(np.round(window[:50, 0], 6))


def _discarded_afdb_cpsc(source, files):
    import wfdb
    rows, windows = [], []
    for name in files:
        path = _raw_path(source, name)
        swap = source == 'afdb' and name in _AFDB_SWAPPED
        clean, fs = _clean_record(str(path), swap)
        ann = wfdb.rdann(str(path), 'atr')
        cands = _candidates(_rhythm_labels(source, ann, len(clean)), fs, ann.sample[0])
        with np.load(RAW_SOURCES[source]['npz'] / f'{name}.npz') as d:
            kept = {_key(w) for w in d['segments']} if d['segments'].ndim == 3 else set()
        found = 0
        for s, lab in cands:
            w = clean[s:s + fs * 10]
            if _key(w) in kept:
                found += 1
            else:
                rows.append({'file': name, 'start_sample': s, 'minute': s / fs / 60, 'true': lab, 'fs': fs})
                windows.append(w)
        if found != len(kept):
            warnings.warn(f'{name}: only {found} of {len(kept)} saved windows matched the re-made grid; '
                          'the raw files or preprocessing differ from the ones used to make the npz')
    return rows, windows


def _discarded_icentia(pid, n_frac=0.05, seed=0):
    """Replays segment_icentia's random N subsampling (default n_frac/seed of that script), then compares with
    the npz by (seg_idx, start_sample). Windows dropped by the subsampling are not counted as discarded."""
    import wfdb
    patient_dir = _ICENTIA_RAW / pid[:3] / pid
    if not patient_dir.exists():
        raise FileNotFoundError(f'{patient_dir} not found: mount the Seagate or set _ICENTIA_RAW')
    with np.load(_ICENTIA_DATA / f'{pid}.npz') as d:
        kept = set(zip(d['seg_idx'].tolist(), d['start_sample'].tolist()))
    rng = np.random.default_rng([seed, int(pid[1:])])
    rows, sampled_total = [], 0
    for hea in sorted(patient_dir.glob('*.hea')):
        base = hea.with_suffix('')
        if not base.with_suffix('.atr').exists():
            continue
        n = wfdb.rdheader(str(base)).sig_len
        ann = wfdb.rdann(str(base), 'atr')
        cands = _candidates(_rhythm_labels('icentia', ann, n), 250, 0)
        labs = np.array([lab for _, lab in cands])
        keep = (labs == 'AFIB') | (rng.random(len(labs)) < n_frac)      # same draws, same order as the script
        seg = int(base.name[-2:])
        for (s, lab), k in zip(cands, keep):
            if not k:
                continue
            sampled_total += 1
            if (seg, s) not in kept:
                rows.append({'file': base.name, 'seg_idx': seg, 'start_sample': s, 'minute': s / 15000,
                             'true': lab, 'fs': 250, '_base': str(base)})
    if sampled_total - len(rows) != len(kept):
        warnings.warn(f'{pid}: replayed sampling explains {sampled_total - len(rows)} of {len(kept)} saved windows; '
                      'n_frac/seed probably differ from the ones used (pass them to discarded_windows)')
    return rows


_NK0212_PYTHON = Path.home() / 'venvs/nk0212/bin/python'    # neurokit2 0.2.12, the version behind the AFDB/CPSC files

_GRADE_SCRIPT = r'''
import sys, warnings, numpy as np, neurokit2 as nk
warnings.simplefilter("ignore")
d = np.load(sys.argv[1])
out = []
for w, fs in zip(d["windows"], d["fs"]):
    try:
        _, info = nk.ecg_process(w, sampling_rate=int(fs))
        out.append(nk.ecg_quality(w, rpeaks=info["ECG_R_Peaks"], sampling_rate=int(fs), method="zhao2018"))
    except Exception as e:
        out.append("error: " + type(e).__name__)
print("\n".join(out))
'''


def _grades(windows, fs, python=None):
    """zhao2018 grade of channel 0 for each window (or the error that made the script drop it), computed by
    `python` (default: the interpreter running this code) so the neurokit2 version can match the original run."""
    import subprocess
    import tempfile
    import os
    python = str(python or sys.executable)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'w.npz')
        np.savez(path, windows=np.stack([w[:, 0] for w in windows]), fs=np.asarray(fs))
        # Jupyter sets MPLBACKEND to its inline backend, which the other environment doesn't have:
        # give the child process a plain non-interactive backend instead (it draws nothing).
        env = {**os.environ, 'MPLBACKEND': 'Agg'}
        env.pop('PYTHONPATH', None)
        res = subprocess.run([python, '-c', _GRADE_SCRIPT, path], capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f'Grading with {python} failed:\n{res.stderr[-500:]}')
    return res.stdout.strip().split('\n')


def discarded_windows(source, record, n=6, seed=0, label=None, n_frac=0.05, icentia_seed=0):
    """Some windows a record lost in preprocessing because of the quality check.

    source: 'afdb', 'cpsc' or 'icentia'.
    record: AFDB record ('04746'); CPSC file ('data_25_13') or case ('25', all its files); Icentia patient ('p00060').
    n, seed: how many discarded windows to return (random sample); label: 'AF' or 'N' only.
    n_frac, icentia_seed: the Icentia script's N subsampling settings (its defaults).
    Returns (summary, table, windows): candidate/kept/discarded counts per class, one row per returned window
    (with its zhao2018 grade, computed with the neurokit2 version used originally), and the filtered windows [n, 10*fs, leads].
    """
    if source == 'icentia':
        rows = _discarded_icentia(record, n_frac, icentia_seed)
        windows = None
        with np.load(_ICENTIA_DATA / f'{record}.npz') as d:
            kept_labels = d['labels']
    else:
        files = [record]
        if source == 'cpsc' and not record.startswith('data_'):
            files = [f.replace('.npz', '') for f in _case_to_files(PROJECT / 'BGM_GitHub/preprocess/data_split/cpsc2021')[record]]
        rows, windows = _discarded_afdb_cpsc(source, files)
        kept_labels = np.concatenate([np.load(RAW_SOURCES[source]['npz'] / f'{f}.npz')['labels'] for f in files])

    table = pd.DataFrame(rows)
    to_short = {'AFIB': 'AF', 'N': 'N'}
    kept_counts = pd.Series([to_short[str(l)] for l in kept_labels]).value_counts()
    disc_counts = table['true'].map(to_short).value_counts() if len(table) else pd.Series(dtype=int)
    summary = pd.DataFrame({'kept': kept_counts, 'discarded': disc_counts}).fillna(0).astype(int)
    summary['discarded %'] = (100 * summary['discarded'] / (summary['kept'] + summary['discarded'])).round(1)

    if table.empty:
        return summary, table, np.empty((0,))
    table['true'] = table['true'].map(to_short)
    pool = table.index if label is None else table.index[table['true'] == label]
    pick = np.sort(np.random.default_rng(seed).choice(pool, size=min(n, len(pool)), replace=False)) if len(pool) else []
    table = table.loc[pick]

    if source == 'icentia':
        chosen = [_clean_record(r['_base'], False)[0][r['start_sample']:r['start_sample'] + 2500]
                  for _, r in table.iterrows()]
        table = table.drop(columns='_base')
    else:
        chosen = [windows[i] for i in pick]
    # AFDB/CPSC were segmented with neurokit2 0.2.12, whose grading is stricter than later versions; Icentia
    # with the installed version. Grade with the matching one so the grade explains the original decision.
    grader = _NK0212_PYTHON if source in ('afdb', 'cpsc') and _NK0212_PYTHON.exists() else None
    if grader is None and source in ('afdb', 'cpsc'):
        warnings.warn(f'{_NK0212_PYTHON} not found: grading with the installed neurokit2, which may disagree')
    table['grade'] = _grades(chosen, table['fs'].tolist(), grader) if chosen else []
    return summary, table, np.stack(chosen) if chosen else np.empty((0,))


def plot_discarded(source, record, n=6, seed=0, label=None, **kwargs):
    """Plot windows discarded by the quality check, preprocessed like the kept ones (resampled to 250 Hz,
    z-scored) next to the spectrogram the model would have received. Shows the kept/discarded counts first.
    Arguments as in discarded_windows. Returns (summary, table)."""
    from IPython.display import display
    summary, table, windows = discarded_windows(source, record, n, seed, label, **kwargs)
    print(f'{source} {record}: windows kept vs discarded by the quality check')
    display(summary)
    if table.empty:
        print('No discarded windows' + (f' with true class {label}' if label else ''))
        return summary, table

    fig, axes = plt.subplots(len(table), 2, figsize=(14, 2.4 * len(table)),
                             gridspec_kw={'width_ratios': [3, 1]}, squeeze=False)
    for (ax_ecg, ax_spec), w, (_, row) in zip(axes, windows, table.iterrows()):
        normalise = ECGAugment(target_fs=FS, original_fs=int(row['fs']), prob=0.0, norm_type='zscore')
        x = normalise(w)
        img = AFDataset._generate_image(None, x)
        t = np.arange(len(x)) / FS
        where = f"{row['file']} @ {row['minute']:.1f} min"
        ax_ecg.plot(t, x[:, 0], lw=0.6, color='k')
        ax_ecg.set_xlim(0, t[-1])
        ax_ecg.set_ylabel('z-score')
        ax_ecg.set_title(f"{where} | true {row['true']} | discarded ({row['grade']})",
                         color='tab:red', fontsize=10, loc='left')
        for spine in ax_ecg.spines.values():
            spine.set_edgecolor('tab:red')
        ax_spec.imshow(img, origin='lower', aspect='auto', extent=[0, t[-1], 0, 40])
        ax_spec.set_ylabel('Hz')
    axes[-1, 0].set_xlabel('time (s)')
    axes[-1, 1].set_xlabel('time (s)')
    fig.suptitle(f'{source} {record}: windows removed by the quality check (lead used for the check)', fontsize=12)
    fig.tight_layout()
    plt.show()
    return summary, table


# ---------------------------------------------------------------------------
# Atrial flutter windows (made by preprocess/segment_BIHAF_afl.py and segment_CPSC2021_afl.py)
# ---------------------------------------------------------------------------
# Same pipeline as the AF/N windows (ecg_clean, 10 s grid, zhao2018 'Unacceptable' dropped), saved to
# <dataset>/data_afl/<record>.npz with segments, labels and starts (start sample at the raw rate).
# No model has seen these windows, so they have no predictions: the plots show the ECG and spectrogram only.

AFL_SOURCES = {
    'afdb': {'afl_dir': PROJECT / 'BGM_GitHub/afdb_dataset/data_afl', 'npz': _AFDB_DATA, 'fs': 250},
    'cpsc': {'afl_dir': PROJECT / 'BGM_GitHub/cpsc2021_dataset/data_afl', 'npz': _CPSC_DATA, 'fs': 200},
}
_LABEL_NAMES = {'AFL': 'AFL', 'AFIB': 'AF', 'N': 'N'}
_LABEL_COLOURS = {'AFL': 'tab:purple', 'AFIB': 'tab:red', 'N': 'tab:blue'}


def afl_summary(source):
    """AFL windows kept per record, next to the record's kept AF and N windows. Records whose flutter gave
    no window that passed the quality check are listed with 0."""
    cfg = AFL_SOURCES[source]
    if not cfg['afl_dir'].exists():
        raise FileNotFoundError(f"{cfg['afl_dir']} does not exist: run preprocess/segment_"
                                f"{'BIHAF' if source == 'afdb' else 'CPSC2021'}_afl.py with ~/venvs/nk0212")
    rows = []
    for f in sorted(cfg['afl_dir'].glob('*.npz')):
        n_afl = len(np.load(f)['labels'])
        other = cfg['npz'] / f.name
        labels = np.load(other)['labels'] if other.exists() else np.array([])
        rows.append({'record': f.stem, 'AFL': n_afl,
                     'AF': int((labels == 'AFIB').sum()), 'N': int((labels == 'N').sum())})
    table = pd.DataFrame(rows).set_index('record')
    table.loc['Total'] = table.sum()
    return table


def _afl_windows(source, record=None, n=6, seed=0):
    """(segments, table) for n random AFL windows of one record, or of all records if record is None."""
    cfg = AFL_SOURCES[source]
    files = [cfg['afl_dir'] / f'{record}.npz'] if record is not None else sorted(cfg['afl_dir'].glob('*.npz'))
    segs, rows = [], []
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f'{f} not found. Records with flutter: {list(afl_summary(source).index[:-1])}')
        d = np.load(f)
        for i, (s, start) in enumerate(zip(d['segments'], d['starts'])):
            segs.append(s)
            rows.append({'record': f.stem, 'index': i, 'true': 'AFL', 'minute': start / cfg['fs'] / 60})
    if not rows:
        return [], pd.DataFrame(columns=['record', 'index', 'true', 'minute'])
    pick = np.sort(np.random.default_rng(seed).choice(len(rows), size=min(n, len(rows)), replace=False))
    return [segs[i] for i in pick], pd.DataFrame([rows[i] for i in pick]).reset_index(drop=True)


def _same_record_windows(source, records, seed=0):
    """One kept AF and one kept N window from each record (for comparison), if the record has them.
    Their start time is not saved in the AF/N npz files, so minute is NaN."""
    cfg = AFL_SOURCES[source]
    rng = np.random.default_rng(seed)
    segs, rows = [], []
    for rec in records:
        f = cfg['npz'] / f'{rec}.npz'
        if not f.exists():
            continue
        d = np.load(f)
        for lab in ('AFIB', 'N'):
            idx = np.flatnonzero(d['labels'] == lab)
            if len(idx):
                i = int(rng.choice(idx))
                segs.append(d['segments'][i])
                rows.append({'record': rec, 'index': i, 'true': lab, 'minute': np.nan})
    return segs, pd.DataFrame(rows)


def plot_afl(source, record=None, n=6, seed=0, compare=False):
    """Plot atrial flutter windows next to the spectrogram the model would receive (resampled to 250 Hz,
    z-scored, 128x128 STFT 0-40 Hz, as in test.py).

    source: 'afdb' or 'cpsc'
    record: one record/file (e.g. '04936', 'data_25_13'); None = sample from every record
    n: number of AFL windows (random, fixed by seed)
    compare: also plot one kept AF and one kept N window from each record shown, for contrast
    Returns the table of plotted windows (index = row in that record's npz).
    """
    segs, table = _afl_windows(source, record, n, seed)
    if not segs:
        print(f'No AFL windows for {source} {record or ""}: none passed the quality check')
        return table
    if compare:
        extra_segs, extra = _same_record_windows(source, table['record'].unique(), seed)
        segs, table = segs + extra_segs, pd.concat([table, extra], ignore_index=True)

    normalise = ECGAugment(target_fs=FS, original_fs=AFL_SOURCES[source]['fs'], prob=0.0, norm_type='zscore')
    fig, axes = plt.subplots(len(segs), 2, figsize=(14, 2.4 * len(segs)),
                             gridspec_kw={'width_ratios': [3, 1]}, squeeze=False)
    for (ax_ecg, ax_spec), w, (_, row) in zip(axes, segs, table.iterrows()):
        x = normalise(w)
        img = AFDataset._generate_image(None, x)
        t = np.arange(len(x)) / FS
        colour = _LABEL_COLOURS[row['true']]
        when = '' if np.isnan(row['minute']) else f" @ {row['minute']:.1f} min"
        ax_ecg.plot(t, x[:, 0], lw=0.6, color='k')
        ax_ecg.set_xlim(0, t[-1])
        ax_ecg.set_ylabel('z-score')
        ax_ecg.set_title(f"{row['record']}{when} | window {row['index']} | {_LABEL_NAMES[row['true']]}",
                         color=colour, fontsize=10, loc='left')
        for spine in ax_ecg.spines.values():
            spine.set_edgecolor(colour)
        ax_spec.imshow(img, origin='lower', aspect='auto', extent=[0, t[-1], 0, 40])
        ax_spec.set_ylabel('Hz')
    axes[-1, 0].set_xlabel('time (s)')
    axes[-1, 1].set_xlabel('time (s)')
    fig.suptitle(f"{source} {record or 'all records'}: atrial flutter windows"
                 + (' (with AF and N from the same record)' if compare else ''), fontsize=12)
    fig.tight_layout()
    plt.show()
    return table


# ---------------------------------------------------------------------------
# Grad-CAM (following the authors' BGM_GitHub/interpretability.py, but in eval mode)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_model(path):
    model = GMCNet()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    # eval(): BatchNorm uses its learned statistics and dropout is off. The authors' script used train(),
    # which normalises each window by its own statistics and changes the predictions being explained.
    return model.eval()


def _scale(cam):
    """Min-max scale each window's map to [0, 1], as the authors do. All-zero maps stay zero."""
    flat = cam.flatten(1)
    lo, hi = flat.min(1).values, flat.max(1).values
    shape = (-1,) + (1,) * (cam.dim() - 1)
    return ((cam - lo.view(shape)) / (hi - lo).clamp_min(1e-12).view(shape)).numpy(), hi.numpy()


def _gradcam(model, xs, imgs, target=None):
    """Grad-CAM on both branches for a batch of windows.

    xs: [B, 2500, C] normalised ECG; imgs: [B, 128, 128, 3] spectrograms.
    target: class index per window, or None to explain the model's own prediction.
    Layers: the BatchNorm before the last ReLU of each branch (expand3_time[-2], expand3_freq[-2]),
    as in the authors' script. That ReLU is in-place and shares memory with the stored activations,
    so they end up post-ReLU; the gradient is taken at the BatchNorm output. A full backward hook
    would fail here because of the in-place ReLU, so the older register_backward_hook is used.
    """
    x_t = torch.tensor(xs[:, :, :1], dtype=torch.float32)                  # lead 1 only, as AFDataset
    x_f = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2)
    layers = {'time': model.expand3_time[-2], 'freq': model.expand3_freq[-2]}
    acts, grads, handles = {}, {}, []
    for name, layer in layers.items():
        handles.append(layer.register_forward_hook(
            lambda m, i, o, name=name: acts.__setitem__(name, o.detach())))
        handles.append(layer.register_backward_hook(
            lambda m, gi, go, name=name: grads.__setitem__(name, go[0].detach())))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            logits, _, _ = model(x_t, x_f)
            cls = logits.argmax(1) if target is None else torch.as_tensor(target, dtype=torch.long)
            # In eval mode each window's logit depends only on that window, so one backward pass
            # over the summed scores gives every window its own gradients.
            logits.gather(1, cls[:, None]).sum().backward()
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)

    with torch.no_grad():
        w_t = grads['time'].mean(dim=2, keepdim=True)                       # [B, 256, 1]
        cam_t = torch.relu((w_t * acts['time']).sum(1))                     # [B, 313]
        cam_t = F.interpolate(cam_t[:, None], size=xs.shape[1], mode='linear', align_corners=True)[:, 0]
        w_f = grads['freq'].mean(dim=(2, 3), keepdim=True)                  # [B, 256, 1, 1]
        cam_f = torch.relu((w_f * acts['freq']).sum(1))                     # [B, h, w]
        cam_f = F.interpolate(cam_f[:, None], size=imgs.shape[1:3], mode='bilinear', align_corners=False)[:, 0]
        cam_t, max_t = _scale(cam_t)
        cam_f, max_f = _scale(cam_f)
        p_af = torch.softmax(logits, 1)[:, 1].numpy()
    return {'cam_t': cam_t, 'cam_f': cam_f, 'max_t': max_t, 'max_f': max_f,
            'p_af': p_af, 'cls': cls.numpy()}


def gradcam_windows(dataset, fold, patient, show='incorrect', n=6, windows=None, run=None, seed=0,
                    target='pred', checkpoint=None, tol=1e-3, batch_size=32, label=None):
    """Grad-CAM maps for some of one test patient's windows.

    Window choice (show / n / windows / seed / label) works as in plot_windows.
    target: class to explain: 'pred' (the model's prediction, as in the paper), 'true', 'AF' or 'N'.
    checkpoint: explicit .pth instead of the one in DATASETS (see checkpoint_path).
    tol: largest allowed gap between the recomputed P(AF) and test_predictions.csv. A bigger gap means
         the checkpoint or preprocessing is not the one that produced the predictions, so it raises.

    Returns (rows, maps). rows: the windows' rows of load_patient's table plus 'cam_for' (class explained),
    'p_AF_now' (recomputed P(AF)) and 'empty_t' / 'empty_f' (True if nothing in the window supports
    that class, so the map is all zero). maps: dict of arrays, one entry per window:
    'x' [n, 2500] z-scored lead 1, 'img' [n, 128, 128, 3] spectrogram,
    'cam_t' [n, 2500] and 'cam_f' [n, 128, 128] scaled to [0, 1] per window.
    """
    cfg = DATASETS[dataset]
    segments, table = load_patient(dataset, fold, patient, run)
    windows = _choose_windows(table, show, n, windows, seed, label)
    if not windows:
        raise ValueError(f"No {show} {label or ''} windows for {patient}")
    rows = table.loc[windows].copy()
    model = _load_model(checkpoint_path(dataset, fold, run, checkpoint))

    normalise = _normaliser(cfg)
    xs = np.stack([normalise(segments[i]) for i in windows])
    imgs = np.stack([AFDataset._generate_image(None, x) for x in xs])
    to_idx = {'N': 0, 'AF': 1}
    targets = {'pred': None,
               'true': rows['true'].map(to_idx).to_numpy(),
               'AF': np.ones(len(rows), dtype=int),
               'N': np.zeros(len(rows), dtype=int)}[target]

    outs = []
    for s in range(0, len(windows), batch_size):
        sl = slice(s, s + batch_size)
        outs.append(_gradcam(model, xs[sl], imgs[sl], None if targets is None else targets[sl]))
    out = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}

    gap = np.abs(out['p_af'] - rows['p_AF'].to_numpy())
    if gap.max() > tol:
        worst = windows[int(gap.argmax())]
        raise RuntimeError(f'Recomputed P(AF) differs from test_predictions.csv by {gap.max():.4f} '
                           f'(window {worst}): this checkpoint or preprocessing did not produce these predictions')

    rows['cam_for'] = [CLASSES[c] for c in out['cls']]
    rows['p_AF_now'] = out['p_af']
    rows['empty_t'] = out['max_t'] == 0
    rows['empty_f'] = out['max_f'] == 0
    maps = {'x': xs[:, :, 0], 'img': imgs, 'cam_t': out['cam_t'], 'cam_f': out['cam_f']}
    return rows, maps


def plot_gradcam(dataset, fold, patient, show='incorrect', n=6, windows=None, run=None, seed=0,
                 target='pred', checkpoint=None, alpha=0.5, heatmap=True, label=None):
    """Plot Grad-CAM on the ECG (as in the paper's Figure 6) and on the spectrogram, one row per window.

    Arguments as in gradcam_windows; alpha is the heatmap opacity.
    heatmap=False draws the same windows without Grad-CAM (plot_windows: ECG and the colour spectrogram
    the model received), so you can switch between the two views with one argument.
    label: only windows whose true class is 'AF' or 'N' (None = both).
    Each map is scaled to [0, 1] within its own window: compare where attention falls, not how strong
    it looks, across windows. Returns the windows' rows (see gradcam_windows).
    """
    if not heatmap:
        return plot_windows(dataset, fold, patient, show, n, windows, run, seed, label)
    rows, maps = gradcam_windows(dataset, fold, patient, show, n, windows, run, seed, target, checkpoint,
                                 label=label)

    fig, axes = plt.subplots(len(rows), 2, figsize=(15, 2.5 * len(rows)), squeeze=False,
                             gridspec_kw={'width_ratios': [3, 1]}, layout='constrained')
    for k, ((ax_ecg, ax_spec), (i, row)) in enumerate(zip(axes, rows.iterrows())):
        x, t = maps['x'][k], np.arange(maps['x'].shape[1]) / FS
        pad = 0.05 * (x.max() - x.min())
        ylim = (x.min() - pad, x.max() + pad)
        title, colour = _window_title(i, row)
        note = ' (no evidence: map empty)' if row['empty_t'] else ''

        ax_ecg.imshow(maps['cam_t'][k][None, :], cmap=CAM_CMAP, vmin=0, vmax=1, alpha=alpha,
                      aspect='auto', extent=[0, t[-1], *ylim])
        ax_ecg.plot(t, x, lw=0.7, color='k')
        ax_ecg.set_xlim(0, t[-1])
        ax_ecg.set_ylim(*ylim)
        ax_ecg.set_ylabel('z-score')
        ax_ecg.set_title(f"{title} | CAM for {row['cam_for']}{note}", color=colour, fontsize=10, loc='left')
        for spine in ax_ecg.spines.values():
            spine.set_edgecolor(colour)

        # Spectrogram in grey so the heatmap reads clearly (the authors' script does the same)
        ax_spec.imshow(maps['img'][k].mean(-1), cmap='gray', origin='lower', aspect='auto',
                       extent=[0, t[-1], 0, 40])
        ax_spec.imshow(maps['cam_f'][k], cmap=CAM_CMAP, vmin=0, vmax=1, alpha=alpha * 0.8,
                       origin='lower', aspect='auto', extent=[0, t[-1], 0, 40])
        ax_spec.set_ylabel('Hz')

    axes[-1, 0].set_xlabel('time (s)')
    axes[-1, 1].set_xlabel('time (s)')
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=CAM_CMAP), ax=axes, shrink=0.6,
                 label='Grad-CAM (scaled per window)')
    run_name = run or DATASETS[dataset]['runs'][fold]
    fig.suptitle(f'{dataset} fold {fold} | {patient} | {run_name}', fontsize=12)
    plt.show()
    return rows


# ---------------------------------------------------------------------------
# Interactive explorer (Jupyter): pick dataset, fold, patient, windows and view, then Plot
# ---------------------------------------------------------------------------

def available_folds(dataset):
    """Folds of a dataset whose run has a test_predictions.csv (zero-shot sets don't cover every fold)."""
    cfg = DATASETS[dataset]
    return [f for f, run in cfg['runs'].items()
            if (cfg['results_dir'] / run / 'test_predictions.csv').exists()]


def explorer():
    """Dropdowns and toggles around plot_gradcam. Run explorer() as the last line of a notebook cell.

    Dataset -> fold -> patient update each other; the patient list is the fold's test set.
    Heatmaps off draws the plain windows (no model needed); on runs Grad-CAM for the chosen target.
    """
    import ipywidgets as w
    from IPython.display import display

    style = {'description_width': '70px'}
    dataset = w.Dropdown(options=list(DATASETS), value='afdb_spec', description='Dataset', style=style)
    fold = w.Dropdown(description='Fold', style=style, layout=w.Layout(width='160px'))
    patient = w.Dropdown(description='Patient', style=style, layout=w.Layout(width='220px'))
    show = w.ToggleButtons(options=['incorrect', 'correct', 'all'], value='incorrect', description='Windows',
                           style=style)
    label = w.ToggleButtons(options=[('AF + N', None), ('AF', 'AF'), ('N', 'N')], value=None,
                            description='True class', style=style)
    heatmap = w.Checkbox(value=True, description='Grad-CAM heatmaps', indent=False)
    target = w.Dropdown(options=['pred', 'true', 'AF', 'N'], value='pred', description='CAM for', style=style,
                        layout=w.Layout(width='160px'))
    n = w.IntSlider(value=4, min=1, max=12, description='How many', style=style)
    seed = w.IntText(value=0, description='Seed', style=style, layout=w.Layout(width='140px'))
    button = w.Button(description='Plot', button_style='primary', icon='play')
    status = w.HTML()
    out = w.Output()

    def update_folds(*_):
        folds = available_folds(dataset.value)
        fold.options = folds
        if folds:
            fold.value = folds[0]
        update_patients()       # the fold value may not change (e.g. 1 -> 1), so refresh the patients here
        data_dir = DATASETS[dataset.value]['data_dir']
        status.value = ('' if data_dir.exists() else
                        f'<b style="color:#c00">Data folder not found: {data_dir} (is the drive mounted?)</b>')
        if not folds:
            patient.options = []
            status.value += '<br><b style="color:#c00">No test results for any fold of this dataset</b>'

    def update_patients(*_):
        if fold.value is None:
            return
        patient.options = _test_patients(DATASETS[dataset.value], fold.value)

    def toggle_target(*_):
        target.disabled = not heatmap.value

    def plot(_):
        out.clear_output(wait=True)
        with out:
            if patient.value is None:
                print('Pick a dataset, fold and patient first')
                return
            try:
                print(f'{dataset.value} | fold {fold.value} | patient {patient.value}')
                display(patient_summary(dataset.value, fold.value, patient.value))
                plot_gradcam(dataset.value, fold.value, patient.value, show=show.value, n=n.value,
                             seed=seed.value, target=target.value, heatmap=heatmap.value, label=label.value)
            except Exception as e:      # show the message in the panel instead of a traceback
                print(f'{type(e).__name__}: {e}')

    dataset.observe(update_folds, 'value')
    fold.observe(update_patients, 'value')
    heatmap.observe(toggle_target, 'value')
    button.on_click(plot)
    update_folds()
    update_patients()

    controls = w.VBox([
        w.HBox([dataset, fold, patient]),
        show, label,
        w.HBox([heatmap, target]),
        w.HBox([n, seed, button]),
        status,
    ])
    return w.VBox([controls, out])


# ---------------------------------------------------------------------------
# t-SNE of the penultimate-layer features (as in MGCNet arXiv v2, Figure 7)
# ---------------------------------------------------------------------------
# The paper embeds "feature representations extracted from the penultimate layer" of a CPSC-trained model
# for two test sets: CPSC (in-domain, the fold's held-out cases) and CPSC_AFDB (cross-domain: AFDB windows),
# coloured by class and domain, with KDE contours and confidence ellipses. It gives no layer size or t-SNE
# settings, so here 'penultimate' = the 128-d hidden layer of the classifier (after its ReLU, the input to
# the last Linear); the 512-d fused vector and the 256-d time/frequency embeddings are also available.
#
# Cross-domain windows leave out, by default, the other dataset's records that this fold used for validation
# (picking the best epoch): CPSC models validate on AFDB fold k's val records, AFDB models on CPSC fold k's
# val cases (except the _afdb_only runs, which had none). What is left was never seen in any way.

_SPLITS = {'AFDB': (_AFDB_SPLIT, _AFDB_DATA, 250),
           'CPSC': (PROJECT / 'BGM_GitHub/preprocess/data_split/cpsc2021', _CPSC_DATA, 200)}
_AFL_DIRS = {'AFDB': AFL_SOURCES['afdb']['afl_dir'], 'CPSC': AFL_SOURCES['cpsc']['afl_dir']}
FEATURE_LAYERS = ('penultimate', 'fused', 'time', 'freq')
# paper colours: Normal red (in-domain) / green (cross), AF blue / yellow; flutter added in purple / pink
_TSNE_COLOURS = {('in', 'N'): 'tab:red', ('cross', 'N'): 'tab:green',
                 ('in', 'AF'): 'tab:blue', ('cross', 'AF'): 'gold',
                 ('in', 'AFL'): 'tab:purple', ('cross', 'AFL'): 'tab:pink'}


def _domain_of(dataset):
    data_dir = DATASETS[dataset]['data_dir']
    for name, (_, d, _) in _SPLITS.items():
        if data_dir == d:
            return name
    raise ValueError(f'{dataset}: t-SNE domains are only set up for models tested on AFDB or CPSC2021')


def _split_files(domain, fold, which):
    """npz files of one domain: 'val' = fold's held-out patients, 'all', or 'not val' (everything else)."""
    split_dir, data_dir, _ = _SPLITS[domain]
    with open(split_dir / f'fold_{fold}.json') as f:
        split = json.load(f)
    patients = {'val': split['val'], 'all': split['train'] + split['val'], 'not val': split['train']}[which]
    cfg = {'split_dir': split_dir, 'data_dir': data_dir}
    return [p for pid in patients for p in _patient_files(cfg, pid)]


def _sample_windows(files, n_per_class, rng, afl_files=()):
    """Uniform sample of up to n_per_class windows of each class (N, AF, and AFL from afl_files)."""
    index = []
    for path in files:
        for i, lab in enumerate(np.load(path)['labels']):
            index.append((path, i, 'AF' if lab == 'AFIB' else 'N'))
    for path in afl_files:
        index += [(path, i, 'AFL') for i in range(len(np.load(path)['labels']))]
    index = pd.DataFrame(index, columns=['file', 'window', 'true'])
    picks = [rng.choice(g.index, size=min(n_per_class, len(g)), replace=False) for _, g in index.groupby('true')]
    index = index.loc[np.sort(np.concatenate(picks))].reset_index(drop=True)
    segments = []
    for path, g in index.groupby('file', sort=False):
        with np.load(path) as d:
            segments.append(d['segments'][g['window'].to_numpy()])
    order = np.concatenate([g.index.to_numpy() for _, g in index.groupby('file', sort=False)])
    index = index.loc[order].reset_index(drop=True)
    # start time is only saved in the AFL files (starts, at the raw sampling rate)
    minutes = {}
    for path, g in index.groupby('file', sort=False):
        with np.load(path) as d:
            if 'starts' in d:
                fs = _SPLITS['CPSC'][2] if 'cpsc' in str(path) else _SPLITS['AFDB'][2]
                minutes.update({(path, w): d['starts'][w] / fs / 60 for w in g['window']})
    index['minute'] = [minutes.get((p, w), np.nan) for p, w in zip(index['file'], index['window'])]
    index['file'] = [p.stem for p in index['file']]
    return np.concatenate(segments), index


def _file_to_patient(domain):
    """npz file stem -> patient: the AFDB record, or the CPSC2021 case that holds the file."""
    case_to_files = _case_to_files(_SPLITS[domain][0])
    if case_to_files is None:
        return lambda stem: stem
    lookup = {name.removesuffix('.npz'): case for case, names in case_to_files.items() for name in names}
    return lambda stem: lookup.get(stem, stem)


def _features(model, xs, imgs, layer, batch_size=64):
    """Features of one layer for a batch of preprocessed windows, plus P(AF)."""
    store = {}
    hooks = [model.classifier[1].register_forward_hook(lambda m, i, o: store.__setitem__('penultimate', o)),
             model.classifier[0].register_forward_pre_hook(lambda m, i: store.__setitem__('fused', i[0]))]
    feats, p_af = [], []
    try:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for s in range(0, len(xs), batch_size):
                x_t = torch.tensor(xs[s:s + batch_size, :, :1], dtype=torch.float32)       # lead 1, as AFDataset
                x_f = torch.tensor(imgs[s:s + batch_size], dtype=torch.float32).permute(0, 3, 1, 2)
                logits, t_global, f_global = model(x_t, x_f)
                store.update(time=t_global, freq=f_global)
                feats.append(store[layer].numpy().copy())
                p_af.append(torch.softmax(logits, 1)[:, 1].numpy())
    finally:
        for h in hooks:
            h.remove()
    return np.concatenate(feats), np.concatenate(p_af)


def penultimate_features(dataset, fold, layer='penultimate', n_per_class=300, cross=True, afl=False,
                         exclude_val=True, run=None, checkpoint=None, seed=0, batch_size=64):
    """Features of one checkpoint for in-domain test windows and (cross=True) the other dataset's windows.

    dataset: a model in DATASETS tested on AFDB or CPSC2021 ('afdb_acc', 'afdb_spec', 'cpsc')
    layer: 'penultimate' (128-d classifier hidden layer), 'fused' (512-d [time, freq] concatenation),
           'time' or 'freq' (256-d embeddings Zt / Zf used by the contrastive loss)
    n_per_class: windows per class and domain (uniform random sample)
    afl: also embed the atrial flutter windows (data_afl) of the same records
    exclude_val: leave out the cross-domain records this fold used for validation (see above)
    Returns (features [n, d], table with domain / set / dataset / patient / file / window (row in that npz) /
    minute (AFL windows only) / true / pred / p_AF).
    """
    if layer not in FEATURE_LAYERS:
        raise ValueError(f'layer must be one of {FEATURE_LAYERS}')
    rng = np.random.default_rng(seed)
    home = _domain_of(dataset)
    sets = [('in', home, _split_files(home, fold, 'val'))]
    if cross:
        other = next(d for d in _SPLITS if d != home)
        sets.append(('cross', other, _split_files(other, fold, 'not val' if exclude_val else 'all')))
    model = _load_model(checkpoint_path(dataset, fold, run, checkpoint))

    feats, tables = [], []
    for domain, source, files in sets:
        afl_files = [_AFL_DIRS[source] / f'{p.stem}.npz' for p in files] if afl else []
        afl_files = [p for p in afl_files if p.exists()]
        segments, table = _sample_windows(files, n_per_class, rng, afl_files)
        normalise = ECGAugment(target_fs=FS, original_fs=_SPLITS[source][2], prob=0.0, norm_type='zscore')
        xs = np.stack([normalise(s) for s in segments])
        imgs = np.stack([AFDataset._generate_image(None, x) for x in xs])
        f, p = _features(model, xs, imgs, layer, batch_size)
        name = home if domain == 'in' else f'{home}_{source}'          # paper naming: CPSC, CPSC_AFDB
        table = table.assign(domain=domain, set=name, dataset=source,
                             patient=table['file'].map(_file_to_patient(source)),
                             p_AF=p, pred=np.where(p > 0.5, 'AF', 'N'))
        feats.append(f)
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)[['domain', 'set', 'dataset', 'patient', 'file', 'window', 'minute',
                                                  'true', 'pred', 'p_AF']]
    return np.concatenate(feats), table


def _ellipse_points(pts, n_std=2.0, n=100):
    """Outline of the n_std covariance ellipse of a group (about 86% of a 2-D Gaussian at n_std=2)."""
    if len(pts) < 3:
        return None
    vals, vecs = np.linalg.eigh(np.cov(pts.T))
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)]) * (n_std * np.sqrt(np.maximum(vals, 0)))[:, None]
    return (vecs @ circle).T + pts.mean(0)


def _kde_lines(pts, levels=(0.5, 0.9)):
    """Contour lines enclosing roughly the densest 50% and 90% of a group's points (list of [m, 2] arrays)."""
    import contourpy
    from scipy.stats import gaussian_kde
    if len(pts) < 10:
        return []
    kde = gaussian_kde(pts.T)
    dens = kde(pts.T)
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.1 * (hi - lo)
    gx, gy = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], 120),
                         np.linspace(lo[1] - pad[1], hi[1] + pad[1], 120))
    gz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    gen = contourpy.contour_generator(gx, gy, gz)
    return [line for th in np.quantile(dens, 1 - np.array(levels)) for line in gen.lines(th)]


def _misclassified(table):
    """AF/N windows whose prediction differs from the label (AFL windows have no correct answer)."""
    return (table['true'].isin(['N', 'AF']) & (table['pred'] != table['true'])).to_numpy()


def _hover_text(table):
    minute = table['minute'].map(lambda m: '' if pd.isna(m) else f'<br>start: {m:.1f} min')
    return ('<b>' + table['set'] + ' ' + table['true'] + '</b>'
            + '<br>dataset: ' + table['dataset'] + '<br>patient: ' + table['patient'].astype(str)
            + '<br>file: ' + table['file'] + '<br>segment (row in npz): ' + table['window'].astype(str) + minute
            + '<br>pred: ' + table['pred'] + '  P(AF) = ' + table['p_AF'].map('{:.3f}'.format)).to_numpy()


def _draw_tsne_matplotlib(emb, table, title, contours, ellipses, mark_errors, ax=None):
    show = ax is None
    if show:
        fig, ax = plt.subplots(figsize=(8, 7))
    for (domain, cls), colour in _TSNE_COLOURS.items():
        sel = ((table['domain'] == domain) & (table['true'] == cls)).to_numpy()
        if not sel.any():
            continue
        ax.scatter(*emb[sel].T, s=6, alpha=0.5, color=colour,
                   label=f"{table.loc[sel, 'set'].iloc[0]} {cls} (n={sel.sum()})")
        for line in (_kde_lines(emb[sel]) if contours else []):
            ax.plot(*line.T, color=colour, lw=1.0, alpha=0.8)
        outline = _ellipse_points(emb[sel]) if ellipses else None
        if outline is not None:
            ax.plot(*outline.T, color=colour, lw=1.8, ls='--')
    if mark_errors:
        wrong = _misclassified(table)
        ax.scatter(*emb[wrong].T, s=40, facecolors='none', edgecolors='k', linewidths=1,
                   label=f'misclassified (n={wrong.sum()})')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=1.5, loc='best')
    ax.set_title(title, fontsize=10)
    if show:
        plt.tight_layout()
        plt.show()


def _tsne_window(row):
    """Raw npz window behind one t-SNE point, and its sampling rate."""
    domain = row['dataset']
    if row['true'] == 'AFL':
        path = _AFL_DIRS[domain] / f"{row['file']}.npz"
    else:
        path = _SPLITS[domain][1] / f"{row['file']}.npz"
    with np.load(path) as d:
        return d['segments'][int(row['window'])], _SPLITS[domain][2]


def plot_tsne_window(table, i, model_path, gradcam=False, target='pred', alpha=0.5):
    """ECG and spectrogram of row i of a plot_tsne table, preprocessed as the model saw it, optionally with
    Grad-CAM (target: 'pred', 'true', 'AF' or 'N'; AFL windows have no true AF/N class, so 'true' explains
    the prediction). Called when a point of the interactive t-SNE is clicked."""
    row = table.loc[i]
    seg, fs = _tsne_window(row)
    x = ECGAugment(target_fs=FS, original_fs=fs, prob=0.0, norm_type='zscore')(seg)
    img = AFDataset._generate_image(None, x)
    t = np.arange(len(x)) / FS
    wrong = row['true'] in ('N', 'AF') and row['pred'] != row['true']
    colour = 'tab:red' if wrong else ('tab:green' if row['true'] in ('N', 'AF') else _TSNE_COLOURS[('in', 'AFL')])
    when = '' if pd.isna(row['minute']) else f" @ {row['minute']:.1f} min"
    title = (f"{row['set']} | {row['dataset']} patient {row['patient']} | {row['file']} segment {row['window']}{when}"
             f" | true {row['true']} → pred {row['pred']} (P(AF)={row['p_AF']:.3f})"
             + (' ✗' if wrong else ' ✓' if row['true'] in ('N', 'AF') else ''))

    cam = None
    if gradcam:
        to_idx = {'N': 0, 'AF': 1}
        cls = {'pred': None, 'AF': [1], 'N': [0],
               'true': [to_idx[row['true']]] if row['true'] in to_idx else None}[target]
        cam = _gradcam(_load_model(model_path), x[None], img[None], cls)
        title += f" | CAM for {CLASSES[int(cam['cls'][0])]}" + (' (map empty)' if cam['max_t'][0] == 0 else '')

    fig, (ax_ecg, ax_spec) = plt.subplots(1, 2, figsize=(15, 3), gridspec_kw={'width_ratios': [3, 1]},
                                          layout='constrained')
    pad = 0.05 * (x[:, 0].max() - x[:, 0].min())
    ylim = (x[:, 0].min() - pad, x[:, 0].max() + pad)
    if cam is not None:
        ax_ecg.imshow(cam['cam_t'][0][None, :], cmap=CAM_CMAP, vmin=0, vmax=1, alpha=alpha, aspect='auto',
                      extent=[0, t[-1], *ylim])
        ax_spec.imshow(img.mean(-1), cmap='gray', origin='lower', aspect='auto', extent=[0, t[-1], 0, 40])
        ax_spec.imshow(cam['cam_f'][0], cmap=CAM_CMAP, vmin=0, vmax=1, alpha=alpha * 0.8, origin='lower',
                       aspect='auto', extent=[0, t[-1], 0, 40])
    else:
        ax_spec.imshow(img, origin='lower', aspect='auto', extent=[0, t[-1], 0, 40])
    ax_ecg.plot(t, x[:, 0], lw=0.7, color='k')
    ax_ecg.set_xlim(0, t[-1])
    ax_ecg.set_ylim(*ylim)
    ax_ecg.set_ylabel('z-score')
    ax_ecg.set_xlabel('time (s)')
    ax_ecg.set_title(title, color=colour, fontsize=9, loc='left')
    for spine in ax_ecg.spines.values():
        spine.set_edgecolor(colour)
    ax_spec.set_ylabel('Hz')
    ax_spec.set_xlabel('time (s)')
    plt.show()


def _draw_tsne_plotly(emb, table, title, contours, ellipses, mark_errors, model_path=None):
    """Interactive version: hover a point for its dataset, patient, file and segment; click legend entries to
    hide a group (its contours and ellipse go with it). With model_path (a notebook with ipywidgets and
    anywidget), clicking a point shows its ECG and spectrogram below the plot, with optional Grad-CAM."""
    import plotly.graph_objects as go
    from matplotlib.colors import to_hex
    hover = _hover_text(table)
    traces, rows_of = [], []                      # rows_of[k] = table rows behind trace k (None for lines)
    for (domain, cls), colour in _TSNE_COLOURS.items():
        sel = ((table['domain'] == domain) & (table['true'] == cls)).to_numpy()
        if not sel.any():
            continue
        colour = to_hex(colour)
        name = f"{table.loc[sel, 'set'].iloc[0]} {cls} (n={sel.sum()})"
        traces.append(go.Scattergl(x=emb[sel, 0], y=emb[sel, 1], mode='markers', name=name, legendgroup=name,
                                   marker=dict(size=5, color=colour, opacity=0.6),
                                   hovertext=hover[sel], hoverinfo='text'))
        rows_of.append(np.flatnonzero(sel))
        lines = _kde_lines(emb[sel]) if contours else []
        outline = _ellipse_points(emb[sel]) if ellipses else None
        for line, dash in [(l, 'solid') for l in lines] + ([(outline, 'dash')] if outline is not None else []):
            traces.append(go.Scatter(x=line[:, 0], y=line[:, 1], mode='lines', legendgroup=name, showlegend=False,
                                     line=dict(color=colour, width=1.8 if dash == 'dash' else 1, dash=dash),
                                     hoverinfo='skip'))
            rows_of.append(None)
    if mark_errors:
        wrong = _misclassified(table)
        traces.append(go.Scattergl(x=emb[wrong, 0], y=emb[wrong, 1], mode='markers',
                                   name=f'misclassified (n={wrong.sum()})',
                                   marker=dict(size=11, symbol='circle-open', color='black', line=dict(width=1.5)),
                                   hovertext=hover[wrong], hoverinfo='text'))
        rows_of.append(np.flatnonzero(wrong))
    layout = dict(title=dict(text=title, font=dict(size=13)), width=850, height=720, template='plotly_white',
                  legend=dict(font=dict(size=10)), hoverlabel=dict(font_size=11))
    axes = dict(showticklabels=False, showgrid=False, zeroline=False)

    if model_path is None:
        fig = go.Figure(traces, layout)
        fig.update_xaxes(**axes)
        fig.update_yaxes(**axes, scaleanchor='x')
        fig.show()
        return fig

    import ipywidgets as w
    from IPython.display import display
    fig = go.FigureWidget(traces, layout)
    fig.update_xaxes(**axes)
    fig.update_yaxes(**axes, scaleanchor='x')
    gradcam = w.Checkbox(value=False, description='Grad-CAM', indent=False)
    target = w.Dropdown(options=[('explain prediction', 'pred'), ('explain true class', 'true'),
                                 ('explain AF', 'AF'), ('explain N', 'N')], value='pred', layout=w.Layout(width='190px'))
    status = w.HTML('<i>Click a point (e.g. a ringed misclassified one) to show its ECG and spectrogram.</i>')
    out = w.Output()
    state = {'row': None}

    def show(i):
        state['row'] = i
        with out:
            out.clear_output(wait=True)
            try:
                plot_tsne_window(table, i, model_path, gradcam.value, target.value)
            except Exception as e:                      # keep the widget alive if one window fails
                print(f'Could not show row {i}: {type(e).__name__}: {e}')

    def on_click(trace, points, _):
        if not points.point_inds:                       # the callback fires for every trace; skip the others
            return
        rows = rows_of[fig.data.index(trace)]
        if rows is None:
            return
        i = int(rows[points.point_inds[0]])
        if i != state['row']:
            show(i)

    for trace, rows in zip(fig.data, rows_of):
        if rows is not None:
            trace.on_click(on_click)

    def redraw(_):
        if state['row'] is not None:
            show(state['row'])
    gradcam.observe(redraw, names='value')
    target.observe(redraw, names='value')

    panel = w.VBox([fig, w.HBox([gradcam, target]), status, out])
    panel.show_row = show                               # also callable by hand: panel.show_row(table index)
    display(panel)
    return panel


def feature_separation(features, table, k=10):
    """Numbers to go with the picture, computed in the original feature space (t-SNE distorts distances):
      class silhouette: how well N and AF separate (-1..1, higher = better), per domain and overall
      domain kNN acc: within each class, how well a k-NN tells the two domains apart
                      (0.5 = fully mixed / domain-invariant, 1 = domains form separate clusters)."""
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    rows = []
    for name, part in [('all', table)] + list(table.groupby('set')):
        sub = part[part['true'].isin(['N', 'AF'])]
        if sub['true'].nunique() == 2:
            rows.append({'measure': 'class silhouette (N vs AF)', 'subset': name,
                         'value': silhouette_score(features[sub.index], sub['true'])})
    if table['domain'].nunique() == 2:
        for cls, part in table.groupby('true'):
            if part['domain'].nunique() == 2 and part['domain'].value_counts().min() >= k:
                knn = KNeighborsClassifier(n_neighbors=k)
                acc = cross_val_score(knn, features[part.index], part['domain'], cv=5).mean()
                rows.append({'measure': f'domain kNN acc (k={k}, 5-fold)', 'subset': cls, 'value': acc})
    return pd.DataFrame(rows).round(3)


def plot_tsne(dataset, fold, layer='penultimate', n_per_class=300, cross=True, afl=False, exclude_val=True,
              run=None, checkpoint=None, seed=0, perplexity=30, contours=True, ellipses=True,
              mark_errors=False, interactive=False, ax=None):
    """t-SNE of one checkpoint's features, in the style of MGCNet v2 Figure 7: in-domain test windows and
    cross-domain windows, coloured by class and domain, with KDE contours (densest 50% / 90%) and 2-sigma
    covariance ellipses. Arguments as in penultimate_features; perplexity is passed to sklearn's TSNE
    (PCA initialisation, automatic learning rate, fixed seed). Features are standardised first.

    mark_errors: ring the AF/N windows the model misclassifies (AFL windows have no correct answer)
    interactive: draw with plotly instead of matplotlib; hovering a point shows its dataset, patient, file,
                 segment (row in that npz; start time too for AFL windows), prediction and P(AF). Clicking a
                 point shows that window's ECG and spectrogram below the plot, with a Grad-CAM option
                 (needs a notebook with ipywidgets and anywidget)
    ax: matplotlib axes to draw into (matplotlib only); the figure is then not shown, so more can be added
    Returns (embedding [n, 2], table, separation) where separation = feature_separation(...).
    """
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    features, table = penultimate_features(dataset, fold, layer, n_per_class, cross, afl, exclude_val,
                                           run, checkpoint, seed)
    z = StandardScaler().fit_transform(features)
    keep = z.std(0) > 0                                    # dead ReLU units are constant: drop them
    emb = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto',
               random_state=seed).fit_transform(z[:, keep])

    run_name = run or DATASETS[dataset]['runs'][fold]
    title = f'{run_name}: t-SNE of {layer} features ({int(keep.sum())}/{len(keep)} active dims)'
    if interactive:
        _draw_tsne_plotly(emb, table, title, contours, ellipses, mark_errors,
                          model_path=checkpoint_path(dataset, fold, run, checkpoint))
    else:
        _draw_tsne_matplotlib(emb, table, title, contours, ellipses, mark_errors, ax)
    return emb, table, feature_separation(features, table)
