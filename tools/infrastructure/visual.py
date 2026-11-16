import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path('/Users/tristanmartin/Desktop/IIB PROJECT')

# Reuse the exact preprocessing from training/testing, so the plots show what the model saw
sys.path.insert(0, str(PROJECT / 'trials'))
from data.dataset import AFDataset      # noqa: E402
from data.ecg_augment import ECGAugment  # noqa: E402

FS = 250
CLASSES = {0: 'N', 1: 'AF'}

# split_fold: fold_k.json to read (None = same as the fold argument).
# run: results subfolder for a fold; '{fold}' is filled in. Override with run=... in the functions.
DATASETS = {
    'afdb': {
        'split_dir':   PROJECT / 'BGM_GitHub/preprocess/data_split/bihaf',
        'data_dir':    PROJECT / 'BGM_GitHub/afdb_dataset/data',
        'results_dir': PROJECT / 'BGM_GitHub/new_runs/results',
        'run':         'bgm_afdb_{fold}_afdb_only',
        'split_fold':  None,
    },
    'icentia_100': {
        'split_dir':   PROJECT / 'trials/preprocess/data_split/icentia_100',
        'data_dir':    Path('/Volumes/seagate_hard/icentia_100_npz'),
        'results_dir': PROJECT / 'trials/icentia_runs/results/icentia_100',
        'run':         'bgm_icentia_100_{fold}',
        'split_fold':  None,
    },
    # Zero-shot: one split (fold_1 = all 97 patients); 'fold' picks the AFDB checkpoint
    'icentia_zeroshot': {
        'split_dir':   PROJECT / 'trials/preprocess/data_split/icentia_zeroshot',
        'data_dir':    Path('/Volumes/seagate_hard/icentia_100_npz'),
        'results_dir': PROJECT / 'trials/icentia_runs/results/zeroshot_pilot100',
        'run':         'bgm_afdb_{fold}_afdb_only',
        'split_fold':  1,
    },
}

_normalise = ECGAugment(target_fs=FS, original_fs=FS, prob=0.0, norm_type='zscore')


def _test_patients(cfg, fold):
    split_fold = cfg['split_fold'] or fold
    with open(cfg['split_dir'] / f'fold_{split_fold}.json') as f:
        return json.load(f)['val']


def load_patient(dataset, fold, patient, run=None):
    """Return (segments, table) for one test patient.

    table has one row per window: true label, prediction, P(AF), correct, plus
    seg_idx/start_sample when the npz has them (Icentia11k).
    """
    cfg = DATASETS[dataset]
    run = run or cfg['run'].format(fold=fold)
    preds = pd.read_csv(cfg['results_dir'] / run / 'test_predictions.csv')
    patients = _test_patients(cfg, fold)
    if patient not in patients:
        raise ValueError(f'{patient} is not in the test set of {dataset} fold {fold}')

    # test.py doesn't shuffle: rows follow the val list, patient by patient
    # (AFDataset skips npz files with no windows, so they take no rows)
    offset = 0
    for pid in patients:
        with np.load(cfg['data_dir'] / f'{pid}.npz') as d:
            n = len(d['labels'])
            if pid == patient:
                segments = d['segments']
                labels = (d['labels'] == 'AFIB').astype(int)
                extra = {k: d[k] for k in ('seg_idx', 'start_sample') if k in d}
                break
        offset += n

    rows = preds.iloc[offset:offset + n].reset_index(drop=True)
    if len(rows) != n or not np.array_equal(rows['label'].to_numpy(), labels):
        raise RuntimeError(f'Predictions in {run} do not line up with {patient}.npz: '
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


def plot_windows(dataset, fold, patient, show='incorrect', n=6, windows=None, run=None, seed=0):
    """Plot ECG windows next to the spectrograms the model received.

    show: 'incorrect', 'correct' or 'all' (ignored if windows is given)
    n: how many windows to draw (random sample, fixed by seed)
    windows: explicit list of window indices (the row numbers in load_patient's table)
    """
    segments, table = load_patient(dataset, fold, patient, run)

    if windows is None:
        pool = {'incorrect': table.index[~table['correct']],
                'correct':   table.index[table['correct']],
                'all':       table.index}[show]
        if len(pool) == 0:
            print(f'No {show} windows for {patient}')
            return table
        rng = np.random.default_rng(seed)
        windows = np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False))

    t = np.arange(segments.shape[1]) / FS
    fig, axes = plt.subplots(len(windows), 2, figsize=(14, 2.4 * len(windows)),
                             gridspec_kw={'width_ratios': [3, 1]}, squeeze=False)
    for (ax_ecg, ax_spec), i in zip(axes, windows):
        row = table.loc[i]
        x = _normalise(segments[i])                    # z-scored, as in test.py
        img = AFDataset._generate_image(None, x)       # 128x128 RGB STFT, 0-40 Hz
        colour = 'tab:green' if row['correct'] else 'tab:red'

        ax_ecg.plot(t, x[:, 0], lw=0.6, color='k')
        ax_ecg.set_xlim(0, t[-1])
        ax_ecg.set_ylabel('z-score')
        where = f" | seg {row['seg_idx']} @ {row['start_sample'] / FS / 60:.1f} min" if 'seg_idx' in row else ''
        ax_ecg.set_title(f"window {i}{where} | true {row['true']} → pred {row['pred']} "
                         f"(P(AF)={row['p_AF']:.3f}) {'✓' if row['correct'] else '✗'}",
                         color=colour, fontsize=10, loc='left')
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
