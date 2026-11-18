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


def _choose_windows(table, show, n, windows, seed):
    """Explicit window indices, or a seeded random sample of 'incorrect' / 'correct' / 'all' windows."""
    if windows is not None:
        return list(windows)
    pool = {'incorrect': table.index[~table['correct']],
            'correct':   table.index[table['correct']],
            'all':       table.index}[show]
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


def plot_windows(dataset, fold, patient, show='incorrect', n=6, windows=None, run=None, seed=0):
    """Plot ECG windows next to the spectrograms the model received.

    show: 'incorrect', 'correct' or 'all' (ignored if windows is given)
    n: how many windows to draw (random sample, fixed by seed)
    windows: explicit list of window indices (the row numbers in load_patient's table)
    """
    segments, table = load_patient(dataset, fold, patient, run)
    windows = _choose_windows(table, show, n, windows, seed)
    if not windows:
        print(f'No {show} windows for {patient}')
        return table
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
                    target='pred', checkpoint=None, tol=1e-3, batch_size=32):
    """Grad-CAM maps for some of one test patient's windows.

    Window choice (show / n / windows / seed) works as in plot_windows.
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
    windows = _choose_windows(table, show, n, windows, seed)
    if not windows:
        raise ValueError(f'No {show} windows for {patient}')
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
                 target='pred', checkpoint=None, alpha=0.5):
    """Plot Grad-CAM on the ECG (as in the paper's Figure 6) and on the spectrogram, one row per window.

    Arguments as in gradcam_windows; alpha is the heatmap opacity.
    Each map is scaled to [0, 1] within its own window: compare where attention falls, not how strong
    it looks, across windows. Returns the windows' rows (see gradcam_windows).
    """
    rows, maps = gradcam_windows(dataset, fold, patient, show, n, windows, run, seed, target, checkpoint)

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
