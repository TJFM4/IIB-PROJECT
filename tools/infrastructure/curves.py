"""Training curves (loss, accuracy, learning rate) per epoch, read from the TensorBoard event files.

Each run folder can hold several event files: one per time train.py was started with that title (a restart,
or a rerun that reused the title). Each file is an 'attempt', numbered from 1 in start order; attempt=-1 is the
most recent one with data. Run names are folder paths under a root, e.g. 'bgm_cpsc_4_authcopy' or
'icentia_100/bgm_icentia_100_1'.

Tags written by train.py: Loss/{train,valid} = total loss = CE + lambda x SCL; LossCE/*, LossSCL/* (SCL is the
unweighted contrastive loss) and SCL_<part>/* (intra_time, intra_freq, inter_modal) only in runs started after
the loss logging was added (21 Nov); Accuracy/* in %; 'Learning Rate'. Runs with no validation set (_afdb_only)
have no */valid tags. The checkpoint train.py keeps is the epoch with the best validation accuracy.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = Path('/Users/tristanmartin/Desktop/IIB PROJECT')
RUN_ROOTS = {
    'mine':    PROJECT / 'BGM_GitHub/new_runs/runs',       # my runs (see BGM_GitHub/new_runs/runs.md)
    'author':  PROJECT / 'BGM_GitHub/runs',                # the author's downloaded TensorBoard logs
    'icentia': PROJECT / 'trials/icentia_runs/runs',       # Icentia11k runs
}
COLOURS = {'train': 'tab:blue', 'valid': 'tab:orange'}


def _event_files(run, source=None):
    sources = [source] if source else list(RUN_ROOTS)
    for s in sources:
        folder = RUN_ROOTS[s] / run
        if folder.is_dir():
            files = sorted(folder.glob('events.out.tfevents.*'), key=lambda p: int(p.name.split('.')[3]))
            if files:
                return s, files
    raise FileNotFoundError(f"No run '{run}' in {', '.join(str(RUN_ROOTS[s]) for s in sources)}")


def _read(path):
    """{tag: Series indexed by epoch} for one event file."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(path), size_guidance={'scalars': 0})
    ea.Reload()
    return {tag: pd.Series({e.step: e.value for e in ea.Scalars(tag)}) for tag in ea.Tags()['scalars']}


def list_runs(source=None):
    """Every run folder with TensorBoard logs: number of attempts and epochs logged in the latest one."""
    rows = []
    for s, root in RUN_ROOTS.items():
        if (source and s != source) or not root.is_dir():
            continue
        folders = sorted({f.parent for f in root.rglob('events.out.tfevents.*')})
        for folder in folders:
            rows.append({'source': s, 'run': folder.relative_to(root).as_posix(),
                         'attempts': len(list(folder.glob('events.out.tfevents.*')))})
    return pd.DataFrame(rows)


def attempts(run, source=None):
    """One row per event file of a run: when it started, epochs logged, best validation accuracy and
    whether the CE / contrastive parts were logged."""
    s, files = _event_files(run, source)
    rows = []
    for i, f in enumerate(files):
        curves = _read(f)
        epochs = len(curves.get('Loss/train', []))
        va = curves.get('Accuracy/valid')
        rows.append({
            'attempt': i + 1, 'source': s,
            'started': datetime.fromtimestamp(int(f.name.split('.')[3])).strftime('%d %b %Y %H:%M'),
            'epochs': epochs,
            'best valid acc': None if va is None or va.empty else f'{va.max():.2f}@{int(va.idxmax())}',
            'CE/SCL parts': 'LossCE/train' in curves,
            'file': f.name,
        })
    return pd.DataFrame(rows).set_index('attempt')


def load_curves(run, attempt=-1, source=None):
    """DataFrame indexed by epoch with one column per logged tag. attempt: 1, 2, ... as in attempts(run);
    -1 = latest attempt with any epoch."""
    _, files = _event_files(run, source)
    data = [_read(f) for f in files]
    nonempty = [i for i, d in enumerate(data) if len(d.get('Loss/train', []))]
    if not nonempty:
        raise ValueError(f'{run}: no epochs logged in any attempt')
    if attempt == -1:
        attempt = nonempty[-1] + 1
    if not 1 <= attempt <= len(files):
        raise ValueError(f'{run} has attempts 1-{len(files)}')
    df = pd.DataFrame(data[attempt - 1]).sort_index()
    df.index.name = 'epoch'
    df.attrs.update(run=run, attempt=attempt, n_attempts=len(files))
    return df


def _best_epoch(df):
    if 'Accuracy/valid' in df:
        return int(df['Accuracy/valid'].idxmax()), 'best valid acc (checkpoint)'
    return int(df.index.max()), 'last epoch (no validation set)'


def _pair(ax, df, tag, title, ylabel, logy=False):
    """Train and valid curves of one tag on one axis."""
    drawn = False
    for split in ('train', 'valid'):
        col = f'{tag}/{split}'
        if col in df:
            ax.plot(df.index, df[col], marker='o', ms=3, color=COLOURS[split], label=split)
            drawn = True
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale('log')
    return drawn


def plot_losses(run, attempt=-1, source=None, logy=False):
    """Loss over epochs for one training run.

    Panels: total loss (train / valid), accuracy, learning rate, and, if the run logged them, the
    cross-entropy and contrastive (SCL, unweighted) parts and the three SCL terms.
    The dashed line marks the epoch whose weights were kept (best validation accuracy).
    attempt: 1, 2, ... as in attempts(run); -1 = latest. attempt='all' overlays every attempt (total loss and
    accuracy only), e.g. to compare a rerun with the original.
    """
    if attempt == 'all':
        return _plot_attempts(run, source, logy)
    df = load_curves(run, attempt, source)
    parts = 'LossCE/train' in df
    scl_terms = sorted({c.split('/')[0] for c in df if c.startswith('SCL_')})
    n_panels = 3 + (2 if parts else 0) + (1 if scl_terms else 0)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.6 * nrows), squeeze=False)
    axes = axes.ravel()
    best, best_label = _best_epoch(df)

    _pair(axes[0], df, 'Loss', 'Total loss (CE + λ·SCL)', 'loss', logy)
    _pair(axes[1], df, 'Accuracy', 'Accuracy', '%')
    axes[2].plot(df.index, df['Learning Rate'], color='k', marker='o', ms=3)
    axes[2].set_title('Learning rate', fontsize=10)
    k = 3
    if parts:
        _pair(axes[k], df, 'LossCE', 'Cross-entropy', 'loss', logy)
        _pair(axes[k + 1], df, 'LossSCL', 'Contrastive loss (SCL, unweighted)', 'loss', logy)
        k += 2
    if scl_terms:
        ax = axes[k]
        for term, colour in zip(scl_terms, ('tab:green', 'tab:purple', 'tab:brown')):
            for split, ls in (('train', '-'), ('valid', '--')):
                col = f'{term}/{split}'
                if col in df:
                    ax.plot(df.index, df[col], ls=ls, marker='o', ms=2, color=colour,
                            label=f"{term.removeprefix('SCL_')} {split}")
        ax.set_title('SCL terms (solid = train, dashed = valid)', fontsize=10)
        ax.set_ylabel('loss')
        k += 1
    for ax in axes[:k]:
        ax.axvline(best, color='grey', ls='--', lw=1)
        ax.set_xlabel('epoch')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    for ax in axes[k:]:
        ax.axis('off')

    acc = df['Accuracy/valid'].max() if 'Accuracy/valid' in df else None
    note = f'{best_label}: epoch {best}' + (f' ({acc:.2f}%)' if acc is not None else '')
    fig.suptitle(f"{run} (attempt {df.attrs['attempt']} of {df.attrs['n_attempts']}): {len(df)} epochs, {note}",
                 fontsize=12)
    fig.tight_layout()
    plt.show()
    return df


def _plot_attempts(run, source=None, logy=False):
    info = attempts(run, source)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    colours = plt.cm.tab10.colors
    for i, row in info.iterrows():
        if row['epochs'] == 0:
            continue
        df = load_curves(run, i, source)
        c = colours[(i - 1) % 10]
        label = f"attempt {i} ({row['started']})"
        for split, ls in (('train', '-'), ('valid', '--')):
            if f'Loss/{split}' in df:
                axes[0].plot(df.index, df[f'Loss/{split}'], ls=ls, marker='o', ms=3, color=c, label=f'{label} {split}')
            if f'Accuracy/{split}' in df:
                axes[1].plot(df.index, df[f'Accuracy/{split}'], ls=ls, marker='o', ms=3, color=c)
        if 'Accuracy/valid' in df:
            best = int(df['Accuracy/valid'].idxmax())
            axes[1].plot(best, df['Accuracy/valid'].max(), marker='*', ms=14, color=c)
    axes[0].set_title('Total loss (solid = train, dashed = valid)', fontsize=10)
    axes[1].set_title('Accuracy (%), ★ = kept checkpoint', fontsize=10)
    if logy:
        axes[0].set_yscale('log')
    for ax in axes:
        ax.set_xlabel('epoch')
    axes[0].legend(fontsize=7)
    fig.suptitle(f'{run}: every attempt', fontsize=12)
    fig.tight_layout()
    plt.show()
    return info
