"""Examine the bidirectional gating modules (BGMs) of a trained GMCNet.

Each BGM (model/bgm.py) gates one branch with a summary of the other:
    gate_time = sigmoid(W_ft @ GAP(x_freq) + b_ft)    [B, C]  multiplies the time channels
    gate_freq = sigmoid(W_tf @ GAP(x_time) + b_tf)    [B, C]  multiplies the frequency channels
The gated features are downsampled and concatenated with an ungated ("bypass") downsample of the same
features; expand{k}_{time,freq} then mixes the two halves.

Two views:
  1. Parameters: W and b of each gate's Linear layer, and how much weight the expand conv puts on the
     BGM half vs the bypass half (bgm_parameters, parameter_summary, plot_gate_matrices).
  2. Gate values: the sigmoid outputs for real test windows, i.e. the per-channel weights actually
     applied, split by true class, plus each half's contribution to the expand conv output
     (gate_values, afl_gate_values, gate_summary, plot_gate_* functions).

Channel numbers are only meaningful within one checkpoint: two trainings can learn the same channels
in a different order, so compare folds through summary statistics, not channel by channel.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .visual import (DATASETS, AFL_SOURCES, AFDataset, ECGAugment, FS, _AFDB_DATA, _CPSC_DATA,
                     checkpoint_path, load_patient, _load_model, _normaliser, _test_patients, _patient_files)

BGMS = ('bgm1', 'bgm2', 'bgm3')
# side -> (gate module, what it gates, what it reads)
GATES = {'time': ('gate_time_from_freq', 'time', 'freq'),
         'freq': ('gate_freq_from_time', 'freq', 'time')}
CLASS_COLOURS = {'N': 'tab:blue', 'AF': 'tab:red', 'AFL': 'tab:purple'}


def _model(dataset, fold, run=None, checkpoint=None):
    return _load_model(checkpoint_path(dataset, fold, run, checkpoint))


# ---------------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------------

def bgm_parameters(dataset, fold, run=None, checkpoint=None):
    """{(bgm, side): {'W': [C, C], 'b': [C]}} for every gate. side 'time' = gate on the time branch
    (computed from the frequency features): W[i, j] = effect of pooled freq channel j on time channel i."""
    model = _model(dataset, fold, run, checkpoint)
    out = {}
    for bgm in BGMS:
        for side, (name, _, _) in GATES.items():
            lin = getattr(getattr(model, bgm), name)[2]
            out[(bgm, side)] = {'W': lin.weight.detach().numpy().copy(), 'b': lin.bias.detach().numpy().copy()}
    return out


def _expand_conv(model, k, branch):
    return getattr(model, f'expand{k}_{branch}')[0]


def parameter_summary(dataset, fold, run=None, checkpoint=None):
    """One row per gate: size, weight and bias statistics, the gate value for an all-zero input (sigmoid(b)),
    and the share of the following expand conv's weight norm that sits on the BGM half of its input
    (0.5 = equal; the bypass half is the ungated downsample)."""
    model = _model(dataset, fold, run, checkpoint)
    rows = []
    for k, bgm in enumerate(BGMS, start=1):
        for side, (name, gated, source) in GATES.items():
            lin = getattr(getattr(model, bgm), name)[2]
            W, b = lin.weight.detach().numpy(), lin.bias.detach().numpy()
            conv = _expand_conv(model, k, gated).weight.detach()
            C = conv.shape[1] // 2
            n_bgm, n_byp = conv[:, :C].norm().item(), conv[:, C:].norm().item()
            rows.append({
                'bgm': bgm, 'gate on': gated, 'from': source, 'C': W.shape[0],
                '|W| mean': np.abs(W).mean(), 'W std': W.std(), 'W frobenius': np.linalg.norm(W),
                'b mean': b.mean(), 'b std': b.std(),
                'sigmoid(b) mean': (1 / (1 + np.exp(-b))).mean(),
                'expand weight share on BGM half': n_bgm / (n_bgm + n_byp),
            })
    return pd.DataFrame(rows).set_index(['bgm', 'gate on']).round(4)


def plot_gate_matrices(dataset, fold, run=None, checkpoint=None, sort=True):
    """Heat maps of each gate's weight matrix W (rows = gated channels, columns = pooled channels of the
    other branch), with the bias beside it. sort=True orders rows and columns by their mean weight so
    structure is easier to see (the order is arbitrary anyway)."""
    params = bgm_parameters(dataset, fold, run, checkpoint)
    fig, axes = plt.subplots(len(BGMS), 4, figsize=(15, 3.6 * len(BGMS)),
                             gridspec_kw={'width_ratios': [6, 0.4, 6, 0.4]})
    for r, bgm in enumerate(BGMS):
        for c, side in enumerate(GATES):
            W, b = params[(bgm, side)]['W'], params[(bgm, side)]['b']
            if sort:
                rows_, cols_ = np.argsort(W.mean(1)), np.argsort(W.mean(0))
                W, b = W[rows_][:, cols_], b[rows_]
            lim = np.abs(W).max()
            ax_w, ax_b = axes[r, 2 * c], axes[r, 2 * c + 1]
            im = ax_w.imshow(W, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto', interpolation='nearest')
            _, gated, source = GATES[side]
            ax_w.set_title(f'{bgm}: gate on {gated} (from {source})  C={W.shape[0]}', fontsize=10)
            ax_w.set_xlabel(f'{source} channel (pooled)')
            ax_w.set_ylabel(f'{gated} channel (gated)')
            fig.colorbar(im, ax=ax_w, fraction=0.04)
            blim = np.abs(b).max()
            ax_b.imshow(b[:, None], cmap='RdBu_r', vmin=-blim, vmax=blim, aspect='auto', interpolation='nearest')
            ax_b.set_xticks([])
            ax_b.set_yticks([])
            ax_b.set_title('b', fontsize=10)
    fig.suptitle(f'{dataset} fold {fold}: BGM gate weights' + (' (rows/cols sorted by mean)' if sort else ''),
                 fontsize=12)
    fig.tight_layout()
    plt.show()


def compare_folds(dataset, folds=(1, 2, 3, 4, 5), runs=None):
    """parameter_summary for several folds side by side (runs: optional {fold: run name})."""
    tables = {}
    for f in folds:
        try:
            tables[f] = parameter_summary(dataset, f, (runs or {}).get(f))
        except FileNotFoundError as e:
            print(f'fold {f}: {e}')
    return pd.concat(tables, names=['fold'])


# ---------------------------------------------------------------------------
# 2. Gate values on real windows
# ---------------------------------------------------------------------------

def _prepare(segments, original_fs):
    normalise = ECGAugment(target_fs=FS, original_fs=original_fs, prob=0.0, norm_type='zscore')
    xs = np.stack([normalise(s) for s in segments])
    imgs = np.stack([AFDataset._generate_image(None, x) for x in xs])
    return xs, imgs


def _inputs(xs, imgs):
    """Model inputs as float32 tensors: lead 1 [n, 2500, 1] (as AFDataset) and spectrogram [n, 3, 128, 128]."""
    return (torch.tensor(xs[:, :, :1], dtype=torch.float32),
            torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2).contiguous())


def _run_with_hooks(model, inputs, batch_size=64):
    """Forward pass that records every gate's output and, for each expand conv, the size of the
    contribution from the BGM half and from the bypass half of its input (L2 norm per window)."""
    gates = {key: [] for key in ((bgm, side) for bgm in BGMS for side in GATES)}
    contrib = {(k, br, half): [] for k in (1, 2, 3) for br in ('time', 'freq') for half in ('bgm', 'bypass')}
    p_af = []
    handles = []
    for bgm in BGMS:
        for side, (name, _, _) in GATES.items():
            mod = getattr(getattr(model, bgm), name)
            handles.append(mod.register_forward_hook(
                lambda m, i, o, key=(bgm, side): gates[key].append(o.detach().numpy())))
    for k in (1, 2, 3):
        for br in ('time', 'freq'):
            conv = _expand_conv(model, k, br)

            def hook(m, inp, out, k=k, br=br):
                x = inp[0]
                C = x.shape[1] // 2
                op = F.conv1d if br == 'time' else F.conv2d
                for half, sl in (('bgm', slice(None, C)), ('bypass', slice(C, None))):
                    y = op(x[:, sl], m.weight[:, sl], None, m.stride, m.padding)
                    contrib[(k, br, half)].append(y.flatten(1).norm(dim=1).numpy())
            handles.append(conv.register_forward_hook(hook))
    try:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for s in range(0, len(inputs[0]), batch_size):
                logits, _, _ = model(inputs[0][s:s + batch_size], inputs[1][s:s + batch_size])
                p_af.append(torch.softmax(logits, 1)[:, 1].numpy())
    finally:
        for h in handles:
            h.remove()
    return ({k: np.concatenate(v) for k, v in gates.items()},
            {k: np.concatenate(v) for k, v in contrib.items()},
            np.concatenate(p_af))


def gate_values(dataset, fold, n_per_class=300, run=None, checkpoint=None, seed=0, batch_size=64, tol=1e-3):
    """Gate outputs for a random sample of the fold's test windows (uniform over windows, up to
    n_per_class of each true class; None = every test window, which is slow for large folds).

    Returns a dict:
      'table': one row per window (patient, window index within patient, true, pred, p_AF, correct)
      'gates': {(bgm, side): [n, C]} sigmoid outputs, side 'time' = weights applied to the time channels
      'contrib': {(stage, branch, 'bgm'|'bypass'): [n]} L2 norm of each half's contribution to expand conv
    P(AF) is recomputed and checked against test_predictions.csv (tol), so the gates are the ones behind
    the reported predictions.
    """
    cfg = DATASETS[dataset]
    patients = _test_patients(cfg, fold)
    # labels only (cheap) to draw a uniform sample before loading any segments
    index = []
    for pid in patients:
        labels = np.concatenate([np.load(p)['labels'] for p in _patient_files(cfg, pid)])
        index += [(pid, i, 'AF' if l == 'AFIB' else 'N') for i, l in enumerate(labels)]
    index = pd.DataFrame(index, columns=['patient', 'window', 'true'])
    rng = np.random.default_rng(seed)
    if n_per_class is not None:
        picks = [rng.choice(g.index, size=min(n_per_class, len(g)), replace=False)
                 for _, g in index.groupby('true')]
        index = index.loc[np.sort(np.concatenate(picks))]

    segments, rows = [], []
    for pid, g in index.groupby('patient', sort=False):
        segs, table = load_patient(dataset, fold, pid, run)
        segments.append(segs[g['window'].to_numpy()])
        rows.append(table.loc[g['window'], ['true', 'pred', 'p_AF', 'correct']]
                    .assign(patient=pid, window=g['window'].to_numpy()))
    table = pd.concat(rows, ignore_index=True)[['patient', 'window', 'true', 'pred', 'p_AF', 'correct']]

    inputs = _inputs(*_prepare(np.concatenate(segments), cfg['original_fs']))
    gates, contrib, p_af = _run_with_hooks(_model(dataset, fold, run, checkpoint), inputs, batch_size)
    gap = np.abs(p_af - table['p_AF'].to_numpy()).max()
    if gap > tol:
        raise RuntimeError(f'Recomputed P(AF) differs from test_predictions.csv by {gap:.4f}: '
                           'this checkpoint or preprocessing did not produce these predictions')
    return {'table': table, 'gates': gates, 'contrib': contrib, 'inputs': inputs,
            'dataset': dataset, 'fold': fold, 'run': run, 'checkpoint': checkpoint}


def afl_gate_values(dataset, fold, test_patients_only=True, run=None, checkpoint=None, batch_size=64):
    """Gate outputs for the atrial flutter windows (data_afl) of the dataset the model is tested on.
    The model never saw AFL windows, but test_patients_only=True also drops records whose AF/N windows
    were in this fold's training set. pred / p_AF are the model's AF-vs-N output for each AFL window."""
    cfg = DATASETS[dataset]
    source = {_AFDB_DATA: 'afdb', _CPSC_DATA: 'cpsc'}.get(cfg['data_dir'])
    if source is None:
        raise ValueError(f'No AFL windows for {dataset} (only AFDB and CPSC2021 have them)')
    afl_dir = AFL_SOURCES[source]['afl_dir']
    allowed = None
    if test_patients_only:
        allowed = {p.stem for pid in _test_patients(cfg, fold) for p in _patient_files(cfg, pid)}
    segments, rows = [], []
    for f in sorted(afl_dir.glob('*.npz')):
        if allowed is not None and f.stem not in allowed:
            continue
        d = np.load(f)
        segments += list(d['segments'])
        rows += [{'patient': f.stem, 'window': i, 'true': 'AFL'} for i in range(len(d['labels']))]
    if not rows:
        print(f'No AFL windows from {dataset} fold {fold} test records' if test_patients_only else 'No AFL windows')
        return None
    table = pd.DataFrame(rows)
    inputs = _inputs(*_prepare(segments, AFL_SOURCES[source]['fs']))
    gates, contrib, p_af = _run_with_hooks(_model(dataset, fold, run, checkpoint), inputs, batch_size)
    table['p_AF'] = p_af
    table['pred'] = np.where(p_af > 0.5, 'AF', 'N')
    table['correct'] = np.nan
    return {'table': table, 'gates': gates, 'contrib': contrib, 'inputs': inputs,
            'dataset': dataset, 'fold': fold, 'run': run, 'checkpoint': checkpoint}


def combine(*results):
    """Join several gate_values / afl_gate_values results from the same model into one."""
    results = [r for r in results if r is not None]
    first = results[0]
    return {'table': pd.concat([r['table'] for r in results], ignore_index=True),
            'gates': {k: np.concatenate([r['gates'][k] for r in results]) for k in first['gates']},
            'contrib': {k: np.concatenate([r['contrib'][k] for r in results]) for k in first['contrib']},
            'inputs': tuple(torch.cat([r['inputs'][i] for r in results]) for i in (0, 1)),
            **{k: first[k] for k in ('dataset', 'fold', 'run', 'checkpoint')}}


def gate_ablation(res, batch_size=64):
    """Do the gates' input-dependent values matter? Re-run the windows with every gate's output replaced:
      'channel mean': each channel's gate fixed to its mean over these windows (same scale, no dependence
                      on the input)
      'other class':  AF windows get the N mean gate and N windows the AF mean (AFL windows get the AF mean)
    and compare with the model as trained. If accuracy and P(AF) barely change, the gate acts as a fixed
    per-channel scale, and the following BatchNorm absorbs that scale, so the BGM behaves like an ordinary
    conv path. Returns one row per setting: accuracy (AF/N windows only), mean |change in P(AF)|, and the
    number of windows whose prediction flips."""
    model = _model(res['dataset'], res['fold'], res['run'], res['checkpoint'])
    true = res['table']['true'].to_numpy()
    base_p = res['table']['p_AF'].to_numpy()
    labelled = np.isin(true, ['AF', 'N'])
    means = {c: {k: g[true == c].mean(0) for k, g in res['gates'].items()} for c in ('AF', 'N') if (true == c).any()}
    settings = {'channel mean': {k: np.broadcast_to(g.mean(0), g.shape) for k, g in res['gates'].items()}}
    if len(means) == 2:
        settings['other class'] = {k: np.where((true == 'AF')[:, None], means['N'][k], means['AF'][k])
                                   for k in res['gates']}

    def accuracy(p):
        return ((p > 0.5) == (true == 'AF'))[labelled].mean() if labelled.any() else np.nan

    rows = [{'setting': 'as trained', 'accuracy': accuracy(base_p), 'mean |dP(AF)|': 0.0, 'flips': 0}]
    for name, replacement in settings.items():
        state = {'s': 0}
        handles = []
        for bgm in BGMS:
            for side, (mod_name, _, _) in GATES.items():
                mod = getattr(getattr(model, bgm), mod_name)
                handles.append(mod.register_forward_hook(
                    lambda m, i, o, key=(bgm, side): torch.tensor(
                        replacement[key][state['s']:state['s'] + o.shape[0]], dtype=o.dtype)))
        p = []
        try:
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for s in range(0, len(true), batch_size):
                    state['s'] = s
                    logits, _, _ = model(res['inputs'][0][s:s + batch_size], res['inputs'][1][s:s + batch_size])
                    p.append(torch.softmax(logits, 1)[:, 1].numpy())
        finally:
            for h in handles:
                h.remove()
        p = np.concatenate(p)
        rows.append({'setting': name, 'accuracy': accuracy(p), 'mean |dP(AF)|': np.abs(p - base_p).mean(),
                     'flips': int(((p > 0.5) != (base_p > 0.5)).sum())})
    return pd.DataFrame(rows).set_index('setting').round(4)


def _classes(res):
    return [c for c in ('N', 'AF', 'AFL') if c in set(res['table']['true'])]


def _cohens_d(a, b):
    pooled = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2)
    return (a.mean(0) - b.mean(0)) / np.where(pooled > 0, pooled, np.nan)


def gate_summary(res):
    """One row per gate:
      mean / channel spread: average gate value and how much it differs between channels
      window spread: std across windows, averaged over channels (about 0 = the gate is a fixed per-channel
                     scale, the same for every input; larger = it really depends on the other branch)
      <0.1 / >0.9: share of gate values that nearly switch a channel off / pass it unchanged
      mean <class>: average gate value per true class
      max |d| AF-N: largest per-channel effect size between AF and N windows (Cohen's d), and how many
                    channels have |d| > 0.8 (a 'large' difference)
      max |mean AF - mean N|: the largest per-channel difference in gate value itself. A large d with a tiny
                    difference means the gate separates the classes consistently but by a negligible amount.
    """
    true = res['table']['true'].to_numpy()
    rows = []
    for (bgm, side), g in res['gates'].items():
        row = {'bgm': bgm, 'gate on': side, 'C': g.shape[1], 'mean': g.mean(),
               'channel spread': g.mean(0).std(), 'window spread': g.std(0).mean(),
               '<0.1': (g < 0.1).mean(), '>0.9': (g > 0.9).mean()}
        for c in _classes(res):
            row[f'mean {c}'] = g[true == c].mean()
        if {'AF', 'N'} <= set(true):
            d = _cohens_d(g[true == 'AF'], g[true == 'N'])
            row['max |d| AF-N'] = np.nanmax(np.abs(d))
            row['channels |d|>0.8'] = int((np.abs(d) > 0.8).sum())
            # d is relative to a very small spread when the gates barely move, so also give the size of the
            # largest difference in gate units
            row['max |mean AF - mean N|'] = np.abs(g[true == 'AF'].mean(0) - g[true == 'N'].mean(0)).max()
        rows.append(row)
    return pd.DataFrame(rows).set_index(['bgm', 'gate on']).round(3)


def plot_gate_profiles(res, sort=True):
    """Mean gate value of every channel, per true class (shaded: 25-75th percentile across windows).
    Channels sorted by their overall mean, so channels the gate mostly closes are on the left."""
    true = res['table']['true'].to_numpy()
    fig, axes = plt.subplots(len(BGMS), 2, figsize=(15, 3.2 * len(BGMS)), squeeze=False)
    for r, bgm in enumerate(BGMS):
        for c, side in enumerate(GATES):
            g = res['gates'][(bgm, side)]
            order = np.argsort(g.mean(0)) if sort else np.arange(g.shape[1])
            ax = axes[r, c]
            for cls in _classes(res):
                gc = g[true == cls][:, order]
                x = np.arange(gc.shape[1])
                ax.plot(x, gc.mean(0), color=CLASS_COLOURS[cls], lw=1.2, label=f'{cls} (n={len(gc)})')
                ax.fill_between(x, *np.percentile(gc, [25, 75], axis=0), color=CLASS_COLOURS[cls], alpha=0.15)
            _, gated, source = GATES[side]
            ax.set_title(f'{bgm}: weights on {gated} channels (computed from {source})', fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, g.shape[1] - 1)
            ax.set_ylabel('gate value')
            ax.set_xlabel('channel' + (' (sorted by mean gate)' if sort else ''))
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"{res['dataset']} fold {res['fold']}: BGM gate values per channel", fontsize=12)
    fig.tight_layout()
    plt.show()


def plot_gate_distributions(res, bins=50):
    """Histogram of all gate values (every channel, every window) per gate and class: mass near 0 or 1
    means the gates act as switches, mass in the middle means they rescale."""
    true = res['table']['true'].to_numpy()
    fig, axes = plt.subplots(len(BGMS), 2, figsize=(12, 2.8 * len(BGMS)), squeeze=False)
    for r, bgm in enumerate(BGMS):
        for c, side in enumerate(GATES):
            g, ax = res['gates'][(bgm, side)], axes[r, c]
            for cls in _classes(res):
                ax.hist(g[true == cls].ravel(), bins=bins, range=(0, 1), density=True, histtype='step',
                        lw=1.4, color=CLASS_COLOURS[cls], label=cls)
            ax.set_title(f'{bgm}: gate on {GATES[side][1]}', fontsize=10)
            ax.set_xlabel('gate value')
    axes[0, 0].legend()
    fig.suptitle(f"{res['dataset']} fold {res['fold']}: distribution of gate values", fontsize=12)
    fig.tight_layout()
    plt.show()


def plot_class_separation(res, a='AF', b='N', top=10):
    """Per-channel effect size (Cohen's d) between the gate values of class a and class b, sorted.
    |d| > 0.8 (dotted) is conventionally 'large'. Returns the top channels of each gate."""
    true = res['table']['true'].to_numpy()
    fig, axes = plt.subplots(len(BGMS), 2, figsize=(14, 2.6 * len(BGMS)), squeeze=False)
    tops = []
    for r, bgm in enumerate(BGMS):
        for c, side in enumerate(GATES):
            g, ax = res['gates'][(bgm, side)], axes[r, c]
            d = _cohens_d(g[true == a], g[true == b])
            order = np.argsort(d)
            ax.bar(np.arange(len(d)), d[order], color=np.where(d[order] > 0, CLASS_COLOURS.get(a, 'k'),
                                                               CLASS_COLOURS.get(b, 'k')), width=1.0)
            for y in (-0.8, 0.8):
                ax.axhline(y, color='k', ls=':', lw=0.8)
            ax.set_title(f'{bgm}: gate on {GATES[side][1]}  (d > 0: higher for {a})', fontsize=10)
            ax.set_xlim(-0.5, len(d) - 0.5)
            ax.set_ylabel("Cohen's d")
            for ch in np.argsort(-np.abs(d))[:top]:
                tops.append({'bgm': bgm, 'gate on': GATES[side][1], 'channel': int(ch), 'd': d[ch],
                             f'mean {a}': g[true == a][:, ch].mean(), f'mean {b}': g[true == b][:, ch].mean()})
    axes[-1, 0].set_xlabel('channel (sorted by d)')
    axes[-1, 1].set_xlabel('channel (sorted by d)')
    fig.suptitle(f"{res['dataset']} fold {res['fold']}: which gate channels differ between {a} and {b}",
                 fontsize=12)
    fig.tight_layout()
    plt.show()
    return pd.DataFrame(tops).round(3)


def path_usage(res):
    """Share of each expand conv's output that comes from the BGM (gated) half of its input, per class:
    ||conv_bgm(x)|| / (||conv_bgm(x)|| + ||conv_bypass(x)||), averaged over windows. 0.5 = both halves
    contribute equally; near 0 = the network mostly ignores the gated path."""
    true = res['table']['true'].to_numpy()
    rows = []
    for k in (1, 2, 3):
        for br in ('time', 'freq'):
            nb, ny = res['contrib'][(k, br, 'bgm')], res['contrib'][(k, br, 'bypass')]
            share = nb / (nb + ny)
            row = {'stage': f'bgm{k} -> expand{k}', 'branch': br, 'all': share.mean()}
            for c in _classes(res):
                row[c] = share[true == c].mean()
            rows.append(row)
    return pd.DataFrame(rows).set_index(['stage', 'branch']).round(3)


def plot_path_usage(res):
    table = path_usage(res)
    cols = [c for c in ('N', 'AF', 'AFL') if c in table.columns]
    ax = table[cols].plot.bar(figsize=(10, 3.5), color=[CLASS_COLOURS[c] for c in cols], rot=0)
    ax.axhline(0.5, color='k', ls=':', lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel('share from BGM half')
    ax.set_xticklabels([f'{s.split()[0]} {b}' for s, b in table.index])
    ax.set_title(f"{res['dataset']} fold {res['fold']}: how much of each expand conv's output comes from the "
                 'gated path', fontsize=11)
    plt.tight_layout()
    plt.show()
    return table
