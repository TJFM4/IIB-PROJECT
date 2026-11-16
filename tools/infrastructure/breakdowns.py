import json
from pathlib import Path
import pandas as pd

PROJECT = Path('/Users/tristanmartin/Desktop/IIB PROJECT')

# Add a line here for each new split folder
SPLIT_DIRS = {
    'afdb':             PROJECT / 'BGM_GitHub/preprocess/data_split/bihaf',
    'icentia_100':      PROJECT / 'trials/preprocess/data_split/icentia_100',
    'icentia_zeroshot': PROJECT / 'trials/preprocess/data_split/icentia_zeroshot',
}

pd.set_option('display.float_format', '{:.3f}'.format)

def load_split(dataset, fold):
    split_dir = SPLIT_DIRS[dataset]
    with open(split_dir / 'label_counts.json') as f:
        counts = json.load(f)
    with open(split_dir / f'fold_{fold}.json') as f:
        folds = json.load(f)
    return counts, {'Train': folds['train'], 'Test': folds['val']}


def n_folds(dataset):
    return len(list(SPLIT_DIRS[dataset].glob('fold_*.json')))


def subset_stats(counts, pids):
    n = sum(counts[p]['N'] for p in pids)
    af = sum(counts[p]['AFIB'] for p in pids)
    af_pids = [p for p in pids if counts[p]['AFIB'] > 0]
    top = max(af_pids, key=lambda p: counts[p]['AFIB'], default=None)
    return {
        'patients': len(pids),
        'AF patients': len(af_pids),
        'N patients': len(pids) - len(af_pids),
        'N windows': n,
        'AF windows': af,
        'total windows': n + af,
        'AF share (windows)': af / (n + af) if n + af else float('nan'),
        'AF share (patients)': len(af_pids) / len(pids) if pids else float('nan'),
        'top AF patient': top,
        'top AF patient share of AF windows': counts[top]['AFIB'] / af if top else float('nan'),
    }

# SINGLE FOLD
def breakdown(dataset, fold):
    counts, subsets = load_split(dataset, fold)
    subsets['All'] = subsets['Train'] + subsets['Test']
    return pd.DataFrame({name: subset_stats(counts, pids) for name, pids in subsets.items()}).T


def patient_table(dataset, fold, subset='Test', af_only=False):
    counts, subsets = load_split(dataset, fold)
    df = pd.DataFrame({p: counts[p] for p in subsets[subset]}).T.rename(columns={'AFIB': 'AF'})
    df['AF share'] = df['AF'] / (df['N'] + df['AF'])
    if af_only:
        df = df[df['AF'] > 0]
    return df.sort_values('AF', ascending=False)

# FIVE FOLD COMPARISONS
def fold_summary(dataset, subset='Test'):
    rows = {}
    for k in range(1, n_folds(dataset) + 1):
        counts, subsets = load_split(dataset, k)
        rows[f'Fold {k}'] = subset_stats(counts, subsets[subset])
    return pd.DataFrame(rows).T