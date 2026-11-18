import json
from pathlib import Path

import pandas as pd

# Relative results paths are read from here, so they work from any notebook folder
PROJECT = Path('/Users/tristanmartin/Desktop/IIB PROJECT')

# Column name -> value from metrics.json. Class 1 = AF, so 'recall' is AF recall.
# specificity_per_class is [AF recall, N recall]; its second entry is the true specificity (AF positive).
METRICS = {
    'Acc':      lambda m: m['accuracy'],
    'AUC':      lambda m: m['auc'],
    'F1':       lambda m: m['f1_score'],
    'Prec':     lambda m: m['precision'],
    'Recall':   lambda m: m['recall'],
    'Spec (N)': lambda m: m.get('specificity', m['specificity_per_class'][1]),
}


def fold_results(results, folds=range(1, 6)):
    """Per-fold metrics for one set of results folders (one row per fold that has a metrics.json).

    results: path to a results folder with '{fold}' in it,
             e.g. 'BGM_GitHub/new_runs/results/bgm_afdb_{fold}_afdb_only',
             or a list of folder paths, one per fold (for sets that mix folders across folds).
             Relative paths are taken from the project root; absolute paths are used as they are.
    """
    folds = list(folds)
    paths = results if isinstance(results, (list, tuple)) else [str(results).format(fold=f) for f in folds]
    if len(paths) != len(folds):
        raise ValueError(f'{len(paths)} results folders given for {len(folds)} folds')

    rows = {}
    for fold, path in zip(folds, paths):
        metrics_file = PROJECT / path / 'metrics.json'
        if metrics_file.exists():
            m = json.loads(metrics_file.read_text())
            rows[fold] = {'path': str(path), **{col: get(m) for col, get in METRICS.items()}}
    return pd.DataFrame.from_dict(rows, orient='index').rename_axis('fold')


def fold_average(results, folds=range(1, 6), std=False, decimals=4):
    """Mean of the per-fold test metrics for each set of results folders, one row per set.

    results: list of results-folder paths (see fold_results), or a dict {row label: path}.
             List entries are labelled by their path with '{fold}' shown as 'N'.
    std:     also add a '<metric> std' column (sample std across folds).
    Folds without a metrics.json are skipped and reported; the 'Folds' column shows which were averaged.

    Example:
        fold_average({
            'Mine, AFDB only': 'BGM_GitHub/new_runs/results/bgm_afdb_{fold}_afdb_only',
            'Author':          'BGM_GitHub/results/bgm_afdb_{fold}',
        })
    """
    if not isinstance(results, dict):
        results = {str(r).replace('{fold}', 'N') if isinstance(r, (str, Path)) else ' / '.join(map(str, r)): r
                   for r in results}

    rows = {}
    for label, paths in results.items():
        per_fold = fold_results(paths, folds)
        missing = [f for f in folds if f not in per_fold.index]
        if missing:
            print(f'{label}: no metrics.json for fold(s) {missing}')
        if per_fold.empty:
            continue

        row = {'Folds': ' '.join(map(str, per_fold.index))}
        for col in METRICS:
            row[col] = per_fold[col].mean()
            if std:
                row[f'{col} std'] = per_fold[col].std()
        rows[label] = row

    return pd.DataFrame.from_dict(rows, orient='index').round(decimals)
