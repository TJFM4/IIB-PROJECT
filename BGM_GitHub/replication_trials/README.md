# Replication trials

Attempts to reproduce the author's AFDB 5-fold results (accuracy .9989 / .9817 / .9858 / .9748 / .9980). Each folder has its configs, a run script and a README. Run scripts work from any directory; outputs go to `new_runs/{runs,checkpoint,results}` and logs to `new_runs/logs`.

| Folder | Folds | What changes |
|---|---|---|
| `authcopy_seeds/` | 4 | `_authcopy` with seeds 2 and 3 |
| `authcopy_exact/` | 1–5 | Author's December settings (lr 1e-4 for folds 4–5), CPSC validation |
| `authcopy_ex_minCPSC/` | 4–5 | December settings, no validation, last epoch |
| `authcopy_ex_afdbval/` | 2–5 | December settings, validated on 3 held-out AFDB training records |

Earlier runs are not in this folder: `_authcopy` and `_authcopy_oldcpsc` configs are in git history before commit e6ff752, and `_afdb_only` is `config/bgm/bgm_afdb_N.yaml`.

## Best test accuracy per fold (as of 8 Nov 2026)

| Fold | Best run | Acc | AUC | Author |
|---|---|---|---|---|
| 1 | `_afdb_only` | .9989 | 1.000 | .9989 |
| 2 | `_afdb_only` | .9475 | .9791 | .9817 |
| 3 | `_afdb_only` | .9458 | .9846 | .9858 |
| 4 | `_authcopy_oldcpsc` | .9335 | .9862 | .9748 |
| 5 | Mine (`bgm_afdb_5`, lr 1e-4, old CPSC) | .9359 | .9897 | .9980 |

Picking the best of about 10 setups per fold on test results is optimistic. The best single setup across all five folds is `_afdb_only` (mean .9239).

In each of folds 2–5 one large AF-heavy test record decides the score (06995, 08215, 04936, 07162); the other test records are near-perfect in most runs.
