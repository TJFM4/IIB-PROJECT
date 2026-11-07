# Author's December settings, folds 4 and 5, without CPSC validation

Same as `authcopy_exact` folds 4 and 5 (lr 1e-4, InverseTimeLR, SCL λ = 0.01, seed 3407, vertical flip on), with two changes:

- `dataset.val` (CPSC2021) is commented out, so nothing is validated during training.
- `training.selection: "last"`: all 25 epochs run and the final epoch is saved and tested. With no validation there is no early stopping.

| Config | Compare with |
|---|---|
| `bgm_afdb_4_authcopy_ex_minCPSC.yaml` | `_authcopy_exact` .8759 (best-CPSC epoch 18), author .9748 |
| `bgm_afdb_5_authcopy_ex_minCPSC.yaml` | `_authcopy_exact` .8531 (best-CPSC epoch 22), author .9980 |

These differ from the `_afdb_only` runs, which also have no validation but use lr 1e-3 and vertical flip off.

## Run

From `BGM_GitHub/`, plugged in:

```bash
bash authcopy_ex_minCPSC/run_folds.sh      # folds 4 and 5, about 5 h
```

Log: `new_runs/logs/authcopy_ex_minCPSC_folds45.log`. Add both runs to `new_runs/runs.md`.
