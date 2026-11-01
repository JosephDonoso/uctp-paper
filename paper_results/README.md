# Results used in the manuscript

This folder holds the merged records of every run reported in the manuscript and its
supplement (11,530 runs, 0 failures), produced by `tools/campaigns.py` on
Windows 10 (build 19045), AMD Ryzen 7 7700X, Python 3.12.13, NumPy 2.4.6 and
Numba 0.68.1, with four concurrent processes.

Regenerate every table, figure and cited number with

```powershell
.venv\Scripts\python.exe -m pip install -e ".[paper]"
.venv\Scripts\python.exe tools\paper_analysis.py --results paper_results --out paper_out
```

The outputs are identical to the tables of the manuscript.

| Folder | Content |
|---|---|
| `00_environment/` | versions, test log, solver fingerprint |
| `01_pilot_concurrency/` | 30 s runs alone and with 4 and 6 copies |
| `02_pilot_convergence/` | 30 pilot instances (generator seed 5001), records up to 3000 iterations, recommendation (2500) |
| `A_initial_construction/` | R / PO / CM construction, up to 5000 retries |
| `B_iteration_budgets/` | Embedded up to 2500 iterations; `checkpoints.csv` holds the exact solutions at 25–2000 iterations |
| `C_synthetic_30s/` | Embedded and Decoupled-25/50/100, seven weights, 250 instances |
| `D_real_30s/` | real instance, seven weights, 50 seeds |
| `E_real_quality_time/` | real instance, 30 / 120 / 300 s, ten seeds |
| `F_real_ls_ablation/`, `G_base_ls_ablation/` | with and without the student-adjustment LS |
| `H_base_five_seeds/` | five seeds on BASE |

`SEEDS.json` lists every seed, `PLAN.md` the planned runs, `VALIDATION.md` the
consistency checks, and `progress.csv` / `session_log.txt` the execution log
(local paths removed).

The `solver_revision` stored in each record (`eb48b2db…`) is a SHA-256 of the
`uctp/*.py` files as checked out on Windows (CRLF line endings) at commit
`19b9544`. Later commits change only the experiment orchestration, the analysis
script and documentation (plus one line in `uctp/experiments/runner.py` that
removes a leftover temporary file); the solver code is identical apart from line
endings, so a checkout with LF endings yields a different fingerprint.

The per-run JSON records with full solutions (about 8 GB, 614 MB compressed) are
not stored in Git; they are available from the authors on request.
