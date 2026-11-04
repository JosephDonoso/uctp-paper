# Records of the v8 manuscript (revision experiments)

Flat records of the experiments added in the revision (E1–E7 in the supplement), from
which `tools/paper_analysis_v8.py` regenerates the new tables, figures and cited numbers:

```
python tools/paper_analysis_v8.py --data paper_results_v8/data --out paper_out_v8
```

The campaigns were run with `tools/campaigns_v7.py` (rounds 1 and 2, `tools/run_v7.ps1`;
see docs/CAMPAIGNS.md) into `results_v7/` (about 5 GB, not included). Each stage was
flattened with `tools/extract_v7.py`.

| File | Content | Produced by |
|---|---|---|
| `P3.csv` | Decoupled-50 convergence pilot (3000 iterations) | round 1, stage P3 |
| `ssp_timing.json` | Time per SSP solve and scaling (E6) | `tools/ssp_timing.py` |
| `E1c.csv` | Embedded with and without common random numbers | round 1, stage E1c |
| `E5a_real_converged.csv` | Real instance, 2500 iterations, with checkpoints | round 2, stage E5a |
| `E5b_synthetic_converged.csv` | Synthetic instances, 2500 iterations, with checkpoints | round 2, stage E5b |
| `E2a_trajectories.csv`, `E2a_trajectories_trace.jsonl` | Search trajectories at alpha 0.25 and 1 | round 2, stage E2a |
| `E2b_warm_start.csv` | Alpha 1 started from the final alpha-0.25 timetables | round 2, stage E2b |
| `E2c_neighborhood_200.csv` | Alpha 1 with 200 neighbours | round 2, stage E2c |
| `E3_sequential.csv` | Sequential baseline, 30 s and 2500 iterations | round 2, stage E3 (run with the corrected proxy, commit 14847c4) |
| `reeval_own.csv` | Final timetables of the earlier campaigns (paper_results/) re-evaluated with 50 orders | `tools/extract_solutions.py` + `tools/reevaluate_solutions.py` |
| `e1a_noise.csv`, `reeval_ls_on.csv`, `reeval_ls_off.csv`, `e4b_calibration.csv`, `e4b_partial.csv` | Noise on fixed timetables (E1a), crossed LS evaluation (E4a) and partial adjustment (E4b) | `tools/v7_noise_ls.py` |

In every CSV, `rec_*` are the values computed by the search and `ree_*` the means over 50
fresh SSP orders (the values reported in the manuscript).

The supplementary tables of the earlier 30-second campaigns, re-evaluated, are produced by
replacing the final metrics of `paper_results/` with `reeval_own.csv` and running the
original analysis with the re-evaluated SSP baselines:

```
python tools/reevaluated_paper_results.py --results paper_results --reeval paper_results_v8/data/reeval_own.csv --out work/results_reeval
python tools/paper_analysis.py --results work/results_reeval --baselines paper_results_v8/real_baselines_reeval.json --out work/paper_out_reeval
```

The persistence tables across budgets (cm_by_budget, iteration steps) use the recorded
values of `paper_results/` (`python tools/paper_analysis.py --results paper_results`).
