# Records of the v9 manuscript (pending runs)

Results of the two runs scripted in `tools/run_v9_pending.ps1`, which move the remaining
earlier-protocol results of the manuscript to the main protocol. They were run with the
four processes split by `ronda_v9.ps1` (real instance in two halves of the 50 E5a seeds,
BASE with and without LS); each run is independent, so the results equal those of the
sequential script.

| File | Content | Produced by |
|---|---|---|
| `reeval_ls_on.csv`, `reeval_ls_off.csv` | LS ablation under the main protocol: Embedded with and without LS, alpha 0.25 and 1, real instance (50 seeds of E5a) and BASE (50 instances, seed 703), 2500 iterations, common random numbers; every final timetable evaluated with 50 orders with LS (`on`) and without LS (`off`) | `uctp.experiments run` (stage `LS_v9`) + `tools/extract_solutions.py` + `tools/reevaluate_solutions.py --ls on/off` |
| `e4b_calibration.csv` | Partial adjustment: calibration of p on the institutional timetable | `tools/v7_noise_ls.py --partial-stage E5a_real_converged` |
| `e4b_partial.csv` | Final alpha-0.25 timetables of E5a (2500 iterations) under partial adjustment p = 0.6; the institutional row uses the same draws as the calibration row | same command |

The runs with LS reproduce the corresponding E5a and E5b runs (same seeds): the final
conflict counts coincide in all 200 runs. Their re-evaluated values differ slightly from
those of E5a and E5b because the 50 re-evaluation orders are drawn independently (real
instance at alpha 0.25: F_ref 0.279 in both; BASE at alpha 1: 2.462 here against 2.457).

The v9 tables `ls_crossed.tex` and `partial_final.tex` are generated from these records,
together with the other v9 tables, by

```
python tools/revision_checks.py --data paper_results_v8/data --data-v9 paper_results_v9/data --out paper_out_v9
```
