# Validation report (2026-11-01T03:27:04)

Environment: Python 3.12.13, NumPy 2.4.6, Numba 0.68.1, solver revision `eb48b2dbbae76200`, Windows-10-10.0.19045-SP0, 16 logical CPUs, workers 4.

| Stage | Planned runs | Recorded | Solved | Construction failed | Other | Revisions | Problems |
|---|---|---|---|---|---|---|---|
| P2 Pilot: iteration convergence | 270 | 270 | 270 | 0 | 0 | 1 | none |
| D Real instance, 30 s, 7 weights | 700 | 700 | 700 | 0 | 0 | 1 | none |
| E Real instance, quality-time curve | 180 | 180 | 180 | 0 | 0 | 1 | none |
| F Real instance, LS ablation | 200 | 200 | 200 | 0 | 0 | 1 | none |
| C Synthetic, approaches x weights, 30 s | 7000 | 7000 | 7000 | 0 | 0 | 1 | none |
| G BASE, LS ablation | 200 | 200 | 200 | 0 | 0 | 1 | none |
| H BASE, five seeds | 250 | 250 | 250 | 0 | 0 | 1 | none |
| A Initial construction (R/PO/CM) | 750 | 750 | 750 | 0 | 0 | 1 | none |
| B Optimization up to 2500 iterations | 2250 | 2250 | 2250 | 0 | 0 | 1 | none |

Concurrency while time-budget shards were running (share of their wall-clock time, from progress.csv): 1 processes 0.0%, 2 processes 0.0%, 3 processes 0.0%, 4 processes 100.0%.

Overall: complete and consistent.
