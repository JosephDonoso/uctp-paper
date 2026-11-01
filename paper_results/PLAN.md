# Experiment plan

Workers: 4. Estimates are for the authors' Ryzen 7 7700X; parallel time includes a 10 % allowance.

| Order | Stage | Description | Shards | Runs | Hours (1 process) | Hours (parallel) |
|---|---|---|---|---|---|---|
| 0 | Environment | tests, versions, Numba warm-up | – | – | 0.0 | 0.0 |
| 1 | P1 Pilot: concurrency | same 30 s run alone and with 4 and 6 copies | – | 11 | 0.0 | 0.03 |
| 2 | P2 Pilot: iteration convergence | 6 independent instances per group (generator seed 5001), R/PO/CM, alpha 0.5/1/2, 3000 iterations; recommends the maximum iteration budget. | 30 | 270 | 3.0 | 0.8 |
| 3 | D Real instance, 30 s, 7 weights | Embedded and Decoupled-50, alpha 0, 0.125, 0.25, 0.5, 1, 2, 4, 50 seeds. | 50 | 700 | 6.0 | 1.6 |
| 4 | E Real instance, quality-time curve | Embedded and Decoupled-50, alpha 0/0.25/1, 30/120/300 s, first 10 seeds. | 10 | 180 | 7.5 | 2.1 |
| 5 | F Real instance, LS ablation | Embedded, alpha 0.25/1, local search on/off, 30 s. | 50 | 200 | 1.7 | 0.5 |
| 6 | C Synthetic, approaches x weights, 30 s | Embedded and Decoupled-25/50/100, 7 weights, CM with 500 retries, seed 703. | 250 | 7000 | 59.5 | 16.4 |
| 7 | G BASE, LS ablation | Embedded, alpha 0.25/1, local search on/off, 30 s, seed 703. | 50 | 200 | 1.7 | 0.5 |
| 8 | H BASE, five seeds | Embedded, alpha 1, 30 s, seeds 703, 215303898, 1366270267, 664565471, 2054631744. | 50 | 250 | 2.1 | 0.6 |
| 9 | A Initial construction (R/PO/CM) | R/PO/CM with up to 5000 retries (results for 100 and 500 retries follow from the nested attempt counts), initial quality at alpha 1, seed 703. | 250 | 750 | 0.3 | 0.1 |
| 10 | B Optimization up to 2500 iterations | R/PO/CM, alpha 0.5/1/2, 2500 iterations with exact checkpoints at 25, 50, 100, 200, 500, 1000, 1500, 2000, 500 construction retries, seed 703. | 250 | 2250 | 20.7 | 5.7 |

Total: 102.5 h with one process; about 28.2 h with 4 processes, i.e. 4 sessions of up to 8 h.

Seeds and datasets: SEEDS.json. Each session continues where the previous one stopped.
