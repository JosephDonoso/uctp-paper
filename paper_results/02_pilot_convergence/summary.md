# Convergence pilot (270 solved runs)

Median and quartiles of the relative improvement of F between budgets (%).

| From | To 3000 median | p75 | share of runs improving > 1 % |
|---|---|---|---|
| 500 | 26.39 | 36.35 | 87.7 |
| 1000 | 12.08 | 24.80 | 70.5 |
| 1500 | 0.91 | 13.62 | 49.8 |
| 2000 | 0.00 | 5.31 | 31.4 |
| 2500 | 0.00 | 0.00 | 14.6 |

Iteration of the last improvement: median 1481, p75 2134, p90 2675.

Rule: the smallest budget whose median further improvement up to 3000 iterations is at most 1 % and whose 75th percentile is at most 5 %.

**Recommended maximum iteration budget: 2500.**
