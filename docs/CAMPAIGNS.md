# Experiment campaigns of the article

`tools/campaigns.py` reruns every experiment of the manuscript from scratch on
the bundled datasets, in parallel and resumably, and stores seeds, raw records
and provenance in an order that co-investigators can validate.

## Setup (Windows, once)

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1
```

It requires Python 3.12.13, creates `.venv`, installs the solver with Numba
plus pytest, pandas, matplotlib and SciPy, runs the tests and prints the plan.

## Running

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_experiments.ps1
```

Each call is a session of at most eight hours (`-MaxHours`) and continues
where the previous one stopped; repeat it until it reports completion (about
four sessions with four processes). Equivalent direct command:
`.venv\Scripts\python.exe tools\campaigns.py run`.

Order within the first session:

1. Environment: tests, versions, Numba compilation (`results/00_environment`).
   Later sessions refuse to continue if Python, NumPy, Numba or the solver
   source changed.
2. Concurrency pilot: the same 30-second real-instance run alone and with 4 and
   6 simultaneous copies (`results/01_pilot_concurrency`). Keep the computer
   otherwise idle. Up to four workers are used if their loss of iterations is at
   most 15 %.
3. Convergence pilot: 6 independent instances per group (generator seed 5001),
   R/PO/CM, alpha 0.5/1/2, 3000 iterations with checkpoints; it recommends the
   maximum iteration budget for campaign B (`results/02_pilot_convergence`).
4. Time-budget campaigns, real instance first: D (30 s, 7 weights, 50 seeds),
   E (quality-time curve 30/120/300 s, 10 seeds), F (LS ablation), C (synthetic,
   Embedded and Decoupled-25/50/100, 7 weights), G (LS ablation on BASE), H (five
   seeds on BASE).
5. Iteration-budget campaigns: A (initial construction, R/PO/CM with up to 5000
   retries) and B (R/PO/CM, alpha 0.5/1/2, up to the recommended budget with
   exact checkpoints at 25, 50, 100, 200, 500, 1000 and 1500 iterations).

`python tools/campaigns.py plan` lists runs, shards and time estimates.

## Resumption and sessions

Every campaign is split into shards: one synthetic instance or one seed of the
real instance. A shard is an ordinary `uctp.experiments run` grid with its own
folder, manifest and one JSON record per run, written atomically. Restarting
skips finished runs and repeats only the runs that were in progress. A session
stops launching time-budget shards after `--max-hours` or when a file named
`STOP` exists in `results/`; running shards finish, and iteration-budget shards
fill the free processes meanwhile, so time-budget runs always share the
computer with the same number of processes. `progress.csv` logs the number of
simultaneous processes at the start and end of every shard. On Windows the
script keeps the computer awake; a run whose optimizer time exceeds its budget by
more than 10 s is flagged by the validation.

## Results layout

```
results/
  PLAN.md, SEEDS.json, session_log.txt, progress.csv, VALIDATION.md
  00_environment/        environment.json, environment_history.jsonl, tests.txt
  01_pilot_concurrency/  result.json and the individual runs
  02_pilot_convergence/  instances/, shards/, merged/, summary.md, recommendation.json
  <stage>/shards/<shard>/  manifest.json, records/<run_id>.json, raw.csv, status.json
  <stage>/shards/<shard>.log.txt
  <stage>/merged/        raw.csv, raw.jsonl, metrics_long.csv, trace.csv, checkpoints.csv
```

`SEEDS.json` lists every seed: synthetic runs use 703; BASE five-seed runs use
703, 215303898, 1366270267, 664565471 and 2054631744; the real instance uses
`random.Random(703).sample(range(1, 2**32), 50)` (the first ten for the
quality-time curve). Records keep the full solutions (`--save-solutions`) of the
time-budget campaigns and of B, so endpoint metrics can be re-evaluated without
optimizing again.

## After the last session

```powershell
.venv\Scripts\python.exe tools\campaigns.py validate   # VALIDATION.md
.venv\Scripts\python.exe tools\campaigns.py package    # results_package_<date>.zip
```

The validation checks, per stage, planned against recorded runs, a single
solver revision, input checksums against `datasets/Synthetic/SHA256SUMS`, the
real-instance seeds, duplicate runs and over-budget runs.
