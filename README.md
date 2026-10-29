# UCTP Paper

Source code and datasets accompanying the article on university course timetabling and student sectioning. Student section changes are modeled under an explicit adjustment assumption.

Solves the institutional timetable `X` and student-to-section assignments `Y` through initial construction, the Student Sectioning Problem (SSP), and Tabu Search (TS). Implements **Random**, **Prioritization Only**, **Compatibility Matrix**, **Embedded**, and **Decoupled-k**, using the manuscript's terminology.

Includes the **curricular instance generator** for five synthetic groups, a de-identified institutional instance and its de-identified source workbook. Synthetic datasets are generated locally and are not included in the repository.

Output contains solutions, initial/final quality metrics, timings, counters and execution metadata. The repository includes general experiment and analysis commands that export raw records and statistical tables. Figures are left to researchers.

## Installation

Requires **Python 3.12.13** and **NumPy 2.4.6**. From the repository root, using that Python version:

```powershell
python --version
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

`python --version` must report `3.12.13`; the package requires this exact version. Install the interpreter before creating the virtual environment. Commands below use the environment directly, without activation. On Linux or macOS, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.

**Verified environment:** package installation and tests have been run on Windows with Python 3.12.13 and NumPy 2.4.6.

### Optional compiled SSP

Installing the `fast` extra (`pip install -e .[fast]`) enables a Numba kernel for the SSP and the fixed-assignment objective. It reproduces the pure-Python solver exactly: same assignments, objective values and random-number state for the same seed (checked by `tests/test_ssp_kernel.py`). The first call compiles the kernel and caches it on disk; run one short warm-up before timed experiments. Set `UCTP_PURE_PYTHON=1` to force the pure-Python path.

## Solving an instance

Run the included synthetic example with Compatibility Matrix and Embedded:

```powershell
.venv\Scripts\python.exe -m uctp --input examples/small.json --output output/embedded.json --iterations 200 --seed 703
```

To select another initial construction strategy:

```powershell
.venv\Scripts\python.exe -m uctp --input examples/small.json --output output/random.json --strategy Random --iterations 200 --seed 703
.venv\Scripts\python.exe -m uctp --input examples/small.json --output output/prioritization.json --strategy "Prioritization Only" --iterations 200 --seed 703
```

Run Decoupled-50 with four search blocks:

```powershell
.venv\Scripts\python.exe -m uctp --input examples/small.json --output output/decoupled.json --approach Decoupled --ssp-interval 50 --search-blocks 4 --seed 703
```

`--ssp-interval 50` allows up to 50 TS cycles before the next SSP solution. A block may also end because of stagnation, lack of moves, or the time limit. `--search-blocks 4` limits the number of search blocks, rather than total iterations. Tabu memory persists across blocks, and the best evaluated `X,Y` pair is retained.

Add `--time-limit 30` to set a nominal time budget. Optimizer timing starts after initial construction. The limit is checked after accepted moves and search blocks, so a run may exceed the nominal budget. Other stopping limits remain active. Neighbor evaluation is allowed to finish before stopping.

Display the full command-line help:

```powershell
.venv\Scripts\python.exe -m uctp --help
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--input`, `--output` | Instance path and solution destination | Required |
| `--input-format` | `instance` or `legacy` | `instance` |
| `--strategy` | Exact initial construction strategy name | `Compatibility Matrix` |
| `--approach` | `Embedded` or `Decoupled` | `Embedded` |
| `--alpha` | Nonnegative weight in the internal objective | `1` |
| `--max-retries` | Construction retries after the initial attempt | `500` |
| `--iterations` | TS cycle limit for Embedded | `200` |
| `--ssp-interval` | TS cycle limit per block in Decoupled-k | Required for Decoupled |
| `--search-blocks` | Maximum number of search blocks in Decoupled | Required for Decoupled |
| `--max-stagnant-iterations` | Accepted cycles without improving the TS incumbent | `200` |
| `--neighborhood-size` | Distinct relocation attempts per neighborhood, from 1 to 50; fixed lectures excluded | `50` |
| `--time-limit` | Nominal optimizer time in seconds | No time limit |
| `--seed` | Seed for Python random and NumPy | No fixed seed |
| `--local-search` | SSP local refinement: `on` or backtracking-only `off` | `on` |
| `--total-iterations` | Additional accepted-move cap across all search blocks | No additional cap |
| `--group`, `--run-id` | Research dataset label and unique run identifier | Parent directory; generated identifier |
| `--details` | Export student attendance/cost and section enrollment records | Off |
| `--trace` | Record native objective values at accepted TS moves | Off |
| `--max-daily-load` | Override legacy daily workload; default for English-schema teachers without a limit | Input value; fallback `8` blocks |
| `--max-block-load` | Institutional capacity for simultaneous lectures | `15` |

Zero retries allow one construction attempt; 500 retries allow 501 attempts. Retries continue the random sequence initialized by `--seed`. Reproducibility requires identical input, environment, parameters, and iteration budgets. Runs with a time budget also depend on machine performance.

Each neighborhood samples up to 50 distinct active-lecture/destination moves uniformly without replacement. Preassigned lectures are excluded; the destination differs from the current block. If the finite pool has fewer moves, all are attempted. Infeasible moves consume an attempt and are not replaced: this does not guarantee 50 feasible neighbors. Repetitions can occur across different search iterations, but not within one neighborhood.

The institutional method parameters are `beta=0.25`, `CV_tol=0.10`, and `W_tol=4`. Tabu capacity is `ceil(sqrt(n_move))`, where preassigned lectures do not count as movable. Execution settings are supplied through command-line arguments.

## Institutional instance

`datasets/Real/real.json` uses the English schema (`--input-format instance`). It contains 70 courses, 104 sections, 279 weekly lectures, 48 teachers and 1,209 students, including 1,170 with modeled course requests. Six complete courses have fixed schedules (25 lectures). The input retains its teacher availability, day restrictions and daily limits: four blocks for teachers, with one confirmed six-block exception. Do not replace these stored limits with a uniform override.

All course, section, lecture, teacher, student and curricular-level identifiers have been replaced by neutral labels. No names, student numbers, original course codes or re-identification key are included. Enrollment demand and timetable structure are retained for solving; this file contains course requests, not the observed student-to-section assignments or the complete observed timetable. Identifier removal does not provide a formal guarantee against re-identification from external structural information.

```powershell
.venv\Scripts\python.exe -m uctp --input datasets/Real/real.json --input-format instance --output output/real.json --strategy "Compatibility Matrix" --approach Embedded --alpha 0.25 --iterations 200 --seed 703
```

`datasets/Real/Semester_2025_1_anonymized.xlsx` contains the de-identified source workbook, preserving its two worksheets and recorded enrollments and schedules. Its `U` student, `T` teacher and `C` modeled-course labels match `real.json`; `X` labels identify courses outside the modeled catalog, `P` programs and `R` rooms. Section references use `C001-S001` format. The workbook is a source-data companion, not a solver input; it retains source records rather than replacing them with the prepared instance. No correspondence key or original personal identifiers are distributed.

## Generating instances

The generator takes command-line parameters and writes instance JSON files.
It uses one procedure, with no calibration profiles or configuration files.
Output contains only restrictions, times, courses, teachers and students;
it does not include generator metadata, metrics or a generated timetable.

Generate one accepted BASE instance:

```powershell
.venv\Scripts\python.exe -m uctp.generation --output output/generated --groups BASE --count 1 --seed 1
```

Generate 50 accepted instances for each group:

```powershell
.venv\Scripts\python.exe -m uctp.generation --output output/generated_datasets --groups Dispersion-Low BASE Dispersion-High Offer-Low Offer-High --count 50 --seed 1
```

Read the generated files with `--input-format legacy`:

```powershell
.venv\Scripts\python.exe -m uctp --input output/generated/BASE/instance_001.json --input-format legacy --output output/generated_solution.json --iterations 200 --seed 703
```

| Parameter | Default |
| --- | --- |
| `--output` | Required destination; existing files are never overwritten |
| `--groups` | `BASE` |
| `--count` | `50` accepted instances per group |
| `--seed` | `1`, first candidate seed per group |
| `--max-candidates` | No limit; optional cap on candidates tested |
| `--construction-retries` | `20`, after the first attempt per strategy |
| `--teachers`, `--students` | `20`, `500` |
| `--courses-per-level` | `4 6 6 6 6 2` |
| `--student-load-mean`, `--student-load-sd` | `3.2`, `1.6`, before rounding/truncation to 0–9 |
| `--section-sd` | `0.8`, before rounding/truncation to 1–4 sections |
| `--weekly-lectures-mean`, `--weekly-lectures-sd` | `2.7`, `0.7`, before rounding/truncation to 2–4 lectures |
| `--level-decay` | `0.85` |
| `--teacher-availability` | `0.7`, probability that a block is available |
| `--preassignment-probability` | `0.10`, whole-course preassignment attempt probability |

Every student uses the curricular sampling SD of its group; there is no individual SD draw.
Teacher availability is drawn per block, with at least one available block retained per day.

Curricular sampling SD is **0.4 / 0.8 / 1.2** for Dispersion-Low / BASE /
Dispersion-High, and 0.8 for both offer groups. Section-count means are
1.5, except Offer-Low (1.2) and Offer-High (1.8). Each course's sections
share its weekly lecture count. Each course has one or two teachers.
With `N_days = 5` teaching days, each generated group uses one shared daily teacher limit: `floor(M / (N_days - 2)) + 1`, where M is the largest teacher weekly lecture count across all accepted instances in that group. The same limit applies to every teacher in every instance of the group. Candidate replacements can change M, so the group is rechecked before its files are saved. Complete construction feasibility is required.

Before saving an instance, construction tries Prioritization Only,
Compatibility Matrix and Random, in that order. It saves only complete,
institutionally feasible instances. Rejected candidates advance the seed;
therefore `--count 50` can test more than 50 seeds. Candidate seeds restart
from `--seed` for each group and are printed when the completed group is saved. A candidate cap leaves the current incomplete group unsaved; completed groups are retained. Keep the
command and console output when recording an experiment's provenance.

Synthetic datasets are not bundled. Use the generation commands above to create them locally. Read their JSON files with `--input-format legacy`; the adapter reads UTF-8 with optional BOM support and converts `eventos` and `recursos` into the internal schema. [Generation procedure](docs/GENERATION.md).

## Input data and solution format

The English input schema is illustrated in [examples/small.json](examples/small.json):

- `courses`: courses identified by `id`, containing `sections` and their `lectures`. Each lecture has an `id` and a `teachers` list of teacher identifiers.
- `teachers`: teachers identified by `id`. The optional `available_blocks` field defaults to all time blocks when omitted; an empty list means no availability. Individual institutional limits `max_days` and `max_daily_load` are optional; the latter is expressed in time blocks. When `max_days` is omitted, the limit is `N_days - 2` for up to six lectures and `N_days - 1` otherwise (`N_days = 5`).
- `students`: students with `id` and `courses`, in the order of their course requirements. Observed section assignments are not inputs to the SSP.
- `curricular_levels`: curricular levels with `id` and `courses`. Courses outside these levels are also scheduled.

Time blocks range from `Mon:1` to `Fri:7`, covering five days with seven nominal blocks per day. `Thu:4` is protected. A lecture may have a `preassigned_block`, such as `Mon:1`; the remaining lectures of its course are still scheduled. Optional `distinct_day` and `separated_day` flags constrain marked lectures within the same section to different days or to days separated by at least one intervening day, respectively.

Identifiers are strings and must be unique within their scope. List order is preserved and may affect tie-breaking. JSON files describe institutional data; operational search parameters are supplied through the command line.

Each SSP solution processes students sequentially in a newly randomized order, without replacement, using Python's random generator controlled by `--seed`. Backtracking and local refinement for one student finish before the next student is processed. Original student identifiers and indices remain unchanged. Requested courses are stably sorted by increasing section count for backtracking, and sections by increasing current enrollment; local refinement shuffles the student's course order.

To read another historical instance:

```powershell
.venv\Scripts\python.exe -m uctp --input output/generated/BASE/instance_001.json --input-format legacy --output output/base_001.json --iterations 200 --seed 703
```

The adapter recognizes historical `eventos` and `recursos` entities, predefined schedules, and day restrictions. It preserves the historical handling of repeated students and empty teacher availability. The global instance field `restricciones.MAX_CARGA_DIARIA` supplies a fallback in **time blocks per teacher per day**; a missing value defaults to `8`. A teacher record under `recursos.maestros` can specify its own positive integer `max_daily_load`. New generated instances use a common `MAX_CARGA_DIARIA` across each group, calculated from the highest teacher weekly load in that group; they do not export individual teacher daily limits. Legacy daily fields under `restricciones.PROFESORES` remain ignored. An explicit `--max-daily-load` overrides every legacy teacher limit. Preserve the limits stored in the included real instance: four blocks daily, with one approved six-block exception. The workload-dependent maximum-day limit remains active. Stored datasets keep their original limits; generating new instances does not rewrite them.

Simultaneous lecture capacity comes from `--max-block-load`; valid individual maximum-day limits are preserved. Historical time data must use the institutional five-day, seven-block format. Spanish entity keys are retained at the historical data boundary, including generator output; code identifiers and command names use English. The synthetic section field in that schema is retained as data; SSP builds its own section assignments from course requirements.

The result contains `X` (lectures, time blocks, and teachers), `Y` (students, courses, and sections), status, stopping reason, effective parameters, seed, versions, and input/source hashes. `initial_metrics` and `metrics` report quality before TS and for the returned solution; `diagnostics` reports construction/optimizer/extraction times, retries, accepted TS moves, search blocks and SSP calls. `--details` adds student and section records; `--trace` adds native objective observations at accepted moves. Input identifiers are preserved. A solution file does not replace the original instance or become a preassignment automatically.

The program refuses to overwrite an existing destination. Exit codes are `0` for a solved instance, `1` for exhausted construction attempts or incompatible preassignments, and `2` for invalid arguments or input. Construction failures are written as result records with null quality metrics and empty assignments. Unavoidable student conflicts do not imply an infeasible institutional timetable.

## Experiments and general analysis

Run any combination of instances, initializers, approaches, weights, seeds, LS ablations and budgets through `uctp.experiments`. For example, a small iteration-budget experiment:

```powershell
.venv\Scripts\python.exe -m uctp.experiments run --inputs examples/small.json --output output/research_demo --group Demo --strategies Random "Compatibility Matrix" --approaches Embedded Decoupled-2 --alphas 0.25 1 --local-search on off --seeds 703 704 --iteration-budgets 2 4 --details --save-solutions
```

Analyze the resulting runs, comparing approaches, weights, LS settings and budgets while holding the other recorded factors fixed:

```powershell
.venv\Scripts\python.exe -m uctp.experiments analyze --inputs output/research_demo --output output/research_tables --compare approach alpha local_search budget --unit run --bootstrap 1000
```

This example uses `--unit run` because it repeats seeds on one instance. For comparisons across synthetic instances, the default `--unit instance` aggregates repeated seeds within each instance before inference.

The exports include **raw CSV, full JSONL, long-format metric data, descriptives, success rates, paired values, factor contrasts and machine-readable analysis JSON**. Optional raw student, section and trace tables support further analysis. No images are generated.

See [docs/RESEARCH.md](docs/RESEARCH.md) for timed grids, resume, filters, custom CSV inputs, metric definitions, statistical assumptions and all output formats. The installed `uctp-experiments` command exposes the same interface.

## Project structure and checks

`domain.py` represents entities and decisions; `instance_io.py` and `legacy_input.py` handle data; `constraints.py` checks institutional feasibility; `affinity.py`, `compatibility.py`, and `construction.py` construct the timetable; `sectioning.py` solves the SSP; `student_cost.py` and `objective.py` provide the internal evaluations required by the search; `tabu.py` implements TS; `planner.py` coordinates the approaches; and `cli.py` exposes the command-line interface.

`generation/instance_generator.py` assembles synthetic candidates; `parameters.py` defines inputs and groups; `sampling.py`, `student_demand.py`, and `preassignments.py` handle their respective generation steps; and `generation/cli.py` accepts candidates and writes datasets.

`metrics.py` extracts endpoint values; `reporting.py` builds portable run records. `experiments/runner.py` executes grids; `records.py` handles raw formats; `analysis.py`, `comparisons.py`, and `statistics.py` produce general analysis tables.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover complete assignments, constraints, partial preassignments, retries, CM and affinity rules, SSP behavior with unavoidable conflicts and repeatability. Generator tests cover structure, seeded sampling and output protection. Research tests verify objective consistency, unchanged seeded decisions with recording, failure retention, iteration caps, resume, statistical units, exact tests and raw-data round trips. They do not constitute a reproduction of the full experimental study or a certificate of optimality.
