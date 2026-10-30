# Research execution and analysis

Run commands from the repository root after the README installation steps. Python **3.12.13** and NumPy **2.4.6** are the only runtime requirements. Execution settings are command-line arguments; generated manifests and result JSON files are output data, not configuration inputs.

## Collecting one run

The standard solver command now exports initial/final metrics and diagnostics. Add `--details` for student and section data, and `--trace` for native objective values at accepted TS moves:

```powershell
.venv\Scripts\python.exe -m uctp --input examples/small.json --output output/detailed_run.json --iterations 25 --seed 703 --group Demo --details --trace
```

`--local-search off` omits only the SSP refinement after backtracking. The default is `on`. All results identify this setting. Construction failure produces a record with status `construction_failed`, null quality metrics and empty assignments. Invalid arguments or malformed input return an error instead of a fabricated research result.

## Running a grid

Use files, dataset directories, or both in `--inputs`. Directories are searched recursively for instance JSON. Each instance's parent directory supplies its group label; `--group` overrides it. Historical datasets require `--input-format legacy`.

After generating the synthetic groups with the README command into `output/generated_datasets`, the following command is a **planning example**. Use `--dry-run` to inspect the number of conditions before launching a large experiment:

```powershell
.venv\Scripts\python.exe -m uctp.experiments run --inputs output/generated_datasets/BASE output/generated_datasets/Offer-Low --input-format legacy --output output/strategy_study --strategies Random "Prioritization Only" "Compatibility Matrix" --approaches Embedded Decoupled-50 --alphas 0.25 0.5 1 --local-search on off --seeds 703 704 705 --iteration-budgets 25 50 200 --dry-run
```

Remove `--dry-run` to execute it. Runs are sequential. `Decoupled-k` accepts any positive integer k. Iteration budgets cap the **total accepted TS moves across blocks**, including a shortened final block when necessary; failed neighborhoods and initial states do not count as accepted moves. Other stopping rules still apply.

For time budgets, replace `--iteration-budgets` with `--time-budgets`, for example:

```powershell
.venv\Scripts\python.exe -m uctp.experiments run --inputs examples/small.json --output output/timed_demo --approaches Embedded Decoupled-2 --alphas 1 --seeds 703 --time-budgets 0.1 0.2 --iteration-ceiling 1000000 --details --save-solutions
```

Time is measured after construction and includes SSP work required by optimization. Endpoint metric extraction and file export occur after optimization. Initial assignment capture is excluded from the reported optimizer time. The nominal target is checked at the algorithm's stopping checkpoints, so it can be exceeded. Timed runs also have a secondary total-iteration ceiling; `--iteration-ceiling` defaults to 1000000. Batch stagnation defaults to the total iteration cap; set `--stagnation` explicitly for a different limit. The single-run CLI retains its README defaults.

`--max-retries` defaults to 500 retries after the initial attempt. `--neighborhood-size`, `--max-daily-load` and `--max-block-load` have the same units as in the solver. With legacy inputs, daily load comes from the global `restricciones.MAX_CARGA_DIARIA` field (eight blocks if absent); `--max-daily-load` explicitly overrides it. Generated synthetic instances store the institutional limit of four blocks. The real English-schema instance stores four blocks per teacher with one six-block exception; preserve these stored values instead of imposing a uniform override. The exported `metadata.max_daily_load` records the effective uniform limit, and `teacher_daily_loads` records every teacher's limit. Record custom research labels using repeatable `--tag name=value`; analysis exposes them as `tag_name` fields. Instead of explicit `--seeds`, use `--runs N --master-seed S` to draw a reproducible list of unique seeds. Each instance and treatment uses that same seed list.

Each completed condition is saved immediately under `records/`, including failed constructions. Raw tables are exported when the grid finishes or is interrupted with Ctrl+C. To inspect partial runs, analyze the directory while the remaining grid is still running. `status.json` records completion, interruption or error. Resume with the same command plus `--resume`; changed inputs, code or arguments are rejected. Existing terminal failures are retained, not retried until they succeed. Use a new destination for a changed design.

`--save-solutions` retains initial and final X,Y in each batch record. Initial and final timetable hashes are always recorded for successful runs. `--details` retains student and section observations. These can be large; choose them when additional assignment-level calculations are needed. Initial and final scalar metrics are always recorded. `--trace` is optional and records the native current objective and within-block incumbent after each accepted move. It does not call SSP or evaluate quality again. Tracing adds recording overhead and is included as a comparison control. Trace objective values are compared at fixed alpha; use endpoint F_ref for comparisons across alpha.

## Raw formats and fields

| File | Content |
| --- | --- |
| `records/<run_id>.json` | One condition: status, inputs, hashes, seed, all effective solver parameters, initial/final metrics and diagnostics; optional assignments/details/trace |
| `manifest.json` | Grid specification, ordered seed list, input hashes and solver fingerprint |
| `raw.csv` | One flat row per attempted run; missing values remain blank |
| `raw.jsonl` | Full run records, one JSON object per line |
| `metrics_long.csv` | One numeric observation per run, stage and metric; includes final, initial and diagnostic values |
| `students.csv`, `sections.csv` | Optional raw endpoint details linked by run_id |
| `trace.csv` | Optional accepted-move native objective observations linked by run_id |

Core endpoint metrics:

| Field | Definition |
| --- | --- |
| `R_imb` | Mean population section-enrollment SD over multi-section courses divided by the corresponding mean reference SD (10% of each course's mean enrollment), with denominator floor 1e-6 |
| `R_dis` | Total modeled student inconvenience cost divided by four times the number of students with course requirements, with denominator floor 1e-6 |
| `F` | Native search objective: `R_imb ** alpha * R_dis` |
| `F_ref` | Common reference criterion: per-run `R_imb * R_dis`, independent of the optimization weight |
| `conflicts` | Sum over students of excess attendance records per nominal block: `len(blocks) - len(set(blocks))`; three simultaneous records contribute two conflicts, not three pairs |
| `imbalance_observed`, `imbalance_reference` | Unnormalized imbalance components |
| `dissatisfaction_total`, `dissatisfaction_reference` | Unnormalized dissatisfaction components |
| `students`, `active_students`, `courses`, `parallel_courses` | Entity counts for the evaluated instance |
| `requested_courses`, `assigned_courses`, `scheduled_lectures` | Coverage counts; course requests are student-course records |

Initial metrics describe the first SSP assignment before TS. Final metrics describe the returned best X,Y pair. Mean F_ref must be obtained by averaging the per-run products, not multiplying group means. Attendance retains the same lecture-record multiplicity and nominal-block interpretation as the solver. Raw student records include the attendance list and modeled inconvenience cost; raw section records include enrollment, course mean and population SD. No section capacities are inferred.

Diagnostics include `construction_seconds`, `optimizer_seconds`, `extraction_seconds`, `total_seconds`, `retries_used`, `construction_attempts`, `completed_iterations`, `search_blocks_completed` and `ssp_calls`. Total time includes loading, setup, metadata and extraction, but excludes writing the result file. Endpoint extraction does not consume Python/NumPy random draws. Recording cannot make time-budget executions independent of machine load.

## General analysis

Supply one or more result directories, JSON records, JSONL exports or flat CSV files. A directory containing `records/` reads those files, avoiding double-counting its raw exports. Equivalent run IDs are deduplicated; conflicting records are rejected. CSV inputs must contain `status` (`solved` for successful runs), requested numeric metrics and the fields used for grouping or pairing. Arbitrary additional scalar columns can be summarized or used as research factors. Missing pair identities must be supplied explicitly or analyzed in independent mode.

```powershell
.venv\Scripts\python.exe -m uctp.experiments analyze --inputs output/research_demo --output output/research_filtered --where alpha=1 --where local_search=on --compare approach budget --metrics F_ref R_imb R_dis conflicts optimizer_seconds completed_iterations --unit run --bootstrap 1000
```

Repeated `--where` filters are combined with AND; comma-separated values within one filter mean OR. Numeric comparisons normalize numeric representations. Boolean filters accept true/false or on/off. `--group-by` selects summary dimensions. Its default retains recorded experimental controls, including group, approach, alpha, initializer, LS, budgets, retries, source revision and institutional limits. Quality summaries use successful records; failures remain in raw data and success-rate tables. Add `--include-failed-metrics` to summarize nonmissing failed-run diagnostics, such as construction time. This does not create missing quality values.

`--compare` varies each requested factor separately while holding the other recorded controls fixed. It supports approach, alpha, strategy, local_search, budget, group, retries and custom tag fields. Native F and initial_F cannot be contrasted across alpha: use F_ref. For iteration-budget comparisons, the duplicate derived iteration-cap field is not held fixed; timed comparisons retain their secondary cap. `--strata` permits an explicit alternative set of fixed comparison dimensions. An empty contrast table means that selected records contain no comparable factor levels or no eligible successful observations.

### Matching and statistical units

Paired comparisons default to `--pair-by input_sha256 seed`. Duplicate pair keys are errors, rather than implicit averaging of incompatible conditions. Quality uses jointly successful pairs; reliability retains all matched attempts. Unmatched counts are exported. The same seed labels do not prove that every subsequent random draw was shared.

The default `--unit instance` first matches runs and then averages paired differences within each instance, assigning equal weight to instances. Summary confidence intervals similarly resample instance means. Repeated seeds therefore do not become additional independent institutions. `--unit run` analyzes run-to-run variability; use it for repeated seeds on a single fixed instance and interpret inference conditionally on that instance. With fewer than two units, no confidence interval or inferential p value is reported.

For groups containing different instances, choose independent comparisons rather than inventing pairings:

```powershell
.venv\Scripts\python.exe -m uctp.experiments analyze --inputs output/strategy_study --output output/group_comparison --where approach=Embedded --where alpha=1 --where local_search=on --compare group --comparison-mode independent --unit instance
```

This command assumes the earlier planned grid was executed. `--instance-field` and `--pair-by` can use alternative identity columns in custom CSV data. Independence/exchangeability assumptions remain the researcher's responsibility.

### Statistical tables

| File | Content |
| --- | --- |
| `summary.csv` | Unit-level mean, sample SD, median, quartiles, range and pointwise 95% bootstrap CI; run means/SDs and sample counts are also retained |
| `success_rates.csv` | Attempted, solved and failed run counts and success proportions |
| `contrasts.csv` | Factor labels, strata, sample counts, mean differences, pointwise CIs, tests and Holm-adjusted p values |
| `reliability.csv` | Paired binary success contrasts and a separate Holm family |
| `paired_values.csv` | Individual matched run values and differences used in paired comparisons |
| `analysis.json` | Nested machine-readable descriptives, contrasts and method specification |
| `METHODS.md` | Effective analysis arguments, units, missing-data rules and statistical definitions |

Paired quality tests are exact two-sided sign tests, excluding absolute differences <=1e-8. They test the direction of differences, not equality of means. Paired confidence intervals are percentile bootstrap intervals for mean unit differences. Independent comparisons use a mean-difference permutation test and independent bootstrap; permutations are exhaustive when their count fits within the requested replicate count, otherwise Monte Carlo with the plus-one correction. The default is 10000 replicates and analysis seed 20261026.

Holm correction is applied jointly to valid quality contrasts produced by one analysis invocation. Paired success tests form a separate family. Under `--unit instance`, success means at least one successful matched seed for that instance; under `--unit run`, it means that individual run succeeded. Reliability uses the exact discordant-pair test (McNemar). Success rates in `success_rates.csv` always remain per attempted run.

Bootstrap intervals are pointwise and not multiplicity-adjusted. Left-minus-right differences are explicit; negative is favorable for quality metrics but not automatically for throughput. A zero observed SD does not justify creating an infinite standardized effect. Post-selection results do not establish a universally optimal alpha. The raw exports support other analyses chosen by researchers; this tool does not silently fit factorial or mixed-effects models.
