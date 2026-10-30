"""Run every experiment of the manuscript from scratch, resumably and in parallel.

The campaigns are split into small shards (one synthetic instance or one seed of
the real instance per shard). Each shard is an ordinary ``uctp.experiments run``
grid with its own output folder, so it is resumable at the level of single
runs: if a session stops for any reason, only the run that was in progress is
repeated the next time (finished runs are written atomically and skipped).

Commands (from the repository root, with the project's Python):

    python tools/campaigns.py plan        # stages, runs, seeds and time estimates
    python tools/campaigns.py run         # run or continue (default: 4 workers, 8 h session)
    python tools/campaigns.py status      # progress per stage
    python tools/campaigns.py merge       # merged CSV tables per stage
    python tools/campaigns.py validate    # completeness and provenance report
    python tools/campaigns.py package     # zip for co-investigators

Order of ``run``: environment check, concurrency pilot (machine otherwise idle),
convergence pilot, then the time-budget campaigns (real instance first) and
finally the iteration-budget campaigns. Iteration-budget shards also act as
fillers so that the number of simultaneous processes stays constant while the
last time-budget shards of a session finish. A session stops launching new
time-budget shards after ``--max-hours`` (default 8) or when a file named STOP
exists in the results folder; running shards are allowed to finish.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
GROUPS = ["Dispersion-Low", "BASE", "Dispersion-High", "Offer-Low", "Offer-High"]
ALPHAS = ["0", "0.125", "0.25", "0.5", "1", "2", "4"]
SYNTHETIC_SEED = 703
FIVE_SEEDS = [703, 215303898, 1366270267, 664565471, 2054631744]
REAL_MASTER_SEED = 703
REAL_RUNS = 50
CURVE_SEEDS = 10
PILOT_INSTANCE_SEED = 5001
PILOT_INSTANCES = 6
PILOT_BUDGET = 3000
PILOT_CHECKPOINTS = [25, 50, 100, 200, 500, 1000, 1500, 2000, 2500]
DEFAULT_ITERATION_BUDGET = 2000
SHORT_BUDGETS = [25, 50, 100, 200, 500, 1000, 1500]
# Time estimates measured on the authors' Ryzen 7 7700X (single process).
SECONDS_PER_ITERATION = 0.013   # Embedded on calibrated synthetic instances
RUN_OVERHEAD = 0.6              # setup, construction and extraction per run
CONCURRENCY_PENALTY = 1.10
# Smoke test of the machinery only (tiny budgets, two instances per group); never for results.
SMOKE = False


# ----------------------------------------------------------------------------- model
@dataclass
class Shard:
    stage: str
    name: str
    kind: str                      # "time" or "iterations"
    args: list[str]
    runs: int
    seconds: float                 # single-process estimate
    output: Path = field(default=None)  # type: ignore[assignment]


@dataclass
class Stage:
    key: str
    folder: str
    title: str
    kind: str
    shards: list[Shard]
    description: str


def _real_seeds() -> list[int]:
    # Same rule as ``uctp.experiments run --runs 50 --master-seed 703``.
    return random.Random(REAL_MASTER_SEED).sample(range(1, 2**32), REAL_RUNS)


def _synthetic_files(data: Path) -> dict[str, list[Path]]:
    files = {g: sorted((data / g).glob("instance_*.json")) for g in GROUPS}
    return {g: v[:2] for g, v in files.items()} if SMOKE else files


def iteration_budget(results: Path, override: int | None) -> int:
    if override:
        return override
    path = results / "02_pilot_convergence" / "recommendation.json"
    if path.is_file():
        return int(json.loads(path.read_text(encoding="utf-8"))["recommended_budget"])
    return DEFAULT_ITERATION_BUDGET


def build_stages(results: Path, data: Path, budget: int) -> list[Stage]:
    real = str(ROOT / "datasets" / "Real" / "real.json")
    files = _synthetic_files(data)
    seeds = _real_seeds()[:2] if SMOKE else _real_seeds()
    sec = "1" if SMOKE else "30"
    curve = ["1", "2", "3"] if SMOKE else ["30", "120", "300"]
    pilot_budget = 60 if SMOKE else PILOT_BUDGET
    pilot_points = [5, 10, 20, 30, 40, 50] if SMOKE else PILOT_CHECKPOINTS
    stages: list[Stage] = []

    def synthetic(stage, folder, title, kind, extra, runs, seconds, groups=GROUPS, seed_args=None):
        shards = []
        for group in groups:
            for path in files[group]:
                args = ["--inputs", str(path), "--input-format", "legacy", "--group", group] + extra
                args += seed_args or ["--seeds", str(SYNTHETIC_SEED)]
                shards.append(Shard(stage, f"{group}/{path.stem}", kind, args, runs, seconds))
        return shards

    def per_real_seed(stage, kind, extra, runs, seconds, chosen):
        return [
            Shard(stage, f"seed_{seed}", kind,
                  ["--inputs", real, "--input-format", "instance", "--seeds", str(seed)] + extra,
                  runs, seconds)
            for seed in chosen
        ]

    # Pilot: convergence (independent instances, generated with another seed).
    pilot_data = results / "02_pilot_convergence" / "instances"
    pilot_shards = []
    for group in GROUPS:
        for index in range(1, (1 if SMOKE else PILOT_INSTANCES) + 1):
            path = pilot_data / group / f"instance_{index:03d}.json"
            args = ["--inputs", str(path), "--input-format", "legacy", "--group", group,
                    "--strategies", "Random", "Prioritization Only", "Compatibility Matrix",
                    "--approaches", "Embedded", "--alphas", "0.5", "1", "2",
                    "--seeds", str(SYNTHETIC_SEED), "--iteration-budgets", str(pilot_budget),
                    "--checkpoints", *map(str, pilot_points), "--trace"]
            pilot_shards.append(Shard("P2", f"{group}/instance_{index:03d}", "iterations", args, 9,
                                      9 * (PILOT_BUDGET * SECONDS_PER_ITERATION + RUN_OVERHEAD)))
    stages.append(Stage("P2", "02_pilot_convergence", "Pilot: iteration convergence", "iterations",
                        pilot_shards,
                        f"{PILOT_INSTANCES} independent instances per group (generator seed "
                        f"{PILOT_INSTANCE_SEED}), R/PO/CM, alpha 0.5/1/2, {PILOT_BUDGET} iterations; "
                        "recommends the maximum iteration budget."))

    t = lambda budget_s, runs: runs * (budget_s + RUN_OVERHEAD)  # noqa: E731
    stages.append(Stage("D", "D_real_30s", "Real instance, 30 s, 7 weights", "time",
                        per_real_seed("D", "time",
                                      ["--approaches", "Embedded", "Decoupled-50", "--alphas", *ALPHAS,
                                       "--time-budgets", sec, "--save-solutions"],
                                      14, t(30, 14), seeds),
                        f"Embedded and Decoupled-50, alpha {', '.join(ALPHAS)}, {REAL_RUNS} seeds."))
    stages.append(Stage("E", "E_real_quality_time", "Real instance, quality-time curve", "time",
                        per_real_seed("E", "time",
                                      ["--approaches", "Embedded", "Decoupled-50", "--alphas", "0", "0.25", "1",
                                       "--time-budgets", *curve, "--save-solutions", "--trace"],
                                      18, 6 * (450 + 3 * RUN_OVERHEAD), seeds[:CURVE_SEEDS]),
                        f"Embedded and Decoupled-50, alpha 0/0.25/1, 30/120/300 s, first {CURVE_SEEDS} seeds."))
    stages.append(Stage("F", "F_real_ls_ablation", "Real instance, LS ablation", "time",
                        per_real_seed("F", "time",
                                      ["--approaches", "Embedded", "--alphas", "0.25", "1",
                                       "--local-search", "on", "off", "--time-budgets", sec,
                                       "--save-solutions"],
                                      4, t(30, 4), seeds),
                        "Embedded, alpha 0.25/1, local search on/off, 30 s."))
    stages.append(Stage("C", "C_synthetic_30s", "Synthetic, approaches x weights, 30 s", "time",
                        synthetic("C", "C_synthetic_30s", "", "time",
                                  ["--approaches", "Embedded", "Decoupled-25", "Decoupled-50", "Decoupled-100",
                                   "--alphas", *ALPHAS, "--time-budgets", sec, "--save-solutions"],
                                  28, t(30, 28)),
                        "Embedded and Decoupled-25/50/100, 7 weights, CM with 500 retries, seed 703."))
    stages.append(Stage("G", "G_base_ls_ablation", "BASE, LS ablation", "time",
                        synthetic("G", "", "", "time",
                                  ["--approaches", "Embedded", "--alphas", "0.25", "1",
                                   "--local-search", "on", "off", "--time-budgets", sec, "--save-solutions"],
                                  4, t(30, 4), groups=["BASE"]),
                        "Embedded, alpha 0.25/1, local search on/off, 30 s, seed 703."))
    stages.append(Stage("H", "H_base_five_seeds", "BASE, five seeds", "time",
                        synthetic("H", "", "", "time",
                                  ["--approaches", "Embedded", "--alphas", "1", "--time-budgets", sec,
                                   "--save-solutions"],
                                  5, t(30, 5), groups=["BASE"],
                                  seed_args=["--seeds", *map(str, FIVE_SEEDS)]),
                        "Embedded, alpha 1, 30 s, seeds " + ", ".join(map(str, FIVE_SEEDS)) + "."))
    stages.append(Stage("A", "A_initial_construction", "Initial construction (R/PO/CM)", "iterations",
                        synthetic("A", "", "", "iterations",
                                  ["--strategies", "Random", "Prioritization Only", "Compatibility Matrix",
                                   "--approaches", "Embedded", "--alphas", "1", "--max-retries", "5000",
                                   "--iteration-budgets", "1"],
                                  3, 3 * (1.5 + SECONDS_PER_ITERATION)),
                        "R/PO/CM with up to 5000 retries (results for 100 and 500 retries follow from the "
                        "nested attempt counts), initial quality at alpha 1, seed 703."))
    checkpoints = [k for k in ([5, 10, 20] if SMOKE else SHORT_BUDGETS) if k < budget]
    stages.append(Stage("B", "B_iteration_budgets", f"Optimization up to {budget} iterations", "iterations",
                        synthetic("B", "", "", "iterations",
                                  ["--strategies", "Random", "Prioritization Only", "Compatibility Matrix",
                                   "--approaches", "Embedded", "--alphas", "0.5", "1", "2",
                                   "--iteration-budgets", str(budget), "--checkpoints", *map(str, checkpoints),
                                   "--save-solutions"],
                                  9, 9 * (budget * SECONDS_PER_ITERATION + RUN_OVERHEAD)),
                        f"R/PO/CM, alpha 0.5/1/2, {budget} iterations with exact checkpoints at "
                        f"{', '.join(map(str, checkpoints))}, 500 construction retries, seed 703."))
    for stage in stages:
        for shard in stage.shards:
            shard.output = results / stage.folder / "shards" / shard.name.replace("/", "__")
    return stages


# ----------------------------------------------------------------------------- helpers
def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log(results: Path, message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with (results / "session_log.txt").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def event(results: Path, **row) -> None:
    path = results / "progress.csv"
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["time", "event", "stage", "shard", "kind",
                                                    "active_total", "active_time", "returncode"])
        if new:
            writer.writeheader()
        writer.writerow({"time": now(), **row})


def shard_done(shard: Shard) -> bool:
    status = shard.output / "status.json"
    if not status.is_file():
        return False
    data = json.loads(status.read_text(encoding="utf-8"))
    return data.get("status") == "complete" and data.get("completed") == data.get("planned")


def shard_completed_runs(shard: Shard) -> int:
    records = shard.output / "records"
    return len(list(records.glob("*.json"))) if records.is_dir() else 0


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- data and environment
def prepare_data(data: Path) -> None:
    archive = ROOT / "datasets" / "Synthetic" / "synthetic_datasets.zip"
    sums = ROOT / "datasets" / "Synthetic" / "SHA256SUMS"
    if not (data / "BASE").is_dir():
        data.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                target = (data.parent / member.filename).resolve()
                if not str(target).startswith(str(data.parent.resolve())):
                    raise RuntimeError(f"Unsafe path in archive: {member.filename}")
            zf.extractall(data.parent)
    expected = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip().removeprefix("./")] = digest
    for name, digest in expected.items():
        if sha256(data / name) != digest:
            raise RuntimeError(f"Checksum mismatch: {name}. Delete {data} and run again.")


def environment(results: Path, workers: int) -> None:
    folder = results / "00_environment"
    folder.mkdir(parents=True, exist_ok=True)
    tests = subprocess.run([PY, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True)
    (folder / "tests.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    if tests.returncode != 0:
        raise RuntimeError(f"Tests failed; see {folder / 'tests.txt'}.")
    probe = subprocess.run(
        [PY, "-c", "import json,sys,numpy,numba;from uctp.reporting import solver_revision;"
                   "import uctp._ssp_kernel;"
                   "print(json.dumps(dict(python=sys.version.split()[0],numpy=numpy.__version__,"
                   "numba=numba.__version__,solver_revision=solver_revision())))"],
        cwd=ROOT, capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout.strip().splitlines()[-1])
    git = None
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True).stdout.strip() or None
    except OSError:
        pass
    info.update(
        recorded_at=now(), platform=platform.platform(), processor=platform.processor(),
        machine=platform.machine(), logical_cpus=os.cpu_count(), workers=workers, git_commit=git,
        tests=tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else "",
    )
    history = folder / "environment_history.jsonl"
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(info) + "\n")
    first = folder / "environment.json"
    if not first.exists():
        first.write_text(json.dumps(info, indent=2), encoding="utf-8")
    else:
        saved = json.loads(first.read_text(encoding="utf-8"))
        for key in ("python", "numpy", "numba", "solver_revision"):
            if saved[key] != info[key]:
                raise RuntimeError(
                    f"{key} changed since the first session ({saved[key]} -> {info[key]}). "
                    "All campaigns must use the same environment and code.")


# ----------------------------------------------------------------------------- pilots
def pilot_concurrency(results: Path, workers: int, seconds: int, levels: list[int]) -> dict:
    folder = results / "01_pilot_concurrency"
    result_path = folder / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    folder.mkdir(parents=True, exist_ok=True)
    log(results, f"Concurrency pilot: levels {levels}, {seconds} s each. Keep the computer otherwise idle.")
    real = ROOT / "datasets" / "Real" / "real.json"
    rows = []
    for level in levels:
        procs = []
        for copy in range(1, level + 1):
            out = folder / f"level{level}_copy{copy}.json"
            if out.exists():
                out.unlink()
            procs.append(subprocess.Popen(
                [PY, "-m", "uctp", "--input", str(real), "--output", str(out), "--approach", "Embedded",
                 "--alpha", "1", "--time-limit", str(seconds), "--iterations", "1000000",
                 "--max-stagnant-iterations", "1000000", "--seed", str(SYNTHETIC_SEED)],
                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for proc in procs:
            proc.wait()
        its = [json.loads((folder / f"level{level}_copy{c}.json").read_text(encoding="utf-8"))
               ["diagnostics"]["completed_iterations"] for c in range(1, level + 1)]
        rows.append(dict(processes=level, iterations=its, mean=statistics.mean(its)))
        log(results, f"  {level} process(es): mean {statistics.mean(its):.0f} iterations {its}")
    base = rows[0]["mean"]
    for row in rows:
        row["loss_percent"] = round(100 * (1 - row["mean"] / base), 1)
    acceptable = [row["processes"] for row in rows if row["loss_percent"] <= 15]
    result = dict(recorded_at=now(), seconds=seconds, rows=rows, threshold_percent=15,
                  recommended_workers=max(acceptable) if acceptable else 1)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def generate_pilot_instances(results: Path) -> None:
    target = results / "02_pilot_convergence" / "instances"
    count = 1 if SMOKE else PILOT_INSTANCES
    if all((target / g / f"instance_{count:03d}.json").is_file() for g in GROUPS):
        return
    if target.exists():
        shutil.rmtree(target)
    subprocess.run([PY, "-m", "uctp.generation", "--output", str(target), "--groups", *GROUPS,
                    "--count", str(count), "--seed", str(PILOT_INSTANCE_SEED)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def summarize_pilot(results: Path, stage: Stage) -> None:
    folder = results / stage.folder
    pilot_budget = 60 if SMOKE else PILOT_BUDGET
    candidates = [20, 30, 40, 50] if SMOKE else [500, 1000, 1500, 2000, 2500]
    rows = []
    for shard in stage.shards:
        for record_path in sorted((shard.output / "records").glob("*.json")):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("status") != "solved":
                continue
            values = {c["iteration_budget"]: c["metrics"]["F"] for c in record["checkpoints"]}
            values[pilot_budget] = record["metrics"]["F"]
            trace = record.get("trace", [])
            best, last = float("inf"), 0
            for step in trace:
                if step["current_F"] < best - 1e-12:
                    best, last = step["current_F"], step["iteration"]
            rows.append(dict(values=values, last=last, alpha=record["metadata"]["parameters"]["alpha"]))
    if not rows:
        raise RuntimeError("Convergence pilot produced no solved runs.")

    def gain(a, b):
        return [100 * (r["values"][a] - r["values"][b]) / r["values"][a] for r in rows if r["values"][a] > 0]

    def q(values, p):
        values = sorted(values)
        return values[min(len(values) - 1, int(p * (len(values) - 1) + 0.5))]

    lines = [f"# Convergence pilot ({len(rows)} solved runs)", "",
             "Median and quartiles of the relative improvement of F between budgets (%).", "",
             f"| From | To {pilot_budget} median | p75 | share of runs improving > 1 % |", "|---|---|---|---|"]
    recommended = None
    for k in candidates:
        g = gain(k, pilot_budget)
        share = 100 * sum(x > 1 for x in g) / len(g)
        lines.append(f"| {k} | {statistics.median(g):.2f} | {q(g, 0.75):.2f} | {share:.1f} |")
        if recommended is None and statistics.median(g) <= 1.0 and q(g, 0.75) <= 5.0:
            recommended = k
    recommended = recommended or pilot_budget
    lasts = [r["last"] for r in rows]
    lines += ["", f"Iteration of the last improvement: median {statistics.median(lasts):.0f}, "
                  f"p75 {q(lasts, 0.75):.0f}, p90 {q(lasts, 0.9):.0f}.", "",
              "Rule: the smallest budget whose median further improvement up to "
              f"{pilot_budget} iterations is at most 1 % and whose 75th percentile is at most 5 %.",
              "", f"**Recommended maximum iteration budget: {recommended}.**"]
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "recommendation.json").write_text(
        json.dumps(dict(recommended_budget=recommended, solved_runs=len(rows), recorded_at=now()), indent=2),
        encoding="utf-8")
    log(results, f"Convergence pilot: recommended maximum iteration budget {recommended} "
                 f"(see {folder / 'summary.md'}).")


# ----------------------------------------------------------------------------- scheduler
def run_shards(results: Path, shards: list[Shard], workers: int, deadline: float) -> bool:
    """Run pending shards; return True when all of them are complete."""
    pending_time = [s for s in shards if s.kind == "time" and not shard_done(s)]
    pending_iter = [s for s in shards if s.kind == "iterations" and not shard_done(s)]
    running: dict[subprocess.Popen, Shard] = {}
    failures = 0
    stop_file = results / "STOP"
    try:
        while pending_time or pending_iter or running:
            draining = time.time() >= deadline or stop_file.exists()
            active_time = sum(1 for s in running.values() if s.kind == "time")
            while len(running) < workers:
                if pending_time and not draining:
                    shard = pending_time.pop(0)
                elif pending_iter and (not draining or active_time > 0):
                    shard = pending_iter.pop(0)
                else:
                    break
                shard.output.parent.mkdir(parents=True, exist_ok=True)
                stream = (shard.output.parent / f"{shard.output.name}.log.txt").open("a", encoding="utf-8")
                stream.write(f"\n=== {now()} start ===\n")
                stream.flush()
                proc = subprocess.Popen(
                    [PY, "-m", "uctp.experiments", "run", *shard.args, "--output", str(shard.output),
                     "--resume"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                proc._uctp_stream = stream  # type: ignore[attr-defined]
                running[proc] = shard
                active_time = sum(1 for s in running.values() if s.kind == "time")
                event(results, event="start", stage=shard.stage, shard=shard.name, kind=shard.kind,
                      active_total=len(running), active_time=active_time, returncode="")
            if not running:
                break
            time.sleep(2)
            for proc in [p for p in running if p.poll() is not None]:
                shard = running.pop(proc)
                proc._uctp_stream.close()  # type: ignore[attr-defined]
                event(results, event="end", stage=shard.stage, shard=shard.name, kind=shard.kind,
                      active_total=len(running), active_time=sum(1 for s in running.values() if s.kind == "time"),
                      returncode=proc.returncode)
                if proc.returncode != 0:
                    failures += 1
                    log(results, f"Shard failed (exit {proc.returncode}): {shard.stage} {shard.name}; "
                                 f"see {shard.output.name}.log.txt. It will be retried next session.")
    except KeyboardInterrupt:
        log(results, "Interrupted: stopping running shards. Finished runs are kept; the runs in "
                     "progress will be repeated next time.")
        for proc in running:
            proc.terminate()
        for proc in running:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise
    return all(shard_done(s) for s in shards) and failures == 0


# ----------------------------------------------------------------------------- merge / validate
def merge_stage(results: Path, stage: Stage) -> int:
    sys.path.insert(0, str(ROOT))
    from uctp.experiments.records import export_raw

    records = []
    for shard in stage.shards:
        for path in sorted((shard.output / "records").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            for key in ("X", "Y", "initial_solution"):
                record.pop(key, None)
            records.append(record)
    if not records:
        return 0
    merged = results / stage.folder / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    export_raw(merged, records)
    return len(records)


def status_rows(stages: list[Stage]) -> list[dict]:
    rows = []
    for stage in stages:
        planned = sum(s.runs for s in stage.shards)
        done = sum(min(shard_completed_runs(s), s.runs) for s in stage.shards)
        remaining = sum(s.seconds * (1 - min(shard_completed_runs(s), s.runs) / s.runs) for s in stage.shards)
        rows.append(dict(stage=stage.key, title=stage.title, kind=stage.kind, shards=len(stage.shards),
                         planned=planned, done=done, remaining_seconds=remaining,
                         total_seconds=sum(s.seconds for s in stage.shards)))
    return rows


def validate(results: Path, stages: list[Stage], data: Path) -> bool:
    lines = [f"# Validation report ({now()})", ""]
    ok = True
    env = results / "00_environment" / "environment.json"
    if env.is_file():
        info = json.loads(env.read_text(encoding="utf-8"))
        lines += [f"Environment: Python {info['python']}, NumPy {info['numpy']}, Numba {info['numba']}, "
                  f"solver revision `{info['solver_revision'][:16]}`, {info['platform']}, "
                  f"{info['logical_cpus']} logical CPUs, workers {info['workers']}.", ""]
    expected = {}
    for line in (ROOT / "datasets" / "Synthetic" / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[digest] = name.strip().removeprefix("./")
    expected[sha256(ROOT / "datasets" / "Real" / "real.json")] = "Real/real.json"
    pilot_dir = results / "02_pilot_convergence" / "instances"
    pilot_hashes = {sha256(p) for p in pilot_dir.rglob("*.json")} if pilot_dir.is_dir() else set()
    real_seeds = set(_real_seeds())
    lines += ["| Stage | Planned runs | Recorded | Solved | Construction failed | Other | Revisions | Problems |",
              "|---|---|---|---|---|---|---|---|"]
    for stage in stages:
        planned = sum(s.runs for s in stage.shards)
        counts = {"solved": 0, "construction_failed": 0, "other": 0}
        revisions, problems, seen = set(), [], set()
        for shard in stage.shards:
            for path in sorted((shard.output / "records").glob("*.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                meta = record["metadata"]
                status = record.get("status")
                counts[status if status in counts else "other"] += 1
                revisions.add(meta["solver_revision"])
                key = meta["run_id"]
                if key in seen:
                    problems.append(f"duplicate run {key}")
                seen.add(key)
                digest = meta["input_sha256"]
                if stage.key == "P2":
                    if digest not in pilot_hashes:
                        problems.append(f"unknown pilot input {meta['instance']}")
                elif digest not in expected:
                    problems.append(f"input not in SHA256SUMS: {meta['instance']}")
                limit = meta["parameters"].get("time_limit_seconds")
                spent = (record.get("diagnostics") or {}).get("optimizer_seconds")
                if limit and spent and spent > limit + 10:
                    problems.append(f"run {key} used {spent:.0f} s for a {limit:.0f} s budget "
                                    "(computer asleep or overloaded?)")
                seed = meta["parameters"]["seed"]
                if stage.key in {"D", "E", "F"} and seed not in real_seeds:
                    problems.append(f"unexpected seed {seed}")
        recorded = sum(counts.values())
        if recorded != planned or problems or len(revisions) > 1:
            ok = False
        lines.append(f"| {stage.key} {stage.title} | {planned} | {recorded} | {counts['solved']} | "
                     f"{counts['construction_failed']} | {counts['other']} | {len(revisions)} | "
                     f"{'; '.join(sorted(set(problems))[:5]) or 'none'} |")
    progress = results / "progress.csv"
    if progress.is_file():
        with progress.open(encoding="utf-8") as stream:
            starts = [r for r in csv.DictReader(stream) if r["event"] == "start" and r["kind"] == "time"]
        if starts:
            loads = [int(r["active_total"]) for r in starts]
            lines += ["", f"Time-budget shards started with {min(loads)}–{max(loads)} simultaneous processes "
                          f"(median {statistics.median(loads):.0f}); see progress.csv."]
    lines += ["", "Overall: " + ("complete and consistent." if ok else "INCOMPLETE or inconsistent (see table).")]
    (results / "VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return ok


def write_plan(results: Path, stages: list[Stage], workers: int, budget: int) -> None:
    seeds = _real_seeds()
    seeds_doc = dict(
        synthetic_instances=dict(archive="datasets/Synthetic/synthetic_datasets.zip",
                                 generator_command="python -m uctp.generation --output <dir> --groups "
                                 + " ".join(GROUPS) + " --count 50 --seed 1"),
        synthetic_run_seed=SYNTHETIC_SEED,
        base_five_seeds=FIVE_SEEDS,
        real_seeds=dict(rule=f"random.Random({REAL_MASTER_SEED}).sample(range(1, 2**32), {REAL_RUNS})",
                        values=seeds),
        real_quality_time_seeds=seeds[:CURVE_SEEDS],
        pilot=dict(instance_generator_seed=PILOT_INSTANCE_SEED, instances_per_group=PILOT_INSTANCES,
                   run_seed=SYNTHETIC_SEED, concurrency_test_seed=SYNTHETIC_SEED),
        iteration_budget=budget,
    )
    (results / "SEEDS.json").write_text(json.dumps(seeds_doc, indent=2), encoding="utf-8")
    rows = status_rows(stages)
    total = sum(r["total_seconds"] for r in rows)
    lines = ["# Experiment plan", "",
             f"Workers: {workers}. Estimates are for the authors' Ryzen 7 7700X; parallel time includes a "
             f"{int((CONCURRENCY_PENALTY - 1) * 100)} % allowance.", "",
             "| Order | Stage | Description | Shards | Runs | Hours (1 process) | Hours (parallel) |",
             "|---|---|---|---|---|---|---|",
             "| 0 | Environment | tests, versions, Numba warm-up | – | – | 0.0 | 0.0 |",
             "| 1 | P1 Pilot: concurrency | same 30 s run alone and with 4 and 6 copies | – | 11 | 0.0 | 0.03 |"]
    for order, (stage, row) in enumerate(zip(stages, rows), start=2):
        lines.append(f"| {order} | {stage.key} {stage.title} | {stage.description} | {row['shards']} | "
                     f"{row['planned']} | {hours(row['total_seconds'])} | "
                     f"{hours(row['total_seconds'] * CONCURRENCY_PENALTY / workers)} |")
    parallel = total * CONCURRENCY_PENALTY / workers
    lines += ["", f"Total: {hours(total)} h with one process; about {hours(parallel)} h with {workers} "
                  f"processes, i.e. {int(parallel // (8 * 3600)) + 1} sessions of up to 8 h.", "",
              "Seeds and datasets: SEEDS.json. Each session continues where the previous one stopped."]
    (results / "PLAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ----------------------------------------------------------------------------- commands
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["plan", "concurrency", "run", "status", "merge", "validate", "package"])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--data", type=Path, default=ROOT / "work" / "Synthetic",
                        help="Folder where the bundled synthetic datasets are extracted.")
    parser.add_argument("--workers", type=int, help="Simultaneous processes (default: pilot result or 4).")
    parser.add_argument("--max-hours", type=float, default=8.0,
                        help="Stop launching time-budget shards after this many hours (default 8).")
    parser.add_argument("--only", nargs="+", help="Run only these stages (e.g. D E).")
    parser.add_argument("--iteration-budget", type=int, help="Override the pilot's recommended budget.")
    parser.add_argument("--skip-concurrency-pilot", action="store_true")
    parser.add_argument("--concurrency-seconds", type=int, default=30)
    parser.add_argument("--force", action="store_true",
                        help="concurrency: repeat the test even if a result already exists.")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.smoke:
        global SMOKE
        SMOKE = True
        if args.results == ROOT / "results":
            args.results = ROOT / "results_smoke"
    results: Path = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    data: Path = args.data.resolve()
    prepare_data(data)
    concurrency = results / "01_pilot_concurrency" / "result.json"
    workers = args.workers or (json.loads(concurrency.read_text(encoding="utf-8"))["recommended_workers"]
                               if concurrency.is_file() else 4)
    workers = max(1, min(workers, 6))
    if args.workers is None and concurrency.is_file():
        workers = min(workers, 4)
    budget = iteration_budget(results, args.iteration_budget)
    stages = build_stages(results, data, budget)
    if args.only:
        stages = [s for s in stages if s.key in set(args.only)]

    if args.command == "plan":
        write_plan(results, build_stages(results, data, budget), workers, budget)
        return 0
    if args.command == "concurrency":
        environment(results, workers)
        if args.force and concurrency.is_file():
            concurrency.unlink()
        result = pilot_concurrency(results, workers, 3 if SMOKE else args.concurrency_seconds,
                                   [1, 2] if SMOKE else [1, 4, 6])
        print("\nProcesses  Mean iterations  Loss vs alone")
        for row in result["rows"]:
            print(f"{row['processes']:9d}  {row['mean']:15.0f}  {row['loss_percent']:12.1f} %")
        chosen = min(result["recommended_workers"], 4)
        print(f"\nUp to {result['recommended_workers']} processes lose at most {result['threshold_percent']} %; "
              f"run_experiments will use {chosen} workers (override with -Workers). "
              f"Saved in {concurrency}.")
        return 0
    if args.command == "status":
        for row in status_rows(stages):
            print(f"{row['stage']:3s} {row['title']:45s} {row['done']:6d}/{row['planned']:<6d} runs  "
                  f"remaining ~{hours(row['remaining_seconds'] * CONCURRENCY_PENALTY / workers)} h "
                  f"with {workers} workers")
        return 0
    if args.command == "merge":
        for stage in stages:
            print(f"{stage.key}: merged {merge_stage(results, stage)} records")
        return 0
    if args.command == "validate":
        return 0 if validate(results, build_stages(results, data, budget), data) else 1
    if args.command == "package":
        for stage in stages:
            merge_stage(results, stage)
        validate(results, stages, data)
        name = results.parent / f"results_package_{dt.date.today().isoformat()}.zip"
        with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(results.rglob("*")):
                if path.is_file() and path.name != "STOP":
                    zf.write(path, path.relative_to(results.parent))
            for extra in [ROOT / "datasets" / "Synthetic" / "SHA256SUMS",
                          ROOT / "datasets" / "Synthetic" / "generation_log.txt"]:
                zf.write(extra, extra.relative_to(ROOT))
        print(f"Package written: {name}")
        return 0

    # run
    if os.name == "nt":
        # Keep Windows awake while the session runs: a sleeping computer would corrupt
        # the wall-clock budgets of the runs in progress.
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    started = time.time()
    deadline = started + args.max_hours * 3600
    (results / "STOP").unlink(missing_ok=True)
    log(results, f"Session start: workers {workers}, maximum {args.max_hours} h, results in {results}.")
    environment(results, workers)
    if not args.only and not args.skip_concurrency_pilot and not concurrency.is_file():
        result = pilot_concurrency(results, workers, 3 if SMOKE else args.concurrency_seconds,
                                   [1, 2] if SMOKE else [1, 4, 6])
        log(results, f"Concurrency pilot: up to {result['recommended_workers']} processes lose at most 15 %.")
        if args.workers is None:
            workers = min(result["recommended_workers"], 4)
            log(results, f"Using {workers} workers.")
    write_plan(results, build_stages(results, data, budget), workers, budget)
    pilot = next((s for s in stages if s.key == "P2"), None)
    if pilot is not None and not (results / pilot.folder / "recommendation.json").is_file():
        generate_pilot_instances(results)
        log(results, "Convergence pilot running.")
        if not run_shards(results, pilot.shards, workers, deadline):
            log(results, "Session ended before the convergence pilot finished. Run again to continue.")
            return 2
        summarize_pilot(results, pilot)
        if args.iteration_budget is None:
            budget = iteration_budget(results, None)
            stages = build_stages(results, data, budget)
            if args.only:
                stages = [s for s in stages if s.key in set(args.only)]
            write_plan(results, stages, workers, budget)
    campaign = [s for s in stages if s.key != "P2"]
    shards = [shard for stage in campaign for shard in stage.shards]
    complete = run_shards(results, shards, workers, deadline)
    for stage in stages:
        merge_stage(results, stage)
    elapsed = (time.time() - started) / 3600
    if complete:
        validate(results, build_stages(results, data, budget), data)
        log(results, f"All campaigns complete ({elapsed:.1f} h this session). Next: python tools/campaigns.py package")
        return 0
    remaining = sum(r["remaining_seconds"] for r in status_rows(campaign)) * CONCURRENCY_PENALTY / workers
    log(results, f"Session finished after {elapsed:.1f} h. Remaining about {hours(remaining)} h. "
                 "Run the same command again to continue.")
    return 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Finished runs are saved; run the same command again to continue.", flush=True)
        sys.exit(130)
