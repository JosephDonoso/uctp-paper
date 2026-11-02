"""Experiments of the v7 revision (E1c, E2, E3, E5, E6), resumably and in parallel.

Same machinery as ``tools/campaigns.py`` (shards, four processes, sessions of at
most eight hours, STOP file, Ctrl+C), in a separate results folder
``results_v7`` because the solver code differs from the published campaigns.

Round 1 (run first):
  T    SSP timing and scaling (E6), alone on the idle computer.
  P3   Pilot: Decoupled-50 convergence up to 3000 iterations (30 pilot instances,
       alpha 0.25 and 1); recommends its iteration budget with the rule of P2.
  E1c  Common random numbers (CRN) versus the published search: Embedded on the
       50 BASE instances and 20 real seeds, alpha 0.25 and 1, 2500 iterations
       (exact checkpoints) and 30 s.

Round 2 (after round 1 has been analysed; needs results_v7/decision.json,
written by the authors, e.g. {"crn": true}):
  E5a  Real instance, Embedded and Decoupled-50, alpha 0/0.25/1, 50 seeds.
  E5b  Synthetic, Embedded and Decoupled-50 at the seven weights; Decoupled-1
       and Decoupled-5 at alpha 0/0.25/1.
  E2a  Trajectories (R_imb and R_dis per iteration), 5 real seeds and 5 BASE
       instances, alpha 0.25 and 1.
  E2c  Alpha 1 with 200 neighbours per iteration, 10 real seeds.
  E3   Sequential baseline (proxy objective, one SSP at the end), 250 synthetic
       instances and 50 real seeds, 30 s and the iteration budget.
  E2b  Alpha 1 started from the final alpha-0.25 timetables of E5a (10 seeds)
       and E5b (50 BASE instances); runs after E5a and E5b.

All iteration budgets are B* = max(2500, recommendation of P3), with exact
checkpoints at 25-2000; every final timetable and checkpoint is re-evaluated
with 50 fresh SSP orders (``--reevaluate 50``).

Usage (repository root, project Python):
    python tools/campaigns_v7.py run --round 1
    python tools/campaigns_v7.py status --round 1
    python tools/campaigns_v7.py run --round 2
    python tools/campaigns_v7.py package
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaigns as base  # noqa: E402
from campaigns import (  # noqa: E402
    GROUPS, PY, ROOT, Shard, Stage, environment, hours, log, merge_stage, prepare_data,
    run_shards, shard_completed_runs, shard_done, status_rows,
)

ALPHAS = ["0", "0.125", "0.25", "0.5", "1", "2", "4"]
SEED = 703
CHECKPOINTS = [25, 50, 100, 200, 500, 1000, 1500, 2000]
REEVAL = ["--reevaluate", "50"]
E1C_REAL_SEEDS = 20
E2_REAL_SEEDS = 5
E2_BASE_INSTANCES = 5
E2B_REAL_SEEDS = 10
E2C_REAL_SEEDS = 10
BASE_BUDGET = 2500
# Rough single-process seconds per run, for planning only.
SEC = {"emb_real": 110, "emb_syn": 45, "dec_real": 40, "dec_syn": 18, "small": 8}
SMOKE = False  # machinery test only: tiny budgets, few instances; never for results


def real_seeds() -> list[int]:
    return base._real_seeds()


def budget(results: Path) -> int:
    if SMOKE:
        return 40
    path = results / "P3_pilot_decoupled" / "recommendation.json"
    recommended = 0
    if path.is_file():
        recommended = int(json.loads(path.read_text(encoding="utf-8"))["recommended_budget"])
    return max(BASE_BUDGET, recommended)


def decision(results: Path) -> dict | None:
    path = results / "decision.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def build(results: Path, data: Path, round_: int) -> list[Stage]:
    real = str(ROOT / "datasets" / "Real" / "real.json")
    files = {g: sorted((data / g).glob("instance_*.json"))[: 1 if SMOKE else None] for g in GROUPS}
    seeds = real_seeds()[: 2 if SMOKE else None]
    sec = "2" if SMOKE else "30"
    pilot_budget = 60 if SMOKE else base.PILOT_BUDGET
    pilot_points = [10, 20, 30, 40, 50] if SMOKE else base.PILOT_CHECKPOINTS
    stages: list[Stage] = []

    def syn(stage, group, path, kind, extra, runs, seconds, seed_args=None):
        args = ["--inputs", str(path), "--input-format", "legacy", "--group", group] + extra
        args += seed_args or ["--seeds", str(SEED)]
        return Shard(stage, f"{group}/{path.stem}", kind, args, runs, seconds)

    def realshard(stage, seed, kind, extra, runs, seconds, suffix=""):
        return Shard(stage, f"seed_{seed}{suffix}", kind,
                     ["--inputs", real, "--input-format", "instance", "--seeds", str(seed)] + extra,
                     runs, seconds)

    def iters(b, points=True):
        return ["--iteration-budgets", str(b)] + (["--checkpoints", *map(str, [k for k in CHECKPOINTS if k < b])]
                                                  if points else [])

    if round_ == 1:
        pilot_dir = results / "02_pilot_convergence" / "instances"
        shards = []
        for group in GROUPS:
            for index in range(1, (1 if SMOKE else base.PILOT_INSTANCES) + 1):
                path = pilot_dir / group / f"instance_{index:03d}.json"
                shards.append(syn("P3", group, path, "iterations",
                                  ["--approaches", "Decoupled-50", "--alphas", "0.25", "1",
                                   "--iteration-budgets", str(pilot_budget),
                                   "--checkpoints", *map(str, pilot_points), "--trace"],
                                  2, 2 * 25))
        stages.append(Stage("P3", "P3_pilot_decoupled", "Pilot: Decoupled-50 convergence", "iterations",
                            shards, "Decoupled-50, alpha 0.25/1, 3000 iterations, 30 pilot instances."))
        shards = []
        for crn in (False, True):
            flag = ["--common-random-numbers"] if crn else []
            tag = "crn" if crn else "std"
            common = ["--approaches", "Embedded", "--alphas", "0.25", "1", "--save-solutions"] + REEVAL + flag
            for path in files["BASE"]:
                s = syn("E1c", "BASE", path, "iterations", common + iters(budget(results) if SMOKE else BASE_BUDGET),
                        2, 2 * SEC["emb_syn"])
                s.name += f"__{tag}_it"
                shards.append(s)
                s = syn("E1c", "BASE", path, "time", common + ["--time-budgets", sec], 2, 2 * 32)
                s.name += f"__{tag}_30s"
                shards.append(s)
            for seed in seeds[:E1C_REAL_SEEDS]:  # noqa: B007
                shards.append(realshard("E1c", seed, "iterations",
                                        common + iters(budget(results) if SMOKE else BASE_BUDGET), 2,
                                        2 * SEC["emb_real"], f"__{tag}_it"))
                shards.append(realshard("E1c", seed, "time", common + ["--time-budgets", sec], 2, 2 * 32,
                                        f"__{tag}_30s"))
        stages.append(Stage("E1c", "E1c_crn", "Common random numbers vs. standard", "mixed", shards,
                            "Embedded, BASE (50) and real (20 seeds), alpha 0.25/1, 2500 iterations and 30 s, "
                            "with and without common random numbers."))
    else:
        d = decision(results)
        if d is None:
            raise SystemExit("Round 2 needs results_v7/decision.json (written after analysing round 1).")
        b = budget(results)
        crn = ["--common-random-numbers"] if d.get("crn") else []
        save = ["--save-solutions"] + REEVAL
        shards = []
        for seed in seeds:
            shards.append(realshard("E5a", seed, "iterations",
                                    ["--approaches", "Embedded", "--alphas", "0", "0.25", "1"] + crn + save + iters(b),
                                    3, 3 * SEC["emb_real"] * b / 2500, "__emb"))
            shards.append(realshard("E5a", seed, "iterations",
                                    ["--approaches", "Decoupled-50", "--alphas", "0", "0.25", "1"] + save + iters(b),
                                    3, 3 * SEC["dec_real"] * b / 2500, "__d50"))
        stages.append(Stage("E5a", "E5a_real_converged", f"Real, {b} iterations", "iterations", shards,
                            "Embedded and Decoupled-50, alpha 0/0.25/1, 50 seeds."))
        shards = []
        for group in GROUPS:
            for path in files[group]:
                s = syn("E5b", group, path, "iterations",
                        ["--approaches", "Embedded", "--alphas", *ALPHAS] + crn + save + iters(b),
                        7, 7 * SEC["emb_syn"] * b / 2500)
                s.name += "__emb"
                shards.append(s)
                s = syn("E5b", group, path, "iterations",
                        ["--approaches", "Decoupled-50", "--alphas", *ALPHAS] + save + iters(b),
                        7, 7 * SEC["dec_syn"] * b / 2500)
                s.name += "__d50"
                shards.append(s)
                s = syn("E5b", group, path, "iterations",
                        ["--approaches", "Decoupled-1", "Decoupled-5", "--alphas", "0", "0.25", "1"] + save + iters(b),
                        6, 6 * SEC["dec_syn"] * b / 2500)
                s.name += "__d1d5"
                shards.append(s)
        stages.append(Stage("E5b", "E5b_synthetic_converged", f"Synthetic, {b} iterations", "iterations",
                            shards, "Embedded and Decoupled-50 at 7 weights; Decoupled-1/5 at 0/0.25/1."))
        shards = [realshard("E2a", seed, "iterations",
                            ["--approaches", "Embedded", "--alphas", "0.25", "1", "--trace", "--trace-components"]
                            + crn + save + iters(b, False), 2, 2 * SEC["emb_real"] * b / 2500)
                  for seed in seeds[:E2_REAL_SEEDS]]
        shards += [syn("E2a", "BASE", path, "iterations",
                       ["--approaches", "Embedded", "--alphas", "0.25", "1", "--trace", "--trace-components"]
                       + crn + save + iters(b, False), 2, 2 * SEC["emb_syn"] * b / 2500)
                   for path in files["BASE"][:E2_BASE_INSTANCES]]
        stages.append(Stage("E2a", "E2a_trajectories", "Alpha diagnosis: trajectories", "iterations", shards,
                            "Embedded, alpha 0.25/1, per-iteration R_imb and R_dis."))
        shards = [realshard("E2c", seed, "iterations",
                            ["--approaches", "Embedded", "--alphas", "1", "--neighborhood-size", "200"]
                            + crn + save + iters(b), 1, 4 * SEC["emb_real"] * b / 2500)
                  for seed in seeds[:E2C_REAL_SEEDS]]
        stages.append(Stage("E2c", "E2c_neighborhood_200", "Alpha diagnosis: 200 neighbours", "iterations",
                            shards, "Embedded, alpha 1, 200 neighbours, 10 real seeds."))
        shards = []
        for group in GROUPS:
            for path in files[group]:
                s = syn("E3", group, path, "time", ["--approaches", "Sequential", "--alphas", "1",
                                                    "--time-budgets", sec] + save, 1, 32)
                s.name += "__30s"
                shards.append(s)
                s = syn("E3", group, path, "iterations", ["--approaches", "Sequential", "--alphas", "1"]
                        + save + iters(b, False), 1, SEC["small"])
                s.name += "__it"
                shards.append(s)
        for seed in seeds:
            shards.append(realshard("E3", seed, "time", ["--approaches", "Sequential", "--alphas", "1",
                                                         "--time-budgets", sec] + save, 1, 32, "__30s"))
            shards.append(realshard("E3", seed, "iterations", ["--approaches", "Sequential", "--alphas", "1"]
                                    + save + iters(b, False), 1, SEC["small"], "__it"))
        stages.append(Stage("E3", "E3_sequential", "Sequential baseline", "mixed", shards,
                            "Proxy-objective TS and one final SSP; 30 s and the iteration budget."))
        warm = results / "E2b_warm_start" / "warm_start.json"
        shards = []
        if warm.is_file():
            shards = [realshard("E2b", seed, "iterations",
                                ["--approaches", "Embedded", "--alphas", "1", "--warm-start", str(warm)]
                                + crn + save + iters(b), 1, SEC["emb_real"] * b / 2500)
                      for seed in seeds[:E2B_REAL_SEEDS]]
            shards += [syn("E2b", "BASE", path, "iterations",
                           ["--approaches", "Embedded", "--alphas", "1", "--warm-start", str(warm)]
                           + crn + save + iters(b), 1, SEC["emb_syn"] * b / 2500)
                       for path in files["BASE"]]
        stages.append(Stage("E2b", "E2b_warm_start", "Alpha diagnosis: warm start", "iterations", shards,
                            "Embedded alpha 1 from the final alpha-0.25 timetables (E5a, E5b)."))
    for stage in stages:
        for shard in stage.shards:
            shard.output = results / stage.folder / "shards" / shard.name.replace("/", "__")
    return stages


def write_warm_start(results: Path, stages: list[Stage]) -> Path:
    """Final alpha-0.25 Embedded timetables of E5a (first seeds) and E5b BASE, by instance and seed."""
    target = results / "E2b_warm_start" / "warm_start.json"
    data: dict[str, dict[str, list]] = {}
    wanted_real = set(real_seeds()[:E2B_REAL_SEEDS])
    for stage in stages:
        if stage.key not in ("E5a", "E5b"):
            continue
        for shard in stage.shards:
            for path in (shard.output / "records").glob("*.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                meta = record["metadata"]
                p = meta["parameters"]
                if record.get("status") != "solved" or meta["approach"] != "Embedded" or p["alpha"] != 0.25:
                    continue
                if stage.key == "E5a" and p["seed"] not in wanted_real:
                    continue
                if stage.key == "E5b" and meta["group"] != "BASE":
                    continue
                rows = [[x["course"], x["section"], x["lecture"], x["block"]] for x in record["X"]]
                data.setdefault(meta["input_sha256"], {})[str(p["seed"])] = rows
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data), encoding="utf-8")
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["run", "status", "package"])
    parser.add_argument("--round", type=int, choices=[1, 2], default=1)
    parser.add_argument("--results", type=Path, default=ROOT / "results_v7")
    parser.add_argument("--data", type=Path, default=ROOT / "work" / "Synthetic")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.smoke:
        global SMOKE
        SMOKE = base.SMOKE = True
    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    data = args.data.resolve()
    prepare_data(data)
    workers = max(1, min(args.workers, 6))

    if args.command == "package":
        name = results.parent / f"results_v7_package_{dt.date.today().isoformat()}.zip"
        with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(results.rglob("*")):
                if path.is_file() and path.name != "STOP":
                    zf.write(path, path.relative_to(results.parent))
        print(f"Package written: {name}")
        return 0
    if args.round == 1:
        base.generate_pilot_instances(results)
    stages = build(results, data, args.round)
    if args.command == "status":
        for row in status_rows(stages):
            print(f"{row['stage']:4s} {row['title']:42s} {row['done']:6d}/{row['planned']:<6d} runs  "
                  f"remaining ~{hours(row['remaining_seconds'] * 1.1 / workers)} h with {workers} workers")
        return 0

    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    started = time.time()
    deadline = started + args.max_hours * 3600
    (results / "STOP").unlink(missing_ok=True)
    log(results, f"Session start (round {args.round}): workers {workers}, maximum {args.max_hours} h.")
    environment(results, workers)
    if args.round == 1:
        timing = results / "03_ssp_timing" / "ssp_timing.json"
        if not timing.is_file():
            log(results, "SSP timing (E6), single process. Keep the computer otherwise idle.")
            subprocess.run([PY, str(ROOT / "tools" / "ssp_timing.py"), "--data", str(data),
                            "--out", str(timing.parent)]
                           + (["--solves", "5", "--seconds", "1"] if SMOKE else []), cwd=ROOT, check=True)
        pilot = next(s for s in stages if s.key == "P3")
        if not (results / pilot.folder / "recommendation.json").is_file():
            if not run_shards(results, pilot.shards, workers, deadline):
                log(results, "Session ended before the Decoupled pilot finished. Run again to continue.")
                return 2
            base.summarize_pilot(results, pilot)
        stages = build(results, data, 1)
        main_shards = [s for st in stages if st.key != "P3" for s in st.shards]
    else:
        main_shards = [s for st in stages if st.key != "E2b" for s in st.shards]
    complete = run_shards(results, main_shards, workers, deadline)
    if complete and args.round == 2:
        stages = build(results, data, 2)
        warm = results / "E2b_warm_start" / "warm_start.json"
        if not warm.is_file():
            write_warm_start(results, stages)
            stages = build(results, data, 2)
        e2b = next(s for s in stages if s.key == "E2b")
        complete = run_shards(results, e2b.shards, workers, deadline)
    elapsed = (time.time() - started) / 3600
    if complete:
        log(results, f"Round {args.round} complete ({elapsed:.1f} h this session). "
                     "Next: python tools/campaigns_v7.py package")
        return 0
    remaining = sum(r["remaining_seconds"] for r in status_rows(build(results, data, args.round))) * 1.1 / workers
    log(results, f"Session finished after {elapsed:.1f} h. Remaining about {hours(remaining)} h. "
                 "Run the same command again to continue.")
    return 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Finished runs are saved; run the same command again to continue.", flush=True)
        sys.exit(130)
