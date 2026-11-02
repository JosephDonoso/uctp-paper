"""Time per SSP solve and its scaling with the number of students (experiment E6).

Run alone on an otherwise idle computer (one process). Measures:
* milliseconds per SSP solve (mean, 95th percentile) with local search, on the
  initial CM timetables of the first ten BASE instances and on the real
  instance (initial CM timetable and the observed timetable X_real);
* the same on the real instance with random subsets of 25, 50, 75 and 100 % of
  its students (subset seed 703), together with the Embedded iterations
  completed in 30 s at alpha 0.25 (seed 703).

Usage: python tools/ssp_timing.py --data work/Synthetic --out results_v7/03_ssp_timing
"""

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from real_observed import observed  # noqa: E402
from uctp.domain import Solution  # noqa: E402
from uctp.instance_io import load_instance  # noqa: E402
from uctp.objective import SearchObjective  # noqa: E402
from uctp.planner import SolveOptions, solve  # noqa: E402
from uctp.sectioning import StudentSectioning  # noqa: E402


def time_solves(instance, X, solves, seed=703):
    objective = SearchObjective(instance, alpha=1, cv_tolerance=0.10, idle_tolerance=4)
    ssp = StudentSectioning(instance, objective, local_search=True)
    rng = random.Random(seed)
    ssp.solve(Solution(dict(X), {}), rng=rng)  # warm-up
    times = []
    for _ in range(solves):
        solution = Solution(dict(X), {})
        started = time.perf_counter()
        ssp.solve(solution, rng=rng)
        times.append(1000 * (time.perf_counter() - started))
    times.sort()
    return dict(solves=solves, mean_ms=statistics.fmean(times),
                p95_ms=times[int(0.95 * (len(times) - 1))], students=len(instance.students))


def initial_X(instance, seed=703):
    result = solve(instance, SolveOptions(seed=seed, iterations_per_search=1))
    return result.initial_solution.X if result.initial_solution else result.solution.X


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--solves", type=int, default=200)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"base": [], "real": {}, "scaling": []}
    for path in sorted((args.data / "BASE").glob("instance_*.json"))[:10]:
        instance = load_instance(path, input_format="legacy", max_daily_load=None, max_block_load=15)
        X = {k: v for k, v in solve(instance, SolveOptions(seed=703, iterations_per_search=1)).solution.X.items() if v}
        result["base"].append(dict(instance=path.name, **time_solves(instance, X, args.solves)))
        print("BASE", path.name, round(result["base"][-1]["mean_ms"], 3), flush=True)
    real = load_instance(ROOT / "datasets" / "Real" / "real.json", input_format="instance",
                         max_daily_load=4, max_block_load=15)
    X_cm = {k: v for k, v in solve(real, SolveOptions(seed=703, iterations_per_search=1)).solution.X.items() if v}
    result["real"]["initial_cm"] = time_solves(real, X_cm, args.solves)
    obs_instance, X_obs, _ = observed()
    result["real"]["observed"] = time_solves(obs_instance, X_obs, args.solves)
    print("real", result["real"], flush=True)
    students = list(range(len(real.students)))
    for fraction in (0.25, 0.5, 0.75, 1.0):
        chosen = sorted(random.Random(703).sample(students, round(fraction * len(students))))
        subset = replace(real, students=tuple(real.students[i] for i in chosen))
        timing = time_solves(subset, X_cm, args.solves)
        run = solve(subset, SolveOptions(alpha=0.25, seed=703, iterations_per_search=10**6,
                                         max_stagnant_iterations=10**6,
                                         time_limit_seconds=args.seconds))
        result["scaling"].append(dict(fraction=fraction, iterations_30s=run.diagnostics["completed_iterations"],
                                      ssp_calls_30s=run.diagnostics["ssp_calls"], **timing))
        print("scaling", result["scaling"][-1], flush=True)
    (args.out / "ssp_timing.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
