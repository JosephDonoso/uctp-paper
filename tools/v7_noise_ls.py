"""E1a (SSP noise on fixed timetables) and E4b (partial student adjustment).

E1a: 100 fresh student orders on X_real and on selected final timetables;
     mean, SD and CV of each metric.
E4b: probability p that a student applies the local search; p* is the grid
     value whose mean R_imb on X_real is closest to the observed 1.760. The
     final Embedded and Decoupled-50 timetables at alpha 0.25 of the real
     instance are then evaluated with p*.

Usage:
    python tools/v7_noise_ls.py --solutions extract/*.jsonl.gz --out reeval --workers 2
"""

import argparse
import csv
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import reevaluate_solutions as rs  # noqa: E402
from real_observed import observed  # noqa: E402
from uctp.reevaluation import FIELDS, reevaluate  # noqa: E402

REAL_SEEDS_FIRST10 = None  # filled from SEEDS.json order (first ten real seeds)


def _load(paths):
    rows = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream)
    return rows


def _task(job):
    label, kind, payload, alpha, orders, key, ls, p = job
    if kind == "real_observed":
        instance, X, _ = observed()
    else:
        instance = rs.instance_for(payload["input_sha256"])
        X = rs.timetable(instance, payload["X"])
    result = reevaluate(instance, X, alpha=alpha, orders=orders, key=key,
                        local_search=ls, local_search_probability=p)
    return dict(label=label, alpha=alpha, p="" if p is None else p, local_search=ls, **result)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--solutions", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True, help="SEEDS.json of the campaigns.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    rows = _load(args.solutions)
    seeds = json.loads(args.seeds.read_text(encoding="utf-8"))
    first10 = seeds["real_quality_time_seeds"]
    catalog = rs.instance_catalog()
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- E1a
    jobs = [("X_real (observed timetable)", "real_observed", None, 1.0, 100, "noise:xreal", True, None)]
    for row in rows:
        if row["stage"] == "D_real_30s" and row["approach"] == "Embedded" and row["alpha"] in (0.25, 1.0) \
                and row["seed"] in first10:
            jobs.append((f"Real Embedded a={row['alpha']} seed {row['seed']}", "run", row,
                         row["alpha"], 100, f"noise:{row['run_id']}", True, None))
    base = sorted({r["instance"] for r in rows if r["stage"] == "C_synthetic_30s" and r["group"] == "BASE"})[:10]
    for row in rows:
        if row["stage"] == "C_synthetic_30s" and row["group"] == "BASE" and row["alpha"] == 1.0 \
                and row["instance"] in base:
            jobs.append((f"BASE {row['approach']} a=1 {row['instance']}", "run", row, 1.0, 100,
                         f"noise:{row['run_id']}", True, None))
    with ProcessPoolExecutor(args.workers, initializer=rs._init, initargs=(catalog,)) as pool:
        results = list(pool.map(_task, jobs))
    _write(args.out / "e1a_noise.csv", results)
    print(f"E1a: {len(results)} timetables", flush=True)

    # ---- E4b calibration on X_real
    grid = [round(0.1 * i, 1) for i in range(11)]
    jobs = [(f"X_real p={p}", "real_observed", None, 1.0, 50, f"partial:xreal:{p}", True, p) for p in grid]
    with ProcessPoolExecutor(args.workers, initializer=rs._init, initargs=(catalog,)) as pool:
        calibration = list(pool.map(_task, jobs))
    _write(args.out / "e4b_calibration.csv", calibration)
    best = min(calibration, key=lambda r: abs(r["R_imb_mean"] - 1.760))
    p_star = float(best["p"])
    print(f"E4b: p* = {p_star} (R_imb {best['R_imb_mean']:.3f})", flush=True)
    jobs = [(f"{row['approach']} a=0.25 seed {row['seed']}", "run", row, 0.25, 50,
             f"partial:{row['run_id']}", True, p_star)
            for row in rows if row["stage"] == "D_real_30s" and row["alpha"] == 0.25]
    jobs.append(("X_real (observed timetable)", "real_observed", None, 0.25, 50, "partial:xreal:star", True, p_star))
    with ProcessPoolExecutor(args.workers, initializer=rs._init, initargs=(catalog,)) as pool:
        partial = list(pool.map(_task, jobs))
    _write(args.out / "e4b_partial.csv", partial)
    return 0


def _write(path, results):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    raise SystemExit(main())
