"""Re-solve the SSP on saved final timetables and compare with reported metrics.

Use on a campaign run with --save-solutions. For every solved record, the final
timetable X is rebuilt, the SSP (same local-search setting as the run) is solved
on it with K independent seeds, and endpoint metrics are measured. Decoupled
records from the original code report a Y computed for an earlier timetable;
this script gives their metrics under the same enrollment model as Embedded.
For Embedded it also shows the bias of keeping the single SSP draw that the
search selected.

Example:
    python tools/reevaluate_records.py --campaign results/real --instances datasets/Real \
        --draws 10 --output reevaluation.csv
"""

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

from uctp.domain import Solution
from uctp.instance_io import load_instance
from uctp.metrics import measure_solution
from uctp.objective import SearchObjective
from uctp.sectioning import StudentSectioning

KEYS = ["F_ref", "R_imb", "R_dis", "conflicts"]


def rebuild_X(instance, timetable):
    course_idx = {c.course_id: i for i, c in enumerate(instance.courses)}
    block_idx = {b: i for i, b in enumerate(instance.block_ids)}
    X = {}
    for row in timetable:
        c = course_idx[row["course"]]
        sections = instance.courses[c].sections
        s = next(i for i, sec in enumerate(sections) if sec.section_id == row["section"])
        lectures = sections[s].lectures
        l = next(i for i, lec in enumerate(lectures) if lec.lecture_id == row["lecture"])
        X[c, s, l, block_idx[row["block"]]] = 1
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, required=True, help="Campaign output directory.")
    parser.add_argument(
        "--instances", type=Path, nargs="+", required=True,
        help="Directories or files with the instance JSONs used by the campaign.",
    )
    parser.add_argument("--draws", type=int, default=10, help="SSP seeds per record.")
    parser.add_argument("--seed-base", type=int, default=20261029)
    parser.add_argument("--input-format", default="instance")
    parser.add_argument("--max-daily-load", type=int, default=None)
    parser.add_argument("--max-block-load", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_hash = {}
    for entry in args.instances:
        for path in ([entry] if entry.is_file() else sorted(entry.rglob("*.json"))):
            by_hash[hashlib.sha256(path.read_bytes()).hexdigest()] = path
    cache = {}
    rows = []
    records = sorted((args.campaign / "records").glob("*.json"))
    skipped = 0
    for n, record_path in enumerate(records, 1):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "solved" or not record.get("X"):
            skipped += 1
            continue
        meta = record["metadata"]
        path = by_hash.get(meta["input_sha256"])
        if path is None:
            raise SystemExit(f"Instance for {record_path.name} not found (sha256 mismatch).")
        if path not in cache:
            cache[path] = load_instance(
                path, input_format=args.input_format,
                max_daily_load=args.max_daily_load, max_block_load=args.max_block_load,
            )
        instance = cache[path]
        params = meta["parameters"]
        alpha = params["alpha"]
        objective = SearchObjective(instance, alpha=alpha, cv_tolerance=0.10, idle_tolerance=4)
        ssp = StudentSectioning(instance, objective, local_search=params.get("local_search", True))
        X = rebuild_X(instance, record["X"])
        draws = []
        for k in range(args.draws):
            random.seed(args.seed_base + k)
            solution = Solution(X.copy(), {})
            ssp.solve(solution)
            draws.append(measure_solution(instance, solution, alpha=alpha)["metrics"])
        reported = record.get("stale_metrics") or record["metrics"]
        row = dict(
            run_id=meta["run_id"], group=meta.get("group"), instance=meta.get("instance"),
            approach=meta.get("approach"), alpha=alpha, seed=params.get("seed"),
            local_search=params.get("local_search"),
            iterations=record.get("diagnostics", {}).get("completed_iterations"),
        )
        for key in KEYS:
            values = [d[key] for d in draws]
            row[f"reported_{key}"] = reported[key]
            row[f"resolved_{key}_mean"] = statistics.fmean(values)
            row[f"resolved_{key}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
        if n % 50 == 0:
            print(f"{n}/{len(records)}", flush=True)
    if not rows:
        raise SystemExit("No solved records with saved X (was the campaign run with --save-solutions?).")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} records re-evaluated, {skipped} skipped -> {args.output}")


if __name__ == "__main__":
    main()
