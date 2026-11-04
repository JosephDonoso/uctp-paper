"""Copy of paper_results/ with the final metrics replaced by their re-evaluated means.

The earlier campaigns (A-H, without common random numbers) report the metrics that the
search computed with a single SSP order. ``reeval_own.csv`` (in paper_results_v8/data)
holds the means over 50 fresh orders of every saved final timetable. This script writes
a copy of paper_results in which F, F_ref, R_imb, R_dis and conflicts of each final record
are those means, so that ``tools/paper_analysis.py`` produces the re-evaluated tables of
the supplement. Checkpoints (B_iteration_budgets/merged/checkpoints.csv) keep the
recorded values, which are the ones compared across budgets.

Example:
    python tools/reevaluated_paper_results.py --results paper_results \
        --reeval paper_results_v8/data/reeval_own.csv --out work/results_reeval
    python tools/paper_analysis.py --results work/results_reeval \
        --baselines paper_results_v8/real_baselines_reeval.json --out work/paper_out_reeval
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

METRICS = {"F": "F_mean", "F_ref": "F_ref_mean", "R_imb": "R_imb_mean", "R_dis": "R_dis_mean",
           "conflicts": "conflicts_mean"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--reeval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists.")
    shutil.copytree(args.results, args.out)
    reeval: dict[tuple[str, str], dict] = {}
    with args.reeval.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            reeval[(row["stage"], row["run_id"])] = row
    for raw in sorted(args.out.glob("*/merged/raw.csv")):
        stage = raw.parts[-3]
        with raw.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        replaced = 0
        for row in rows:
            match = reeval.get((stage, row["run_id"]))
            if match is None or row.get("status") != "solved":
                continue
            for metric, column in METRICS.items():
                row[metric] = match[column]
            replaced += 1
        with raw.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{stage}: {replaced}/{len(rows)} records re-evaluated")


if __name__ == "__main__":
    main()
