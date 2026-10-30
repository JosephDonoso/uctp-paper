"""Input descriptives for the institutional instance and synthetic groups.

Reproduces the descriptive tables of the manuscript directly from instance
files. Statistics pool entities within each dataset; preassignment
percentages are calculated per instance; standard deviations are population
standard deviations; teacher unavailability uses the 35 nominal weekly blocks.

Usage:
    python tools/input_descriptives.py --real datasets/Real/real.json \
        --synthetic DIR [--groups BASE ...] [--latex OUT.tex] [--csv OUT.csv]
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from uctp.legacy_input import convert_legacy_instance

NOMINAL_BLOCKS = 35
GROUPS = ["BASE", "Dispersion-Low", "Dispersion-High", "Offer-Low", "Offer-High"]
MEASURES = [
    ("requests", "Requested courses per student"),
    ("sections", "Sections per course"),
    ("lectures_section", "Weekly lectures per section"),
    ("lectures_course", "Weekly lectures per course (all sections)"),
    ("pre_lectures", "File-recorded preassigned lectures (\\%)"),
    ("pre_courses", "File-recorded courses with preassignments (\\%)"),
    ("unavailable", "Unavailable teacher blocks (\\%)"),
    ("teacher_load", "Weekly blocks per teacher"),
]


def _instance_values(data: dict) -> dict[str, list[float]]:
    values = {key: [] for key, _ in MEASURES}
    values["requests"] = [len(s["courses"]) for s in data["students"]]
    loads = Counter()
    lectures = pre = pre_courses = 0
    for course in data["courses"]:
        values["sections"].append(len(course["sections"]))
        total = 0
        has_pre = False
        for section in course["sections"]:
            values["lectures_section"].append(len(section["lectures"]))
            total += len(section["lectures"])
            for lecture in section["lectures"]:
                lectures += 1
                if lecture.get("preassigned_block"):
                    pre += 1
                    has_pre = True
                for teacher in lecture.get("teachers", []):
                    loads[teacher] += 1
        values["lectures_course"].append(total)
        pre_courses += has_pre
    values["pre_lectures"] = [100 * pre / lectures]
    values["pre_courses"] = [100 * pre_courses / len(data["courses"])]
    for teacher in data["teachers"]:
        available = teacher.get("available_blocks")
        count = NOMINAL_BLOCKS if available is None else len(set(available))
        values["unavailable"].append(100 * (NOMINAL_BLOCKS - count) / NOMINAL_BLOCKS)
        values["teacher_load"].append(loads[teacher["id"]])
    return values


def _dataset_values(paths: list[Path], legacy: bool) -> dict[str, list[float]]:
    pooled = {key: [] for key, _ in MEASURES}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if legacy:
            data = convert_legacy_instance(data)
        for key, values in _instance_values(data).items():
            pooled[key].extend(values)
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=GROUPS)
    parser.add_argument("--latex", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    datasets = [("Real", [args.real], False)] + [
        (group, sorted((args.synthetic / group).glob("*.json")), True) for group in args.groups
    ]
    rows = []
    for name, paths, legacy in datasets:
        pooled = _dataset_values(paths, legacy)
        for key, label in MEASURES:
            a = np.asarray(pooled[key], dtype=float)
            rows.append((name, label, len(a), a.mean(), a.std(), a.min(), a.max()))
    for row in rows:
        print(f"{row[0]:16s} {row[1]:48s} {row[2]:6d} {row[3]:8.3f} {row[4]:8.3f} {row[5]:8.3f} {row[6]:8.3f}")
    if args.latex:
        with args.latex.open("w", encoding="utf-8") as out:
            for name, label, n, mean, sd, low, high in rows:
                shown = name.replace("-", "\\-")
                out.write(f"{shown} & {label} & {n} & {mean:.3f} & {sd:.3f} & {low:.3f} & {high:.3f} \\\\\n")
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(["dataset", "measure", "n", "mean", "sd", "min", "max"])
            writer.writerows(rows)


if __name__ == "__main__":
    main()
