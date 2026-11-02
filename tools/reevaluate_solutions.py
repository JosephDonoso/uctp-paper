"""Re-evaluate final timetables with fresh SSP student orders (experiment E1).

Each recorded run optimized against noisy SSP evaluations and reports the
best value it found. This tool fixes the final timetable X of every run and
solves the SSP again with ``--orders`` new student orders, so that the reported
quality is an unbiased estimate for that timetable. Orders are drawn from
``random.Random(f"reeval:{run_id}:{k}")``, so results are reproducible and
independent of the run's own random stream.

Input is the compact ``*.jsonl.gz`` produced by ``tools/extract_solutions.py``
(one line per run with its metadata, recorded metrics and final X).

Usage:
    python tools/reevaluate_solutions.py --solutions extract/*.jsonl.gz \
        --orders 50 --ls own --out reeval/reeval_own.csv --workers 4
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import random
import statistics
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uctp.domain import Solution  # noqa: E402
from uctp.instance_io import load_instance, parse_instance  # noqa: E402
from uctp.legacy_input import convert_legacy_instance  # noqa: E402
from uctp.metrics import measure_solution  # noqa: E402
from uctp.objective import SearchObjective  # noqa: E402
from uctp.sectioning import StudentSectioning  # noqa: E402

FIELDS = ("R_imb", "R_dis", "conflicts", "F_ref", "F")


def instance_catalog() -> dict[str, tuple[str, bytes]]:
    """sha256 -> (format, raw bytes) for the real instance and the 250 synthetic ones."""
    catalog = {}

    def add(fmt: str, raw: bytes) -> None:
        # Records store the checksum of the file as read on the machine that ran
        # it; a Windows checkout may have CRLF line endings.
        lf = raw.replace(b"\r\n", b"\n")
        for variant in (lf, lf.replace(b"\n", b"\r\n")):
            catalog[hashlib.sha256(variant).hexdigest()] = (fmt, lf)

    add("instance", (ROOT / "datasets" / "Real" / "real.json").read_bytes())
    with zipfile.ZipFile(ROOT / "datasets" / "Synthetic" / "synthetic_datasets.zip") as archive:
        for name in archive.namelist():
            if name.endswith(".json"):
                add("legacy", archive.read(name))
    return catalog


_CACHE: dict = {}


def instance_for(sha: str):
    if sha not in _CACHE:
        fmt, raw = _CATALOG[sha]
        data = json.loads(raw.decode("utf-8-sig"))
        if fmt == "legacy":
            daily = data["restricciones"]["MAX_CARGA_DIARIA"]
            data = convert_legacy_instance(data)
            instance = parse_instance(data, max_daily_load=daily, max_block_load=15)
        else:
            instance = parse_instance(data, max_daily_load=4, max_block_load=15)
        _CACHE[sha] = instance
    return _CACHE[sha]


def timetable(instance, rows) -> dict:
    """Recorded X (identifiers) -> index-based X."""
    courses = {c.course_id: i for i, c in enumerate(instance.courses)}
    blocks = {b: i for i, b in enumerate(instance.block_ids)}
    X = {}
    for course, section, lecture, block in rows:
        c = courses[course]
        s = next(j for j, sec in enumerate(instance.courses[c].sections) if sec.section_id == section)
        l = next(k for k, lec in enumerate(instance.courses[c].sections[s].lectures) if lec.lecture_id == lecture)
        X[(c, s, l, blocks[block])] = 1
    return X


def evaluate(instance, X, *, alpha, orders, key, local_search, probability=None):
    """Mean and SD of each metric over ``orders`` SSP solves of a fixed X."""
    objective = SearchObjective(instance, alpha=alpha, cv_tolerance=0.10, idle_tolerance=4)
    ssp = StudentSectioning(
        instance, objective, local_search=local_search, local_search_probability=probability
    )
    values = {f: [] for f in FIELDS}
    for k in range(orders):
        solution = Solution(dict(X), {})
        ssp.solve(solution, rng=random.Random(f"reeval:{key}:{k}"))
        m = measure_solution(instance, solution, alpha=alpha)["metrics"]
        for f in FIELDS:
            values[f].append(float(m[f]))
    out = {}
    for f in FIELDS:
        out[f"{f}_mean"] = statistics.fmean(values[f])
        out[f"{f}_sd"] = statistics.stdev(values[f]) if orders > 1 else 0.0
    return out


def _work(args):
    row, orders, mode = args
    instance = instance_for(row["input_sha256"])
    X = timetable(instance, row["X"])
    if mode == "own":
        ls, p = bool(row["local_search"]), None
    elif mode == "on":
        ls, p = True, None
    elif mode == "off":
        ls, p = False, None
    else:
        ls, p = True, float(mode)
    result = evaluate(
        instance, X, alpha=float(row["alpha"]), orders=orders, key=row["run_id"],
        local_search=ls, probability=p,
    )
    rec = row.get("metrics") or {}
    out = {
        "run_id": row["run_id"], "stage": row["stage"], "instance": row["instance"],
        "group": row["group"], "approach": row["approach"], "strategy": row["strategy"],
        "alpha": row["alpha"], "seed": row["seed"], "local_search": row["local_search"],
        "budget_kind": row["budget_kind"], "budget": row["budget"], "eval_mode": mode,
        "orders": orders,
    }
    for f in FIELDS:
        out[f"recorded_{f}"] = rec.get(f)
    out.update(result)
    return out


def _init(catalog):
    global _CATALOG
    _CATALOG = catalog


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solutions", nargs="+", type=Path, required=True)
    parser.add_argument("--orders", type=int, default=50)
    parser.add_argument("--ls", default="own", help="own, on, off, or a probability in (0, 1).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--filter", default=None, help="Python expression on `row` selecting runs.")
    args = parser.parse_args(argv)
    rows = []
    for path in args.solutions:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("status") != "solved" or not row.get("X"):
                    continue
                if args.filter and not eval(args.filter, {}, {"row": row}):
                    continue
                rows.append(row)
    done = set()
    if args.out.exists():
        with args.out.open(encoding="utf-8") as stream:
            done = {r["run_id"] for r in csv.DictReader(stream)}
    rows = [r for r in rows if r["run_id"] not in done]
    print(f"{len(rows)} runs to evaluate ({len(done)} already in {args.out}).", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    catalog = instance_catalog()
    new = not args.out.exists()
    with args.out.open("a", encoding="utf-8", newline="") as stream, ProcessPoolExecutor(
        args.workers, initializer=_init, initargs=(catalog,)
    ) as pool:
        writer = None
        for i, result in enumerate(
            pool.map(_work, [(r, args.orders, args.ls) for r in rows], chunksize=4), 1
        ):
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(result))
                if new:
                    writer.writeheader()
            writer.writerow(result)
            if i % 200 == 0:
                stream.flush()
                print(f"  {i}/{len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
