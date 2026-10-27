"""Read portable run JSON/JSONL/CSV and export flat or detailed raw tables."""

import csv
import hashlib
import json
from pathlib import Path

CONDITION_FIELDS = [
    "group",
    "approach",
    "alpha",
    "strategy",
    "local_search",
    "trace",
    "budget_kind",
    "budget",
    "iteration_budget",
    "secondary_iteration_cap",
    "max_retries",
    "max_stagnant_iterations",
    "neighborhood_size",
    "max_daily_load",
    "max_block_load",
    "solver_revision",
    "input_format",
    "student_order",
    "python",
    "numpy",
    "beta",
    "cv_tolerance",
    "idle_tolerance",
]


def flatten_record(record: dict) -> dict:
    """Flatten metadata, scalar metrics and diagnostics without losing failures."""
    if "metadata" not in record:
        return record.copy()
    meta = record["metadata"]
    parameters = meta.get("parameters", {})
    row = {
        key: meta.get(key)
        for key in [
            "run_id",
            "group",
            "approach",
            "instance",
            "input_sha256",
            "initial_X_sha256",
            "final_X_sha256",
            "input_format",
            "solver_revision",
            "budget_kind",
            "budget",
            "iteration_budget",
            "max_daily_load",
            "max_block_load",
            "student_order",
            "python",
            "numpy",
        ]
    }
    row.update(
        {
            key: parameters.get(key)
            for key in [
                "alpha",
                "strategy",
                "local_search",
                "trace",
                "seed",
                "max_retries",
                "iterations_per_search",
                "search_blocks",
                "total_iteration_limit",
                "time_limit_seconds",
                "max_stagnant_iterations",
                "neighborhood_size",
            ]
        }
    )
    row.update(status=record["status"], stop_reason=record.get("stop_reason"))
    row.update(meta.get("constants", {}))
    row["secondary_iteration_cap"] = (
        meta.get("iteration_budget") if meta.get("budget_kind") == "seconds" else None
    )
    row.update({key: None for key in ["F", "F_ref", "R_imb", "R_dis", "conflicts"]})
    row.update(record.get("metrics") or {})
    row.update(
        {"initial_" + key: value for key, value in (record.get("initial_metrics") or {}).items()}
    )
    row.update(record.get("diagnostics") or {})
    row.update({"tag_" + key: value for key, value in meta.get("tags", {}).items()})
    return row


def write_table(path: Path, rows: list[dict], *, empty_fields: list[str] | None = None) -> None:
    """Write a CSV with the union of scalar columns, including missing values."""
    fields = sorted({key for row in rows for key in row}) if rows else (empty_fields or ["run_id"])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_raw(output: Path, records: list[dict]) -> list[dict]:
    """Export flat runs plus optional endpoint student and section tables."""
    rows = [flatten_record(record) for record in records]
    write_table(output / "raw.csv", rows)
    parameter_fields = set(CONDITION_FIELDS) | {
        "seed",
        "iterations_per_search",
        "search_blocks",
        "total_iteration_limit",
        "time_limit_seconds",
        "max_stagnant_iterations",
        "neighborhood_size",
    }
    longitudinal = []
    for row in rows:
        identifiers = {
            key: row.get(key)
            for key in [
                "run_id",
                "group",
                "approach",
                "alpha",
                "strategy",
                "local_search",
                "budget_kind",
                "budget",
                "input_sha256",
                "seed",
                "status",
            ]
        }
        for key, value in row.items():
            if (
                isinstance(value, (int, float))
                and key not in parameter_fields
                and not key.startswith("tag_")
            ):
                stage = "initial" if key.startswith("initial_") else "final"
                if key.endswith("_seconds") or key in {
                    "retries_used",
                    "construction_attempts",
                    "completed_iterations",
                    "search_blocks_completed",
                    "ssp_calls",
                }:
                    stage = "diagnostic"
                longitudinal.append(
                    identifiers
                    | dict(stage=stage, metric=key.removeprefix("initial_"), value=value)
                )
    write_table(
        output / "metrics_long.csv",
        longitudinal,
        empty_fields=["run_id", "stage", "metric", "value"],
    )
    with (output / "raw.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    for entity in ("students", "sections"):
        details = []
        for record, row in zip(records, rows, strict=True):
            for item in record.get("details", {}).get(entity, []):
                scalars = {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in item.items()
                }
                details.append({"run_id": row["run_id"], **scalars})
        if details:
            write_table(output / f"{entity}.csv", details)
    trace = [
        {"run_id": row["run_id"], **step}
        for record, row in zip(records, rows, strict=True)
        for step in record.get("trace", [])
    ]
    if trace:
        write_table(output / "trace.csv", trace)
    return rows


def _csv_value(key: str, value: str):
    if value == "":
        return None
    if key in {
        "group",
        "approach",
        "strategy",
        "instance",
        "input_sha256",
        "initial_X_sha256",
        "final_X_sha256",
        "run_id",
        "input_format",
        "solver_revision",
        "budget_kind",
        "status",
        "stop_reason",
        "student_order",
        "python",
        "numpy",
    } or key.startswith("tag_"):
        return value
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_records(inputs: list[Path]) -> list[dict]:
    """Load exported runs, deduplicating equal run IDs and rejecting conflicts.

    Directories prefer their records/ files to avoid double-counting raw
    CSV/JSONL exports of the same runs. Other JSON is not executed or treated
    as configuration. Explicit CSV inputs may include custom scalar columns.
    """
    paths = set()
    for path in inputs:
        if path.is_dir():
            if (path / "records").is_dir():
                paths.update((path / "records").glob("*.json"))
            elif (path / "raw.jsonl").is_file():
                paths.add(path / "raw.jsonl")
            elif (path / "raw.csv").is_file():
                paths.add(path / "raw.csv")
            else:
                paths.update(path.glob("*.json"))
        elif path.is_file():
            paths.add(path)
        else:
            raise ValueError(f"Input not found: {path}")
    unique = {}
    for path in sorted(paths):
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                records = [
                    {key: _csv_value(key, value) for key, value in row.items()}
                    for row in csv.DictReader(stream)
                ]
        elif path.suffix.lower() == ".jsonl":
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
        elif path.suffix.lower() == ".json":
            records = [json.loads(path.read_text(encoding="utf-8-sig"))]
        else:
            raise ValueError(f"Unsupported result format: {path}")
        for index, record in enumerate(records):
            if not isinstance(record, dict) or "status" not in record:
                continue
            row = flatten_record(record)
            identifier = (
                row.get("run_id")
                or hashlib.sha256(f"{path.resolve()}:{index}".encode()).hexdigest()
            )
            if "metadata" in record:
                record["metadata"]["run_id"] = identifier
            else:
                record["run_id"] = identifier
            if identifier in unique:
                previous = {
                    k: v for k, v in flatten_record(unique[identifier]).items() if v is not None
                }
                current = {k: v for k, v in flatten_record(record).items() if v is not None}
                if previous != current:
                    raise ValueError(
                        f"Conflicting records for run_id={identifier}; select one source format."
                    )
                previous_record = unique[identifier]
                for key in ("X", "Y", "initial_solution", "details", "trace"):
                    if (
                        key in previous_record
                        and key in record
                        and previous_record[key] != record[key]
                    ):
                        raise ValueError(f"Conflicting {key} for run_id={identifier}.")
                if "metadata" in record:
                    unique[identifier] = (
                        (previous_record | record) if "metadata" in previous_record else record
                    )
                continue
            unique[identifier] = record
    if not unique:
        raise ValueError("No run records found. Supply exported result JSON, JSONL or raw CSV.")
    return list(unique.values())
