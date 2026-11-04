"""Extract the records of one v7 campaign stage to a flat CSV (one row per final or checkpoint).

Columns ``rec_*`` are the metrics recorded by the search and ``ree_*`` / ``ree_sd_*`` the
means and standard deviations over the re-evaluation orders. When the records contain a
search trace, it is written next to the CSV as ``<name>_trace.jsonl``.

Example (from the repository root):
    python tools/extract_v7.py results_v7 E5b_synthetic_converged paper_results_v8/data/E5b_synthetic_converged.csv
"""

import csv
import json
import sys
from pathlib import Path


def main() -> None:
    results, stage, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    rows = []
    trace_path = out.replace(".csv", "_trace.jsonl")
    for path in sorted(results.glob(stage + "/shards/*/records/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        meta, params = record["metadata"], record["metadata"]["parameters"]
        base = dict(run_id=meta["run_id"], shard=path.parent.parent.name, instance=meta["instance"],
                    group=meta["group"], approach=meta["approach"], alpha=params["alpha"], seed=params["seed"],
                    crn=params.get("common_random_numbers"), neighborhood=params["neighborhood_size"],
                    warm=params.get("initial_timetable") is not None, budget_kind=meta["budget_kind"],
                    budget=meta["budget"], status=record["status"])
        diagnostics = record.get("diagnostics") or {}
        for key in ("completed_iterations", "optimizer_seconds", "optimizer_cpu_seconds", "ssp_calls"):
            base[key] = diagnostics.get(key)

        def add(row, metrics, reeval):
            for name in ("F", "F_ref", "R_imb", "R_dis", "conflicts"):
                row["rec_" + name] = (metrics or {}).get(name)
                row["ree_" + name] = (reeval or {}).get(name + "_mean")
                row["ree_sd_" + name] = (reeval or {}).get(name + "_sd")

        row = dict(base, point="final")
        add(row, record.get("metrics"), record.get("reevaluation"))
        rows.append(row)
        for checkpoint in record.get("checkpoints", []):
            row = dict(base, point=checkpoint["iteration_budget"],
                       completed_iterations=checkpoint["iteration_budget"],
                       optimizer_seconds=checkpoint.get("optimizer_seconds"),
                       optimizer_cpu_seconds=checkpoint.get("optimizer_cpu_seconds"),
                       ssp_calls=checkpoint.get("ssp_calls"))
            add(row, checkpoint.get("metrics"), checkpoint.get("reevaluation"))
            rows.append(row)
        if "trace" in record:
            with open(trace_path, "a", encoding="utf-8") as stream:
                for step in record["trace"]:
                    stream.write(json.dumps(dict(run_id=meta["run_id"], **step)) + "\n")
    with open(out, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(stage, len(rows))


if __name__ == "__main__":
    main()
