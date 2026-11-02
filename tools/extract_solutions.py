"""Compact copy of the final timetables of campaign records (input of the re-evaluation tools).

Writes one JSON line per run (metadata, recorded metrics, final X as
[course, section, lecture, block]) to a gzip file; resumable (Python >= 3.8,
standard library only).

Usage: python tools/extract_solutions.py results D_real_30s extract/D_real_30s.jsonl.gz
"""

import gzip
import json
import sys
from pathlib import Path


def main(results: Path, stage: str, out: Path) -> None:
    done = set()
    if out.exists():
        with gzip.open(out, "rt", encoding="utf-8") as stream:
            done = {json.loads(line)["run_id"] for line in stream}
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(out, "at", encoding="utf-8") as writer:
        for path in sorted((results / stage / "shards").glob("*/records/*.json")):
            if path.stem in done:
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            meta = record["metadata"]
            params = meta["parameters"]
            row = dict(
                run_id=meta["run_id"], stage=stage, instance=meta["instance"], group=meta.get("group"),
                input_sha256=meta["input_sha256"], approach=meta.get("approach"),
                strategy=params["strategy"], alpha=params["alpha"], seed=params["seed"],
                local_search=params["local_search"], budget_kind=meta.get("budget_kind"),
                budget=meta.get("budget"), status=record["status"], metrics=record.get("metrics"),
                diagnostics={k: (record.get("diagnostics") or {}).get(k)
                             for k in ("completed_iterations", "optimizer_seconds", "ssp_calls")},
                X=[[x["course"], x["section"], x["lecture"], x["block"]] for x in (record.get("X") or [])],
            )
            writer.write(json.dumps(row) + "\n")
            count += 1
    print(stage, "new", count, "total", len(done) + count)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
