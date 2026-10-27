"""Sequential, resumable grids over instances, treatments, seeds and budgets."""

import hashlib
import itertools
import json
import math
import platform
import random
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..construction import InitialStrategy
from ..instance_io import load_instance, write_document
from ..planner import Approach, SolveOptions, validate_options
from ..reporting import execute_run, solver_revision
from .records import export_raw


def _approach(value: str) -> tuple[Approach, int | None]:
    if value == "Embedded":
        return Approach.EMBEDDED, None
    if value.startswith("Decoupled-") and value[10:].isdigit() and int(value[10:]) > 0:
        return Approach.DECOUPLED, int(value[10:])
    raise ValueError("Use Embedded or Decoupled-k with a positive integer k.")


def run_grid(args: Namespace) -> dict:
    """Execute a CLI-specified grid with immutable manifests and raw records.

    Runs are sequential. Repeated invocations resume only with identical
    inputs, source fingerprint and arguments; terminal failures are retained.
    No optimizer metric is used to select which runs enter the data.
    """
    files = sorted(
        {
            p.resolve()
            for supplied in args.inputs
            for p in (supplied.rglob("*.json") if supplied.is_dir() else [supplied])
        }
    )
    if not files or any(not path.is_file() for path in files):
        raise ValueError("No instance JSON files found.")
    seeds = (
        args.seeds
        if args.seeds is not None
        else random.Random(args.master_seed).sample(range(1, 2**32), args.runs)
    )
    if len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Use unique integer seeds in [0, 2**32).")
    if args.runs < 1 or args.iteration_ceiling < 1:
        raise ValueError("runs and iteration-ceiling must be positive.")
    budgets = (
        [("seconds", value) for value in args.time_budgets]
        if args.time_budgets is not None
        else [("iterations", value) for value in args.iteration_budgets or [200]]
    )
    if any(not math.isfinite(value) or value <= 0 for _, value in budgets):
        raise ValueError("Budgets must be positive and finite.")
    tags = {}
    for item in args.tag:
        if "=" not in item:
            raise ValueError("Tags use NAME=VALUE.")
        key, value = item.split("=", 1)
        if not key.isidentifier() or key in tags:
            raise ValueError("Tag names must be unique identifiers.")
        tags[key] = value
    inputs, identities = [], set()
    for path in files:
        group = args.group or path.parent.name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (group, digest) in identities:
            raise ValueError(
                "Duplicate instance content within one group; avoid duplicate experimental units."
            )
        identities.add((group, digest))
        load_instance(
            path,
            input_format=args.input_format,
            max_daily_load=args.max_daily_load,
            max_block_load=args.max_block_load,
        )
        inputs.append(dict(path=str(path), sha256=digest, group=group))
    revision = solver_revision()
    jobs = []
    for source, seed, strategy, approach, alpha, local_search, (kind, budget) in itertools.product(
        inputs, seeds, args.strategies, args.approaches, args.alphas, args.local_search, budgets
    ):
        integration, interval = _approach(approach)
        total = args.iteration_ceiling if kind == "seconds" else int(budget)
        per_search = interval or total
        blocks = math.ceil(total / per_search) if integration == Approach.DECOUPLED else 1
        options = SolveOptions(
            strategy=InitialStrategy(strategy),
            approach=integration,
            alpha=alpha,
            max_retries=args.max_retries,
            iterations_per_search=per_search,
            search_blocks=blocks,
            total_iteration_limit=total,
            time_limit_seconds=budget if kind == "seconds" else None,
            max_stagnant_iterations=args.stagnation if args.stagnation is not None else total,
            neighborhood_size=args.neighborhood_size,
            seed=seed,
            local_search=local_search == "on",
            trace=args.trace,
        )
        validate_options(options)
        specification = dict(
            instance_sha256=source["sha256"],
            group=source["group"],
            parameters=asdict(options),
            input_format=args.input_format,
            max_daily_load=args.max_daily_load,
            max_block_load=args.max_block_load,
            tags=tags,
            solver_revision=revision,
        )
        identifier = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[
            :24
        ]
        jobs.append((identifier, source, options))
    if len({job[0] for job in jobs}) != len(jobs):
        raise ValueError("Repeated treatment levels produce duplicate runs.")
    options = {
        key: value
        for key, value in vars(args).items()
        if key not in {"output", "resume", "dry_run", "inputs"}
    }
    manifest = dict(
        schema_version=1,
        python=platform.python_version(),
        numpy=np.__version__,
        planned_runs=len(jobs),
        solver_revision=revision,
        inputs=inputs,
        seeds=seeds,
        options=options,
    )
    print(f"Planned runs: {len(jobs)} (sequential).", flush=True)
    if args.dry_run:
        return manifest
    output = args.output
    manifest_path = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume or not manifest_path.is_file():
            raise ValueError(
                "Output is not empty; use a new directory or --resume for the same grid."
            )
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Inputs, code or settings differ from the saved grid.")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_document(manifest_path, manifest)
    records_dir = output / "records"
    records_dir.mkdir(exist_ok=True)
    status = "complete"
    try:
        for index, (identifier, source, options) in enumerate(jobs, 1):
            destination = records_dir / f"{identifier}.json"
            if destination.exists():
                saved = json.loads(destination.read_text(encoding="utf-8"))
                if saved.get("metadata", {}).get("run_id") != identifier:
                    raise ValueError("Invalid saved record; refusing to skip it.")
                continue
            if (
                solver_revision() != revision
                or hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]
            ):
                raise ValueError("Source code or input changed while the grid was running.")
            document = execute_run(
                Path(source["path"]),
                options,
                input_format=args.input_format,
                group=source["group"],
                run_id=identifier,
                details=args.details,
                include_solution=args.save_solutions,
                max_daily_load=args.max_daily_load,
                max_block_load=args.max_block_load,
                tags=tags,
            )
            temporary = destination.with_suffix(".tmp")
            write_document(temporary, document)
            temporary.replace(destination)
            print(
                f"{index}/{len(jobs)} {source['group']}/{Path(source['path']).name} {document['status']}",
                flush=True,
            )
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status = "error"
        raise
    finally:
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(records_dir.glob("*.json"))
        ]
        export_raw(output, records)
        (output / "status.json").write_text(
            json.dumps(dict(status=status, completed=len(records), planned=len(jobs)), indent=2),
            encoding="utf-8",
        )
    return manifest | {"status": status}
