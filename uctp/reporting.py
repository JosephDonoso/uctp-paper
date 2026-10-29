"""Portable research run records shared by the solver CLI and batch runner."""

import hashlib
import json
import platform
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .construction import ConstructionFailure
from .instance_io import load_instance, solution_document
from .metrics import measure_solution
from .planner import Approach, SolveOptions, solve


def solver_revision() -> str:
    """Fingerprint solver source without depending on Git or analysis code."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def execute_run(
    path: Path,
    options: SolveOptions,
    *,
    input_format: str = "instance",
    group: str | None = None,
    run_id: str | None = None,
    details: bool = False,
    include_solution: bool = True,
    max_daily_load: int | None = None,
    max_block_load: int = 15,
    tags: dict[str, str] | None = None,
) -> dict:
    """Solve once and return raw parameters, success/failure and endpoint metrics.

    Input errors raise exceptions. Exhausted construction produces a failure
    record with null quality metrics. Extraction runs after optimizer timing;
    recording never samples random numbers. Decoupled re-solves the SSP on the
    final timetable inside solve() unless final_ssp is disabled.
    """
    started = time.perf_counter()
    instance = load_instance(
        path,
        input_format=input_format,
        max_daily_load=max_daily_load,
        max_block_load=max_block_load,
    )
    parameters = asdict(options)
    daily_limits = sorted({teacher.max_daily_load for teacher in instance.teachers})
    interval = options.iterations_per_search
    iteration_budget = min(
        options.total_iteration_limit or interval * options.search_blocks,
        interval * options.search_blocks,
    )
    approach = "Embedded" if options.approach == Approach.EMBEDDED else f"Decoupled-{interval}"
    metadata = dict(
        run_id=run_id or uuid.uuid4().hex,
        python=platform.python_version(),
        numpy=np.__version__,
        solver_revision=solver_revision(),
        input_path=str(path),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        instance=path.name,
        input_format=input_format,
        group=group or path.parent.name or "unspecified",
        parameters=parameters,
        constants={"beta": 0.25, "cv_tolerance": 0.1, "idle_tolerance": 4},
        approach=approach,
        max_daily_load=daily_limits[0] if len(daily_limits) == 1 else None,
        teacher_daily_loads={
            teacher.teacher_id: teacher.max_daily_load for teacher in instance.teachers
        },
        max_block_load=max_block_load,
        tags=tags or {},
        budget_kind="seconds" if options.time_limit_seconds is not None else "iterations",
        budget=options.time_limit_seconds
        if options.time_limit_seconds is not None
        else iteration_budget,
        iteration_budget=iteration_budget,
        student_order="random_without_replacement",
    )
    try:
        result = solve(instance, options, collect_metrics=True)
    except ConstructionFailure as error:
        return dict(
            schema_version=1,
            status="construction_failed",
            stop_reason=str(error),
            metadata=metadata,
            metrics=None,
            initial_metrics=None,
            diagnostics=error.diagnostics | {"total_seconds": time.perf_counter() - started},
            X=[],
            Y=[],
        )
    extraction_started = time.perf_counter()
    for name, state in [
        ("initial_X_sha256", result.initial_solution),
        ("final_X_sha256", result.solution),
    ]:
        active = sorted(key for key, value in state.X.items() if value == 1)
        metadata[name] = hashlib.sha256(json.dumps(active).encode()).hexdigest()
    measured = measure_solution(instance, result.solution, alpha=options.alpha, details=details)
    initial = measure_solution(instance, result.initial_solution, alpha=options.alpha)["metrics"]
    if include_solution:
        document = solution_document(
            instance, result.solution, stop_reason=result.stop_reason, metadata=metadata
        )
    else:
        document = dict(status="solved", stop_reason=result.stop_reason, metadata=metadata)
    if include_solution:
        initial_document = solution_document(
            instance,
            result.initial_solution,
            stop_reason="Initial SSP assignment before TS.",
            metadata={},
        )
        document["initial_solution"] = {key: initial_document[key] for key in ["X", "Y"]}
    document.update(measured)
    if result.stale_solution is not None:
        # Decoupled endpoint before the final SSP (the pre-fix reported values).
        document["stale_metrics"] = measure_solution(
            instance, result.stale_solution, alpha=options.alpha
        )["metrics"]
    if options.trace:
        document["trace"] = result.trace
    document.update(
        schema_version=1,
        initial_metrics=initial,
        diagnostics=result.diagnostics
        | {
            "extraction_seconds": time.perf_counter() - extraction_started,
            "total_seconds": time.perf_counter() - started,
        },
    )
    return document
