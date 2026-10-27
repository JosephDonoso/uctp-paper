"""Construction and Embedded/Decoupled orchestration, without result analysis."""

import random
import time
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from .constraints import InstitutionalConstraints
from .construction import ConstructionFailure, InitialStrategy, InitialTimetabling
from .domain import Instance, Solution
from .objective import SearchObjective
from .sectioning import StudentSectioning
from .tabu import TabuSearch


class Approach(StrEnum):
    """SSP integration approaches from the manuscript."""

    EMBEDDED = "Embedded"
    DECOUPLED = "Decoupled"


@dataclass(frozen=True)
class SolveOptions:
    """Explicit execution parameters; no configuration-file loading."""

    strategy: InitialStrategy = InitialStrategy.COMPATIBILITY_MATRIX
    approach: Approach = Approach.EMBEDDED
    alpha: float = 1.0
    max_retries: int = 500
    iterations_per_search: int = 200
    search_blocks: int = 1
    max_stagnant_iterations: int = 200
    neighborhood_size: int = 50
    time_limit_seconds: float | None = None
    seed: int | None = None
    local_search: bool = True
    total_iteration_limit: int | None = None
    trace: bool = False


@dataclass(frozen=True)
class SolveResult:
    """Solution, termination reason and optional endpoint diagnostics."""

    solution: Solution
    stop_reason: str
    diagnostics: dict = field(default_factory=dict, compare=False)
    initial_solution: Solution | None = field(default=None, compare=False)
    trace: list[dict] = field(default_factory=list, compare=False)


def solve(
    instance: Instance, options: SolveOptions, *, collect_metrics: bool = False
) -> SolveResult:
    """Resolve an institutional timetable and section assignments.

    Notes
    -----
    Decoupled alternates SSP with blocks of TS. Each TS call retains its best
    evaluated pair; tabu memory persists across blocks. The nominal time limit
    applies inside each TS call and is also checked after each outer block,
    starting after initial construction. Endpoint extraction is performed by
    the reporting layer after solving; instrumentation does not consume RNG.
    """
    validate_options(options)
    if options.seed is not None:
        random.seed(options.seed)
        np.random.seed(options.seed)
    constraints = InstitutionalConstraints(instance)
    objective = SearchObjective(instance, alpha=options.alpha, cv_tolerance=0.10, idle_tolerance=4)
    sectioning = StudentSectioning(instance, objective, local_search=options.local_search)
    construction = InitialTimetabling(
        instance,
        constraints,
        strategy=options.strategy,
        max_retries=options.max_retries,
        beta=0.25,
        cv_tolerance=0.10,
    )
    search = TabuSearch(
        instance,
        constraints,
        objective,
        sectioning,
        embedded=options.approach == Approach.EMBEDDED,
        neighborhood_size=options.neighborhood_size,
        max_stagnant_iterations=options.max_stagnant_iterations,
        time_limit_seconds=options.time_limit_seconds,
        collect_trace=options.trace,
    )
    construction_started = time.perf_counter()
    try:
        solution = construction.solve()
    except ConstructionFailure as error:
        error.diagnostics = {
            "construction_seconds": time.perf_counter() - construction_started,
            "retries_used": construction._retry_count,
            "construction_attempts": construction._retry_count + 1,
            "completed_iterations": 0,
            "search_blocks_completed": 0,
            "ssp_calls": 0,
        }
        raise
    construction_seconds = time.perf_counter() - construction_started
    best_solution = solution.copy()
    best_objective = float("inf")
    start_time = time.time()
    optimizer_started = time.perf_counter()
    initial_solution = None
    capture_seconds = 0.0
    for _ in range(options.search_blocks):
        sectioning.solve(solution)
        if collect_metrics and initial_solution is None:
            capture_started = time.perf_counter()
            initial_solution = solution.copy()
            capture_seconds = time.perf_counter() - capture_started
            start_time += capture_seconds
        limit = options.iterations_per_search
        if options.total_iteration_limit is not None:
            limit = min(limit, options.total_iteration_limit - search.completed_iterations)
        reason = search.solve(solution, max_iterations=limit)
        current_objective = objective.evaluate(solution)
        if current_objective < best_objective:
            best_objective = current_objective
            best_solution = solution.copy()
        if (
            options.time_limit_seconds is not None
            and time.time() - start_time >= options.time_limit_seconds
        ):
            reason = "Time limit reached."
            break
        if (
            options.total_iteration_limit is not None
            and search.completed_iterations >= options.total_iteration_limit
        ):
            reason = "Total iteration limit reached."
            break
    else:
        if options.search_blocks > 1:
            reason = "Search-block limit reached."
    diagnostics = {
        "construction_seconds": construction_seconds,
        "optimizer_seconds": time.perf_counter() - optimizer_started - capture_seconds,
        "retries_used": construction._retry_count,
        "construction_attempts": construction._retry_count + 1,
        "completed_iterations": search.completed_iterations,
        "search_blocks_completed": search.search_calls,
        "ssp_calls": sectioning.solve_calls,
    }
    return SolveResult(best_solution, reason, diagnostics, initial_solution, search.trace)


def validate_options(options: SolveOptions) -> None:
    """Reject unsupported or ambiguous execution settings before solving."""
    import math

    if not isinstance(options.strategy, InitialStrategy) or not isinstance(
        options.approach, Approach
    ):
        raise ValueError("Strategy and approach must use the declared labels.")
    if not math.isfinite(options.alpha) or options.alpha < 0:
        raise ValueError("alpha must be finite and nonnegative.")
    for name in (
        "max_retries",
        "iterations_per_search",
        "search_blocks",
        "max_stagnant_iterations",
        "neighborhood_size",
    ):
        value = getattr(options, name)
        minimum = 0 if name == "max_retries" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if options.neighborhood_size > 50:
        raise ValueError("neighborhood_size cannot exceed the manuscript limit of 50.")
    if type(options.local_search) is not bool:
        raise ValueError("local_search must be boolean.")
    if type(options.trace) is not bool:
        raise ValueError("trace must be boolean.")
    if options.total_iteration_limit is not None and (
        type(options.total_iteration_limit) is not int or options.total_iteration_limit < 1
    ):
        raise ValueError("total_iteration_limit must be a positive integer.")
    if options.approach == Approach.EMBEDDED and options.search_blocks != 1:
        raise ValueError("Embedded uses one search block.")
    if options.time_limit_seconds is not None:
        if not math.isfinite(options.time_limit_seconds) or options.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be finite and positive.")
    if options.seed is not None and (
        type(options.seed) is not int or not 0 <= options.seed < 2**32
    ):
        raise ValueError("seed must be an integer in [0, 2**32).")
