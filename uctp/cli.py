"""Command-line resolution interface; inputs are data, settings are arguments."""

import argparse
import json
from pathlib import Path

from .construction import InitialStrategy
from .instance_io import write_document
from .planner import Approach, SolveOptions, validate_options
from .reporting import execute_run


def build_parser() -> argparse.ArgumentParser:
    """Describe the supported solving command and its explicit parameters."""
    defaults = SolveOptions()
    parser = argparse.ArgumentParser(
        description="Resolve X and Y using the manuscript's UCTP procedure."
    )
    parser.add_argument("--input", type=Path, required=True, help="Instance JSON path (data only).")
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination solution JSON path."
    )
    parser.add_argument("--input-format", choices=("instance", "legacy"), default="instance")
    parser.add_argument(
        "--strategy",
        choices=list(InitialStrategy),
        default=defaults.strategy,
        help="Initial strategy; default: Compatibility Matrix.",
    )
    parser.add_argument("--approach", choices=list(Approach), default=defaults.approach)
    parser.add_argument(
        "--alpha", type=float, default=defaults.alpha, help="Objective weight; default: 1."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=defaults.max_retries,
        help="Failed construction retries after the initial attempt; default: 500.",
    )
    parser.add_argument("--iterations", type=int, help="Embedded TS iteration limit; default: 200.")
    parser.add_argument(
        "--ssp-interval", type=int, help="Required for Decoupled-k: k TS cycles per block."
    )
    parser.add_argument(
        "--search-blocks", type=int, help="Required for Decoupled: maximum SSP/TS blocks."
    )
    parser.add_argument(
        "--max-stagnant-iterations", type=int, default=defaults.max_stagnant_iterations
    )
    parser.add_argument(
        "--neighborhood-size",
        type=int,
        default=defaults.neighborhood_size,
        help="Relocation attempts per neighborhood, from 1 to 50; default: 50.",
    )
    parser.add_argument(
        "--time-limit", type=float, help="Nominal optimizer seconds; excludes construction."
    )
    parser.add_argument("--seed", type=int, help="Seed for Python random and NumPy.")
    parser.add_argument(
        "--local-search",
        choices=("on", "off"),
        default="on",
        help="Enable SSP local refinement, or use backtracking only.",
    )
    parser.add_argument(
        "--total-iterations", type=int, help="Optional accepted-move cap across all blocks."
    )
    parser.add_argument("--group", help="Dataset label recorded for analysis.")
    parser.add_argument("--run-id", help="Optional unique identifier for the research run.")
    parser.add_argument(
        "--trace", action="store_true", help="Record native-objective values at accepted TS moves."
    )
    parser.add_argument(
        "--details", action="store_true", help="Export per-student and per-section raw data."
    )
    parser.add_argument(
        "--max-daily-load",
        type=int,
        default=None,
        help="Daily blocks: override legacy MAX_CARGA_DIARIA; otherwise read input (fallback: 8).",
    )
    parser.add_argument(
        "--max-block-load", type=int, default=15, help="Simultaneous lecture capacity; default: 15."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Resolve an instance; print only operational status and output location."""
    parser = build_parser()
    args = parser.parse_args(argv)
    approach = Approach(args.approach)
    if approach == Approach.DECOUPLED:
        if args.ssp_interval is None or args.search_blocks is None:
            parser.error("Decoupled requires --ssp-interval and --search-blocks.")
        if args.iterations is not None:
            parser.error("Use --ssp-interval for Decoupled, not --iterations.")
        iterations = args.ssp_interval
        search_blocks = args.search_blocks
    else:
        if args.ssp_interval is not None or args.search_blocks is not None:
            parser.error("--ssp-interval and --search-blocks apply only to Decoupled.")
        iterations = (
            args.iterations if args.iterations is not None else SolveOptions().iterations_per_search
        )
        search_blocks = 1
    options = SolveOptions(
        strategy=InitialStrategy(args.strategy),
        approach=approach,
        alpha=args.alpha,
        max_retries=args.max_retries,
        iterations_per_search=iterations,
        search_blocks=search_blocks,
        max_stagnant_iterations=args.max_stagnant_iterations,
        neighborhood_size=args.neighborhood_size,
        time_limit_seconds=args.time_limit,
        seed=args.seed,
        local_search=args.local_search == "on",
        total_iteration_limit=args.total_iterations,
        trace=args.trace,
    )
    try:
        validate_options(options)
        if args.input.resolve() == args.output.resolve():
            raise ValueError("Input and output must be different files.")
        if args.output.exists():
            raise ValueError("The output already exists; choose a new destination.")
        document = execute_run(
            args.input,
            options,
            input_format=args.input_format,
            group=args.group,
            run_id=args.run_id,
            details=args.details,
            max_daily_load=args.max_daily_load,
            max_block_load=args.max_block_load,
        )
        write_document(args.output, document)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    solved = document["status"] == "solved"
    print(
        f"{'Solved.' if solved else 'Unresolved.'} {document['stop_reason']} Result: {args.output}"
    )
    return 0 if solved else 1
