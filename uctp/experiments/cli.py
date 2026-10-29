"""General experiment and analysis CLI; no configuration files or plots."""

import argparse
from pathlib import Path

from ..construction import InitialStrategy
from .analysis import analyze
from .runner import run_grid


def build_parser() -> argparse.ArgumentParser:
    """Describe grid execution and general result analysis arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run or resume a sequential grid, exporting raw results.")
    run.add_argument("--inputs", type=Path, nargs="+", required=True)
    run.add_argument("--input-format", choices=("instance", "legacy"), default="instance")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--strategies", nargs="+", choices=list(InitialStrategy), default=["Compatibility Matrix"]
    )
    run.add_argument("--approaches", nargs="+", default=["Embedded", "Decoupled-50"])
    run.add_argument("--alphas", type=float, nargs="+", default=[1])
    run.add_argument("--local-search", nargs="+", choices=("on", "off"), default=["on"])
    seeds = run.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", type=int, nargs="+")
    seeds.add_argument(
        "--runs", type=int, default=1, help="Number of seeds drawn from master-seed."
    )
    run.add_argument("--master-seed", type=int, default=703)
    budgets = run.add_mutually_exclusive_group()
    budgets.add_argument("--time-budgets", type=float, nargs="+")
    budgets.add_argument("--iteration-budgets", type=int, nargs="+")
    run.add_argument(
        "--iteration-ceiling",
        type=int,
        default=1000000,
        help="Secondary total-move cap for timed runs; default: 1000000.",
    )
    run.add_argument("--max-retries", type=int, default=500)
    run.add_argument(
        "--stagnation", type=int, help="Nonimproving-cycle limit; default: total iteration cap."
    )
    run.add_argument("--neighborhood-size", type=int, default=50)
    run.add_argument(
        "--max-daily-load",
        type=int,
        help="Override legacy daily blocks; otherwise read MAX_CARGA_DIARIA (fallback: 8).",
    )
    run.add_argument("--max-block-load", type=int, default=15)
    run.add_argument(
        "--group", help="Override the default label from each instance's parent directory."
    )
    run.add_argument(
        "--tag", action="append", default=[], help="Research label NAME=VALUE, repeatable."
    )
    run.add_argument("--details", action="store_true")
    run.add_argument("--save-solutions", action="store_true")
    run.add_argument(
        "--no-final-ssp",
        action="store_true",
        help="Decoupled: report the block-start Y instead of re-solving the SSP on the final X.",
    )
    run.add_argument(
        "--trace", action="store_true", help="Record native objective values at accepted moves."
    )
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume", action="store_true")
    analysis = commands.add_parser(
        "analyze", help="Export raw data, descriptives and optional comparisons."
    )
    analysis.add_argument("--inputs", type=Path, nargs="+", required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--metrics", nargs="+", default=["F_ref", "R_imb", "R_dis", "conflicts"])
    analysis.add_argument(
        "--group-by", nargs="+", help="Summary dimensions; default: recorded experimental controls."
    )
    analysis.add_argument(
        "--where", action="append", default=[], help="Filter FIELD=VALUE1,VALUE2; repeat for AND."
    )
    analysis.add_argument(
        "--compare",
        nargs="+",
        default=[],
        help="Factors to contrast separately, holding others fixed.",
    )
    analysis.add_argument("--comparison-mode", choices=("paired", "independent"), default="paired")
    analysis.add_argument("--pair-by", nargs="+", default=["input_sha256", "seed"])
    analysis.add_argument("--unit", choices=("instance", "run"), default="instance")
    analysis.add_argument("--instance-field", default="input_sha256")
    analysis.add_argument(
        "--strata", nargs="*", help="Explicit fixed comparison dimensions; default: other controls."
    )
    analysis.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Bootstrap/permutation replicates; default: 10000.",
    )
    analysis.add_argument("--analysis-seed", type=int, default=20261026)
    analysis.add_argument(
        "--include-failed-metrics",
        action="store_true",
        help="Include recorded nonmissing diagnostics of failed runs, e.g. construction time.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute a research command; report invalid input with exit code two."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_grid(args)
            return 130 if result.get("status") == "interrupted" else 0
        result = analyze(
            args.inputs,
            args.output,
            metrics=args.metrics,
            group_by=args.group_by,
            filters=args.where,
            compare=args.compare,
            mode=args.comparison_mode,
            pair_by=args.pair_by,
            unit=args.unit,
            instance_field=args.instance_field,
            strata=args.strata,
            replicates=args.bootstrap,
            seed=args.analysis_seed,
            include_failed=args.include_failed_metrics,
        )
        print(
            f"Analyzed {result['methods']['records']} runs; {len(result['contrasts'])} quality contrasts. Output: {args.output}"
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0
