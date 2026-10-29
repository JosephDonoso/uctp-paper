"""Regenerate the manuscript tables (LaTeX) and figures (PDF) from raw campaign results.

Every number in the generated files comes from ``raw.csv`` exported by
``python -m uctp.experiments run`` (or from the baseline JSON for the fixed
institutional references). No value is typed by hand.

Subcommands
-----------
real-weights       Real instance: outcomes per approach and weight at one budget
                   (table + weight-response figure + comparison with baselines).
quality-time       Real instance: outcomes and iterations per time budget
                   (table + quality-time figure).
synthetic-changes  Synthetic groups: paired change from the lowest to the highest
                   weight per approach (bootstrap CI) and iteration throughput.

Requires pandas and matplotlib (not dependencies of the solver).

Examples
--------
python tools/paper_artifacts.py real-weights --raw results/real/raw.csv --budget 30 \
    --baselines tools/real_baselines.json --alpha-compare 0.25 --out paper_out
python tools/paper_artifacts.py quality-time --raw results/real_curve/raw.csv --out paper_out
python tools/paper_artifacts.py synthetic-changes --raw results/synthetic/raw.csv \
    --budget 30 --approaches Embedded Decoupled-50 --out paper_out
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["F_ref", "R_imb", "R_dis", "conflicts"]
LABELS = {
    "F_ref": r"$F_{\mathrm{ref}}$",
    "R_imb": r"$R_{\mathit{imb}}$",
    "R_dis": r"$R_{\mathit{dis}}$",
    "conflicts": "Conflicts",
}
PLOT_LABELS = {"F_ref": "$F_{ref}$", "R_imb": "$R_{imb}$", "R_dis": "$R_{dis}$", "conflicts": "Conflicts"}
DECIMALS = {"F_ref": 3, "R_imb": 3, "R_dis": 3, "conflicts": 2}


# ----------------------------------------------------------------------------- data
def load(raw: Path, *, budget_kind: str | None = None, budget: float | None = None) -> pd.DataFrame:
    df = pd.read_csv(raw)
    df = df[df["status"] == "solved"].copy()
    for column in ["alpha", "budget", "completed_iterations", "optimizer_seconds", *METRICS]:
        df[column] = pd.to_numeric(df[column])
    if budget_kind is not None:
        df = df[df["budget_kind"] == budget_kind]
    if budget is not None:
        df = df[np.isclose(df["budget"], budget)]
    if df.empty:
        raise SystemExit("No solved runs match the requested filters.")
    return df


def fmt(value: float, metric: str) -> str:
    return f"{value:.{DECIMALS[metric]}f}"


def t_interval(values: np.ndarray) -> tuple[float, float]:
    from scipy import stats

    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    half = stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)
    return values.mean() - half, values.mean() + half


def bootstrap_mean(values: np.ndarray, *, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def alpha_text(alpha: float) -> str:
    return f"{alpha:g}"


# ------------------------------------------------------------------- real: weights
def real_weights(args) -> None:
    df = load(args.raw, budget_kind="seconds", budget=args.budget)
    if args.group:
        df = df[df["group"] == args.group]
    approaches = args.approaches or sorted(df["approach"].unique())
    alphas = sorted(df["alpha"].unique())
    lines = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcrrrrr@{}}",
        r"\hline",
        r"Approach & $\alpha$ & $n$ & " + " & ".join(LABELS[m] for m in METRICS) + r" \\",
        r"\hline",
    ]
    for approach in approaches:
        for alpha in alphas:
            cell = df[(df["approach"] == approach) & np.isclose(df["alpha"], alpha)]
            if cell.empty:
                continue
            values = [
                f"{fmt(cell[m].mean(), m)} $\\pm$ {fmt(cell[m].std(ddof=1), m)}" for m in METRICS
            ]
            lines.append(
                f"{approach} & {alpha_text(alpha)} & {len(cell)} & " + " & ".join(values) + r" \\"
            )
    lines += [r"\hline", r"\end{tabular*}", ""]
    write(args.out / f"real_weights_{args.budget:g}s.tex", "\n".join(lines))

    baselines = json.loads(args.baselines.read_text(encoding="utf-8")) if args.baselines else {}
    _weight_figure(df, approaches, alphas, baselines, args.out / f"real_weight_response_{args.budget:g}s.pdf")

    if args.alpha_compare is not None:
        _baseline_table(df, approaches, args.alpha_compare, baselines, args)


def _weight_figure(df, approaches, alphas, baselines, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    positions = np.arange(len(alphas))
    width = 0.12
    for axis, metric in zip(axes.flat, METRICS):
        for offset, approach in enumerate(approaches):
            means, lows, highs = [], [], []
            for alpha in alphas:
                values = df[(df["approach"] == approach) & np.isclose(df["alpha"], alpha)][metric].to_numpy()
                low, high = t_interval(values)
                means.append(values.mean() if len(values) else np.nan)
                lows.append(low)
                highs.append(high)
            means, lows, highs = map(np.array, (means, lows, highs))
            x = positions + (offset - (len(approaches) - 1) / 2) * width
            axis.errorbar(x, means, yerr=[means - lows, highs - means], marker="o", capsize=3, label=approach)
        styles = [("--", "0.2"), (":", "0.35"), ("-.", "0.55")]
        for (name, values), (style, color) in zip(baselines.items(), styles * 3):
            if metric in values:
                axis.axhline(values[metric], linestyle=style, linewidth=1, color=color, label=name)
        axis.set_xticks(positions, [alpha_text(a) for a in alphas])
        axis.set_xlabel(r"$\alpha$")
        axis.set_title(PLOT_LABELS[metric])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def _baseline_table(df, approaches, alpha, baselines, args) -> None:
    header = list(baselines) + [f"{a} $\\alpha={alpha_text(alpha)}$" for a in approaches]
    lines = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "r" * len(header) + r"@{}}",
        r"\hline",
        "Metric & " + " & ".join(header) + r" \\",
        r"\hline",
    ]
    improve_rows = []
    for metric in METRICS:
        cells = [fmt(baselines[name][metric], metric) for name in baselines]
        for approach in approaches:
            cell = df[(df["approach"] == approach) & np.isclose(df["alpha"], alpha)][metric]
            cells.append(fmt(cell.mean(), metric) if len(cell) else "--")
        lines.append(f"{LABELS[metric]} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular*}", ""]
    write(args.out / f"real_baselines_alpha{alpha_text(alpha)}.tex", "\n".join(lines))
    # Per-run simultaneous improvement counts against each baseline (all four metrics).
    for approach in approaches:
        runs = df[(df["approach"] == approach) & np.isclose(df["alpha"], alpha)]
        for name, values in baselines.items():
            better = np.ones(len(runs), dtype=bool)
            for metric in METRICS:
                better &= runs[metric].to_numpy() < values[metric]
            improve_rows.append(
                dict(approach=approach, alpha=alpha, baseline=name, runs=len(runs), all_four_better=int(better.sum()))
            )
    pd.DataFrame(improve_rows).to_csv(args.out / f"real_baselines_alpha{alpha_text(alpha)}_counts.csv", index=False)
    print(f"wrote {args.out / f'real_baselines_alpha{alpha_text(alpha)}_counts.csv'}")


# ------------------------------------------------------------------ quality vs time
def quality_time(args) -> None:
    df = load(args.raw, budget_kind="seconds")
    if args.group:
        df = df[df["group"] == args.group]
    approaches = args.approaches or sorted(df["approach"].unique())
    alphas = sorted(df["alpha"].unique())
    budgets = sorted(df["budget"].unique())
    lines = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcrrr" + "r" * len(METRICS) + r"@{}}",
        r"\hline",
        r"Approach & $\alpha$ & Budget (s) & $n$ & Iterations & "
        + " & ".join(LABELS[m] for m in METRICS)
        + r" \\",
        r"\hline",
    ]
    for approach in approaches:
        for alpha in alphas:
            for budget in budgets:
                cell = df[(df["approach"] == approach) & np.isclose(df["alpha"], alpha) & np.isclose(df["budget"], budget)]
                if cell.empty:
                    continue
                values = [fmt(cell[m].mean(), m) for m in METRICS]
                lines.append(
                    f"{approach} & {alpha_text(alpha)} & {budget:g} & {len(cell)} & "
                    f"{cell['completed_iterations'].mean():.0f} & " + " & ".join(values) + r" \\"
                )
    lines += [r"\hline", r"\end{tabular*}", ""]
    write(args.out / "quality_time.tex", "\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, metric in zip(axes.flat, METRICS):
        for approach in approaches:
            for alpha in alphas:
                means, lows, highs = [], [], []
                for budget in budgets:
                    values = df[(df["approach"] == approach) & np.isclose(df["alpha"], alpha) & np.isclose(df["budget"], budget)][metric].to_numpy()
                    low, high = t_interval(values)
                    means.append(values.mean() if len(values) else np.nan)
                    lows.append(low)
                    highs.append(high)
                means, lows, highs = map(np.array, (means, lows, highs))
                axis.errorbar(budgets, means, yerr=[means - lows, highs - means], marker="o", capsize=3,
                              label=f"{approach}, $\\alpha$={alpha_text(alpha)}")
        axis.set_xscale("log")
        axis.minorticks_off()
        axis.set_xticks(budgets, [f"{b:g}" for b in budgets])
        axis.set_xlabel("Optimizer budget (s)")
        axis.set_title(PLOT_LABELS[metric])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = args.out / "quality_time.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ------------------------------------------------------------- synthetic: changes
def synthetic_changes(args) -> None:
    kind = "seconds" if args.budget_kind == "seconds" else "iterations"
    df = load(args.raw, budget_kind=kind, budget=args.budget)
    approaches = args.approaches or sorted(df["approach"].unique())
    df = df[df["approach"].isin(approaches)]
    groups = args.groups or sorted(df["group"].unique())
    low, high = df["alpha"].min(), df["alpha"].max()
    alphas = sorted(df["alpha"].unique())
    lines = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcr" + "r" * len(approaches) + r"@{}}",
        r"\hline",
        "Group & Metric & $n$ & " + " & ".join(approaches) + r" \\",
        r"\hline",
    ]
    throughput = []
    for group in groups:
        sub = df[df["group"] == group]
        # Instances complete across every approach and weight, as in the manuscript.
        needed = len(approaches) * len(alphas)
        counts = sub.groupby("input_sha256").size()
        complete = counts[counts == needed].index
        sub = sub[sub["input_sha256"].isin(complete)]
        n = len(complete)
        if n == 0:
            continue
        for metric in METRICS:
            cells = []
            for k, approach in enumerate(approaches):
                a = sub[sub["approach"] == approach]
                top = a[np.isclose(a["alpha"], high)].set_index("input_sha256")[metric]
                bottom = a[np.isclose(a["alpha"], low)].set_index("input_sha256")[metric]
                change = (top - bottom.reindex(top.index)).to_numpy()
                lo, hi = bootstrap_mean(change, replicates=args.replicates, seed=args.seed + k)
                cells.append(f"{fmt(change.mean(), metric)} [{fmt(lo, metric)}, {fmt(hi, metric)}]")
            lines.append(f"{group} & {LABELS[metric]} & {n} & " + " & ".join(cells) + r" \\")
        for approach in approaches:
            a = sub[sub["approach"] == approach]
            throughput.append(
                dict(group=group, approach=approach, n=n,
                     iterations_mean=a["completed_iterations"].mean(),
                     iterations_sd=a["completed_iterations"].std(ddof=1),
                     time_mean=a["optimizer_seconds"].mean(),
                     time_sd=a["optimizer_seconds"].std(ddof=1))
            )
    lines += [r"\hline", r"\end{tabular*}", ""]
    write(args.out / f"synthetic_changes_{alpha_text(low)}_to_{alpha_text(high)}.tex", "\n".join(lines))
    tp = pd.DataFrame(throughput)
    rows = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrr@{}}",
        r"\hline",
        r"Group & Approach & $n$ & Iterations & Time (s) \\",
        r"\hline",
    ]
    for _, r in tp.iterrows():
        rows.append(
            f"{r.group} & {r.approach} & {r.n} & {r.iterations_mean:.1f} $\\pm$ {r.iterations_sd:.1f} & "
            f"{r.time_mean:.3f} $\\pm$ {r.time_sd:.3f} \\\\"
        )
    rows += [r"\hline", r"\end{tabular*}", ""]
    write(args.out / "synthetic_throughput.tex", "\n".join(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("real-weights")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--budget", type=float, required=True, help="Optimizer budget in seconds.")
    p.add_argument("--group", default=None)
    p.add_argument("--approaches", nargs="+")
    p.add_argument("--baselines", type=Path, help="JSON with fixed baseline metrics.")
    p.add_argument("--alpha-compare", type=float, help="Weight compared with the baselines.")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=real_weights)

    p = sub.add_parser("quality-time")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--group", default=None)
    p.add_argument("--approaches", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=quality_time)

    p = sub.add_parser("synthetic-changes")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--budget-kind", choices=("seconds", "iterations"), default="seconds")
    p.add_argument("--approaches", nargs="+")
    p.add_argument("--groups", nargs="+")
    p.add_argument("--replicates", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=synthetic_changes)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
