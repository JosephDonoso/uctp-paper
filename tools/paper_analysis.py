"""Regenerate every table, figure and cited number of the manuscript from raw records.

Usage (from the repository root, after ``tools/campaigns.py merge``):

    python tools/paper_analysis.py --results results --out paper_out

Inputs are the merged ``raw.csv`` / ``checkpoints.csv`` tables of the campaigns
(see docs/CAMPAIGNS.md) and ``tools/real_baselines.json``. Outputs:

    <out>/tables/*.tex     tabular bodies included by main.tex and supplementary.tex
    <out>/figures/*.pdf    figures
    <out>/numbers.json     every number cited in the text, by key

All resampling uses fixed seeds, so the outputs are deterministic.

Statistical procedures
----------------------
* Exact two-sided sign tests on paired differences, ties excluded.
* Exact McNemar tests (binomial test on discordant pairs).
* Holm adjustment within each stated family.
* 95 % percentile bootstrap intervals of mean paired differences (10,000 resamples).
* Two-factor repeated-measures ANOVA (both factors within instances) computed on
  orthonormal effect contrasts, with the Greenhouse-Geisser correction; partial
  eta squared. Each test is accompanied by a centered bootstrap test: the effect
  contrasts are centered at zero (the null), instances are resampled with
  replacement, and p is the share of bootstrap F statistics at least as large as
  the observed one, (1 + count) / (B + 1).
* Fixed-weight follow-ups: paired t-tests and a centered bootstrap test of the
  mean paired difference, each Holm-adjusted across the family.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

B = 10_000
SEED = 20261026
GROUPS = ["Dispersion-Low", "Dispersion-High", "BASE", "Offer-Low", "Offer-High"]
DISPERSION = ["Dispersion-Low", "BASE", "Dispersion-High"]
OFFER = ["Offer-Low", "BASE", "Offer-High"]
STRATEGIES = {"Random": "R", "Prioritization Only": "PO", "Compatibility Matrix": "CM"}
ALPHAS = [0, 0.125, 0.25, 0.5, 1, 2, 4]
METRICS = ["R_imb", "R_dis", "conflicts", "F_ref"]
TEX = {"F_ref": r"$F_{\mathrm{ref}}$", "R_imb": r"$R_{\mathit{imb}}$", "R_dis": r"$R_{\mathit{dis}}$",
       "conflicts": "Conflicts", "F": "$F$"}
PLOT = {"F_ref": r"$F_{\mathrm{ref}}$", "R_imb": r"$R_{\mathrm{imb}}$", "R_dis": r"$R_{\mathrm{dis}}$",
        "conflicts": "Conflicts", "F": "$F$"}
# Validated categorical slots (blue, orange, aqua, yellow); identity also carried by markers.
COLORS = {"Embedded": "#2a78d6", "Decoupled-50": "#eb6834", "Decoupled-25": "#1baf7a",
          "Decoupled-100": "#eda100", "R": "#2a78d6", "PO": "#eb6834", "CM": "#1baf7a"}
MARKERS = {"Embedded": "o", "Decoupled-50": "s", "Decoupled-25": "^", "Decoupled-100": "D",
           "R": "o", "PO": "s", "CM": "^"}
SHORT = {"Embedded": "Embedded", "Decoupled-25": "D25", "Decoupled-50": "D50", "Decoupled-100": "D100"}
NUM: dict = {}


# ----------------------------------------------------------------------------- helpers
def g(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


def dec(metric: str) -> int:
    return 1 if metric == "conflicts" else 3


def pfmt(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def holm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def sign_test(diff: np.ndarray) -> tuple[int, int, float]:
    """Return (negative count, non-tied count, two-sided exact p)."""
    nonzero = diff[diff != 0]
    negative = int((nonzero < 0).sum())
    n = len(nonzero)
    return negative, n, (stats.binomtest(negative, n, 0.5).pvalue if n else 1.0)


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    return only_a, only_b, (stats.binomtest(only_a, n, 0.5).pvalue if n else 1.0)


def boot_ci(values: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(B, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_mean_p(values: np.ndarray, seed: int = SEED) -> float:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(B, len(values)))
    means = centered[idx].mean(axis=1)
    return float((1 + (np.abs(means) >= abs(values.mean())).sum()) / (B + 1))


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def t_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))
    return values.mean() - half, values.mean() + half


def contrasts(levels: int) -> np.ndarray:
    """Orthonormal contrasts (levels x levels-1), orthogonal to the constant."""
    helmert = np.zeros((levels, levels - 1))
    for j in range(1, levels):
        helmert[:j, j - 1] = 1.0
        helmert[j, j - 1] = -j
        helmert[:, j - 1] /= np.linalg.norm(helmert[:, j - 1])
    return helmert


def rm_anova(cube: np.ndarray, seed: int) -> dict:
    """Two-factor within-subject ANOVA on cube[n, a, b]; returns stats per effect."""
    n, a, b = cube.shape
    flat = cube.reshape(n, a * b)
    ca, cb = contrasts(a), contrasts(b)
    effects = {
        "A": np.kron(ca, np.ones((b, 1)) / math.sqrt(b)),
        "B": np.kron(np.ones((a, 1)) / math.sqrt(a), cb),
        "AxB": np.kron(ca, cb),
    }
    out = {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    for name, contrast in effects.items():
        z = flat @ contrast
        d = z.shape[1]
        mean = z.mean(axis=0)
        ss_effect = n * float((mean ** 2).sum())
        ss_error = float(((z - mean) ** 2).sum())
        df1, df2 = d, d * (n - 1)
        f = (ss_effect / df1) / (ss_error / df2)
        cov = np.cov(z, rowvar=False).reshape(d, d)
        eps = float(np.trace(cov) ** 2 / (d * np.trace(cov @ cov))) if d > 1 else 1.0
        eps = min(1.0, max(eps, 1.0 / d))
        p_gg = float(stats.f.sf(f, eps * df1, eps * df2))
        centered = z - mean
        samples = centered[idx]                       # B x n x d
        smean = samples.mean(axis=1)
        sse = ((samples - smean[:, None, :]) ** 2).sum(axis=(1, 2))
        sf = (n * (smean ** 2).sum(axis=1) / df1) / (sse / df2)
        p_boot = float((1 + (sf >= f).sum()) / (B + 1))
        out[name] = dict(F=f, df1=df1, df2=df2, eps=eps, p_gg=p_gg, p_boot=p_boot,
                         eta=ss_effect / (ss_effect + ss_error))
    return out


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def tabular(spec: str, header: list[str], rows: list[str], width: bool = True) -> str:
    head = (r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + spec + "@{}}") if width \
        else (r"\begin{tabular}{" + spec + "}")
    tail = r"\end{tabular*}" if width else r"\end{tabular}"
    return "\n".join([head, r"\hline", *header, r"\hline", *rows, r"\hline", tail])


def longtable(spec: str, header: str, rows: list[str], caption: str, label: str) -> str:
    cols = spec.count("l") + spec.count("r") + spec.count("c")
    return "\n".join([
        r"\begingroup\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{" + spec + "}",
        r"\caption{" + caption + r"}\label{" + label + r"}\\",
        r"\hline", header, r"\hline", r"\endfirsthead",
        r"\multicolumn{" + str(cols) + r"}{l}{Table \thetable\ (continued)}\\",
        r"\hline", header, r"\hline", r"\endhead",
        r"\hline", r"\endfoot",
        *rows, r"\end{longtable}", r"\endgroup"])


def gname(group: str) -> str:
    return group


def plain_log(ax, axis="y"):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    target = ax.yaxis if axis == "y" else ax.xaxis
    if axis == "y":
        target.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    target.set_minor_formatter(NullFormatter())


def setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
        "grid.color": "#e4e3df", "grid.linewidth": 0.5, "pdf.fonttype": 42,
        "lines.linewidth": 1.5, "lines.markersize": 4,
    })
    return plt


# ----------------------------------------------------------------------------- loading
def load(results: Path, stage: str, name: str = "raw.csv") -> pd.DataFrame:
    df = pd.read_csv(results / stage / "merged" / name)
    if "strategy" in df:
        df["s"] = df["strategy"].map(STRATEGIES)
    return df


# ----------------------------------------------------------------------------- A: construction
def construction(results: Path, out: Path) -> None:
    a = load(results, "A_initial_construction")
    assert (a.status == "solved").all()
    rows = []
    for group in GROUPS:
        for s in ["R", "PO", "CM"]:
            x = a[(a.group == group) & (a.s == s)]
            rows.append(f"{group} & {s} & {len(x)}/{len(x)} & {x.construction_attempts.mean():.2f} & "
                        f"{x.construction_attempts.max():d} & {(x.construction_attempts == 1).mean() * 100:.0f} & "
                        f"{x.construction_seconds.median():.3f} \\\\")
    write(out / "tables/generation.tex", tabular(
        "lcrrrrr", [r"Group & Strategy & Successes & \shortstack{Mean\\attempts} & \shortstack{Max.\\attempts} & "
                    r"\shortstack{First-attempt\\success (\%)} & \shortstack{Median\\time (s)} \\"], rows))
    NUM["construction"] = dict(
        runs=len(a), max_attempts=int(a.construction_attempts.max()),
        first_attempt_share=round(100 * (a.construction_attempts == 1).mean(), 1),
        mean_attempts={s: round(a[a.s == s].construction_attempts.mean(), 3) for s in ["R", "PO", "CM"]},
        median_time={s: round(a[a.s == s].construction_seconds.median(), 3) for s in ["R", "PO", "CM"]},
        within_101=int((a.construction_attempts <= 101).sum()))
    # Initial quality (alpha = 1): all 50 instances per group.
    rows_common, rows_contrast, comp = [], [], []
    p_main, p_comp = [], []
    for group in GROUPS:
        x = a[a.group == group].pivot_table(index="instance", columns="s",
                                            values=["initial_F", "initial_R_imb", "initial_R_dis", "initial_conflicts"])
        for s in ["R", "PO", "CM"]:
            rows_common.append(f"{group} & {s} & {len(x)} & " + " & ".join(
                f"${x[(m, s)].mean():.{d}f} \\pm {x[(m, s)].std(ddof=1):.{d}f}$"
                for m, d in [("initial_F", 2), ("initial_R_imb", 2), ("initial_R_dis", 2), ("initial_conflicts", 1)]) + r" \\")
        for other in ["R", "PO"]:
            diff = (x[("initial_F", "CM")] - x[("initial_F", other)]).to_numpy()
            neg, n, p = sign_test(diff)
            lo, hi = boot_ci(diff)
            rows_contrast.append([group, f"CM--{other}", len(diff), f"{neg}/{n}", diff.mean(), lo, hi, p])
            p_main.append(p)
            for metric in ["initial_R_imb", "initial_R_dis"]:
                d2 = (x[(metric, "CM")] - x[(metric, other)]).to_numpy()
                neg2, n2, p2 = sign_test(d2)
                comp.append([group, other, metric, neg2, n2, d2.mean(), p2])
                p_comp.append(p2)
    adj = holm(p_main)
    rows = [f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]:.2f} & [{r[5]:.2f}, {r[6]:.2f}] & {pfmt(p)} \\\\"
            for r, p in zip(rows_contrast, adj)]
    write(out / "tables/initial_quality_contrasts.tex", tabular(
        "lcrrrrr", [r"Group & Contrast & $n$ & CM improved & Mean $\Delta F$ & 95\% CI & $p_{\mathrm{Holm}}$ \\"], rows))
    write(out / "tables/initial_quality_common.tex", tabular(
        "lcrrrrr", [r"Group & Strategy & $n$ & Initial $F$ & $R_{\mathit{imb}}$ & $R_{\mathit{dis}}$ & Conflicts \\"],
        rows_common))
    adjc = holm(p_comp)
    NUM["initial_quality"] = dict(
        improved_share_range=[round(100 * min(int(r[3].split('/')[0]) / r[2] for r in rows_contrast), 1),
                              round(100 * max(int(r[3].split('/')[0]) / r[2] for r in rows_contrast), 1)],
        significant=int((adj < 0.05).sum()), max_p_holm=float(adj.max()),
        ratio_mean_F={s: round(float(a[a.s == s].initial_F.mean()), 2) for s in ["R", "PO", "CM"]},
        comp_imb_supported=int(sum(1 for c, p in zip(comp, adjc) if c[2] == "initial_R_imb" and p < 0.05 and c[5] < 0)),
        comp_dis_supported=int(sum(1 for c, p in zip(comp, adjc) if c[2] == "initial_R_dis" and p < 0.05 and c[5] < 0)),
        comp_imb_against=int(sum(1 for c, p in zip(comp, adjc) if c[2] == "initial_R_imb" and p < 0.05 and c[5] > 0)),
        comp_dis_against=int(sum(1 for c, p in zip(comp, adjc) if c[2] == "initial_R_dis" and p < 0.05 and c[5] > 0)),
        conflicts_mean={s: round(float(a[a.s == s].initial_conflicts.mean()), 1) for s in ["R", "PO", "CM"]})
    # Figure: initial F distributions.
    plt = setup_plot()
    for family, groups in [("dispersion", DISPERSION), ("offer", OFFER)]:
        fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), sharey=False)
        for ax, group in zip(axes, groups):
            data = [a[(a.group == group) & (a.s == s)].initial_F.to_numpy() for s in ["R", "PO", "CM"]]
            box = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                             medianprops=dict(color="#0b0b0b", linewidth=1))
            for patch, s in zip(box["boxes"], ["R", "PO", "CM"]):
                patch.set_facecolor(COLORS[s]); patch.set_alpha(0.35); patch.set_edgecolor(COLORS[s])
            rng = np.random.default_rng(1)
            for i, (d, s) in enumerate(zip(data, ["R", "PO", "CM"]), start=1):
                ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=5, color=COLORS[s], alpha=0.7,
                           marker=MARKERS[s], linewidths=0)
            ax.set_xticks([1, 2, 3], ["R", "PO", "CM"])
            ax.set_title(group)
            ax.grid(axis="y")
        axes[0].set_ylabel(r"Initial $F$ ($\alpha=1$)")
        fig.tight_layout()
        fig.savefig(out / f"figures/initial_quality_{family}.pdf")
        plt.close(fig)


# ----------------------------------------------------------------------------- B: iteration budgets
def budgets(results: Path, out: Path) -> None:
    raw = load(results, "B_iteration_budgets")
    cp = load(results, "B_iteration_budgets", "checkpoints.csv")
    final_budget = int(raw.iteration_budget.iloc[0])
    keep = ["group", "instance", "s", "alpha"]
    last = raw[keep + ["F", "F_ref", "R_imb", "R_dis", "conflicts", "optimizer_seconds", "initial_F",
                       "initial_R_dis", "initial_conflicts"]].copy()
    last["iteration_budget"] = final_budget
    allb = pd.concat([cp[keep + ["iteration_budget", "F", "F_ref", "R_imb", "R_dis", "conflicts",
                                 "optimizer_seconds"]], last[keep + ["iteration_budget", "F", "F_ref", "R_imb",
                                                                     "R_dis", "conflicts", "optimizer_seconds"]]])
    budgets_list = sorted(allb.iteration_budget.unique())
    NUM["budgets"] = dict(final=final_budget, checkpoints=[int(b) for b in budgets_list[:-1]],
                          runs=len(raw), reached=bool(cp.reached.all()))
    # Persistence at the final budget.
    red = 100 * (raw.initial_F - raw.F) / raw.initial_F
    rows_gain = []
    for group in GROUPS:
        for s in ["R", "PO", "CM"]:
            vals = [red[(raw.group == group) & (raw.s == s) & (raw.alpha == al)].median() for al in [0.5, 1, 2]]
            rows_gain.append(f"{group} & {s} & " + " & ".join(f"{v:.1f}" for v in vals) + r" \\")
    write(out / "tables/optimization_relative_gain.tex", tabular(
        "lcrrr", [r"Group & Strategy & $\alpha=0.5$ & $\alpha=1$ & $\alpha=2$ \\"], rows_gain))
    NUM["persistence"] = dict(all_reduced=bool((raw.F < raw.initial_F).all()),
                              median_reduction_range=[round(float(red.groupby([raw.group, raw.s, raw.alpha]).median().min()), 1),
                                                      round(float(red.groupby([raw.group, raw.s, raw.alpha]).median().max()), 1)])
    def contrast_family(frame: pd.DataFrame, metric: str):
        res = []
        for al in [0.5, 1, 2]:
            for group in GROUPS:
                x = frame[(frame.group == group) & (frame.alpha == al)].pivot_table(
                    index="instance", columns="s", values=metric)
                for other in ["R", "PO"]:
                    diff = (x["CM"] - x[other]).to_numpy()
                    neg, n, p = sign_test(diff)
                    res.append(dict(alpha=al, group=group, other=other, neg=neg, n=n, mean=diff.mean(), p=p,
                                    mR=x["R"].mean(), mPO=x["PO"].mean(), mCM=x["CM"].mean(),
                                    sR=x["R"].std(ddof=1), sPO=x["PO"].std(ddof=1), sCM=x["CM"].std(ddof=1),
                                    diff=diff, nn=len(x)))
        ph = holm([r["p"] for r in res])
        for r, p in zip(res, ph):
            r["ph"] = p
        return res
    fin = allb[allb.iteration_budget == final_budget]
    res = contrast_family(fin, "F")
    rows = []
    for al in [0.5, 1, 2]:
        for group in GROUPS:
            r_r = next(r for r in res if r["alpha"] == al and r["group"] == group and r["other"] == "R")
            r_p = next(r for r in res if r["alpha"] == al and r["group"] == group and r["other"] == "PO")
            rows.append(f"{group} & {al:g} & {r_r['nn']} & ${r_r['mR']:.3f} \\pm {r_r['sR']:.3f}$ & "
                        f"${r_r['mPO']:.3f} \\pm {r_r['sPO']:.3f}$ & ${r_r['mCM']:.3f} \\pm {r_r['sCM']:.3f}$ & "
                        f"{pfmt(r_r['ph'])} & {pfmt(r_p['ph'])} \\\\")
    write(out / "tables/final_quality_weights.tex", tabular(
        "lcrrrrrr", [r" & & & \multicolumn{3}{c}{Final $F$ (mean $\pm$ SD)} & \multicolumn{2}{c}{$p_{\mathrm{Holm}}$} \\",
                     r"Group & $\alpha$ & $n$ & R & PO & CM & CM--R & CM--PO \\"], rows))
    lower = {al: sum(1 for r in res if r["alpha"] == al and r["mean"] < 0) for al in [0.5, 1, 2]}
    NUM["persistence"].update(lower_mean_CM=lower, supported=int(sum(1 for r in res if r["ph"] < 0.05 and r["mean"] < 0)),
                              against=int(sum(1 for r in res if r["ph"] < 0.05 and r["mean"] > 0)),
                              min_p_holm=float(min(r["ph"] for r in res)))
    # Component contrasts at the final budget (90 tests).
    comp_all = []
    for metric in ["R_imb", "R_dis", "conflicts"]:
        for al in [0.5, 1, 2]:
            for group in GROUPS:
                x = fin[(fin.group == group) & (fin.alpha == al)].pivot_table(index="instance", columns="s", values=metric)
                for other in ["R", "PO"]:
                    diff = (x["CM"] - x[other]).to_numpy()
                    neg, n, p = sign_test(diff)
                    lo, hi = boot_ci(diff)
                    comp_all.append(dict(metric=metric, alpha=al, group=group, other=other, neg=neg, n=n,
                                         mean=diff.mean(), lo=lo, hi=hi, p=p))
    ph = holm([c["p"] for c in comp_all])
    for c, p in zip(comp_all, ph):
        c["ph"] = p
    for metric, name in [("R_imb", "imbalance"), ("R_dis", "dissatisfaction"), ("conflicts", "conflicts")]:
        d = 1 if metric == "conflicts" else 3
        rows = []
        for group in GROUPS:
            for al in [0.5, 1, 2]:
                cr = next(c for c in comp_all if c["metric"] == metric and c["group"] == group and c["alpha"] == al and c["other"] == "R")
                cp_ = next(c for c in comp_all if c["metric"] == metric and c["group"] == group and c["alpha"] == al and c["other"] == "PO")
                rows.append(f"{group} & {al:g} & {cr['mean']:.{d}f} & [{cr['lo']:.{d}f}, {cr['hi']:.{d}f}] & {pfmt(cr['ph'])} & "
                            f"{cp_['mean']:.{d}f} & [{cp_['lo']:.{d}f}, {cp_['hi']:.{d}f}] & {pfmt(cp_['ph'])} \\\\")
        write(out / f"tables/final_metric_{name}.tex", tabular(
            "lcrrrrrr", [r" & & \multicolumn{3}{c}{CM--R} & \multicolumn{3}{c}{CM--PO} \\",
                         r"Group & $\alpha$ & Mean diff. & 95\% CI & $p_{\mathrm{Holm}}$ & Mean diff. & 95\% CI & $p_{\mathrm{Holm}}$ \\"], rows))
    sig = [c for c in comp_all if c["ph"] < 0.05]
    NUM["persistence"]["components_significant"] = [
        dict(metric=c["metric"], group=c["group"], alpha=c["alpha"], other=c["other"], mean=round(c["mean"], 3),
             neg=c["neg"], n=c["n"], p=round(c["ph"], 4)) for c in sig]
    inc = {al: dict(dis=int((raw[raw.alpha == al].R_dis > raw[raw.alpha == al].initial_R_dis).sum()),
                    conf=int((raw[raw.alpha == al].conflicts > raw[raw.alpha == al].initial_conflicts).sum()),
                    n=int((raw.alpha == al).sum())) for al in [0.5, 1, 2]}
    NUM["persistence"]["increases"] = inc
    # Supported CM advantage by budget (Holm within each budget, 30 tests).
    rows, by_budget = [], {}
    for k in budgets_list:
        r = contrast_family(allb[allb.iteration_budget == k], "F")
        by_budget[int(k)] = r
        cells = []
        for al in [0.5, 1, 2]:
            sub = [x for x in r if x["alpha"] == al]
            cells.append(f"{sum(x['mean'] < 0 for x in sub)}/10 & {sum(x['ph'] < 0.05 and x['mean'] < 0 for x in sub)}/10")
        rows.append(f"{int(k)} & " + " & ".join(cells) + r" \\")
    write(out / "tables/cm_by_budget.tex", tabular(
        "rrrrrrr", [r" & \multicolumn{2}{c}{$\alpha=0.5$} & \multicolumn{2}{c}{$\alpha=1$} & \multicolumn{2}{c}{$\alpha=2$} \\",
                    r"Iterations & \shortstack{Lower\\mean} & Supported & \shortstack{Lower\\mean} & Supported & "
                    r"\shortstack{Lower\\mean} & Supported \\"], rows))
    NUM["cm_by_budget"] = {int(k): dict(lower=int(sum(x["mean"] < 0 for x in r)),
                                        supported=int(sum(x["ph"] < 0.05 and x["mean"] < 0 for x in r)),
                                        against=int(sum(x["ph"] < 0.05 and x["mean"] > 0 for x in r)))
                           for k, r in by_budget.items()}
    # Short budgets: joint family of 60 contrasts at 25 and 50 iterations (full list in the supplement).
    short = []
    for k in [25, 50]:
        for al in [0.5, 1, 2]:
            for group in GROUPS:
                x = allb[(allb.iteration_budget == k) & (allb.group == group) & (allb.alpha == al)].pivot_table(
                    index="instance", columns="s", values="F")
                for other in ["R", "PO"]:
                    diff = (x["CM"] - x[other]).to_numpy()
                    neg, n, p = sign_test(diff)
                    lo, hi = boot_ci(diff)
                    short.append([group, k, al, other, len(diff), x["CM"].mean(), x[other].mean(), diff.mean(), lo, hi, p])
    ph = np.empty(len(short))
    for k in [25, 50]:  # Holm within each budget (30 contrasts), as in the main-text summary
        idx = [i for i, s in enumerate(short) if s[1] == k]
        ph[idx] = holm([short[i][-1] for i in idx])
    rows = [f"{s[0]} & {s[1]} & {s[2]:g} & {s[3]} & {s[4]} & {s[5]:.3f} & {s[6]:.3f} & {s[7]:.3f} [{s[8]:.3f}, {s[9]:.3f}] & {pfmt(p)} \\\\"
            for s, p in zip(short, ph)]
    write(out / "tables/cm_short_budget_contrasts.tex", longtable(
        "@{}llllrrrlr@{}",
        r"Group & Iter. & $\alpha$ & vs & $n$ & CM mean & Other mean & Mean diff. [95\% CI] & $p_{\mathrm{Holm}}$ \\",
        rows, "All fixed-budget CM objective contrasts at 25 and 50 iterations (Holm adjustment across the 30 contrasts of each budget).",
        "tab:cm_short_budget_contrasts"))
    NUM["short"] = {k: dict(lower=int(sum(1 for s in short if s[1] == k and s[7] < 0)),
                            supported=int(sum(1 for s, p in zip(short, ph) if s[1] == k and p < 0.05 and s[7] < 0)))
                    for k in [25, 50]}
    # Marginal gains between consecutive budgets.
    wide = allb.pivot_table(index=["group", "instance", "s", "alpha"], columns="iteration_budget", values="F")
    secs = allb.pivot_table(index=["group", "instance", "s", "alpha"], columns="iteration_budget", values="optimizer_seconds")
    steps = list(zip(budgets_list[:-1], budgets_list[1:]))
    gain_tables = {}
    for al, label in [(0.5, "half"), (1, "one"), (2, "two")]:
        rows = []
        for group in GROUPS:
            for s in ["R", "PO", "CM"]:
                sub = wide.xs((group, s, al), level=["group", "s", "alpha"])
                vals = [100 * ((sub[k1] - sub[k2]) / sub[k1]).mean() for k1, k2 in steps]
                rows.append(f"{group} & {s} & " + " & ".join(f"{v:.1f}" for v in vals) + r" \\")
        gain_tables[al] = rows
        write(out / f"tables/iteration_gain_{label}.tex", tabular(
            "lc" + "r" * len(steps), [r"Group & Str. & " + " & ".join(f"{int(a)}$\\to${int(b)}" for a, b in steps) + r" \\"], rows))
    step_rows, step_num = [], {}
    for k1, k2 in steps:
        gain = 100 * (wide[k1] - wide[k2]) / wide[k1]
        by_alpha = {al: gain.xs(al, level="alpha") for al in [0.5, 1, 2]}
        cell_means = gain.groupby(level=["group", "s", "alpha"]).mean()
        added = (secs[k2] - secs[k1])
        improved = (wide[k2] < wide[k1]).groupby(level="alpha").sum()
        step_rows.append(f"{int(k1)}$\\to${int(k2)} & " + " & ".join(
            f"{by_alpha[al].mean():.1f}" for al in [0.5, 1, 2]) + f" & {cell_means.min():.1f}--{cell_means.max():.1f} & "
            + " & ".join(f"{int(improved[al])}" for al in [0.5, 1, 2]) + f" & {added.mean():.1f} \\\\")
        step_num[f"{int(k1)}-{int(k2)}"] = dict(
            mean_gain={al: round(float(by_alpha[al].mean()), 2) for al in [0.5, 1, 2]},
            cell_range=[round(float(cell_means.min()), 1), round(float(cell_means.max()), 1)],
            improved={al: int(improved[al]) for al in [0.5, 1, 2]}, added_seconds=round(float(added.mean()), 2))
    write(out / "tables/iteration_steps.tex", tabular(
        "lrrrrrrrr", [r" & \multicolumn{3}{c}{Mean relative gain (\%)} & & \multicolumn{3}{c}{Runs improved (of 750)} & \\",
                      r"Step & $\alpha=0.5$ & $\alpha=1$ & $\alpha=2$ & \shortstack{Range of\\cell means} & $\alpha=0.5$ & "
                      r"$\alpha=1$ & $\alpha=2$ & \shortstack{Added\\time (s)} \\"], step_rows))
    NUM["steps"] = step_num
    NUM["budget_time"] = {int(k): round(float(allb[allb.iteration_budget == k].optimizer_seconds.mean()), 1) for k in budgets_list}
    # Dissatisfaction / conflict increases in late steps
    late = {}
    comp = allb.pivot_table(index=["group", "instance", "s", "alpha"], columns="iteration_budget", values=["R_dis", "conflicts"])
    for k1, k2 in steps[-3:]:
        late[f"{int(k1)}-{int(k2)}"] = {al: dict(
            dis=int((comp.xs(al, level="alpha")[("R_dis", k2)] > comp.xs(al, level="alpha")[("R_dis", k1)]).sum()),
            conf=int((comp.xs(al, level="alpha")[("conflicts", k2)] > comp.xs(al, level="alpha")[("conflicts", k1)]).sum()))
            for al in [0.5, 1, 2]}
    NUM["late_component_increases"] = late
    # Figure: mean F by budget (log scale), per group and weight.
    plt = setup_plot()
    for family, groups in [("dispersion", DISPERSION), ("offer", OFFER)]:
        fig, axes = plt.subplots(3, 3, figsize=(6.6, 5.6))
        for i, al in enumerate([0.5, 1, 2]):
            for j, group in enumerate(groups):
                ax = axes[i, j]
                for s in ["R", "PO", "CM"]:
                    sub = allb[(allb.group == group) & (allb.alpha == al) & (allb.s == s)]
                    m = sub.groupby("iteration_budget").F.mean()
                    ax.plot(m.index, m.values, marker=MARKERS[s], color=COLORS[s], label=s, markersize=3)
                ax.set_xscale("log"); plain_log(ax, "x")
                ax.set_xticks([25, 100, 500, 2500], ["25", "100", "500", "2500"])
                ax.grid(True, which="major")
                if i == 0:
                    ax.set_title(group)
                if j == 0:
                    ax.set_ylabel(f"Mean $F$ ($\\alpha={al:g}$)")
                if i == 2:
                    ax.set_xlabel("Iteration budget")
        axes[0, 0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"figures/iteration_gains_{family}.pdf")
        plt.close(fig)
    # Figure: mean paired final differences (CM minus R/PO) with bootstrap intervals.
    for family, groups in [("dispersion", DISPERSION), ("offer", OFFER)]:
        fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))
        for ax, al in zip(axes, [0.5, 1, 2]):
            ys, labels = [], []
            for gi, group in enumerate(groups):
                for oi, other in enumerate(["R", "PO"]):
                    r = next(x for x in res if x["alpha"] == al and x["group"] == group and x["other"] == other)
                    lo, hi = boot_ci(r["diff"])
                    y = gi * 2.6 + oi
                    ax.errorbar(r["mean"], y, xerr=[[r["mean"] - lo], [hi - r["mean"]]], fmt=MARKERS[other],
                                color=COLORS[other], capsize=2, markersize=4, linewidth=1)
                    ys.append(y); labels.append(f"{group}: CM vs {other}")
            ax.axvline(0, color="#52514e", linewidth=0.6)
            ax.set_yticks(ys, labels if al == 0.5 else [""] * len(ys))
            ax.invert_yaxis()
            ax.set_title(f"$\\alpha={al:g}$")
            ax.set_xlabel("Mean final $F$ difference")
            ax.grid(axis="x")
        fig.tight_layout()
        fig.savefig(out / f"figures/treatment_reliability_{family}.pdf")
        plt.close(fig)


# ----------------------------------------------------------------------------- C: approaches
def approaches(results: Path, out: Path) -> None:
    c = load(results, "C_synthetic_30s")
    assert (c.status == "solved").all()
    NUM["synthetic"] = dict(runs=len(c), instances_per_group=int(c.groupby("group").instance.nunique().min()))

    def cube(frame, factor_levels, metric, group):
        sub = frame[frame.group == group]
        piv = sub.pivot_table(index="instance", columns=["alpha", "approach"], values=metric)
        arr = np.array([[piv[(al, ap)].to_numpy() for ap in factor_levels] for al in ALPHAS])
        return arr.transpose(2, 0, 1)  # n x alpha x approach

    families = {"Decoupled": ["Decoupled-25", "Decoupled-50", "Decoupled-100"],
                "approaches": ["Embedded", "Decoupled-50"]}
    anova = {}
    for fam, levels in families.items():
        rows_all = []
        for metric in METRICS:
            for gi, group in enumerate(GROUPS):
                res = rm_anova(cube(c, levels, metric, group), seed=SEED + gi)
                for eff in ["A", "B", "AxB"]:
                    rows_all.append(dict(metric=metric, group=group, effect=eff, n=50, **res[eff]))
        pg = holm([r["p_gg"] for r in rows_all])
        pb = holm([r["p_boot"] for r in rows_all])
        for r, a1, a2 in zip(rows_all, pg, pb):
            r["p_gg_h"], r["p_b_h"] = a1, a2
        anova[fam] = rows_all
        sym = {"A": r"$\alpha$", "B": "$k$" if fam == "Decoupled" else "$A$",
               "AxB": r"$\alpha\times$" + ("$k$" if fam == "Decoupled" else "$A$")}
        for metric in METRICS:
            rows = [f"{r['group']} & {r['n']} & {sym[r['effect']]} & {r['F']:.2f} & {r['eps']:.3f} & {r['eta']:.3f} & "
                    f"{pfmt(r['p_gg_h']) if r['p_gg_h'] >= 0.001 else '$<0.001$'} & {r['p_b_h']:.4f} \\\\"
                    for r in rows_all if r["metric"] == metric]
            write(out / f"tables/factorial_{fam}_{metric}.tex", tabular(
                "lrlrrrrr", [r"Group & $n$ & Effect & $F$ & $\varepsilon_{\mathrm{GG}}$ & $\eta_p^2$ & "
                             r"$p_{\mathrm{GG,H}}$ & $p_{\mathrm{B,H}}$ \\"], rows))
    def summarize(fam, effect, metric=None):
        rows = [r for r in anova[fam] if r["effect"] == effect and (metric is None or r["metric"] == metric)]
        return dict(gg=int(sum(r["p_gg_h"] < 0.05 for r in rows)), boot=int(sum(r["p_b_h"] < 0.05 for r in rows)),
                    both=int(sum(r["p_gg_h"] < 0.05 and r["p_b_h"] < 0.05 for r in rows)), total=len(rows),
                    eta=[round(min(r["eta"] for r in rows), 3), round(max(r["eta"] for r in rows), 3)])
    NUM["anova"] = {fam: {f"{eff}_{m}": summarize(fam, eff, m) for eff in ["A", "B", "AxB"] for m in METRICS}
                    for fam in families}
    NUM["anova_detail"] = {fam: [dict(metric=r["metric"], group=r["group"], effect=r["effect"], eta=round(r["eta"], 3),
                                      p_gg_h=r["p_gg_h"], p_b_h=r["p_b_h"]) for r in rows]
                           for fam, rows in anova.items()}
    # Endpoint changes alpha 0 -> 4.
    def endpoint(approach, metric, group):
        piv = c[(c.group == group) & (c.approach == approach)].pivot_table(index="instance", columns="alpha", values=metric)
        d = (piv[4] - piv[0]).to_numpy()
        lo, hi = boot_ci(d)
        return d.mean(), lo, hi, 100 * (piv[4].mean() - piv[0].mean()) / piv[0].mean()
    rows, ep = [], {}
    for group in GROUPS:
        for metric in METRICS:
            cells = []
            for ap in ["Embedded", "Decoupled-50"]:
                m, lo, hi, pct = endpoint(ap, metric, group)
                d = dec(metric)
                cells.append(f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]")
                ep[(ap, metric, group)] = pct
            rows.append(f"{group} & {TEX[metric]} & " + " & ".join(cells) + r" \\")
    write(out / "tables/approach_endpoint_changes.tex", tabular("lcrr", [r"Group & Metric & Embedded & D50 \\"], rows))
    rows = []
    for group in GROUPS:
        for ap in ["Decoupled-25", "Decoupled-50", "Decoupled-100"]:
            cells = []
            for metric in METRICS:
                m, lo, hi, pct = endpoint(ap, metric, group)
                d = dec(metric)
                cells.append(f"\\shortstack[r]{{{m:.{d}f}\\\\{{[{lo:.{d}f}, {hi:.{d}f}]}}}}")
                ep[(ap, metric, group)] = pct
            rows.append(f"{group} & {SHORT[ap]} & " + " & ".join(cells) + r" \\")
    write(out / "tables/decoupled_endpoint_changes.tex", tabular(
        "llrrrr", [r"Group & $k$ & " + " & ".join(TEX[m] for m in METRICS) + r" \\"], rows))
    def rng_pct(aps, metric):
        v = [ep[(ap, metric, g_)] for ap in aps for g_ in GROUPS]
        return [round(min(v), 1), round(max(v), 1)]
    NUM["endpoint_pct"] = {"Embedded_imb": rng_pct(["Embedded"], "R_imb"), "D50_imb": rng_pct(["Decoupled-50"], "R_imb"),
                           "Decoupled_imb": rng_pct(["Decoupled-25", "Decoupled-50", "Decoupled-100"], "R_imb"),
                           "Embedded_dis": rng_pct(["Embedded"], "R_dis"), "Embedded_conf": rng_pct(["Embedded"], "conflicts"),
                           "D50_dis": rng_pct(["Decoupled-50"], "R_dis"), "D50_conf": rng_pct(["Decoupled-50"], "conflicts")}
    # 140 follow-up contrasts Embedded - D50.
    fol = []
    for group in GROUPS:
        for metric in METRICS:
            for al in ALPHAS:
                piv = c[(c.group == group) & (c.alpha == al)].pivot_table(index="instance", columns="approach", values=metric)
                d = (piv["Embedded"] - piv["Decoupled-50"]).to_numpy()
                t = stats.ttest_1samp(d, 0.0).pvalue
                fol.append(dict(group=group, metric=metric, alpha=al, n=len(d), mean=d.mean(), ci=boot_ci(d), pt=t,
                                pb=boot_mean_p(d)))
    pt = holm([f["pt"] for f in fol]); pb = holm([f["pb"] for f in fol])
    for f, a1, a2 in zip(fol, pt, pb):
        f["pth"], f["pbh"] = a1, a2
    rows = [f"{f['group']} & {TEX[f['metric']]} & {f['alpha']:g} & {f['n']} & {f['mean']:.3f} & "
            f"[{f['ci'][0]:.3f}, {f['ci'][1]:.3f}] & {pfmt(f['pth'])} & {pfmt(f['pbh'])} \\\\" for f in fol]
    write(out / "tables/supp_embedded_D50_140.tex", longtable(
        "@{}p{1.8cm}lrrrlrr@{}",
        r"Group & Metric & $\alpha$ & $n$ & Mean diff. & 95\% CI & $p_{t,\mathrm{H}}$ & $p_{\mathrm{B,H}}$ \\",
        rows, r"All fixed-weight synthetic contrasts between Embedded and Decoupled-50 (Embedded minus D50; Holm "
              r"adjustment across the 140 contrasts for each procedure).", "tab:supp_embedded_D50_140"))
    def supported(metric, al, sign):
        return [f["group"] for f in fol if f["metric"] == metric and f["alpha"] == al and f["pth"] < 0.05
                and f["pbh"] < 0.05 and np.sign(f["mean"]) == sign]
    NUM["followup"] = {f"{m}_{al:g}": dict(emb_lower=supported(m, al, -1), d50_lower=supported(m, al, 1))
                       for m in METRICS for al in ALPHAS}
    # Weight selection on synthetic data (applied later to the real instance).
    emb = c[c.approach == "Embedded"]
    pooled = emb.groupby("alpha").F_ref.mean()
    per_group = emb.groupby(["group", "alpha"]).F_ref.mean().unstack()
    NUM["alpha_selection"] = dict(pooled={f"{k:g}": round(v, 3) for k, v in pooled.items()},
                                  selected=float(pooled.idxmin()),
                                  per_group_argmin={gg: float(per_group.loc[gg].idxmin()) for gg in GROUPS})
    # Throughput at alpha = 1.
    rows = []
    for group in GROUPS:
        for ap in ["Embedded", "Decoupled-25", "Decoupled-50", "Decoupled-100"]:
            x = c[(c.group == group) & (c.approach == ap) & (c.alpha == 1)]
            rows.append(f"{group} & {ap} & {len(x)} & ${x.completed_iterations.mean():.1f} \\pm {x.completed_iterations.std(ddof=1):.1f}$ & "
                        f"${x.optimizer_seconds.mean():.3f} \\pm {x.optimizer_seconds.std(ddof=1):.3f}$ \\\\")
    write(out / "tables/synthetic_iteration_throughput.tex", tabular(
        "llrrr", [r"Group & Approach & $n$ & \shortstack{Iterations\\mean $\pm$ SD} & \shortstack{Time (s)\\mean $\pm$ SD} \\"], rows))
    thr = c[c.alpha == 1].groupby(["group", "approach"]).completed_iterations.mean()
    NUM["throughput"] = dict(base={ap: round(float(thr[("BASE", ap)]), 1) for ap in families["Decoupled"] + ["Embedded"]},
                             embedded_range=[round(float(thr.xs("Embedded", level="approach").min()), 1),
                                             round(float(thr.xs("Embedded", level="approach").max()), 1)],
                             max_time=round(float(c.optimizer_seconds.max()), 3),
                             mean_time={ap: round(float(c[c.approach == ap].optimizer_seconds.mean()), 3) for ap in c.approach.unique()})
    # Means table (for the text) by approach x alpha pooled and per group.
    NUM["synthetic_means"] = {ap: {f"{al:g}": {m: round(float(c[(c.approach == ap) & (c.alpha == al)][m].mean()), 3)
                                               for m in METRICS} for al in ALPHAS} for ap in c.approach.unique()}
    # Figures: factorial responses.
    plt = setup_plot()
    for fam, levels, fname in [("Decoupled", families["Decoupled"], "01_decoupled"),
                               ("approaches", families["approaches"], "02_embedded_vs_D50")]:
        for family, groups in [("dispersion", DISPERSION), ("offer", OFFER)]:
            fig, axes = plt.subplots(len(METRICS), 3, figsize=(6.6, 7.2), sharex=True)
            for j, group in enumerate(groups):
                for i, metric in enumerate(METRICS):
                    ax = axes[i, j]
                    for ap in levels:
                        sub = c[(c.group == group) & (c.approach == ap)]
                        means, lows, highs = [], [], []
                        for al in ALPHAS:
                            v = sub[sub.alpha == al][metric].to_numpy()
                            lo, hi = t_ci(v)
                            means.append(v.mean()); lows.append(v.mean() - lo); highs.append(hi - v.mean())
                        ax.errorbar(range(len(ALPHAS)), means, yerr=[lows, highs], marker=MARKERS[ap],
                                    color=COLORS[ap], label=SHORT[ap], capsize=1.5, markersize=3, linewidth=1.2)
                    ax.grid(axis="y")
                    if i == 0:
                        ax.set_title(group)
                    if j == 0:
                        ax.set_ylabel(PLOT[metric])
                    if i == len(METRICS) - 1:
                        ax.set_xticks(range(len(ALPHAS)), [f"{a:g}" for a in ALPHAS])
                        ax.set_xlabel(r"$\alpha$")
            axes[0, 0].legend(frameon=False)
            fig.tight_layout()
            fig.savefig(out / f"figures/{fname}_{family}.pdf")
            plt.close(fig)


# ----------------------------------------------------------------------------- H: five seeds
def five_seeds(results: Path, out: Path) -> None:
    h = load(results, "H_base_five_seeds")
    assert (h.status == "solved").all()
    rows, num = [], {}
    for metric in METRICS:
        piv = h.pivot_table(index="instance", columns="seed", values=metric)
        within = math.sqrt((piv.std(axis=1, ddof=1) ** 2).mean())
        between = piv.mean(axis=1).std(ddof=1)
        d = 2 if metric == "conflicts" else 4 if metric == "R_imb" else 3
        rows.append(f"{TEX[metric]} & {len(piv)} & {within:.{d}f} & {between:.{d}f} \\\\")
        num[metric] = dict(within=round(within, 4), between=round(between, 4))
    write(out / "tables/BASE_seed_variability.tex", tabular(
        "lrrr", [r"Metric & Instances & \shortstack{RMS within-instance\\SD across seeds} & \shortstack{SD across\\instance means} \\"], rows))
    NUM["five_seeds"] = dict(runs=len(h), **num)


# ----------------------------------------------------------------------------- real instance
def real(results: Path, out: Path, baselines: dict) -> None:
    d = load(results, "D_real_30s")
    assert (d.status == "solved").all()
    rows = []
    for ap in ["Embedded", "Decoupled-50"]:
        for al in ALPHAS:
            x = d[(d.approach == ap) & (d.alpha == al)]
            rows.append(f"{ap} & {al:g} & {len(x)} & " + " & ".join(
                f"{x[m].mean():.{dec(m)}f} $\\pm$ {x[m].std(ddof=1):.{dec(m)}f}" for m in ["F_ref", "R_imb", "R_dis", "conflicts"])
                + f" & {x.completed_iterations.mean():.0f} \\\\")
    write(out / "tables/real_weights.tex", tabular(
        "lcrrrrrr", [r"Approach & $\alpha$ & $n$ & $F_{\mathrm{ref}}$ & $R_{\mathit{imb}}$ & $R_{\mathit{dis}}$ & Conflicts & Iterations \\"], rows))
    sel = NUM["alpha_selection"]["selected"]
    obs, ssp, ssp0 = baselines["Observed"], baselines["SSP with LS"], baselines["SSP without LS"]
    order = ["F_ref", "R_imb", "R_dis", "conflicts"]
    e = d[(d.approach == "Embedded") & (d.alpha == sel)]
    k = d[(d.approach == "Decoupled-50") & (d.alpha == sel)]
    rows = []
    for m in order:
        dd = 2 if m == "conflicts" else 3
        rows.append(f"{TEX[m]} & {obs[m]:.{dd}f} & {ssp[m]:.{dd}f} & {ssp0[m]:.{dd}f} & "
                    f"{e[m].mean():.{dd}f} $\\pm$ {e[m].std(ddof=1):.{dd}f} & {k[m].mean():.{dd}f} $\\pm$ {k[m].std(ddof=1):.{dd}f} \\\\")
    write(out / "tables/real_selected_alpha.tex", tabular(
        "lrrrrr", [r"Metric & Observed & \shortstack{SSP\\with LS} & \shortstack{SSP\\without LS} & "
                   rf"\shortstack{{Embedded\\$\alpha={sel:g}$}} & \shortstack{{Decoupled-50\\$\alpha={sel:g}$}} \\"], rows))
    def all_better(frame, ref):
        return int(np.all(np.stack([frame[m].to_numpy() < ref[m] for m in order]), axis=0).sum())
    diff_rows, diff_num = [], {}
    for m in order:
        v = e[m].to_numpy() - obs[m]
        lo, hi = t_ci(v)
        dd = 2 if m == "conflicts" else 3
        diff_rows.append(f"{TEX[m]} & {obs[m]:.{dd}f} & {e[m].mean():.{dd}f} & {v.mean():.{dd}f} & [{lo:.{dd}f}, {hi:.{dd}f}] & "
                         f"{100 * v.mean() / obs[m]:.1f} & {int((v < 0).sum())}/{len(v)} \\\\")
        diff_num[m] = dict(pct_vs_observed=round(100 * v.mean() / obs[m], 1),
                           pct_vs_ssp=round(100 * (e[m].mean() - ssp[m]) / ssp[m], 1),
                           improved_vs_observed=int((v < 0).sum()), improved_vs_ssp=int((e[m].to_numpy() < ssp[m]).sum()))
    write(out / "tables/real_observed_differences.tex", tabular(
        "lrrrrrr", [r"Metric & Observed & Embedded mean & Mean diff. & 95\% CI & Change (\%) & Runs improved \\"], diff_rows))
    means = {ap: {f"{al:g}": {m: round(float(d[(d.approach == ap) & (d.alpha == al)][m].mean()), 3) for m in order + ["completed_iterations"]}
                  for al in ALPHAS} for ap in ["Embedded", "Decoupled-50"]}
    sds = {ap: {f"{al:g}": {m: round(float(d[(d.approach == ap) & (d.alpha == al)][m].std(ddof=1)), 3) for m in order}
                for al in ALPHAS} for ap in ["Embedded", "Decoupled-50"]}
    NUM["real"] = dict(runs=len(d), selected_alpha=sel, means=means, sds=sds,
                       emb_all_better_observed=all_better(e, obs), emb_all_better_ssp=all_better(e, ssp),
                       d50_all_better_observed=all_better(k, obs), d50_all_better_ssp=all_better(k, ssp),
                       differences=diff_num,
                       emb_vs_d50={m: dict(diff=round(float((e[m].to_numpy() - k[m].to_numpy()).mean()), 3),
                                           p_t=float(stats.ttest_rel(e[m], k[m]).pvalue)) for m in order},
                       min_fref={ap: dict(alpha=float(d[d.approach == ap].groupby("alpha").F_ref.mean().idxmin()),
                                          value=round(float(d[d.approach == ap].groupby("alpha").F_ref.mean().min()), 3))
                                 for ap in ["Embedded", "Decoupled-50"]},
                       max_time=round(float(d.optimizer_seconds.max()), 3))
    plt = setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6), sharex=True)
    for ax, m in zip(axes.flat, order):
        for ap in ["Embedded", "Decoupled-50"]:
            mm, lo_, hi_ = [], [], []
            for al in ALPHAS:
                v = d[(d.approach == ap) & (d.alpha == al)][m].to_numpy()
                lo, hi = t_ci(v); mm.append(v.mean()); lo_.append(v.mean() - lo); hi_.append(hi - v.mean())
            ax.errorbar(range(len(ALPHAS)), mm, yerr=[lo_, hi_], marker=MARKERS[ap], color=COLORS[ap],
                        label=SHORT[ap], capsize=1.5, markersize=3, linewidth=1.2)
        ax.axhline(obs[m], color="#52514e", linestyle="--", linewidth=0.8, label="Observed")
        ax.axhline(ssp[m], color="#52514e", linestyle=":", linewidth=0.8, label="SSP with LS")
        ax.set_ylabel(PLOT[m]); ax.grid(axis="y")
        if m in ("F_ref", "R_imb"):
            ax.set_yscale("log"); plain_log(ax)
    for ax in axes[1]:
        ax.set_xticks(range(len(ALPHAS)), [f"{a:g}" for a in ALPHAS]); ax.set_xlabel(r"$\alpha$")
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(out / "figures/real_weight_response_30s.pdf"); plt.close(fig)
    # Quality-time curve.
    q = load(results, "E_real_quality_time")
    rows, qt = [], {}
    for ap in ["Embedded", "Decoupled-50"]:
        for al in [0, 0.25, 1]:
            for t in [30, 120, 300]:
                x = q[(q.approach == ap) & (q.alpha == al) & (q.time_limit_seconds == t)]
                rows.append(f"{ap} & {al:g} & {t} & {len(x)} & " + " & ".join(
                    f"{x[m].mean():.{dec(m)}f} $\\pm$ {x[m].std(ddof=1):.{dec(m)}f}" for m in order)
                    + f" & {x.completed_iterations.mean():.0f} \\\\")
                qt[f"{ap}|{al:g}|{t}"] = {m: round(float(x[m].mean()), 3) for m in order + ["completed_iterations"]}
    write(out / "tables/real_quality_time.tex", tabular(
        "lcrrrrrrr", [r"Approach & $\alpha$ & Budget (s) & $n$ & $F_{\mathrm{ref}}$ & $R_{\mathit{imb}}$ & $R_{\mathit{dis}}$ & Conflicts & Iterations \\"], rows))
    # Paired comparison Embedded vs D50 F_ref at each budget, alpha = selected.
    pairs = {}
    for t in [30, 120, 300]:
        pe = q[(q.approach == "Embedded") & (q.alpha == sel) & (q.time_limit_seconds == t)].sort_values("seed")
        pd_ = q[(q.approach == "Decoupled-50") & (q.alpha == sel) & (q.time_limit_seconds == t)].sort_values("seed")
        pairs[t] = int((pe.F_ref.to_numpy() < pd_.F_ref.to_numpy()).sum())
    NUM["quality_time"] = dict(means=qt, emb_lower_fref_pairs=pairs, runs=len(q))
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), sharey=True)
    for ax, al in zip(axes, [0, 0.25, 1]):
        for ap in ["Embedded", "Decoupled-50"]:
            mm, lo_, hi_ = [], [], []
            for t in [30, 120, 300]:
                v = q[(q.approach == ap) & (q.alpha == al) & (q.time_limit_seconds == t)].F_ref.to_numpy()
                lo, hi = t_ci(v); mm.append(v.mean()); lo_.append(v.mean() - lo); hi_.append(hi - v.mean())
            ax.errorbar([30, 120, 300], mm, yerr=[lo_, hi_], marker=MARKERS[ap], color=COLORS[ap], label=SHORT[ap],
                        capsize=1.5, markersize=3, linewidth=1.2)
        ax.set_xscale("log"); ax.set_yscale("log"); plain_log(ax, "x"); plain_log(ax, "y")
        ax.set_xticks([30, 120, 300], ["30", "120", "300"])
        ax.set_title(f"$\\alpha={al:g}$"); ax.set_xlabel("Optimizer budget (s)"); ax.grid(True)
    axes[0].set_ylabel(PLOT["F_ref"]); axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "figures/real_quality_time.pdf"); plt.close(fig)
    # LS ablation.
    f = load(results, "F_real_ls_ablation"); gb = load(results, "G_base_ls_ablation")
    rows, ab = [], {}
    for name, frame in [("BASE", gb), ("Real", f)]:
        for al in [0.25, 1]:
            for ls, label in [(True, "With"), (False, "Without")]:
                x = frame[(frame.alpha == al) & (frame.local_search == ls)]
                rows.append(f"{name} & {al:g} & {label} & " + " & ".join(
                    f"{x[m].mean():.{dec(m)}f} $\\pm$ {x[m].std(ddof=1):.{dec(m)}f}" for m in order) + r" \\")
                ab[f"{name}|{al:g}|{label}"] = {m: round(float(x[m].mean()), 3) for m in order} | {"n": len(x)}
    write(out / "tables/ls_ablation.tex", tabular(
        "llcrrrr", [r"Instance & $\alpha$ & LS & $F_{\mathrm{ref}}$ & $R_{\mathit{imb}}$ & $R_{\mathit{dis}}$ & Conflicts \\"], rows))
    NUM["ls_ablation"] = ab


# ----------------------------------------------------------------------------- pilots
def pilots(results: Path, out: Path) -> None:
    raw = load(results, "02_pilot_convergence")
    cp = load(results, "02_pilot_convergence", "checkpoints.csv")
    final = raw[["run_id", "F"]].rename(columns={"F": "F_final"})
    x = cp.merge(final, on="run_id")
    x = x[(x.iteration_budget >= 500) & (x.F > 0)]  # relative gain undefined when F = 0
    x["gain"] = 100 * (x.F - x.F_final) / x.F
    rows, num = [], {}
    for k, sub in x.groupby("iteration_budget"):
        rows.append(f"{int(k)} & {sub.gain.median():.2f} & {sub.gain.quantile(0.75):.2f} & {100 * (sub.gain > 1).mean():.1f} \\\\")
        num[int(k)] = dict(median=round(float(sub.gain.median()), 2), p75=round(float(sub.gain.quantile(0.75)), 2),
                           share=round(float(100 * (sub.gain > 1).mean()), 1))
    write(out / "tables/pilot_convergence.tex", tabular(
        "rrrr", [r"From budget & \shortstack{Median further\\gain to 3000 (\%)} & \shortstack{75th\\percentile (\%)} & "
                 r"\shortstack{Runs improving\\by $>1$\% (\%)} \\"], rows, width=False))
    conc = json.loads((results / "01_pilot_concurrency" / "result.json").read_text(encoding="utf-8"))
    NUM["pilots"] = dict(convergence=num, convergence_runs=len(raw),
                         concurrency={r["processes"]: dict(mean=r["mean"], loss=r["loss_percent"]) for r in conc["rows"]})
    env = json.loads((results / "00_environment" / "environment.json").read_text(encoding="utf-8"))
    NUM["environment"] = {k: env[k] for k in ("python", "numpy", "numba", "platform", "logical_cpus", "workers")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--baselines", type=Path, default=Path(__file__).with_name("real_baselines.json"))
    parser.add_argument("--out", type=Path, default=Path("paper_out"))
    args = parser.parse_args()
    for sub in ("tables", "figures"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    baselines = json.loads(args.baselines.read_text(encoding="utf-8"))
    pilots(args.results, args.out)
    construction(args.results, args.out)
    budgets(args.results, args.out)
    approaches(args.results, args.out)
    five_seeds(args.results, args.out)
    real(args.results, args.out, baselines)
    (args.out / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float), encoding="utf-8")
    print(f"Wrote tables, figures and numbers.json to {args.out}")


if __name__ == "__main__":
    main()
