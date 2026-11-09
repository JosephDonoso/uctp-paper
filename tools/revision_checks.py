"""Tables added in the v9 revision, computed from the v8 records (no new runs).

1. equal_cpu.tex        Embedded at the 1000-iteration checkpoint against Decoupled-50
                        after 2500 iterations (comparable optimizer CPU time), per group
                        and weight: mean re-evaluated F_ref and paired win counts.
2. noise_converged.tex  Coefficient of variation over the 50 fresh SSP orders of the
                        final 2500-iteration timetables (E5a, E5b).
4. partial_final.tex    Final alpha-0.25 timetables under partial adjustment (p*), with the
                        institutional timetable taken from the calibration grid (same draws).
3. real_paired.tex      Real instance at alpha 0.25: Embedded minus Decoupled-50, paired
                        by seed, with paired t-test and exact sign test (ties excluded).

Usage:
    python tools/revision_checks.py --data paper_results_v8/data --out paper_out_v9
The script also prints every number cited in the text.
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

GROUPS = ["Dispersion-Low", "BASE", "Dispersion-High", "Offer-Low", "Offer-High"]
METRICS = [("F_ref", r"$F_{\mathrm{ref}}$", 3), ("R_imb", r"$R_{\mathit{imb}}$", 3),
           ("R_dis", r"$R_{\mathit{dis}}$", 3), ("conflicts", "Conflicts", 2)]


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def pfmt(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def equal_cpu(data: Path, out: Path, checkpoint: str) -> None:
    d = pd.read_csv(data / "E5b_synthetic_converged.csv")
    e = d[(d.approach == "Embedded") & (d.point.astype(str) == checkpoint)]
    f = d[(d.approach == "Decoupled-50") & (d.point.astype(str) == "final")]
    m = e.merge(f, on=["instance", "group", "alpha"], suffixes=("_e", "_d"))
    cpu_e, cpu_d = m.optimizer_cpu_seconds_e.mean(), m.optimizer_cpu_seconds_d.mean()
    print(f"equal-CPU: Embedded@{checkpoint} {cpu_e:.1f} s vs D50 final {cpu_d:.1f} s (mean CPU)")
    alphas = sorted(m.alpha.unique())
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lc" + "r" * 3 + r"@{}}", r"\hline",
             rf"Group & $\alpha$ & Embedded ({checkpoint} it.) & D50 (2500 it.) & Embedded lower \\", r"\hline"]
    for group in GROUPS:
        for alpha in alphas:
            x = m[(m.group == group) & (m.alpha == alpha)]
            wins = int((x.ree_F_ref_e < x.ree_F_ref_d).sum())
            lines.append(f"{group} & {alpha:g} & {x.ree_F_ref_e.mean():.3f} & {x.ree_F_ref_d.mean():.3f} "
                         f"& {wins}/{len(x)} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular*}"]
    write(out / "tables/equal_cpu.tex", lines)
    mid = m[m.alpha.isin([0.125, 0.25, 0.5])]
    by = mid.groupby(["group", "alpha"]).apply(
        lambda x: pd.Series({"lower_mean": x.ree_F_ref_e.mean() < x.ree_F_ref_d.mean(),
                             "wins": (x.ree_F_ref_e < x.ree_F_ref_d).sum()}), include_groups=False)
    print(f"alpha 0.125-0.5: Embedded lower mean in {int(by.lower_mean.sum())}/{len(by)} group-weight cells; "
          f"wins per cell {int(by.wins.min())}-{int(by.wins.max())} of 50")
    for alpha in (0.125, 0.25, 0.5):
        w = by.xs(alpha, level="alpha").wins
        print(f"  alpha {alpha}: wins {int(w.min())}-{int(w.max())} of 50")


def noise(data: Path, out: Path) -> None:
    rows = []
    for name, file in (("Real", "E5a_real_converged.csv"), ("Synthetic", "E5b_synthetic_converged.csv")):
        d = pd.read_csv(data / file)
        d = d[(d.point.astype(str) == "final") & d.approach.isin(["Embedded", "Decoupled-50"])]
        d = d.assign(cv_imb=d.ree_sd_R_imb / d.ree_R_imb, cv_dis=d.ree_sd_R_dis / d.ree_R_dis)
        for (approach, alpha), x in d.groupby(["approach", "alpha"]):
            rows.append((name, approach, alpha, len(x), 100 * x.cv_imb.mean(), 100 * x.cv_dis.mean()))
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcrrr@{}}", r"\hline",
             r"Instances & Approach & $\alpha$ & $n$ & CV $R_{\mathit{imb}}$ (\%) & CV $R_{\mathit{dis}}$ (\%) \\",
             r"\hline"]
    for r in rows:
        lines.append(f"{r[0]} & {r[1]} & {r[2]:g} & {r[3]} & {r[4]:.1f} & {r[5]:.1f} \\\\")
    lines += [r"\hline", r"\end{tabular*}"]
    write(out / "tables/noise_converged.tex", lines)
    t = pd.DataFrame(rows, columns=["set", "approach", "alpha", "n", "cv_imb", "cv_dis"])
    emb = t[(t.approach == "Embedded") & (t.alpha >= 0.25)]
    print(f"Embedded alpha>=0.25 CV imbalance: real {emb[emb.set == 'Real'].cv_imb.min():.1f}-"
          f"{emb[emb.set == 'Real'].cv_imb.max():.1f}%, synthetic {emb[emb.set == 'Synthetic'].cv_imb.min():.1f}-"
          f"{emb[emb.set == 'Synthetic'].cv_imb.max():.1f}%; dissatisfaction {emb.cv_dis.min():.1f}-{emb.cv_dis.max():.1f}%")


def real_paired(data: Path, out: Path) -> None:
    d = pd.read_csv(data / "E5a_real_converged.csv")
    d = d[(d.point.astype(str) == "final") & (d.alpha == 0.25)]
    e = d[d.approach == "Embedded"].set_index("seed")
    f = d[d.approach == "Decoupled-50"].set_index("seed").loc[e.index]
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}", r"\hline",
             r"Metric & Mean diff. & Lower / higher & $p$ ($t$) & $p$ (sign) \\", r"\hline"]
    for key, label, dec in METRICS:
        x = e[f"ree_{key}"] - f[f"ree_{key}"]
        low, high = int((x < 0).sum()), int((x > 0).sum())
        pt = stats.ttest_rel(e[f"ree_{key}"], f[f"ree_{key}"]).pvalue
        ps = stats.binomtest(low, low + high).pvalue if low + high else 1.0
        lines.append(f"{label} & {x.mean():.{dec}f} & {low}/{high} & {pfmt(pt)} & {pfmt(ps)} \\\\")
        print(f"real alpha 0.25 {key}: diff {x.mean():.3f}, lower {low}, higher {high}, p_t {pt:.4f}, p_sign {ps:.4f}")
    lines += [r"\hline", r"\end{tabular*}"]
    write(out / "tables/real_paired.tex", lines)


def partial_final(data: Path, out: Path) -> None:
    part = pd.read_csv(data / "e4b_partial.csv")
    cal = pd.read_csv(data / "e4b_calibration.csv")
    p_star = float(part.p.iloc[0])
    xreal = cal[(cal.p - p_star).abs() < 1e-9].iloc[0]
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}", r"\hline",
             r"Timetable & $n$ & $R_{\mathit{imb}}$ & $R_{\mathit{dis}}$ & Conflicts & $F_{\mathrm{ref}}$ \\", r"\hline"]
    for who in ("Embedded", "Decoupled-50"):
        x = part[part.label.str.startswith(who)]
        lines.append(f"{who} $\\alpha=0.25$ & {len(x)} & {x.R_imb_mean.mean():.3f} & {x.R_dis_mean.mean():.3f} & "
                     f"{x.conflicts_mean.mean():.1f} & {x.F_ref_mean.mean():.3f} \\\\")
        print(f"partial p={p_star}: {who} F_ref {x.F_ref_mean.mean():.3f} ({len(x)} timetables of e4b_partial.csv)")
    lines.append(f"Institutional & 1 & {xreal.R_imb_mean:.3f} & {xreal.R_dis_mean:.3f} & {xreal.conflicts_mean:.1f} & "
                 f"{xreal.F_ref_mean:.3f} \\\\")
    print(f"partial p={p_star}: institutional F_ref {xreal.F_ref_mean:.3f} (calibration draws)")
    lines += [r"\hline", r"\end{tabular*}"]
    write(out / "tables/partial_final.tex", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", default="1000")
    args = parser.parse_args()
    equal_cpu(args.data, args.out, args.checkpoint)
    noise(args.data, args.out)
    real_paired(args.data, args.out)
    partial_final(args.data, args.out)


if __name__ == "__main__":
    main()
