"""Tables, figures and numbers of the v8 manuscript (revision v7 experiments).

Inputs (``--data`` folder): compact CSV extracts of the v7 campaigns
(``E1c.csv``, ``P3.csv``, ``E5a_real_converged.csv``, ``E5b_synthetic_converged.csv``,
``E2a_trajectories.csv`` and ``E2a_trajectories_trace.jsonl``,
``E2b_warm_start.csv``, ``E2c_neighborhood_200.csv``, ``E3_sequential.csv``,
``ssp_timing.json``) and the re-evaluations of the published campaigns
(``reeval_own.csv``, ``reeval_ls_on.csv``, ``reeval_ls_off.csv``,
``e1a_noise.csv``, ``e4b_calibration.csv``, ``e4b_partial.csv``).
All quality values are means over 50 fresh SSP orders (``ree_*`` columns).

Usage: python tools/paper_analysis_v8.py --data v8_data --out paper_out_v8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_analysis import (  # noqa: E402
    ALPHAS, COLORS, DISPERSION, GROUPS, MARKERS, METRICS, OFFER, PLOT, SEED, TEX, boot_ci, boot_mean_p,
    dec, holm, longtable, pfmt, plain_log, rm_anova, setup_plot, sign_test, t_ci, tabular, write,
)

NUM: dict = {}
SHORT = {"Embedded": "Embedded", "Decoupled-50": "D50", "Decoupled-5": "D5", "Decoupled-1": "D1",
         "Sequential": "Sequential"}
COLORS = dict(COLORS, **{"Decoupled-1": "#1baf7a", "Decoupled-5": "#8a5cd6", "Sequential": "#52514e"})
MARKERS = dict(MARKERS, **{"Decoupled-1": "^", "Decoupled-5": "D", "Sequential": "x"})
OBSERVED = {"F_ref": 13.777, "R_imb": 1.760, "R_dis": 7.828, "conflicts": 201.0}
REE = {m: f"ree_{m}" for m in METRICS}


def bigger_fonts(plt):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                         "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})


def finals(df):
    return df[df.point == "final"].copy()


# ----------------------------------------------------------------------------- E5b
def synthetic(data: Path, out: Path, seq: pd.DataFrame) -> None:
    raw = pd.read_csv(data / "E5b_synthetic_converged.csv")
    f = finals(raw)
    assert (f.status == "solved").all()
    for m in METRICS:
        f[m] = f[REE[m]]
    s = finals(seq)
    s = s[(s.instance != "real.json") & (s.budget_kind == "iterations")].copy()
    for m in METRICS:
        s[m] = s[REE[m]]
    NUM["e5b"] = dict(runs=int(len(f)), sequential_runs=int(len(s)))
    # Means by approach and alpha (pooled and by group).
    rows, means = [], {}
    order = ["Embedded", "Decoupled-50", "Decoupled-5", "Decoupled-1"]
    for ap in order:
        for al in ALPHAS:
            x = f[(f.approach == ap) & (f.alpha == al)]
            if x.empty:
                continue
            means[f"{ap}|{al:g}"] = {m: round(float(x[m].mean()), 3) for m in METRICS}
            rows.append(f"{SHORT[ap]} & {al:g} & {len(x)} & " + " & ".join(
                f"{x[m].mean():.{dec(m)}f}" for m in METRICS) + " & " + " & ".join(
                f"{x[x.group == g][ 'F_ref'].mean():.3f}" for g in GROUPS) + r" \\")
    rows.append(f"Sequential & -- & {len(s)} & " + " & ".join(f"{s[m].mean():.{dec(m)}f}" for m in METRICS)
                + " & " + " & ".join(f"{s[s.group == g]['F_ref'].mean():.3f}" for g in GROUPS) + r" \\")
    means["Sequential"] = {m: round(float(s[m].mean()), 3) for m in METRICS}
    write(out / "tables/conv_means.tex", tabular(
        "lrr" + "r" * 4 + "r" * 5,
        [r"Approach & $\alpha$ & $n$ & " + " & ".join(TEX[m] for m in METRICS) + " & "
         + " & ".join(r"\shortstack{" + g.replace("-", r"-\\") + r"\\$F_{\mathrm{ref}}$}" for g in GROUPS) + r" \\"],
        rows))
    NUM["e5b_means"] = means
    emb = f[f.approach == "Embedded"]
    pooled = emb.groupby("alpha").F_ref.mean()
    per_group = emb.groupby(["group", "alpha"]).F_ref.mean().unstack()
    NUM["alpha_selection"] = dict(pooled={f"{k:g}": round(v, 3) for k, v in pooled.items()},
                                  selected=float(pooled.idxmin()),
                                  per_group_argmin={g: float(per_group.loc[g].idxmin()) for g in GROUPS})
    # Cost.
    cost = f.groupby("approach")[["completed_iterations", "ssp_calls", "optimizer_cpu_seconds", "optimizer_seconds"]].mean()
    NUM["e5b_cost"] = {ap: {k: round(float(v), 1) for k, v in cost.loc[ap].items()} for ap in cost.index}
    NUM["sequential_cost"] = dict(cpu=round(float(s.optimizer_seconds.mean()), 1))
    # Factorial ANOVA Embedded vs D50 (60 tests).
    sub = f[f.approach.isin(["Embedded", "Decoupled-50"])]
    rows_all = []
    for metric in METRICS:
        for gi, group in enumerate(GROUPS):
            piv = sub[sub.group == group].pivot_table(index="instance", columns=["alpha", "approach"], values=metric)
            cube = np.array([[piv[(al, ap)].to_numpy() for ap in ["Embedded", "Decoupled-50"]] for al in ALPHAS])
            res = rm_anova(cube.transpose(2, 0, 1), seed=SEED + gi)
            for eff in ["A", "B", "AxB"]:
                rows_all.append(dict(metric=metric, group=group, effect=eff, n=cube.shape[2], **res[eff]))
    pg, pb = holm([r["p_gg"] for r in rows_all]), holm([r["p_boot"] for r in rows_all])
    for r, a1, a2 in zip(rows_all, pg, pb):
        r["p_gg_h"], r["p_b_h"] = a1, a2
    sym = {"A": r"$\alpha$", "B": "$A$", "AxB": r"$\alpha\times A$"}
    for metric in METRICS:
        rows = [f"{r['group']} & {r['n']} & {sym[r['effect']]} & {r['F']:.2f} & {r['eps']:.3f} & {r['eta']:.3f} & "
                f"{pfmt(r['p_gg_h']) if r['p_gg_h'] >= 0.001 else '$<0.001$'} & {r['p_b_h']:.4f} \\\\"
                for r in rows_all if r["metric"] == metric]
        write(out / f"tables/conv_factorial_{metric}.tex", tabular(
            "lrlrrrrr", [r"Group & $n$ & Effect & $F$ & $\varepsilon_{\mathrm{GG}}$ & $\eta_p^2$ & "
                         r"$p_{\mathrm{GG,H}}$ & $p_{\mathrm{B,H}}$ \\"], rows))
    NUM["e5b_anova"] = {f"{eff}_{m}": dict(
        both=int(sum(r["p_gg_h"] < 0.05 and r["p_b_h"] < 0.05 for r in rows_all if r["effect"] == eff and r["metric"] == m)),
        eta=[round(min(r["eta"] for r in rows_all if r["effect"] == eff and r["metric"] == m), 3),
             round(max(r["eta"] for r in rows_all if r["effect"] == eff and r["metric"] == m), 3)])
        for eff in ["A", "B", "AxB"] for m in METRICS}
    # Follow-ups: Embedded - D50 and Embedded - Sequential at each weight.
    def followups(other_frame, label, fname, caption, key):
        fol = []
        for group in GROUPS:
            for metric in METRICS:
                for al in ALPHAS:
                    e = emb[(emb.group == group) & (emb.alpha == al)].set_index("instance")[metric]
                    o = other_frame(group, al).set_index("instance")[metric]
                    d = (e - o.reindex(e.index)).dropna().to_numpy()
                    fol.append(dict(group=group, metric=metric, alpha=al, n=len(d), mean=d.mean(), ci=boot_ci(d),
                                    pt=stats.ttest_1samp(d, 0.0).pvalue, pb=boot_mean_p(d)))
        pt, pb2 = holm([x["pt"] for x in fol]), holm([x["pb"] for x in fol])
        for x, a1, a2 in zip(fol, pt, pb2):
            x["pth"], x["pbh"] = a1, a2
        rows = [f"{x['group']} & {TEX[x['metric']]} & {x['alpha']:g} & {x['n']} & {x['mean']:.3f} & "
                f"[{x['ci'][0]:.3f}, {x['ci'][1]:.3f}] & {pfmt(x['pth'])} & {pfmt(x['pbh'])} \\\\" for x in fol]
        write(out / f"tables/{fname}.tex", longtable(
            "@{}p{1.8cm}lrrrlrr@{}",
            r"Group & Metric & $\alpha$ & $n$ & Mean diff. & 95\% CI & $p_{t,\mathrm{H}}$ & $p_{\mathrm{B,H}}$ \\",
            rows, caption, f"tab:{fname}"))
        res = {}
        for m in METRICS:
            for al in ALPHAS:
                sel = [x for x in fol if x["metric"] == m and x["alpha"] == al and x["pth"] < 0.05 and x["pbh"] < 0.05]
                res[f"{m}_{al:g}"] = dict(emb_lower=[x["group"] for x in sel if x["mean"] < 0],
                                          other_lower=[x["group"] for x in sel if x["mean"] > 0])
        NUM[key] = res
    d50 = f[f.approach == "Decoupled-50"]
    followups(lambda g, al: d50[(d50.group == g) & (d50.alpha == al)], "D50", "conv_followups_d50",
              r"Fixed-weight contrasts between Embedded (CRN) and Decoupled-50 after 2500 iterations (Embedded "
              r"minus D50; re-evaluated values; Holm adjustment across the 140 contrasts for each procedure).",
              "e5b_followup_d50")
    followups(lambda g, al: s[s.group == g], "Sequential", "conv_followups_seq",
              r"Contrasts between Embedded (CRN) at each weight and the sequential baseline after 2500 iterations "
              r"(Embedded minus Sequential; re-evaluated values; Holm adjustment across the 140 contrasts).",
              "e5b_followup_seq")
    # Endpoint change alpha 0 -> 4 (percent of means) for imbalance.
    def pct(ap, m, g):
        x = f[(f.approach == ap) & (f.group == g)].groupby("alpha")[m].mean()
        return 100 * (x[4] - x[0]) / x[0]
    NUM["e5b_endpoint"] = {f"{ap}_{m}": [round(min(pct(ap, m, g) for g in GROUPS), 1), round(max(pct(ap, m, g) for g in GROUPS), 1)]
                          for ap in ["Embedded", "Decoupled-50"] for m in METRICS}
    # Figures.
    plt = setup_plot()
    bigger_fonts(plt)
    for family, groups in [("dispersion", DISPERSION), ("offer", OFFER)]:
        fig, axes = plt.subplots(len(METRICS), 3, figsize=(7.0, 8.4), sharex=True)
        for j, group in enumerate(groups):
            for i, metric in enumerate(METRICS):
                ax = axes[i, j]
                for ap in ["Embedded", "Decoupled-50"]:
                    x = f[(f.group == group) & (f.approach == ap)]
                    mm, lo_, hi_ = [], [], []
                    for al in ALPHAS:
                        v = x[x.alpha == al][metric].to_numpy()
                        lo, hi = t_ci(v)
                        mm.append(v.mean()); lo_.append(v.mean() - lo); hi_.append(hi - v.mean())
                    ax.errorbar(range(len(ALPHAS)), mm, yerr=[lo_, hi_], marker=MARKERS[ap], color=COLORS[ap],
                                label=SHORT[ap], capsize=1.5, markersize=4, linewidth=1.3)
                ax.axhline(s[s.group == group][metric].mean(), color=COLORS["Sequential"], linestyle="--",
                           linewidth=1.0, label="Sequential")
                ax.grid(axis="y")
                if i == 0:
                    ax.set_title(group)
                if j == 0:
                    ax.set_ylabel(PLOT[metric])
                if i == len(METRICS) - 1:
                    ax.set_xticks(range(len(ALPHAS)), [f"{a:g}".lstrip("0") if 0 < a < 1 else f"{a:g}" for a in ALPHAS],
                                  fontsize=8)
                    ax.set_xlabel(r"$\alpha$")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"figures/conv_embedded_vs_D50_{family}.pdf")
        plt.close(fig)


# ----------------------------------------------------------------------------- E5a, E3 real, E4
def real(data: Path, out: Path, seq: pd.DataFrame) -> None:
    raw = pd.read_csv(data / "E5a_real_converged.csv")
    f = finals(raw)
    for m in METRICS:
        f[m] = f[REE[m]]
    s = finals(seq)
    s = s[(s.instance == "real.json") & (s.budget_kind == "iterations")].copy()
    for m in METRICS:
        s[m] = s[REE[m]]
    noise = pd.read_csv(data / "e1a_noise.csv")
    xr = noise[noise.label.str.startswith("X_real")].iloc[0]
    cal = pd.read_csv(data / "e4b_calibration.csv")
    x0 = cal[cal.p == 0.0].iloc[0]
    ssp = {m: float(xr[f"{m}_mean"]) for m in METRICS}
    ssp0 = {m: float(x0[f"{m}_mean"]) for m in METRICS}
    NUM["baselines"] = dict(observed=OBSERVED, ssp_ls={k: round(v, 3) for k, v in ssp.items()},
                            ssp_no_ls={k: round(v, 3) for k, v in ssp0.items()},
                            ssp_ls_cv_fref=round(100 * float(xr["F_ref_sd"]) / float(xr["F_ref_mean"]), 1))
    sel = 0.25
    e = f[(f.approach == "Embedded") & (f.alpha == sel)].sort_values("seed")
    k = f[(f.approach == "Decoupled-50") & (f.alpha == sel)].sort_values("seed")
    order = ["F_ref", "R_imb", "R_dis", "conflicts"]
    rows = []
    for m in order:
        dd = 1 if m == "conflicts" else 3
        rows.append(f"{TEX[m]} & {OBSERVED[m]:.{dd}f} & {ssp[m]:.{dd}f} & {ssp0[m]:.{dd}f} & "
                    + " & ".join(f"{x[m].mean():.{dd}f} $\\pm$ {x[m].std(ddof=1):.{dd}f}" for x in (e, k, s)) + r" \\")
    write(out / "tables/real_conv_selected.tex", tabular(
        "lrrrrrr", [r"Metric & Observed & \shortstack{SSP\\with LS} & \shortstack{SSP\\without LS} & "
                   r"\shortstack{Embedded\\$\alpha=0.25$} & \shortstack{D50\\$\alpha=0.25$} & Sequential \\"], rows))

    def all_better(frame, ref):
        return int(np.all(np.stack([frame[m].to_numpy() < ref[m] for m in order]), axis=0).sum())
    NUM["real_conv"] = dict(
        means={f"{ap}|{al:g}": {m: round(float(f[(f.approach == ap) & (f.alpha == al)][m].mean()), 3) for m in order}
               for ap in ["Embedded", "Decoupled-50"] for al in [0, 0.25, 1]},
        sequential={m: round(float(s[m].mean()), 3) for m in order},
        all_better={name: dict(observed=all_better(x, OBSERVED), ssp=all_better(x, ssp))
                    for name, x in [("Embedded", e), ("Decoupled-50", k), ("Sequential", s)]},
        pct_vs_ssp={m: round(100 * (e[m].mean() - ssp[m]) / ssp[m], 1) for m in order},
        pct_vs_observed={m: round(100 * (e[m].mean() - OBSERVED[m]) / OBSERVED[m], 1) for m in order},
        emb_vs_d50={m: dict(diff=round(float((e[m].to_numpy() - k[m].to_numpy()).mean()), 3),
                            p_t=float(stats.ttest_rel(e[m], k[m]).pvalue),
                            emb_lower=int((e[m].to_numpy() < k[m].to_numpy()).sum())) for m in order},
        cpu={ap: round(float(f[f.approach == ap].optimizer_cpu_seconds.mean()), 1) for ap in ["Embedded", "Decoupled-50"]},
        solves={ap: round(float(f[f.approach == ap].ssp_calls.mean()), 0) for ap in ["Embedded", "Decoupled-50"]},
        sequential_cpu=round(float(s.optimizer_seconds.mean()), 1))
    # All weights (supplement), with checkpoints re-evaluated.
    rows = []
    for ap in ["Embedded", "Decoupled-50"]:
        for al in [0, 0.25, 1]:
            x = f[(f.approach == ap) & (f.alpha == al)]
            rows.append(f"{SHORT[ap]} & {al:g} & {len(x)} & " + " & ".join(
                f"{x[m].mean():.{dec(m)}f} $\\pm$ {x[m].std(ddof=1):.{dec(m)}f}" for m in order)
                + f" & {x.optimizer_cpu_seconds.mean():.1f} & {x.ssp_calls.mean():.0f} \\\\")
    write(out / "tables/real_conv_weights.tex", tabular(
        "lcrrrrrrr", [r"Approach & $\alpha$ & $n$ & $F_{\mathrm{ref}}$ & $R_{\mathit{imb}}$ & $R_{\mathit{dis}}$ & "
                      r"Conflicts & CPU (s) & SSP solves \\"], rows))
    # Quality versus effort (checkpoints).
    raw["pt"] = raw.point.replace("final", "2500").astype(int)
    curve = raw.groupby(["approach", "alpha", "pt"]).agg(F=("ree_F_ref", "mean"), cpu=("optimizer_cpu_seconds", "mean"),
                                                          solves=("ssp_calls", "mean")).reset_index()
    emb_curve = curve[(curve.approach == "Embedded") & (curve.alpha == sel)]
    d50_final = float(k.F_ref.mean())
    match = emb_curve[emb_curve.F < d50_final].sort_values("pt").iloc[0]
    NUM["real_curve"] = dict(emb_iterations_to_beat_d50=int(match.pt), emb_cpu_to_beat_d50=round(float(match.cpu), 1),
                             emb_solves_to_beat_d50=int(match.solves), d50_final=round(d50_final, 3),
                             points={f"{r.approach}|{r.alpha:g}|{r.pt}": dict(F=round(r.F, 3), cpu=round(r.cpu, 1))
                                     for r in curve.itertuples()})
    plt = setup_plot()
    bigger_fonts(plt)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    ax = axes[0]
    for ap, al, ls in [("Embedded", 0.25, "-"), ("Embedded", 1, "--"), ("Embedded", 0, ":"),
                       ("Decoupled-50", 0.25, "-"), ("Decoupled-50", 1, "--")]:
        c = curve[(curve.approach == ap) & (curve.alpha == al)].sort_values("pt")
        ax.plot(c.cpu, c.F, linestyle=ls, marker=MARKERS[ap], markersize=3, color=COLORS[ap],
                label=f"{SHORT[ap]} $\\alpha={al:g}$")
    for val, lab, sty in [(OBSERVED["F_ref"], "Observed", (0, (4, 2))), (ssp["F_ref"], "SSP with LS", (0, (1, 1))),
                          (ssp0["F_ref"], "SSP without LS", (0, (3, 1, 1, 1)))]:
        ax.axhline(val, color="#52514e", linestyle=sty, linewidth=0.8, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); plain_log(ax, "y")
    ax.set_xlabel("Optimizer CPU time (s)"); ax.set_ylabel(PLOT["F_ref"]); ax.grid(True)
    ax = axes[1]
    tau = ssp0["F_ref"]
    for ap, al, ls in [("Embedded", 0.25, "-"), ("Embedded", 1, "--"), ("Decoupled-50", 0.25, "-")]:
        sub = raw[(raw.approach == ap) & (raw.alpha == al)]
        grid = sorted(sub.pt.unique())
        xs, ys = [], []
        for p in grid:
            g = sub[sub.pt == p]
            xs.append(g.optimizer_cpu_seconds.mean()); ys.append(100 * (g.ree_F_ref <= tau).mean())
        ax.plot(xs, ys, linestyle=ls, marker=MARKERS[ap], markersize=3, color=COLORS[ap], label=f"{SHORT[ap]} $\\alpha={al:g}$")
    ax.set_xscale("log"); ax.set_ylim(-3, 103)
    ax.set_xlabel("Optimizer CPU time (s)"); ax.set_ylabel(f"Runs with $F_{{\\mathrm{{ref}}}}\\leq{tau:.2f}$ (%)")
    ax.grid(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(out / "figures/real_convergence.pdf"); plt.close(fig)
    NUM["real_tau"] = round(tau, 3)
    # E4: 2x2 (optimized with/without LS x evaluated with/without LS) and partial adjustment.
    on = pd.read_csv(data / "reeval_ls_on.csv"); off = pd.read_csv(data / "reeval_ls_off.csv")
    m2 = on.merge(off, on=["run_id", "stage"], suffixes=("_on", "_off"))
    rows, two = [], {}
    for stage, name in [("G_base_ls_ablation", "BASE"), ("F_real_ls_ablation", "Real")]:
        for al in [0.25, 1.0]:
            for ls in [True, False]:
                x = m2[(m2.stage == stage) & (m2.alpha_on == al) & (m2.local_search_on == ls)]
                cells = []
                for ev in ["on", "off"]:
                    cells.append(" / ".join(f"{x[f'{m}_mean_{ev}'].mean():.{2 if m != 'conflicts' else 1}f}"
                                            for m in ["F_ref", "R_imb", "R_dis", "conflicts"]))
                    two[f"{name}|{al:g}|{'with' if ls else 'without'}|eval_{ev}"] = {
                        m: round(float(x[f"{m}_mean_{ev}"].mean()), 3) for m in METRICS}
                rows.append(f"{name} & {al:g} & {'With' if ls else 'Without'} & {cells[0]} & {cells[1]} \\\\")
    write(out / "tables/ls_crossed.tex", tabular(
        "lclll", [r"Instance & $\alpha$ & \shortstack{Optimized\\with LS} & \shortstack{Evaluated with LS\\"
                  r"$F_{\mathrm{ref}}$ / $R_{\mathit{imb}}$ / $R_{\mathit{dis}}$ / conflicts} & "
                  r"\shortstack{Evaluated without LS\\$F_{\mathrm{ref}}$ / $R_{\mathit{imb}}$ / $R_{\mathit{dis}}$ / conflicts} \\"],
        rows))
    NUM["ls_crossed"] = two
    part = pd.read_csv(data / "e4b_partial.csv")
    rows = []
    for _, r in cal.iterrows():
        rows.append(f"{r.p:.1f} & {r.R_imb_mean:.3f} & {r.R_dis_mean:.3f} & {r.conflicts_mean:.1f} & {r.F_ref_mean:.3f} \\\\")
    write(out / "tables/partial_calibration.tex", tabular(
        "rrrrr", [r"$p$ & $R_{\mathit{imb}}$ & $R_{\mathit{dis}}$ & Conflicts & $F_{\mathrm{ref}}$ \\"], rows, width=False))
    pstar = float(part.p.iloc[0])
    NUM["partial"] = dict(p_star=pstar, **{key: {m: round(float(part[part.label.str.startswith(key)][f"{m}_mean"].mean()), 3)
                                                 for m in METRICS} for key in ["Embedded", "Decoupled-50", "X_real"]})


# ----------------------------------------------------------------------------- E1, E2, E6, P3
def noise_and_diagnostics(data: Path, out: Path) -> None:
    noise = pd.read_csv(data / "e1a_noise.csv")
    noise["kind"] = noise.label.str.replace(r" seed \d+| instance_\d+\.json", "", regex=True)
    rows, e1a = [], {}
    for kind, x in noise.groupby("kind", sort=False):
        cv = {m: float((100 * x[f"{m}_sd"] / x[f"{m}_mean"].where(x[f"{m}_mean"] > 0)).fillna(0).mean()) for m in METRICS}
        e1a[kind] = {m: round(v, 1) for m, v in cv.items()}
        shown = kind.replace("X_real", r"$X_{\mathrm{real}}$").replace("a=", r"$\alpha=$")
        rows.append(f"{shown} & {len(x)} & " + " & ".join(f"{x[f'{m}_mean'].mean():.{dec(m)}f} ({cv[m]:.1f})" for m in METRICS) + r" \\")
    write(out / "tables/e1a_noise.tex", tabular(
        "lrrrrr", [r"Timetable & $n$ & " + " & ".join(TEX[m] for m in METRICS) + r" \\"], rows))
    NUM["e1a"] = e1a
    ree = pd.read_csv(data / "reeval_own.csv")
    labels = {"C_synthetic_30s": "Synthetic, 30 s", "D_real_30s": "Real, 30 s", "B_iteration_budgets": "Synthetic, 2500 it."}
    rows, bias = [], {}
    for stage, lab in labels.items():
        x = ree[ree.stage == stage]
        for ap in ["Embedded", "Decoupled-25", "Decoupled-50", "Decoupled-100"]:
            for al in sorted(x.alpha.unique()):
                y = x[(x.approach == ap) & (x.alpha == al)]
                if y.empty or (stage == "B_iteration_budgets" and False):
                    continue
                if stage == "B_iteration_budgets":
                    y = y[y.strategy == "Compatibility Matrix"]
                rec, re_ = y.recorded_F_ref.mean(), y.F_ref_mean.mean()
                bias[f"{stage}|{ap}|{al:g}"] = round(100 * (re_ - rec) / rec, 1)
                if ap in ("Embedded", "Decoupled-50"):
                    rows.append(f"{lab} & {SHORT.get(ap, ap)} & {al:g} & {rec:.3f} & {re_:.3f} & {100 * (re_ - rec) / rec:+.1f} & "
                                f"{y.recorded_R_imb.mean():.3f} & {y.R_imb_mean.mean():.3f} & "
                                f"{y.recorded_R_dis.mean():.3f} & {y.R_dis_mean.mean():.3f} \\\\")
    write(out / "tables/e1b_bias.tex", tabular(
        "llrrrrrrrr", [r"Experiment & Approach & $\alpha$ & \shortstack{$F_{\mathrm{ref}}$\\recorded} & "
                       r"\shortstack{$F_{\mathrm{ref}}$\\re-eval.} & \shortstack{Change\\(\%)} & "
                       r"\shortstack{$R_{\mathit{imb}}$\\recorded} & \shortstack{$R_{\mathit{imb}}$\\re-eval.} & "
                       r"\shortstack{$R_{\mathit{dis}}$\\recorded} & \shortstack{$R_{\mathit{dis}}$\\re-eval.} \\"], rows))
    NUM["e1b_bias"] = bias
    c = finals(pd.read_csv(data / "E1c.csv"))
    c["inst"] = np.where(c.instance == "real.json", "Real", "BASE")
    rows, e1c = [], {}
    for (inst, al, kind), x in c.groupby(["inst", "alpha", "budget_kind"]):
        std = x[x.crn == False].set_index(["instance", "seed"])  # noqa: E712
        crn = x[x.crn == True].set_index(["instance", "seed"])  # noqa: E712
        d = (crn.ree_F_ref - std.ree_F_ref.reindex(crn.index)).to_numpy()
        neg, n, p = sign_test(d)
        lab = "2500 it." if kind == "iterations" else "30 s"
        rows.append(f"{inst} & {al:g} & {lab} & {std.rec_F_ref.mean():.3f} & {std.ree_F_ref.mean():.3f} & "
                    f"{crn.rec_F_ref.mean():.3f} & {crn.ree_F_ref.mean():.3f} & "
                    f"{100 * d.mean() / std.ree_F_ref.mean():+.1f} & {neg}/{n} & {pfmt(p)} \\\\")
        e1c[f"{inst}|{al:g}|{lab}"] = dict(std_rec=round(std.rec_F_ref.mean(), 3), std_ree=round(std.ree_F_ref.mean(), 3),
                                           crn_rec=round(crn.rec_F_ref.mean(), 3), crn_ree=round(crn.ree_F_ref.mean(), 3),
                                           pct=round(100 * d.mean() / std.ree_F_ref.mean(), 1), better=f"{neg}/{n}",
                                           crn_R_dis=round(crn.ree_R_dis.mean(), 3), std_R_dis=round(std.ree_R_dis.mean(), 3),
                                           crn_conf=round(crn.ree_conflicts.mean(), 1), std_conf=round(std.ree_conflicts.mean(), 1))
    write(out / "tables/e1c_crn.tex", tabular(
        "lrlrrrrrrr", [r"Instance & $\alpha$ & Budget & \shortstack{Standard\\recorded} & \shortstack{Standard\\re-eval.} & "
                       r"\shortstack{CRN\\recorded} & \shortstack{CRN\\re-eval.} & \shortstack{Change\\(\%)} & "
                       r"\shortstack{CRN\\better} & $p$ \\"], rows))
    NUM["e1c"] = e1c
    # E2b, E2c.
    w = pd.read_csv(data / "E2b_warm_start.csv")
    w["inst"] = np.where(w.instance == "real.json", "Real", "BASE")
    w["pt"] = w.point.replace("final", "2500").astype(int)
    a = pd.read_csv(data / "E5a_real_converged.csv")
    b = pd.read_csv(data / "E5b_synthetic_converged.csv")
    a, b = finals(a), finals(b)
    start_real = a[(a.approach == "Embedded") & (a.alpha == 0.25)]
    start_real = start_real[start_real.seed.isin(w[w.inst == "Real"].seed.unique())]
    start_base = b[(b.approach == "Embedded") & (b.alpha == 0.25) & (b.group == "BASE")]
    alpha1_real = a[(a.approach == "Embedded") & (a.alpha == 1) & (a.seed.isin(w[w.inst == "Real"].seed.unique()))]
    alpha1_base = b[(b.approach == "Embedded") & (b.alpha == 1) & (b.group == "BASE")]
    rows, e2 = [], {}
    for inst, st, a1 in [("Real", start_real, alpha1_real), ("BASE", start_base, alpha1_base)]:
        x = w[w.inst == inst]
        lines = [("Start: $\\alpha=0.25$ final", st), ("$\\alpha=1$ from start, 100 it.", x[x.pt == 100]),
                 ("$\\alpha=1$ from start, 500 it.", x[x.pt == 500]), ("$\\alpha=1$ from start, 2500 it.", x[x.pt == 2500]),
                 ("$\\alpha=1$ from CM construction, 2500 it.", a1)]
        for lab, y in lines:
            rows.append(f"{inst} & {lab} & {len(y)} & " + " & ".join(f"{y[REE[m]].mean():.{dec(m)}f}" for m in METRICS)
                        + f" & {y.rec_F_ref.mean():.3f} \\\\")
            e2[f"{inst}|{lab}"] = {m: round(float(y[REE[m]].mean()), 3) for m in METRICS} | {"rec_F_ref": round(float(y.rec_F_ref.mean()), 3)}
    cc = finals(pd.read_csv(data / "E2c_neighborhood_200.csv"))
    base50 = a[(a.approach == "Embedded") & (a.alpha == 1) & (a.seed.isin(cc.seed.unique()))]
    for lab, y in [("$\\alpha=1$, 50 neighbours", base50), ("$\\alpha=1$, 200 neighbours", cc)]:
        rows.append(f"Real & {lab} & {len(y)} & " + " & ".join(f"{y[REE[m]].mean():.{dec(m)}f}" for m in METRICS)
                    + f" & {y.rec_F_ref.mean():.3f} \\\\")
        e2[f"Real|{lab}"] = {m: round(float(y[REE[m]].mean()), 3) for m in METRICS} | {"cpu": round(float(y.optimizer_cpu_seconds.mean()), 1)}
    write(out / "tables/e2_alpha.tex", tabular(
        "llrrrrrr", [r"Instance & Run & $n$ & " + " & ".join(TEX[m] for m in METRICS) + r" & \shortstack{$F_{\mathrm{ref}}$\\recorded} \\"], rows))
    NUM["e2"] = e2
    # E2a trajectories figure.
    tr = pd.read_json(data / "E2a_trajectories_trace.jsonl", lines=True)
    meta = pd.read_csv(data / "E2a_trajectories.csv")[["run_id", "instance", "alpha", "seed"]]
    tr = tr.merge(meta, on="run_id")
    plt = setup_plot(); bigger_fonts(plt)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=True)
    for j, inst in enumerate(["real.json", "BASE"]):
        sub = tr[tr.instance == "real.json"] if inst == "real.json" else tr[tr.instance != "real.json"]
        for al in [0.25, 1.0]:
            color = COLORS["Embedded"] if al == 0.25 else COLORS["Decoupled-50"]
            x = sub[sub.alpha == al]
            for i, comp in enumerate(["best_R_imb", "best_R_dis"]):
                # Rolling median over 25 iterations per run removes the iteration-to-iteration
                # fluctuation of the re-solved incumbent; then mean and range across runs.
                runs = x.pivot_table(index="iteration", columns="run_id", values=comp).sort_index()
                runs = runs.rolling(25, min_periods=1).median()
                ax = axes[i, j]
                ax.fill_between(runs.index, runs.min(axis=1), runs.max(axis=1), color=color, alpha=0.18, linewidth=0)
                ax.plot(runs.index, runs.mean(axis=1), color=color, linewidth=1.6,
                        label=rf"$\alpha={al:g}$")
        for i, comp in enumerate(["R_imb", "R_dis"]):
            ax = axes[i, j]
            ax.set_xscale("log"); ax.set_yscale("log"); plain_log(ax, "y"); ax.grid(True)
            if j == 0:
                ax.set_ylabel(PLOT[comp] + " of best solution")
            if i == 0:
                ax.set_title("Real instance (5 seeds)" if inst == "real.json" else "BASE (5 instances)")
            else:
                ax.set_xlabel("Iteration")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "figures/e2_trajectories.pdf"); plt.close(fig)
    # E6 and P3.
    t = json.loads((data / "ssp_timing.json").read_text(encoding="utf-8"))
    rows = [f"BASE (10 instances, initial CM timetable) & 500 & {np.mean([x['mean_ms'] for x in t['base']]):.2f} & "
            f"{np.mean([x['p95_ms'] for x in t['base']]):.2f} & -- \\\\",
            f"Real, initial CM timetable & {t['real']['initial_cm']['students']} & {t['real']['initial_cm']['mean_ms']:.2f} & "
            f"{t['real']['initial_cm']['p95_ms']:.2f} & -- \\\\",
            f"Real, observed timetable & {t['real']['observed']['students']} & {t['real']['observed']['mean_ms']:.2f} & "
            f"{t['real']['observed']['p95_ms']:.2f} & -- \\\\"]
    for sc in t["scaling"]:
        rows.append(f"Real, {100 * sc['fraction']:.0f}\\% of students & {sc['students']} & {sc['mean_ms']:.2f} & "
                    f"{sc['p95_ms']:.2f} & {sc['iterations_30s']} \\\\")
    write(out / "tables/e6_timing.tex", tabular(
        "lrrrr", [r"Instance & Students & \shortstack{Mean\\(ms)} & \shortstack{95th pct.\\(ms)} & "
                  r"\shortstack{Embedded iterations\\in 30 s ($\alpha=0.25$)} \\"], rows))
    NUM["e6"] = dict(base_ms=round(float(np.mean([x["mean_ms"] for x in t["base"]])), 2),
                     real_ms=round(t["real"]["initial_cm"]["mean_ms"], 2),
                     scaling=[dict(students=s["students"], ms=round(s["mean_ms"], 2), it=s["iterations_30s"]) for s in t["scaling"]])
    p3 = pd.read_csv(data / "P3.csv")
    fin = finals(p3)[["run_id", "rec_F"]].rename(columns={"rec_F": "F_final"})
    x = p3[p3.point != "final"].merge(fin, on="run_id")
    x["pt"] = x.point.astype(int)
    x = x[(x.pt >= 500) & (x.rec_F > 0)]
    x["gain"] = 100 * (x.rec_F - x.F_final) / x.rec_F
    rows, p3n = [], {}
    for k, sub in x.groupby("pt"):
        rows.append(f"{k} & {sub.gain.median():.2f} & {sub.gain.quantile(0.75):.2f} & {100 * (sub.gain > 1).mean():.1f} \\\\")
        p3n[int(k)] = dict(median=round(float(sub.gain.median()), 2), p75=round(float(sub.gain.quantile(0.75)), 2))
    write(out / "tables/pilot_decoupled.tex", tabular(
        "rrrr", [r"From budget & \shortstack{Median further\\gain to 3000 (\%)} & \shortstack{75th\\percentile (\%)} & "
                 r"\shortstack{Runs improving\\by $>1$\% (\%)} \\"], rows, width=False))
    NUM["p3"] = p3n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for sub in ("tables", "figures"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    seq = pd.read_csv(args.data / "E3_sequential.csv")
    noise_and_diagnostics(args.data, args.out)
    real(args.data, args.out, seq)
    synthetic(args.data, args.out, seq)
    (args.out / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float), encoding="utf-8")
    print(f"Wrote tables, figures and numbers.json to {args.out}")


if __name__ == "__main__":
    main()
