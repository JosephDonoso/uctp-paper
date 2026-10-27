"""General filtered descriptives and factor comparisons for exported runs."""

import json
import math
from pathlib import Path

import numpy as np

from .comparisons import compare_factors, group_rows, unit_values
from .records import CONDITION_FIELDS, export_raw, flatten_record, read_records, write_table
from .statistics import describe


def analyze(
    inputs: list[Path],
    output: Path,
    *,
    metrics: list[str],
    group_by: list[str] | None,
    filters: list[str],
    compare: list[str],
    mode: str,
    pair_by: list[str],
    unit: str,
    instance_field: str,
    strata: list[str] | None,
    replicates: int,
    seed: int,
    include_failed: bool = False,
) -> dict:
    """Export raw, descriptive and optional inferential tables into an empty folder."""
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Analysis destination must be absent or empty.")
    if replicates < 100 or seed < 0:
        raise ValueError(
            "Use at least 100 bootstrap/permutation replicates and a nonnegative seed."
        )
    records = read_records(inputs)
    for expression in filters:
        if "=" not in expression:
            raise ValueError("Filters use FIELD=VALUE or FIELD=VALUE1,VALUE2.")
        key, values = expression.split("=", 1)
        allowed = {value.strip().lower() for value in values.split(",")}
        if not any(key in flatten_record(r) for r in records):
            raise ValueError(f"Unknown filter field: {key}")

        def matches(value):
            if isinstance(value, bool):
                return any(
                    token in ({"true", "on", "1"} if value else {"false", "off", "0"})
                    for token in allowed
                )
            if isinstance(value, (int, float)):
                for token in allowed:
                    try:
                        if float(token) == value:
                            return True
                    except ValueError:
                        continue
                return False
            return str(value).lower() in allowed

        records = [r for r in records if matches(flatten_record(r).get(key))]
    if not records:
        raise ValueError("No runs match the filters.")
    rows = [flatten_record(r) for r in records]
    available = {key for row in rows for key in row}
    if any(metric not in available for metric in metrics):
        raise ValueError("Some requested metrics are absent from every selected record.")
    fields = (
        group_by
        if group_by is not None
        else (
            [f for f in CONDITION_FIELDS if f in available]
            + sorted(f for f in available if f.startswith("tag_"))
        )
    )
    if any(
        field not in available
        for field in fields + (strata or []) + (pair_by if compare and mode == "paired" else [])
    ):
        raise ValueError("A grouping, strata or pairing field is absent from the inputs.")
    rng = np.random.default_rng(seed)
    summaries, success = [], []
    for labels, group in group_rows(rows, fields):
        solved = [r for r in group if r["status"] == "solved"]
        success.append(
            labels
            | dict(
                attempted=len(group),
                solved=len(solved),
                failed=len(group) - len(solved),
                success_rate=len(solved) / len(group),
            )
        )
        for metric in metrics:
            raw = [
                float(r[metric])
                for r in (group if include_failed else solved)
                if isinstance(r.get(metric), (int, float)) and math.isfinite(r[metric])
            ]
            values = unit_values(group, metric, unit, instance_field, include_failed)
            stats = describe(values, rng, replicates)
            summaries.append(
                labels
                | dict(
                    metric=metric,
                    unit=unit,
                    successful_runs=len(solved),
                    metric_runs=len(raw),
                    run_mean=float(np.mean(raw)) if raw else None,
                    run_sd=float(np.std(raw, ddof=1)) if len(raw) > 1 else None,
                    **stats,
                )
            )
    contrasts, reliability, pairs = compare_factors(
        rows,
        compare,
        metrics,
        mode=mode,
        pair_by=pair_by,
        unit=unit,
        instance_field=instance_field,
        strata=strata,
        rng=rng,
        replicates=replicates,
        include_failed=include_failed,
    )
    methods = dict(
        records=len(rows),
        successful=sum(r["status"] == "solved" for r in rows),
        metrics=metrics,
        group_by=fields,
        filters=filters,
        factors=compare,
        comparison_mode=mode,
        pair_by=pair_by,
        statistical_unit=unit,
        instance_field=instance_field,
        strata=strata,
        bootstrap_permutation_replicates=replicates,
        analysis_seed=seed,
        intervals="pointwise 95% percentile bootstrap, not multiplicity adjusted",
        quality_family="all valid quality contrasts in this invocation; joint Holm",
        reliability_family="paired success contrasts, separate joint Holm family",
        paired_test="exact two-sided sign test, absolute differences <=1e-8 are ties; tests direction, not equality of means",
        independent_test="two-sided independent-unit permutation test for mean difference; exchangeability required",
        instance_unit="equal-weight means per instance; paired contrasts first match runs then average paired differences within instance",
        run_unit="run-to-run variation conditional on tested instances; repeated seeds are not new independent institutions",
        missing_data="all failures retained in raw/success tables; quality uses available successful runs or jointly successful pairs",
        small_samples="fewer than two units: no confidence interval or inferential p value",
        direction="left minus right; lower is favorable for quality metrics, not automatically for throughput",
        formats="raw CSV and full JSONL; CSV tables plus analysis.json; no images",
        include_failed_metrics=include_failed,
    )
    result = dict(
        methods=methods,
        summaries=summaries,
        success_rates=success,
        contrasts=contrasts,
        reliability=reliability,
    )
    output.mkdir(parents=True, exist_ok=True)
    export_raw(output, records)
    for name, values in [
        ("summary", summaries),
        ("success_rates", success),
        ("contrasts", contrasts),
        ("reliability", reliability),
        ("paired_values", pairs),
    ]:
        write_table(
            output / f"{name}.csv",
            values,
            empty_fields=["factor", "left", "right", "metric", "mean_difference", "p", "p_holm"],
        )
    (output / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "METHODS.md").write_text(
        "# Analysis specification\n\n"
        + "\n".join(f"- **{key}:** {value}" for key, value in methods.items())
        + "\n",
        encoding="utf-8",
    )
    return result
