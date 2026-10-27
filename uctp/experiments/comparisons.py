"""Factor contrasts with explicit strata, matching and statistical units."""

import itertools
import json
import math
from collections import defaultdict

import numpy as np

from .records import CONDITION_FIELDS
from .statistics import describe, holm, independent_difference, sign_probability


def _value_key(value):
    if isinstance(value, bool):
        return "boolean", value
    if isinstance(value, (int, float)):
        return "number", value
    if value is None:
        return "missing", ""
    return "text", json.dumps(value, sort_keys=True)


def group_rows(rows: list[dict], fields: list[str]) -> list[tuple[dict, list[dict]]]:
    """Group records by scalar values, keeping labels in their original types."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(_value_key(row.get(f)) for f in fields)].append(row)
    return [({field: group[0].get(field) for field in fields}, group) for group in groups.values()]


def _finite(row: dict, metric: str) -> bool:
    value = row.get(metric)
    return isinstance(value, (int, float)) and math.isfinite(value)


def unit_values(
    rows: list[dict], metric: str, unit: str, instance_field: str, include_failed: bool = False
) -> list[float]:
    """Return run values or equal-weight instance means over successful runs."""
    valid = [r for r in rows if (r["status"] == "solved" or include_failed) and _finite(r, metric)]
    if unit == "run":
        return [float(r[metric]) for r in valid]
    if any(r.get(instance_field) is None for r in valid):
        raise ValueError(
            f"Instance unit requires {instance_field}; use --instance-field or --unit run."
        )
    return [
        float(np.mean([r[metric] for r in group]))
        for _, group in group_rows(valid, [instance_field])
    ]


def _matched(left: list[dict], right: list[dict], pair_by: list[str]) -> tuple:
    maps = []
    for rows in [left, right]:
        mapping = {}
        for row in rows:
            if any(row.get(key) is None for key in pair_by):
                raise ValueError(
                    f"Missing pair key in {pair_by}; explicitly choose valid keys or independent mode."
                )
            key = tuple(_value_key(row[field]) for field in pair_by)
            if key in mapping:
                raise ValueError(
                    f"Duplicate pair key {pair_by}; include seed or other replicate identifiers."
                )
            mapping[key] = row
        maps.append(mapping)
    common = sorted(maps[0].keys() & maps[1].keys())
    return maps, [(maps[0][key], maps[1][key]) for key in common]


def _reliability(pairs: list[tuple], unit: str, instance_field: str) -> dict:
    if unit == "instance":
        grouped = defaultdict(list)
        for left, right in pairs:
            if left.get(instance_field) is None or left.get(instance_field) != right.get(
                instance_field
            ):
                raise ValueError("Paired instance units require matching instance identities.")
            grouped[left[instance_field]].append((left, right))
        outcomes = [
            (
                any(a["status"] == "solved" for a, _ in group),
                any(b["status"] == "solved" for _, b in group),
            )
            for group in grouped.values()
        ]
    else:
        outcomes = [(a["status"] == "solved", b["status"] == "solved") for a, b in pairs]
    left_only = sum(a and not b for a, b in outcomes)
    right_only = sum(b and not a for a, b in outcomes)
    return dict(
        matched_units=len(outcomes),
        left_only_success=left_only,
        right_only_success=right_only,
        both_success=sum(a and b for a, b in outcomes),
        both_failure=sum(not a and not b for a, b in outcomes),
        success_definition="at_least_one_success_among_matched_seeds"
        if unit == "instance"
        else "run_success",
        test="exact_paired_mcnemar",
        p=sign_probability(left_only, right_only) if len(outcomes) > 1 else None,
    )


def compare_factors(
    rows: list[dict],
    factors: list[str],
    metrics: list[str],
    *,
    mode: str,
    pair_by: list[str],
    unit: str,
    instance_field: str,
    strata: list[str] | None,
    rng: np.random.Generator,
    replicates: int,
    include_failed: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Contrast one varying factor at a time, holding other controls fixed.

    Paired sign tests concern direction, while bootstrap intervals describe
    the mean difference. All quality p values form one Holm family; paired
    success contrasts form a separate family. No test runs with <2 units.
    """
    quality, reliability, exported_pairs = [], [], []
    available = {key for row in rows for key in row}
    controls = [f for f in CONDITION_FIELDS if f in available] + sorted(
        f for f in available if f.startswith("tag_")
    )
    for factor in factors:
        if factor not in available or any(row.get(factor) is None for row in rows):
            raise ValueError(f"Comparison factor is missing: {factor}")
        if factor == "alpha" and any(m in {"F", "initial_F"} for m in metrics):
            raise ValueError("Native F changes scale across alpha; compare F_ref across weights.")
        if factor in pair_by and mode == "paired":
            raise ValueError("A varying factor cannot also be a pairing key.")
        fixed = [f for f in (strata if strata is not None else controls) if f != factor]
        if strata is None and factor == "budget":
            fixed = [f for f in fixed if f != "iteration_budget"]
        for labels, group in group_rows(rows, fixed):
            treatments = group_rows(group, [factor])
            for (label_a, left), (label_b, right) in itertools.combinations(treatments, 2):
                context = {
                    **labels,
                    "factor": factor,
                    "left": label_a[factor],
                    "right": label_b[factor],
                    "unit": unit,
                    "comparison_mode": mode,
                }
                pairs = []
                if mode == "paired":
                    maps, pairs = _matched(left, right, pair_by)
                    reliability.append(
                        context
                        | _reliability(pairs, unit, instance_field)
                        | dict(
                            matched_runs=len(pairs),
                            unmatched_left=len(maps[0]) - len(pairs),
                            unmatched_right=len(maps[1]) - len(pairs),
                        )
                    )
                for metric in metrics:
                    item = context | dict(metric=metric)
                    if mode == "independent":
                        a = unit_values(left, metric, unit, instance_field, include_failed)
                        b = unit_values(right, metric, unit, instance_field, include_failed)
                        if not a or not b:
                            continue
                        item.update(
                            left_units=len(a),
                            right_units=len(b),
                            left_mean=float(np.mean(a)),
                            right_mean=float(np.mean(b)),
                            test="independent_mean_permutation",
                        )
                        item.update(independent_difference(a, b, rng, replicates))
                    else:
                        good = [
                            (a, b)
                            for a, b in pairs
                            if (a["status"] == b["status"] == "solved" or include_failed)
                            and _finite(a, metric)
                            and _finite(b, metric)
                        ]
                        if not good:
                            continue
                        paired_units = defaultdict(list)
                        for index, (a, b) in enumerate(good):
                            key = a.get(instance_field) if unit == "instance" else index
                            if unit == "instance" and (key is None or key != b.get(instance_field)):
                                raise ValueError(
                                    "Instance aggregation requires matched instance identities."
                                )
                            paired_units[key].append((float(a[metric]), float(b[metric])))
                            exported_pairs.append(
                                context
                                | dict(
                                    metric=metric,
                                    left_run_id=a["run_id"],
                                    right_run_id=b["run_id"],
                                    instance=a.get(instance_field),
                                    seed=a.get("seed"),
                                    left_value=a[metric],
                                    right_value=b[metric],
                                    difference=a[metric] - b[metric],
                                )
                            )
                        a = [float(np.mean([x for x, _ in v])) for v in paired_units.values()]
                        b = [float(np.mean([y for _, y in v])) for v in paired_units.values()]
                        differences = [x - y for x, y in zip(a, b, strict=True)]
                        stats = describe(differences, rng, replicates)
                        lower = sum(d < -1e-8 for d in differences)
                        higher = sum(d > 1e-8 for d in differences)
                        item.update(
                            matched_successful_runs=len(good),
                            left_units=len(a),
                            right_units=len(b),
                            left_mean=float(np.mean(a)),
                            right_mean=float(np.mean(b)),
                            mean_difference=stats["mean"],
                            ci95_low=stats["ci95_low"],
                            ci95_high=stats["ci95_high"],
                            left_lower_units=lower,
                            left_higher_units=higher,
                            tied_units=len(a) - lower - higher,
                            standardized_mean_difference=stats["mean"] / stats["sd"]
                            if stats["sd"]
                            else None,
                            p=sign_probability(lower, higher) if len(a) > 1 else None,
                            test="exact_paired_sign",
                        )
                    quality.append(item)
    holm(quality)
    holm(reliability)
    return quality, reliability, exported_pairs
