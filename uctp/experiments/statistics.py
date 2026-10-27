"""Explicit bootstrap, exact sign tests and Holm adjustment, using NumPy only."""

import itertools
import math

import numpy as np


def mean_interval(values: list[float], rng: np.random.Generator, replicates: int) -> tuple:
    """Return a pointwise 95% percentile bootstrap CI, or nulls for <2 units."""
    if len(values) < 2:
        return None, None
    data = np.asarray(values, dtype=float)
    means = []
    for start in range(0, replicates, 1000):
        size = min(1000, replicates - start)
        means.extend(data[rng.integers(0, len(data), size=(size, len(data)))].mean(axis=1))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def sign_probability(positive: int, negative: int) -> float:
    """Two-sided exact binomial sign probability, excluding tied units."""
    count = positive + negative
    if not count:
        return 1.0
    return min(
        1.0, sum(math.comb(count, k) for k in range(min(positive, negative) + 1)) / 2 ** (count - 1)
    )


def holm(rows: list[dict], *, key: str = "p", adjusted: str = "p_holm") -> None:
    """Attach joint Holm-adjusted p values to the explicitly supplied family."""
    valid = sorted([row for row in rows if row.get(key) is not None], key=lambda row: row[key])
    previous = 0.0
    for index, row in enumerate(valid):
        previous = min(1.0, max(previous, row[key] * (len(valid) - index)))
        row[adjusted] = previous
    for row in rows:
        row.setdefault(adjusted, None)
        row["family_size"] = len(valid)


def describe(values: list[float], rng: np.random.Generator, replicates: int) -> dict:
    """Compute scalar descriptives and a pointwise bootstrap CI of the mean."""
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(
            n=0,
            mean=None,
            sd=None,
            median=None,
            minimum=None,
            maximum=None,
            q25=None,
            q75=None,
            ci95_low=None,
            ci95_high=None,
        )
    low, high = mean_interval(values, rng, replicates)
    return dict(
        n=len(a),
        mean=float(a.mean()),
        sd=float(a.std(ddof=1)) if len(a) > 1 else None,
        median=float(np.median(a)),
        minimum=float(a.min()),
        maximum=float(a.max()),
        q25=float(np.quantile(a, 0.25)),
        q75=float(np.quantile(a, 0.75)),
        ci95_low=low,
        ci95_high=high,
    )


def independent_difference(
    left: list[float], right: list[float], rng: np.random.Generator, replicates: int
) -> dict:
    """Independent-unit mean difference, bootstrap CI and permutation p value."""
    a, b = np.asarray(left), np.asarray(right)
    difference = float(a.mean() - b.mean())
    if len(a) < 2 or len(b) < 2:
        return dict(mean_difference=difference, ci95_low=None, ci95_high=None, p=None)
    boot = []
    for _ in range(replicates):
        boot.append(
            float(
                a[rng.integers(0, len(a), len(a))].mean()
                - b[rng.integers(0, len(b), len(b))].mean()
            )
        )
    low, high = np.quantile(boot, [0.025, 0.975])
    pooled = np.concatenate([a, b])
    total = math.comb(len(pooled), len(a))
    exceed = 0
    if total <= replicates:
        for indices in itertools.combinations(range(len(pooled)), len(a)):
            selected = np.zeros(len(pooled), dtype=bool)
            selected[list(indices)] = True
            statistic = float(pooled[selected].mean() - pooled[~selected].mean())
            exceed += abs(statistic) >= abs(difference) - 1e-12
        probability = exceed / total
    else:
        for _ in range(replicates):
            shuffled = rng.permutation(pooled)
            statistic = float(shuffled[: len(a)].mean() - shuffled[len(a) :].mean())
            exceed += abs(statistic) >= abs(difference) - 1e-12
        probability = (exceed + 1) / (replicates + 1)
    return dict(
        mean_difference=difference, ci95_low=float(low), ci95_high=float(high), p=probability
    )
