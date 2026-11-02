"""Fresh-order re-evaluation of a fixed timetable (experiment E1).

The SSP processes students in a random order, so the quality of a timetable X
is a random variable. A search reports the best value among many noisy
evaluations, which is biased downwards. Re-evaluating X with new orders gives
an unbiased estimate. Orders come from ``random.Random(f"reeval:{key}:{k}")``:
reproducible and independent of the run's own random stream (the global
generator is never touched).
"""

import random
import statistics

from .domain import Instance, Solution
from .metrics import measure_solution
from .objective import SearchObjective
from .sectioning import StudentSectioning

FIELDS = ("R_imb", "R_dis", "conflicts", "F_ref", "F")


def reevaluate(
    instance: Instance,
    X: dict,
    *,
    alpha: float,
    orders: int,
    key: str,
    local_search: bool = True,
    local_search_probability: float | None = None,
) -> dict:
    """Mean and SD of each metric over ``orders`` SSP solves of a fixed X."""
    objective = SearchObjective(instance, alpha=alpha, cv_tolerance=0.10, idle_tolerance=4)
    sectioning = StudentSectioning(
        instance,
        objective,
        local_search=local_search,
        local_search_probability=local_search_probability,
    )
    values = {field: [] for field in FIELDS}
    for k in range(orders):
        solution = Solution(dict(X), {})
        sectioning.solve(solution, rng=random.Random(f"reeval:{key}:{k}"))
        metrics = measure_solution(instance, solution, alpha=alpha)["metrics"]
        for field in FIELDS:
            values[field].append(float(metrics[field]))
    summary = {"orders": orders}
    for field in FIELDS:
        summary[f"{field}_mean"] = statistics.fmean(values[field])
        summary[f"{field}_sd"] = statistics.stdev(values[field]) if orders > 1 else 0.0
    return summary
