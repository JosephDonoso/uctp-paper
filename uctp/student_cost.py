"""Internal inconvenience cost required by the SSP and search objective."""


def compute_inconvenience_cost(blocks: list[int], *, idle_tolerance: float) -> float:
    """Evaluate Cost(W_s), including overlaps and the lunch break.

    Notes
    -----
    Implements eq:incomodidad, eq:penalizacion and eq:insatisfaccion_individual.
    Duplicate blocks must not be removed: each negative consecutive gap costs
    seven units. The input is not mutated. This is a search primitive, not a
    reporting or extraction function.
    """
    if not blocks:
        return 0
    ordered_blocks = sorted(blocks)
    days = {day_index: [] for day_index in range(5)}
    for block in ordered_blocks:
        days[block // 7].append(block % 7)
    inconvenience = 0
    for day_blocks in days.values():
        if len(day_blocks) < 2:
            continue
        for index in range(len(day_blocks) - 1):
            gap = day_blocks[index + 1] - day_blocks[index] - 1
            if gap < 0:
                gap = 7
            inconvenience += gap
        if day_blocks[0] <= 3 and day_blocks[-1] >= 4:
            inconvenience += 1
    threshold = max(0, float(idle_tolerance))
    if inconvenience > threshold:
        return threshold + (inconvenience - threshold) ** 2
    return inconvenience
