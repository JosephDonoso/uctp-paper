"""Internal inconvenience cost required by the SSP and search objective."""

from functools import lru_cache


def compute_inconvenience_cost(blocks: list[int], *, idle_tolerance: float) -> float:
    """Evaluate Cost(W_s), including overlaps and the lunch break.

    Notes
    -----
    Implements eq:incomodidad, eq:penalizacion and eq:insatisfaccion_individual.
    Duplicate blocks must not be removed: each negative consecutive gap costs
    seven units. The input is not mutated. This is a search primitive, not a
    reporting or extraction function. The cost depends only on the multiset of
    blocks, so results are memoized on the sorted tuple (identical values).
    """
    if not blocks:
        return 0
    return _cost(tuple(sorted(blocks)), float(idle_tolerance))


@lru_cache(maxsize=1 << 20)
def _cost(ordered_blocks: tuple[int, ...], idle_tolerance: float) -> float:
    inconvenience = 0
    previous_day = -1
    previous_slot = 0
    first_slot = 0
    for block in ordered_blocks:
        day, slot = divmod(block, 7)
        if day != previous_day:
            if previous_day >= 0 and first_slot <= 3 and previous_slot >= 4:
                inconvenience += 1
            previous_day, first_slot = day, slot
        else:
            gap = slot - previous_slot - 1
            inconvenience += 7 if gap < 0 else gap
        previous_slot = slot
    if first_slot <= 3 and previous_slot >= 4:
        inconvenience += 1
    threshold = max(0, idle_tolerance)
    if inconvenience > threshold:
        return threshold + (inconvenience - threshold) ** 2
    return inconvenience
