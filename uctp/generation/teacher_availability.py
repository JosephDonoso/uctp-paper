"""Blockwise teacher availability with at least one available block per day."""

import random


def _generate_teacher_availability(
    days: tuple[str, ...], probability: float, rng: random.Random
) -> list[dict]:
    availability = []
    for day in days:
        values = [int(rng.random() < probability) for _ in range(7)]
        if not any(values):
            values[rng.randrange(7)] = 1
        availability.append({day: values})
    return availability
