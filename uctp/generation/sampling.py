"""Reference random sampling and teacher allocation for synthetic instances."""

import random


def _sample_integer(minimum: int, maximum: int, mean: float, sd: float, rng: random.Random) -> int:
    if minimum == maximum:
        return minimum
    while True:
        value = rng.gauss(mean, sd)
        if minimum - 0.5 <= value <= maximum + 0.5:
            return max(minimum, min(maximum, int(round(value))))


def _sample_counts(
    count: int, minimum: int, maximum: int, mean: float, sd: float, rng: random.Random
) -> list[int]:
    return [_sample_integer(minimum, maximum, mean, sd, rng) for _ in range(count)]


def _assign_teachers(
    courses: list[dict], teachers: list[dict], targets: dict[str, int], rng: random.Random
) -> None:
    """Retain coverage, target-guided extras and reference tie-breaking."""
    max_load, max_teachers = 4, 2
    loads = {teacher["nombre"]: 0 for teacher in teachers}
    names = list(loads)
    course_teachers: list[list[str]] = [[] for _ in courses]
    order = list(range(len(courses)))
    rng.shuffle(order)
    for index, name in enumerate(names):
        course_teachers[order[index]].append(name)
        loads[name] += 1
    for course_index in order[len(names) :]:
        candidates = [name for name in names if loads[name] < max_load] or names[:]
        teacher = min(candidates, key=lambda name: (loads[name], name))
        course_teachers[course_index].append(teacher)
        loads[teacher] += 1
    extras = sum(max(0, targets[name] - loads[name]) for name in names)
    candidates = [
        index for index in range(len(courses)) if len(course_teachers[index]) < max_teachers
    ]
    rng.shuffle(candidates)
    for course_index in candidates:
        if extras <= 0:
            break
        present = set(course_teachers[course_index])
        eligible = [name for name in names if name not in present and loads[name] < max_load]
        if not eligible:
            continue
        eligible.sort(key=lambda name: (loads[name] - targets[name], loads[name], name))
        teacher = eligible[0]
        course_teachers[course_index].append(teacher)
        loads[teacher] += 1
        extras -= 1
    for course_index, course in enumerate(courses):
        for section in course["paralelos"]:
            section["maestros"] = course_teachers[course_index][:]
