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


def _assign_section_teachers(
    courses: list[dict],
    rng: random.Random,
    *,
    load_mean: float,
    load_sd: float,
    max_load: int,
    co_teaching_probability: float,
    shared_teacher_probability: float,
) -> list[str]:
    """Assign teachers to sections and return the teacher names in creation order.

    Each section receives one teacher, or two with ``co_teaching_probability``;
    every assigned teacher teaches all lectures of the section. Teachers are
    created with target weekly loads drawn from a rounded, truncated normal on
    ``[2, max_load]`` until the targets cover the weekly teaching demand.
    Courses are visited in random order. A later section of a course reuses the
    staff of the preceding section with ``shared_teacher_probability`` when the
    staff can take it without exceeding ``max_load``; otherwise each place goes
    to the teacher with the largest remaining target among those that stay
    within ``max_load`` (ties broken at random). No teacher exceeds
    ``max_load`` weekly lectures.
    """
    sections = [section for course in courses for section in course["paralelos"]]
    staff_sizes = {
        id(section): 2 if rng.random() < co_teaching_probability else 1 for section in sections
    }
    demand = sum(len(section["clases"]) * staff_sizes[id(section)] for section in sections)
    targets: list[int] = []
    while sum(targets) < demand:
        targets.append(_sample_integer(2, max_load, load_mean, load_sd, rng))
    loads = [0] * len(targets)
    order = list(range(len(courses)))
    rng.shuffle(order)
    for course_index in order:
        previous: list[int] = []
        for position, section in enumerate(courses[course_index]["paralelos"]):
            lectures = len(section["clases"])
            size = staff_sizes[id(section)]
            staff: list[int] = []
            if (
                position > 0
                and len(previous) == size
                and rng.random() < shared_teacher_probability
                and all(loads[t] + lectures <= max_load for t in previous)
            ):
                staff = previous[:]
            while len(staff) < size:
                fitting = [
                    t for t in range(len(targets))
                    if t not in staff and loads[t] + lectures <= max_load
                ]
                if not fitting:
                    targets.append(_sample_integer(2, max_load, load_mean, load_sd, rng))
                    loads.append(0)
                    continue
                staff.append(max(fitting, key=lambda t: (targets[t] - loads[t], rng.random())))
            for teacher in staff:
                loads[teacher] += lectures
            section["maestros"] = staff
            previous = staff
    used = [t for t in range(len(targets)) if loads[t] > 0]
    names = {t: f"DOC{rank:03d}" for rank, t in enumerate(used, start=1)}
    for section in sections:
        section["maestros"] = [names[t] for t in section["maestros"]]
    return [names[t] for t in used]
