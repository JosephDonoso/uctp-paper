"""Geometric base levels with rounded, truncated-normal request counts."""

import random

from .parameters import GenerationOptions
from .sampling import _sample_integer


def _generate_students(
    courses: list[dict], options: GenerationOptions, rng: random.Random
) -> list[dict]:
    level_count = len(options.courses_per_level)
    by_level: list[list[int]] = [[] for _ in range(level_count)]
    for index, course in enumerate(courses):
        by_level[course["_level_index"]].append(index)
    weights = [options.level_decay**index for index in range(level_count)]
    students = []
    for index in range(1, options.student_count + 1):
        count = _sample_integer(0, 9, options.student_load_mean, options.student_load_sd, rng)
        base_level = rng.choices(range(1, level_count + 1), weights=weights, k=1)[0]
        selected = set()
        requests = []
        for _ in range(count):
            level = _sample_integer(1, level_count, base_level, options.curricular_sampling_sd, rng)
            pool = [course for course in by_level[level - 1] if course not in selected]
            if not pool:
                found = False
                for distance in range(1, level_count):
                    for neighbor in (level - distance, level + distance):
                        if 1 <= neighbor <= level_count:
                            nearby = [
                                course
                                for course in by_level[neighbor - 1]
                                if course not in selected
                            ]
                            if nearby:
                                pool, found = nearby, True
                                break
                    if found:
                        break
            if not pool:
                pool = [course for course in range(len(courses)) if course not in selected]
            if not pool:
                break
            course_index = rng.choice(pool)
            selected.add(course_index)
            course = courses[course_index]
            requests.append(
                {"codigo": course["codigo"], "paralelo": rng.choice(course["paralelos"])["codigo"]}
            )
        students.append({"id": f"EST{index:04d}", "asignaturas": requests})
    return students
