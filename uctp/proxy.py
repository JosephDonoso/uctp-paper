"""Timetable-only proxy objective for the sequential baseline (experiment E3).

The sequential approach optimizes X without solving the SSP and assigns
students once at the end. Its proxy counts expected student conflicts:

    P(X) = sum over course pairs c < c' of n_cc' * ov_cc'(X),

where n_cc' is the number of students requesting both courses and ov_cc'(X)
is the fraction of section pairs (p, p') whose lectures share at least one
block. It uses only X and the course requests, never Y.
"""

import numpy as np

from .domain import Instance, Solution


class ProxyObjective:
    """Evaluate P(X); same interface as SearchObjective.evaluate."""

    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        offsets, running = [], 0
        for course in instance.courses:
            offsets.append(running)
            running += len(course.sections)
        self._offsets = offsets
        self._n_sections = running
        shared: dict[tuple[int, int], int] = {}
        for student in instance.students:
            courses = sorted(set(student.course_indices))
            for i, first in enumerate(courses):
                for second in courses[i + 1 :]:
                    shared[first, second] = shared.get((first, second), 0) + 1
        left, right, weight = [], [], []
        for (first, second), count in shared.items():
            n_first = len(instance.courses[first].sections)
            n_second = len(instance.courses[second].sections)
            for p in range(n_first):
                for q in range(n_second):
                    left.append(offsets[first] + p)
                    right.append(offsets[second] + q)
                    weight.append(count / (n_first * n_second))
        self._left = np.array(left, dtype=np.int64)
        self._right = np.array(right, dtype=np.int64)
        self._weight = np.array(weight, dtype=np.float64)

    def evaluate(self, solution: Solution) -> float:
        masks = np.zeros(self._n_sections, dtype=np.int64)
        offsets = self._offsets
        for (course, section, _lecture, block), active in solution.X.items():
            if active == 1:
                masks[offsets[course] + section] |= np.int64(1) << np.int64(block)
        overlap = (masks[self._left] & masks[self._right]) != 0
        return float(self._weight[overlap].sum())
