"""Timetable-only proxy objective for the sequential baseline (experiment E3).

The sequential approach optimizes X without solving the SSP and assigns
students once at the end. Its proxy is the expected number of student
conflicts when every student takes each requested course in a section chosen
uniformly at random:

    P(X) = sum_{c<c'} n_cc' / (|P_c||P_c'|) * sum_{p,q} |B_p & B_q|
         + sum_c n_c / |P_c| * sum_p (lectures of p - distinct blocks of p),

where n_cc' is the number of students requesting both courses, n_c those
requesting c, and B_p the set of blocks of section p. The second term counts
lectures of one section placed in the same block (possible for lectures
without teachers); without it, stacking a section's lectures would lower the
first term while creating conflicts for all of its students. It uses only X
and the course requests, never Y.
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
        demand = [0] * len(instance.courses)
        for student in instance.students:
            courses = sorted(set(student.course_indices))
            for i, first in enumerate(courses):
                demand[first] += 1
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
        self._section_weight = np.array(
            [demand[c] / len(course.sections) for c, course in enumerate(instance.courses)
             for _ in course.sections],
            dtype=np.float64,
        )
        self._lectures = np.array(
            [len(section.lectures) for course in instance.courses for section in course.sections],
            dtype=np.int64,
        )

    def evaluate(self, solution: Solution) -> float:
        masks = np.zeros(self._n_sections, dtype=np.int64)
        offsets = self._offsets
        for (course, section, _lecture, block), active in solution.X.items():
            if active == 1:
                masks[offsets[course] + section] |= np.int64(1) << np.int64(block)
        across = np.bitwise_count(masks[self._left] & masks[self._right]).astype(np.float64)
        within = (self._lectures - np.bitwise_count(masks)).astype(np.float64)
        return float(self._weight @ across + self._section_weight @ within)
