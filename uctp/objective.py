"""Internal objective evaluation for candidate selection; no reporting API."""

import os

import numpy as np

from .domain import Instance, Solution, collect_section_blocks
from .student_cost import compute_inconvenience_cost


class SearchObjective:
    """Evaluate eq:objetivo without accumulating or exporting measurements."""

    def __init__(
        self, instance: Instance, *, alpha: float, cv_tolerance: float, idle_tolerance: float
    ) -> None:
        self.instance = instance
        self.alpha = alpha
        self.cv_tolerance = cv_tolerance
        self.idle_tolerance = idle_tolerance
        offsets, running = [], 0
        for course in instance.courses:
            offsets.append(running)
            running += len(course.sections)
        self._section_offsets = offsets
        self._requested = frozenset(
            (student_index, course_index)
            for student_index, student in enumerate(instance.students)
            for course_index in set(student.course_indices)
        )

    def evaluate(self, solution: Solution) -> float:
        """Return the scalar required for search, leaving X and Y unchanged."""
        kernel = _kernel()
        if kernel is not None:
            return self._evaluate_kernel(kernel, solution)
        section_blocks = collect_section_blocks(self.instance, solution.X)
        enrollments = {
            (course_index, section_index): 0
            for course_index, course in enumerate(self.instance.courses)
            for section_index in range(len(course.sections))
        }
        requested = self._requested
        student_blocks = {index: [] for index in range(len(self.instance.students))}
        for (student_index, course_index, section_index), value in solution.Y.items():
            if value == 1 and (student_index, course_index) in requested:
                student_blocks[student_index].extend(section_blocks[course_index, section_index])
                enrollments[course_index, section_index] += 1
        return self.evaluate_assignment(enrollments, student_blocks)

    def _evaluate_kernel(self, kernel, solution: Solution) -> float:
        # Same value as the Python path: identical enrollments and the same
        # integer-valued individual costs (their sum is exact in any order).
        Y = solution.Y
        cached = getattr(self, "_y_cache", None)
        if cached is None or cached[0] is not Y or cached[1] != Y:
            requested = self._requested
            offsets = self._section_offsets
            entries = sorted(
                (student_index, offsets[course_index] + section_index)
                for (student_index, course_index, section_index), value in Y.items()
                if value == 1 and (student_index, course_index) in requested
            )
            students = np.array([e[0] for e in entries], dtype=np.int64)
            sids = np.array([e[1] for e in entries], dtype=np.int64)
            cached = (Y, dict(Y), students, sids)
            self._y_cache = cached
        section_blocks = collect_section_blocks(self.instance, solution.X)
        sb_ptr, flat, _, _ = kernel.section_arrays(self.instance, section_blocks)
        n_students = len(self.instance.students)
        enrollments = np.zeros(len(sb_ptr) - 1, dtype=np.int64)
        costs = np.zeros(n_students, dtype=np.float64)
        kernel.assignment_costs(
            n_students, cached[2], cached[3], sb_ptr, flat, len(self.instance.block_ids),
            float(max(0, float(self.idle_tolerance))), enrollments, costs,
        )
        enrollment_map = {}
        sid = 0
        for course_index, course in enumerate(self.instance.courses):
            for section_index in range(len(course.sections)):
                enrollment_map[course_index, section_index] = int(enrollments[sid])
                sid += 1
        return self.evaluate_from_costs(enrollment_map, float(costs.sum()))

    def evaluate_assignment(
        self, enrollments: dict[tuple[int, int], int], student_blocks: dict[int, list[int]]
    ) -> float:
        """Evaluate SSP state without rebuilding its just-computed assignment.

        Notes
        -----
        Both aggregate imbalance terms average over C*, as in the manuscript.
        When C* is empty, both terms are zero. This method is consumed only by
        search components, not exposed as a CLI or result metric.
        """
        dissatisfaction = sum(
            compute_inconvenience_cost(
                student_blocks[student_index], idle_tolerance=self.idle_tolerance
            )
            for student_index in range(len(self.instance.students))
        )
        return self.evaluate_from_costs(enrollments, dissatisfaction)

    def evaluate_from_costs(
        self, enrollments: dict[tuple[int, int], int], dissatisfaction: float
    ) -> float:
        """Same objective, given the summed individual costs (integer-valued)."""
        observed_sum = 0
        reference_sum = 0
        parallel_course_count = 0
        for course_index, course in enumerate(self.instance.courses):
            section_count = len(course.sections)
            counts = [
                enrollments[course_index, section_index] for section_index in range(section_count)
            ]
            if section_count > 1:
                mean_enrollment = sum(counts) / section_count
                variance = sum((count - mean_enrollment) ** 2 for count in counts) / section_count
                observed_sum += variance**0.5
                reference_sum += mean_enrollment * self.cv_tolerance
                parallel_course_count += 1
        observed_imbalance = observed_sum / max(parallel_course_count, 1)
        reference_imbalance = reference_sum / max(parallel_course_count, 1)
        active_students = sum(1 for student in self.instance.students if student.course_indices)
        reference_dissatisfaction = active_students * self.idle_tolerance
        imbalance_ratio = observed_imbalance / max(reference_imbalance, 1e-6)
        dissatisfaction_ratio = dissatisfaction / max(reference_dissatisfaction, 1e-6)
        # Kept for optional search traces; never read by the search itself.
        self.last_components = (imbalance_ratio, dissatisfaction_ratio)
        return (imbalance_ratio**self.alpha) * dissatisfaction_ratio


def _kernel():
    if os.environ.get("UCTP_PURE_PYTHON", "") == "1":
        return None
    try:
        from . import _ssp_kernel
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    return _ssp_kernel
