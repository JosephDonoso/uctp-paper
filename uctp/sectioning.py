"""Enrollment-prioritized backtracking and local refinement for the SSP."""

import os
import random

import numpy as np

from .domain import Instance, Solution, collect_section_blocks
from .objective import SearchObjective
from .student_cost import compute_inconvenience_cost

try:  # Optional compiled kernel; results are identical to the Python path.
    from . import _ssp_kernel
except ImportError:  # pragma: no cover - depends on the environment
    _ssp_kernel = None


def kernel_enabled() -> bool:
    """Use the Numba kernel unless unavailable or UCTP_PURE_PYTHON=1."""
    return _ssp_kernel is not None and os.environ.get("UCTP_PURE_PYTHON", "") != "1"


class StudentSectioning:
    """Resolve Y for a fixed X with randomized student processing.

    Notes
    -----
    Each solve visits students once in a newly shuffled order, controlled by
    Python's random generator. Courses are stably sorted by section count;
    branches are stably sorted by current enrollment. Local Search shuffles
    each student's original course list using Python's random generator.
    """

    def __init__(
        self, instance: Instance, objective: SearchObjective, *, local_search: bool = True
    ) -> None:
        self.instance = instance
        self.objective = objective
        self.idle_tolerance = objective.idle_tolerance
        self.local_search = local_search
        self.solve_calls = 0
        self._multi_section_courses = frozenset(
            index for index, course in enumerate(instance.courses) if len(course.sections) > 1
        )
        self._prepare_kernel_data()
        if kernel_enabled():
            _ssp_kernel.warm_up()  # before the optimizer clock starts

    def _prepare_kernel_data(self) -> None:
        instance = self.instance
        n_sections = [len(course.sections) for course in instance.courses]
        self._k_n_sections = np.array(n_sections, dtype=np.int64)
        self._k_sec_off = np.concatenate(([0], np.cumsum(n_sections)[:-1])).astype(np.int64)
        self._k_n_sec_total = int(sum(n_sections))
        pointers, original, ordered = [0], [], []
        for student in instance.students:
            courses = list(student.course_indices)
            original.extend(courses)
            ordered.extend(sorted(courses, key=lambda c: n_sections[c]))
            pointers.append(len(original))
        self._k_stu_ptr = np.array(pointers, dtype=np.int64)
        self._k_stu_courses = np.array(original, dtype=np.int64)
        self._k_stu_sorted = np.array(ordered, dtype=np.int64)
        self._k_slot_student = np.repeat(
            np.arange(len(instance.students)), np.diff(self._k_stu_ptr)
        )
        self._k_n_blocks = len(instance.block_ids)
        if self._k_n_blocks > 62:
            raise ValueError("The SSP kernel supports at most 62 blocks.")

    def _solve_kernel(self, solution: Solution) -> float:
        section_blocks = collect_section_blocks(self.instance, solution.X)
        total = self._k_n_sec_total
        sb_ptr, flat, masks, distinct = _ssp_kernel.section_arrays(self.instance, section_blocks)
        version, internal, gauss = random.getstate()
        mt = np.fromiter(internal, dtype=np.int64, count=624)
        pos = np.array([internal[-1]], dtype=np.int64)
        n_students = len(self.instance.students)
        assigned = np.empty(len(self._k_stu_courses), dtype=np.int64)
        order = np.empty(n_students, dtype=np.int64)
        enrollments = np.zeros(total, dtype=np.int64)
        costs = np.zeros(n_students, dtype=np.float64)
        _ssp_kernel.solve_ssp(
            mt, pos, n_students,
            self._k_stu_ptr, self._k_stu_courses, self._k_stu_sorted,
            self._k_n_sections, self._k_sec_off,
            sb_ptr, flat, masks, distinct,
            self._k_n_blocks, float(max(0, float(self.idle_tolerance))), self.local_search,
            assigned, order, enrollments, costs,
        )
        random.setstate((version, (*mt.tolist(), int(pos[0])), gauss))
        solution.Y = dict.fromkeys(
            zip(
                self._k_slot_student.tolist(),
                self._k_stu_courses.tolist(),
                assigned.tolist(),
            ),
            1,
        )
        enrollment_map = {}
        sid = 0
        for course_index, course in enumerate(self.instance.courses):
            for section_index in range(len(course.sections)):
                enrollment_map[course_index, section_index] = int(enrollments[sid])
                sid += 1
        return self.objective.evaluate_from_costs(enrollment_map, float(costs.sum()))

    def solve(self, solution: Solution) -> float:
        """Replace Y, keep X fixed, and return the internal search objective."""
        self.solve_calls += 1
        if kernel_enabled():
            return self._solve_kernel(solution)
        solution.Y = {}
        self._section_blocks = collect_section_blocks(self.instance, solution.X)
        self._section_masks = {}
        for key, blocks in self._section_blocks.items():
            mask = 0
            for block in blocks:
                mask |= 1 << block
            self._section_masks[key] = (mask, blocks, mask.bit_count() == len(blocks))
        self._enrollments = {
            (course_index, section_index): 0
            for course_index, course in enumerate(self.instance.courses)
            for section_index in range(len(course.sections))
        }
        student_blocks = {index: [] for index in range(len(self.instance.students))}
        student_order = list(range(len(self.instance.students)))
        random.shuffle(student_order)
        for student_index in student_order:
            student = self.instance.students[student_index]
            courses = list(student.course_indices)
            if not courses:
                continue
            ordered_courses = sorted(
                courses, key=lambda course_index: len(self.instance.courses[course_index].sections)
            )
            self._minimum_conflicts = float("inf")
            self._best_sections: dict[int, int] = {}
            self._assign_courses(ordered_courses, 0, 0, {})
            occupied_blocks = []
            for course_index, section_index in self._best_sections.items():
                solution.Y[student_index, course_index, section_index] = 1
                self._enrollments[course_index, section_index] += 1
                occupied_blocks.extend(self._section_blocks[course_index, section_index])
            if self.local_search:
                self._refine_sections(solution, student_index, courses[:], occupied_blocks)
            student_blocks[student_index] = occupied_blocks
        return self.objective.evaluate_assignment(self._enrollments, student_blocks)

    def _assign_courses(
        self,
        pending_courses: list[int],
        branch_mask: int,
        branch_conflicts: int,
        branch_sections: dict[int, int],
        depth: int = 0,
    ) -> None:
        # Same search as the published version; branch membership is kept as a
        # bitmask over blocks, which gives identical conflict counts.
        if depth == len(pending_courses):
            self._minimum_conflicts = branch_conflicts
            self._best_sections = branch_sections
            return
        course_index = pending_courses[depth]
        enrollments = self._enrollments
        sections = sorted(
            range(len(self.instance.courses[course_index].sections)),
            key=lambda section_index: enrollments[course_index, section_index],
        )
        for section_index in sections:
            mask, blocks, distinct = self._section_masks[course_index, section_index]
            if distinct:
                conflicts = (mask & branch_mask).bit_count()
            else:
                conflicts = sum(1 for block in blocks if branch_mask >> block & 1)
            # Equality prunes too, retaining the first complete minimum found.
            if branch_conflicts + conflicts >= self._minimum_conflicts:
                continue
            next_sections = branch_sections.copy()
            next_sections[course_index] = section_index
            self._assign_courses(
                pending_courses,
                branch_mask | mask,
                branch_conflicts + conflicts,
                next_sections,
                depth + 1,
            )

    def _refine_sections(
        self,
        solution: Solution,
        student_index: int,
        courses: list[int],
        occupied_blocks: list[int],
    ) -> None:
        # Same moves, order and acceptance rule as the published version; the
        # current cost is carried instead of recomputed (it is deterministic).
        random.shuffle(courses)
        multi = self._multi_section_courses
        if not any(course_index in multi for course_index in courses):
            return
        section_blocks = self._section_blocks
        tolerance = float(self.idle_tolerance)
        Y = solution.Y
        enrollments = self._enrollments
        assigned = dict(self._best_sections)
        current_cost = compute_inconvenience_cost(occupied_blocks, idle_tolerance=tolerance)
        improved = True
        while improved:
            improved = False
            for course_index in courses:
                if course_index not in multi:
                    continue
                current_section = assigned.get(course_index)
                if current_section is None:
                    continue
                current_blocks = section_blocks[course_index, current_section]
                for next_section in range(len(self.instance.courses[course_index].sections)):
                    if next_section == current_section:
                        continue
                    next_blocks = section_blocks[course_index, next_section]
                    for block in current_blocks:
                        occupied_blocks.remove(block)
                    if not any(block in occupied_blocks for block in next_blocks):
                        occupied_blocks.extend(next_blocks)
                        next_cost = compute_inconvenience_cost(
                            occupied_blocks, idle_tolerance=tolerance
                        )
                        if next_cost < current_cost:
                            Y.pop((student_index, course_index, current_section), None)
                            Y[student_index, course_index, next_section] = 1
                            enrollments[course_index, current_section] -= 1
                            enrollments[course_index, next_section] += 1
                            assigned[course_index] = next_section
                            current_cost = next_cost
                            improved = True
                            break
                        for block in next_blocks:
                            occupied_blocks.remove(block)
                        occupied_blocks.extend(current_blocks)
                    else:
                        occupied_blocks.extend(current_blocks)
