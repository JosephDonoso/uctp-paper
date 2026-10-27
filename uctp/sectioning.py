"""Enrollment-prioritized backtracking and local refinement for the SSP."""

import random

from .domain import Instance, Solution, collect_section_blocks
from .objective import SearchObjective
from .student_cost import compute_inconvenience_cost


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

    def solve(self, solution: Solution) -> float:
        """Replace Y, keep X fixed, and return the internal search objective."""
        self.solve_calls += 1
        solution.Y = {}
        self._section_blocks = collect_section_blocks(self.instance, solution.X)
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
            self._assign_courses(ordered_courses, [], 0, {})
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
        branch_blocks: list[int],
        branch_conflicts: int,
        branch_sections: dict[int, int],
    ) -> None:
        if not pending_courses:
            self._minimum_conflicts = branch_conflicts
            self._best_sections = branch_sections
            return
        remaining_courses = pending_courses[:]
        course_index = remaining_courses.pop(0)
        sections = sorted(
            range(len(self.instance.courses[course_index].sections)),
            key=lambda section_index: self._enrollments[course_index, section_index],
        )
        for section_index in sections:
            blocks = self._section_blocks[course_index, section_index]
            conflicts = sum(1 for block in blocks if block in branch_blocks)
            # Equality prunes too, retaining the first complete minimum found.
            if branch_conflicts + conflicts >= self._minimum_conflicts:
                continue
            next_sections = branch_sections.copy()
            next_sections[course_index] = section_index
            self._assign_courses(
                remaining_courses,
                branch_blocks + blocks,
                branch_conflicts + conflicts,
                next_sections,
            )

    def _refine_sections(
        self,
        solution: Solution,
        student_index: int,
        courses: list[int],
        occupied_blocks: list[int],
    ) -> None:
        random.shuffle(courses)
        improved = True
        while improved:
            improved = False
            for course_index in courses:
                section_count = len(self.instance.courses[course_index].sections)
                current_section = next(
                    (
                        section_index
                        for section_index in range(section_count)
                        if solution.Y.get((student_index, course_index, section_index), 0) == 1
                    ),
                    None,
                )
                if current_section is None:
                    continue
                current_cost = compute_inconvenience_cost(
                    occupied_blocks, idle_tolerance=self.idle_tolerance
                )
                current_blocks = self._section_blocks[course_index, current_section]
                for next_section in range(section_count):
                    if next_section == current_section:
                        continue
                    next_blocks = self._section_blocks[course_index, next_section]
                    # Remove one occurrence, retaining any overlapping lectures.
                    for block in current_blocks:
                        occupied_blocks.remove(block)
                    if sum(1 for block in next_blocks if block in occupied_blocks) == 0:
                        occupied_blocks.extend(next_blocks)
                        next_cost = compute_inconvenience_cost(
                            occupied_blocks, idle_tolerance=self.idle_tolerance
                        )
                        if next_cost < current_cost:
                            solution.Y.pop((student_index, course_index, current_section), None)
                            solution.Y[student_index, course_index, next_section] = 1
                            self._enrollments[course_index, current_section] -= 1
                            self._enrollments[course_index, next_section] += 1
                            improved = True
                            break
                        for block in next_blocks:
                            occupied_blocks.remove(block)
                        occupied_blocks.extend(current_blocks)
                    else:
                        occupied_blocks.extend(current_blocks)
