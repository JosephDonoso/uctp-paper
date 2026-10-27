"""Internal objective evaluation for candidate selection; no reporting API."""

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

    def evaluate(self, solution: Solution) -> float:
        """Return the scalar required for search, leaving X and Y unchanged."""
        section_blocks = collect_section_blocks(self.instance, solution.X)
        enrollments = {
            (course_index, section_index): 0
            for course_index, course in enumerate(self.instance.courses)
            for section_index in range(len(course.sections))
        }
        student_blocks = {}
        for student_index, student in enumerate(self.instance.students):
            blocks = []
            for course_index in student.course_indices:
                for section_index in range(len(self.instance.courses[course_index].sections)):
                    if solution.Y.get((student_index, course_index, section_index), 0) == 1:
                        blocks.extend(section_blocks[course_index, section_index])
                        enrollments[course_index, section_index] += 1
            student_blocks[student_index] = blocks
        return self.evaluate_assignment(enrollments, student_blocks)

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
        dissatisfaction = sum(
            compute_inconvenience_cost(
                student_blocks[student_index], idle_tolerance=self.idle_tolerance
            )
            for student_index in range(len(self.instance.students))
        )
        active_students = sum(1 for student in self.instance.students if student.course_indices)
        reference_dissatisfaction = active_students * self.idle_tolerance
        imbalance_ratio = observed_imbalance / max(reference_imbalance, 1e-6)
        dissatisfaction_ratio = dissatisfaction / max(reference_dissatisfaction, 1e-6)
        return (imbalance_ratio**self.alpha) * dissatisfaction_ratio
