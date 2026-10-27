"""Random, Prioritization Only and Compatibility Matrix initial timetabling."""

import random
from enum import StrEnum

from .affinity import LevelAffinity
from .compatibility import Matrix, build_compatibility_matrix
from .constraints import InstitutionalConstraints
from .domain import Instance, LectureKey, Solution


class InitialStrategy(StrEnum):
    """Public strategy labels from Experimental Setup."""

    RANDOM = "Random"
    PRIORITIZATION_ONLY = "Prioritization Only"
    COMPATIBILITY_MATRIX = "Compatibility Matrix"


class ConstructionFailure(RuntimeError):
    """No complete feasible timetable was built within the retry budget."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.diagnostics: dict = {}


class _RestartConstruction(Exception):
    pass


class InitialTimetabling:
    """Construct X with ordered feasibility checks and stochastic retries."""

    def __init__(
        self,
        instance: Instance,
        constraints: InstitutionalConstraints,
        *,
        strategy: InitialStrategy,
        max_retries: int,
        beta: float,
        cv_tolerance: float,
    ) -> None:
        self.instance = instance
        self.constraints = constraints
        self.strategy = strategy
        self.max_retries = max_retries
        self.cv_tolerance = cv_tolerance
        self._affinity = LevelAffinity(instance, beta=beta)

    def solve(self) -> Solution:
        """Return a complete institutional timetable, with Y initially empty.

        Raises
        ------
        ConstructionFailure
            Preassignments violate constraints or all L+1 attempts fail.

        Notes
        -----
        Retries continue the seeded random stream without reseeding it. Fixed
        lectures are preserved individually, including partially fixed courses.
        """
        self._retry_count = 0
        while True:
            solution = Solution(self.constraints.preassignments.copy())
            if not self.constraints.is_feasible(solution.X):
                raise ConstructionFailure("Preassignments violate institutional constraints.")
            self._scheduled_courses = {}
            self._distinct_days: dict[tuple[int, int], set[int]] = {}
            self._separated_days: dict[tuple[int, int], set[int]] = {}
            for course_index, section_index, lecture_index, block_index in solution.X:
                self._record_lecture_day((course_index, section_index, lecture_index), block_index)
            try:
                if self.strategy == InitialStrategy.RANDOM:
                    self._schedule_randomly(solution)
                else:
                    for group in self._affinity.ordered_groups():
                        courses = []
                        for level_index in group:
                            courses.extend(
                                self.instance.curricular_levels[level_index].course_indices
                            )
                        matrices, ordered_courses = self._prepare_preferences(courses)
                        self._schedule_ordered_courses(solution, matrices, ordered_courses)
                    remaining = [
                        index
                        for index in range(len(self.instance.courses))
                        if index not in self._scheduled_courses
                    ]
                    if remaining:
                        matrices, ordered_courses = self._prepare_preferences(remaining)
                        self._schedule_ordered_courses(solution, matrices, ordered_courses)
                if not self.constraints.is_complete(solution.X) or not self.constraints.is_feasible(
                    solution.X
                ):
                    raise ConstructionFailure(
                        "Construction did not produce a complete feasible timetable."
                    )
                return solution
            except _RestartConstruction:
                continue

    def _restart(self) -> None:
        if self._retry_count >= self.max_retries:
            raise ConstructionFailure("Initial timetabling exhausted its retry budget.")
        self._retry_count += 1
        raise _RestartConstruction

    def _is_fixed(self, lecture_key: LectureKey) -> bool:
        course_index, section_index, lecture_index = lecture_key
        return (
            self.instance.courses[course_index]
            .sections[section_index]
            .lectures[lecture_index]
            .preassigned_block
            is not None
        )

    def _violates_lecture_days(self, lecture_key: LectureKey, block_index: int) -> bool:
        day_index = block_index // self.instance.blocks_per_day
        section_key = lecture_key[:2]
        if lecture_key in self.instance.distinct_day_lectures:
            if day_index in self._distinct_days.get(section_key, set()):
                return True
        if lecture_key in self.instance.separated_day_lectures:
            for previous_day in self._separated_days.get(section_key, set()):
                if abs(day_index - previous_day) <= 1:
                    return True
        return False

    def _record_lecture_day(self, lecture_key: LectureKey, block_index: int) -> None:
        day_index = block_index // self.instance.blocks_per_day
        section_key = lecture_key[:2]
        if lecture_key in self.instance.distinct_day_lectures:
            self._distinct_days.setdefault(section_key, set()).add(day_index)
        if lecture_key in self.instance.separated_day_lectures:
            self._separated_days.setdefault(section_key, set()).add(day_index)

    def _try_block(self, solution: Solution, lecture_key: LectureKey, block_index: int) -> bool:
        if block_index in self.instance.protected_blocks:
            return False
        if self._violates_lecture_days(lecture_key, block_index):
            return False
        key = (*lecture_key, block_index)
        solution.X[key] = 1
        if self.constraints.allows_partial_timetable(solution.X):
            self._record_lecture_day(lecture_key, block_index)
            return True
        solution.X.pop(key, None)
        return False

    def _schedule_randomly(self, solution: Solution) -> None:
        courses = list(range(len(self.instance.courses)))
        random.shuffle(courses)
        for course_index in courses:
            sections = list(range(len(self.instance.courses[course_index].sections)))
            random.shuffle(sections)
            for section_index in sections:
                lectures = list(
                    range(len(self.instance.courses[course_index].sections[section_index].lectures))
                )
                random.shuffle(lectures)
                for lecture_index in lectures:
                    lecture_key = course_index, section_index, lecture_index
                    if self._is_fixed(lecture_key):
                        continue
                    blocks = list(range(len(self.instance.block_ids)))
                    random.shuffle(blocks)
                    for block_index in blocks:
                        if self._try_block(solution, lecture_key, block_index):
                            break
                    else:
                        self._restart()

    def _prepare_preferences(
        self, courses: list[int]
    ) -> tuple[dict[tuple[int, int], Matrix], list[int]]:
        ordered_courses = courses.copy()
        random.shuffle(ordered_courses)
        ordered_courses.sort(
            key=lambda index: len(self.instance.courses[index].sections), reverse=True
        )
        matrices = {}
        for first_index in range(len(ordered_courses)):
            for second_index in range(first_index + 1, len(ordered_courses)):
                first_course = ordered_courses[first_index]
                second_course = ordered_courses[second_index]
                first_count = len(self.instance.courses[first_course].sections)
                second_count = len(self.instance.courses[second_course].sections)
                if self.strategy == InitialStrategy.COMPATIBILITY_MATRIX:
                    first_students = second_students = shared_students = 0
                    for student in self.instance.students:
                        takes_first = first_course in student.course_indices
                        takes_second = second_course in student.course_indices
                        if takes_first and takes_second:
                            shared_students += 1
                        if takes_first:
                            first_students += 1
                        if takes_second:
                            second_students += 1
                    matrix = build_compatibility_matrix(
                        first_count,
                        first_students,
                        second_count,
                        second_students,
                        shared_students,
                        cv_tolerance=self.cv_tolerance,
                    )
                else:
                    matrix = [[0 for _ in range(second_count)] for _ in range(first_count)]
                matrices[first_course, second_course] = matrix
        return matrices, ordered_courses

    def _schedule_ordered_courses(
        self, solution: Solution, matrices: dict[tuple[int, int], Matrix], courses: list[int]
    ) -> None:
        for index, course_index in enumerate(courses):
            if course_index in self._scheduled_courses:
                continue
            self._scheduled_courses[course_index] = True
            previous_courses = [courses[previous] for previous in range(index - 1, -1, -1)]
            for section_index, section in enumerate(self.instance.courses[course_index].sections):
                avoid_lectures = []
                prefer_lectures = []
                for previous_course in previous_courses:
                    matrix = matrices[previous_course, course_index]
                    for previous_section, previous in enumerate(
                        self.instance.courses[previous_course].sections
                    ):
                        relation = matrix[previous_section][section_index]
                        for previous_lecture in range(len(previous.lectures)):
                            key = previous_course, previous_section, previous_lecture
                            if relation == 1:
                                avoid_lectures.append(key)
                            elif relation == -1:
                                prefer_lectures.append(key)
                for lecture_index in range(len(section.lectures)):
                    lecture_key = course_index, section_index, lecture_index
                    if self._is_fixed(lecture_key):
                        continue
                    forbidden = self._find_blocks(solution, avoid_lectures)
                    preferred = self._find_blocks(solution, prefer_lectures)
                    self._schedule_with_preferences(solution, lecture_key, forbidden, preferred)

    def _find_blocks(self, solution: Solution, lectures: list[LectureKey]) -> list[int]:
        blocks = []
        for lecture_key in lectures:
            for block_index in range(len(self.instance.block_ids)):
                if solution.X.get((*lecture_key, block_index), 0) == 1:
                    blocks.append(block_index)
        return blocks

    def _schedule_with_preferences(
        self,
        solution: Solution,
        lecture_key: LectureKey,
        forbidden: list[int],
        preferred: list[int],
    ) -> None:
        preferred = list(preferred)
        random.shuffle(preferred)
        enforce_preferences = True
        reviewed = [0 for _ in self.instance.block_ids]
        preferred_index = 0
        while True:
            block_index = None
            while preferred_index < len(preferred):
                candidate = preferred[preferred_index]
                preferred_index += 1
                if 0 <= candidate < len(reviewed) and reviewed[candidate] == 0:
                    block_index = candidate
                    break
            if block_index is None:
                unreviewed = [index for index, visited in enumerate(reviewed) if visited == 0]
                if unreviewed:
                    block_index = random.choice(unreviewed)
                elif enforce_preferences:
                    enforce_preferences = False
                    reviewed = [0 for _ in self.instance.block_ids]
                    continue
                else:
                    self._restart()
            reviewed[block_index] = 1
            if enforce_preferences and block_index in forbidden:
                continue
            if self._try_block(solution, lecture_key, block_index):
                return
