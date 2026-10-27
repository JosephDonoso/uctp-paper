"""Ordered problem data and sparse decision variables from Problem Formulation."""

from dataclasses import dataclass, field

type LectureKey = tuple[int, int, int]
type Timetable = dict[tuple[int, int, int, int], int]
type SectionAssignment = dict[tuple[int, int, int], int]


@dataclass(frozen=True)
class Lecture:
    """Weekly lecture; teacher indices refer to the instance teacher order."""

    lecture_id: str
    teacher_indices: tuple[int, ...]
    preassigned_block: int | None = None


@dataclass(frozen=True)
class Section:
    """Parallel section with lectures in source order."""

    section_id: str
    lectures: tuple[Lecture, ...]


@dataclass(frozen=True)
class Course:
    """Course in C; section order determines deterministic tie breaking."""

    course_id: str
    sections: tuple[Section, ...]


@dataclass(frozen=True)
class Teacher:
    """Teacher availability and workload limits, measured in blocks and days."""

    teacher_id: str
    available_blocks: frozenset[int]
    max_days: int
    max_daily_load: int


@dataclass(frozen=True)
class Student:
    """Fixed course requirements in their original enrollment order."""

    student_id: str
    course_indices: tuple[int, ...]


@dataclass(frozen=True)
class CurricularLevel:
    """Curricular level Q, distinct from a course in C."""

    level_id: str
    course_indices: tuple[int, ...]


@dataclass(frozen=True)
class Instance:
    """Institutional data; block indices use day-major order.

    Notes
    -----
    This implementation retains the five-day, seven-block framework of the
    source solver. Protected blocks remain indexed but cannot receive lectures.
    No observed student section assignments are supplied to the SSP.
    """

    courses: tuple[Course, ...]
    teachers: tuple[Teacher, ...]
    students: tuple[Student, ...]
    curricular_levels: tuple[CurricularLevel, ...]
    block_ids: tuple[str, ...]
    protected_blocks: frozenset[int]
    max_block_load: int
    distinct_day_lectures: frozenset[LectureKey] = frozenset()
    separated_day_lectures: frozenset[LectureKey] = frozenset()

    @property
    def blocks_per_day(self) -> int:
        """Return the institutional number of nominal blocks per day."""
        return 7


@dataclass
class Solution:
    """Sparse X and Y; insertion order is part of search reproducibility.

    X uses (course, section, lecture, block) indices and Y uses
    (student, course, section) indices. Search can retain zero-valued X entries.
    """

    X: Timetable = field(default_factory=dict)
    Y: SectionAssignment = field(default_factory=dict)

    def copy(self) -> "Solution":
        """Copy both assignment dictionaries without sharing mutable state."""
        return Solution(self.X.copy(), self.Y.copy())


def collect_section_blocks(instance: Instance, X: Timetable) -> dict[tuple[int, int], list[int]]:
    """Read X in course, section, lecture and block order, preserving duplicates."""
    section_blocks = {}
    for course_index, course in enumerate(instance.courses):
        for section_index, section in enumerate(course.sections):
            blocks = []
            for lecture_index in range(len(section.lectures)):
                for block_index in range(len(instance.block_ids)):
                    if X.get((course_index, section_index, lecture_index, block_index), 0) == 1:
                        blocks.append(block_index)
            section_blocks[course_index, section_index] = blocks
    return section_blocks
