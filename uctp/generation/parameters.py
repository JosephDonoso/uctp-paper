"""Explicit generator parameters and the five manuscript dataset groups."""

import math
from dataclasses import dataclass
from enum import StrEnum


class DatasetGroup(StrEnum):
    """Unique dataset groups, with BASE shared by both experimental families."""

    DISPERSION_LOW = "Dispersion-Low"
    BASE = "BASE"
    DISPERSION_HIGH = "Dispersion-High"
    OFFER_LOW = "Offer-Low"
    OFFER_HIGH = "Offer-High"


# (section-count normal location, curricular sampling SD, group index).
# BASE is calibrated to the institutional instance: its location gives an
# expected 1.486 sections per course (section SD 0.8) and its curricular SD
# reproduces the mean within-student SD of requested levels (0.187). Offer-Low
# and Offer-High give 20 % fewer and more expected sections per course than
# BASE; Dispersion-Low and Dispersion-High use half and 1.5 times its SD.
GROUP_PARAMETERS = {
    DatasetGroup.DISPERSION_LOW: (1.17, 0.155, 0),
    DatasetGroup.BASE: (1.17, 0.31, 1),
    DatasetGroup.DISPERSION_HIGH: (1.17, 0.465, 2),
    DatasetGroup.OFFER_LOW: (0.36, 0.31, 3),
    DatasetGroup.OFFER_HIGH: (1.64, 0.31, 5),
}


@dataclass(frozen=True)
class GenerationOptions:
    """Inputs for one synthetic candidate, supplied through code or CLI.

    Notes
    -----
    Availability probability applies to available blocks. Preassignment
    probability selects a whole course for a feasible placement attempt.
    Section, lecture and student counts use independent rounded, truncated
    normals. Teachers are assigned per section, with target weekly loads drawn
    from a rounded, truncated normal; the number of teachers follows from the
    teaching demand. Distribution parameters precede rounding and truncation.
    """

    courses_per_level: tuple[int, ...] = (4, 6, 6, 6, 6, 2)
    teacher_load_mean: float = 2.0
    teacher_load_sd: float = 3.25
    teacher_max_load: int = 12
    co_teaching_probability: float = 0.07
    shared_teacher_probability: float = 0.5
    max_daily_load: int = 4
    student_count: int = 500
    section_mean: float = 1.17
    section_sd: float = 0.8
    weekly_lectures_mean: float = 2.59
    weekly_lectures_sd: float = 0.7
    curricular_sampling_sd: float = 0.31
    level_decay: float = 0.85
    availability_probability: float = 0.66
    preassignment_probability: float = 0.10
    student_load_mean: float = 3.12
    student_load_sd: float = 1.6

    def validate(self) -> None:
        """Reject invalid counts and sampling parameters before generation.

        Raises
        ------
        ValueError
            A count is invalid, or a
            distribution parameter is invalid.
        """
        if not isinstance(self.courses_per_level, tuple) or not self.courses_per_level:
            raise ValueError("courses_per_level must be a nonempty tuple.")
        if any(type(n) is not int or n < 0 for n in self.courses_per_level):
            raise ValueError("Course counts must be nonnegative integers.")
        courses = sum(self.courses_per_level)
        if not math.isfinite(self.student_load_mean) or not 0 <= self.student_load_mean <= 9:
            raise ValueError("student_load_mean must be finite and in [0, 9].")
        if not math.isfinite(self.student_load_sd) or self.student_load_sd < 0:
            raise ValueError("student_load_sd must be finite and nonnegative.")
        if courses < 9:
            raise ValueError("At least nine courses are required by the request distribution.")
        if type(self.teacher_max_load) is not int or not 4 <= self.teacher_max_load <= 34:
            raise ValueError("teacher_max_load must be an integer in [4, 34].")
        if not math.isfinite(self.teacher_load_mean) or not 2 <= self.teacher_load_mean <= self.teacher_max_load:
            raise ValueError("teacher_load_mean must be finite and in [2, teacher_max_load].")
        if not math.isfinite(self.teacher_load_sd) or self.teacher_load_sd < 0:
            raise ValueError("teacher_load_sd must be finite and nonnegative.")
        if type(self.max_daily_load) is not int or self.max_daily_load < 1:
            raise ValueError("max_daily_load must be a positive integer.")
        if type(self.student_count) is not int or self.student_count < 1:
            raise ValueError("student_count must be a positive integer.")
        if not math.isfinite(self.section_mean) or not 0 <= self.section_mean <= 4:
            raise ValueError("section_mean must be finite and between zero and four.")
        if not math.isfinite(self.section_sd) or self.section_sd < 0:
            raise ValueError("section_sd must be finite and nonnegative.")
        if not math.isfinite(self.weekly_lectures_mean) or not 2 <= self.weekly_lectures_mean <= 4:
            raise ValueError("weekly_lectures_mean must be finite and in [2, 4].")
        if (
            not math.isfinite(self.weekly_lectures_sd) or self.weekly_lectures_sd < 0
        ):
            raise ValueError("weekly_lectures_sd must be finite and nonnegative.")
        if not math.isfinite(self.curricular_sampling_sd) or self.curricular_sampling_sd <= 0:
            raise ValueError("curricular_sampling_sd must be finite and positive.")
        if not math.isfinite(self.level_decay) or not 0 < self.level_decay <= 1:
            raise ValueError("level_decay must be in (0, 1].")
        for name in (
            "availability_probability",
            "preassignment_probability",
            "co_teaching_probability",
            "shared_teacher_probability",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")
