"""Observed institutional timetable X_real and enrollment Y_real from the workbook.

`datasets/Real/Semester_2025_1_anonymized.xlsx` keeps the implemented schedule
(sheet Asignaturas, column horario) and the recorded enrollments (sheet
Estudiantes). Lectures follow the order of the schedule entries; when a section
lists more entries than modeled lectures, entries in the same block are
collapsed. Lectures of one section that share a block are collapsed in the
evaluation instance as well, which reproduces the published observed baseline
(R_imb 1.760, R_dis 7.828, 201 conflicts) and the SSP baselines.

Usage: python tools/real_observed.py   (prints the observed metrics)
"""

import re
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uctp.domain import Solution  # noqa: E402
from uctp.instance_io import load_instance  # noqa: E402

_DAYS = {"Lu": "Mon", "Ma": "Tue", "Mi": "Wed", "Ju": "Thu", "Vi": "Fri"}


def _blocks(schedule: str | None) -> list[str]:
    blocks = []
    for entry in (part.strip() for part in (schedule or "").split(",")):
        match = re.match(r"(Lu|Ma|Mi|Ju|Vi) (.*) (\S+)$", entry)
        if not match:
            continue  # room labels and empty entries
        day, hours, _kind = match.groups()
        for first, _last in re.findall(r"(\d+) - (\d+)", hours):
            blocks.append(f"{_DAYS[day]}:{(int(first) + 1) // 2}")
    return blocks


def observed():
    """Return (instance, X_real, Y_real) on the evaluation instance (collapsed lectures)."""
    import openpyxl

    instance = load_instance(
        ROOT / "datasets" / "Real" / "real.json",
        input_format="instance", max_daily_load=4, max_block_load=15,
    )
    workbook = openpyxl.load_workbook(
        ROOT / "datasets" / "Real" / "Semester_2025_1_anonymized.xlsx", read_only=True
    )
    rows = list(workbook["Asignaturas"].iter_rows(values_only=True))[1:]
    schedule = {row[0]: _blocks(row[5]) for row in rows}
    block_index = {block: index for index, block in enumerate(instance.block_ids)}
    courses, X = [], {}
    for c, course in enumerate(instance.courses):
        sections = []
        for s, section in enumerate(course.sections):
            blocks = schedule[f"{course.course_id}-{section.section_id}"]
            if len(blocks) > len(section.lectures):
                blocks = list(dict.fromkeys(blocks))
            if len(blocks) != len(section.lectures):
                raise ValueError(f"Schedule of {course.course_id}-{section.section_id} does not match.")
            kept, seen = [], set()
            for lecture, block in zip(section.lectures, blocks):
                b = block_index[block]
                if lecture.preassigned_block is not None and lecture.preassigned_block != b:
                    raise ValueError("Preassigned block differs from the observed schedule.")
                if b in seen:
                    continue
                seen.add(b)
                X[(c, s, len(kept), b)] = 1
                kept.append(lecture)
            sections.append(replace(section, lectures=tuple(kept)))
        courses.append(replace(course, sections=tuple(sections)))
    keys = {key[:3] for key in X}
    instance = replace(
        instance,
        courses=tuple(courses),
        distinct_day_lectures=frozenset(k for k in instance.distinct_day_lectures if k in keys),
        separated_day_lectures=frozenset(k for k in instance.separated_day_lectures if k in keys),
    )
    sections = {
        (course.course_id, section.section_id): (c, s)
        for c, course in enumerate(instance.courses)
        for s, section in enumerate(course.sections)
    }
    students = {student.student_id: index for index, student in enumerate(instance.students)}
    Y = {}
    for student_id, enrolled in list(workbook["Estudiantes"].iter_rows(values_only=True))[1:]:
        if student_id not in students:
            continue
        index = students[student_id]
        for item in (part.strip() for part in (enrolled or "").split(",")):
            if not item:
                continue
            course_id, section_id = item.split("-")
            if (course_id, section_id) in sections:
                c, s = sections[course_id, section_id]
                if c in instance.students[index].course_indices:
                    Y[(index, c, s)] = 1
    return instance, X, Y


if __name__ == "__main__":
    from uctp.metrics import measure_solution

    instance, X, Y = observed()
    metrics = measure_solution(instance, Solution(X, Y), alpha=1)["metrics"]
    print({key: round(metrics[key], 3) for key in ("R_imb", "R_dis", "conflicts", "F_ref")})
