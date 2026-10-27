"""Read ordered instance data and write solutions without evaluation reports."""

import json
from pathlib import Path

from .domain import Course, CurricularLevel, Instance, Lecture, Section, Solution, Student, Teacher

BLOCK_IDS = tuple(
    f"{day}:{position}" for day in ("Mon", "Tue", "Wed", "Thu", "Fri") for position in range(1, 8)
)


def _unique_ids(records: list[dict], label: str) -> dict[str, int]:
    identifiers = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Each {label} must be an object.")
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Each {label} must have a nonempty string id.")
        if identifier in identifiers:
            raise ValueError(f"Duplicate {label} id: {identifier}")
        identifiers[identifier] = index
    return identifiers


def _check_fields(record: dict, allowed: set[str], label: str) -> None:
    if set(record) - allowed:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(set(record) - allowed))}")


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _resolve_ids(values: list[str], indices: dict[str, int], label: str) -> tuple[int, ...]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"{label} must be a list of string identifiers.")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate reference in {label}.")
    try:
        return tuple(indices[value] for value in values)
    except KeyError as error:
        raise ValueError(f"Unknown identifier in {label}: {error.args[0]}") from error


def load_instance(
    path: Path, *, input_format: str, max_daily_load: int | None, max_block_load: int
) -> Instance:
    """Read instance JSON, with operational limits supplied by the CLI.

    Parameters
    ----------
    input_format
        'instance' for the English schema, or 'legacy' for historical data.
    max_daily_load
        Explicit teacher-load override in blocks per day. None reads legacy
        MAX_CARGA_DIARIA; missing limits default to eight blocks, not hours.
    max_block_load
        Institutional simultaneous-lecture capacity.

    Raises
    ------
    ValueError
        The instance is malformed or references unknown entities.
    """
    with path.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("The instance root must be an object.")
    if input_format == "legacy":
        from .legacy_input import convert_legacy_instance

        try:
            data = convert_legacy_instance(data)
            if max_daily_load is not None:
                for teacher in data["teachers"]:
                    teacher["max_daily_load"] = max_daily_load
        except (KeyError, TypeError, AttributeError, IndexError) as error:
            raise ValueError(f"Malformed historical instance: {error}") from error
    elif input_format != "instance":
        raise ValueError("Unknown input format.")
    return parse_instance(
        data,
        max_daily_load=8 if max_daily_load is None else max_daily_load,
        max_block_load=max_block_load,
    )


def parse_instance(data: dict, *, max_daily_load: int, max_block_load: int) -> Instance:
    """Build ordered domain objects from the English instance schema."""
    if not isinstance(data, dict):
        raise ValueError("The instance root must be an object.")
    allowed = {"courses", "teachers", "students", "curricular_levels"}
    if set(data) - allowed:
        raise ValueError(f"Unknown instance fields: {', '.join(sorted(set(data) - allowed))}")
    for name in allowed:
        if not isinstance(data.get(name), list) or any(
            not isinstance(item, dict) for item in data[name]
        ):
            raise ValueError(f"{name} must be a list of objects.")
    if not data["courses"]:
        raise ValueError("At least one course is required.")
    _positive_int(max_daily_load, "max_daily_load")
    _positive_int(max_block_load, "max_block_load")
    course_indices = _unique_ids(data["courses"], "course")
    teacher_indices = _unique_ids(data["teachers"], "teacher")
    _unique_ids(data["students"], "student")
    _unique_ids(data["curricular_levels"], "curricular level")
    block_indices = {identifier: index for index, identifier in enumerate(BLOCK_IDS)}
    courses = []
    teacher_lecture_counts = {index: 0 for index in teacher_indices.values()}
    distinct_lectures = set()
    separated_lectures = set()
    for course_index, course_data in enumerate(data["courses"]):
        _check_fields(course_data, {"id", "sections"}, "course")
        sections_data = course_data.get("sections")
        if not isinstance(sections_data, list) or not sections_data:
            raise ValueError("Each course needs a nonempty sections list.")
        _unique_ids(sections_data, "section")
        sections = []
        for section_index, section_data in enumerate(sections_data):
            _check_fields(section_data, {"id", "lectures"}, "section")
            lectures_data = section_data.get("lectures")
            if not isinstance(lectures_data, list) or not lectures_data:
                raise ValueError("Each section needs a nonempty lectures list.")
            _unique_ids(lectures_data, "lecture")
            lectures = []
            for lecture_index, lecture_data in enumerate(lectures_data):
                _check_fields(
                    lecture_data,
                    {"id", "teachers", "preassigned_block", "distinct_day", "separated_day"},
                    "lecture",
                )
                teachers = _resolve_ids(
                    lecture_data.get("teachers", []), teacher_indices, "lecture teachers"
                )
                for teacher_index in teachers:
                    teacher_lecture_counts[teacher_index] += 1
                block_id = lecture_data.get("preassigned_block")
                if block_id is not None and (
                    not isinstance(block_id, str) or block_id not in block_indices
                ):
                    raise ValueError(f"Unknown preassigned block: {block_id}")
                lectures.append(Lecture(lecture_data["id"], teachers, block_indices.get(block_id)))
                key = course_index, section_index, lecture_index
                for field, target in (
                    ("distinct_day", distinct_lectures),
                    ("separated_day", separated_lectures),
                ):
                    flag = lecture_data.get(field, False)
                    if type(flag) is not bool:
                        raise ValueError(f"{field} must be a boolean.")
                    if flag:
                        target.add(key)
            sections.append(Section(section_data["id"], tuple(lectures)))
        courses.append(Course(course_data["id"], tuple(sections)))
    teachers = []
    for teacher_index, teacher_data in enumerate(data["teachers"]):
        _check_fields(
            teacher_data, {"id", "available_blocks", "max_days", "max_daily_load"}, "teacher"
        )
        available = _resolve_ids(
            teacher_data.get("available_blocks", list(BLOCK_IDS)), block_indices, "availability"
        )
        automatic_days = 3 if teacher_lecture_counts[teacher_index] <= 6 else 4
        teachers.append(
            Teacher(
                teacher_data["id"],
                frozenset(available),
                _positive_int(teacher_data.get("max_days", automatic_days), "max_days"),
                _positive_int(teacher_data.get("max_daily_load", max_daily_load), "max_daily_load"),
            )
        )
    for record in data["students"]:
        _check_fields(record, {"id", "courses"}, "student")
    for record in data["curricular_levels"]:
        _check_fields(record, {"id", "courses"}, "curricular level")
    students = tuple(
        Student(
            record["id"], _resolve_ids(record.get("courses", []), course_indices, "student courses")
        )
        for record in data["students"]
    )
    levels = tuple(
        CurricularLevel(
            record["id"], _resolve_ids(record.get("courses", []), course_indices, "level courses")
        )
        for record in data["curricular_levels"]
    )
    return Instance(
        tuple(courses),
        tuple(teachers),
        students,
        levels,
        BLOCK_IDS,
        frozenset({block_indices["Thu:4"]}),
        max_block_load,
        frozenset(distinct_lectures),
        frozenset(separated_lectures),
    )


def write_solution(
    path: Path, instance: Instance, solution: Solution, *, stop_reason: str, metadata: dict
) -> None:
    """Write X and Y with original identifiers, preserving existing files."""
    write_document(
        path, solution_document(instance, solution, stop_reason=stop_reason, metadata=metadata)
    )


def solution_document(
    instance: Instance, solution: Solution, *, stop_reason: str, metadata: dict
) -> dict:
    """Serialize a solution with original identifiers, without writing files."""
    timetable = []
    for (course_index, section_index, lecture_index, block_index), active in solution.X.items():
        if active != 1:
            continue
        course = instance.courses[course_index]
        section = course.sections[section_index]
        lecture = section.lectures[lecture_index]
        timetable.append(
            {
                "course": course.course_id,
                "section": section.section_id,
                "lecture": lecture.lecture_id,
                "block": instance.block_ids[block_index],
                "teachers": [
                    instance.teachers[index].teacher_id for index in lecture.teacher_indices
                ],
            }
        )
    assignments = []
    for (student_index, course_index, section_index), active in solution.Y.items():
        if active == 1:
            course = instance.courses[course_index]
            assignments.append(
                {
                    "student": instance.students[student_index].student_id,
                    "course": course.course_id,
                    "section": course.sections[section_index].section_id,
                }
            )
    return {
        "status": "solved",
        "stop_reason": stop_reason,
        "metadata": metadata,
        "X": timetable,
        "Y": assignments,
    }


def write_document(path: Path, document: dict) -> None:
    """Write a result document as JSON; refuse an existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
