"""Explicit boundary for historical Spanish instance keys, never settings files."""

from .instance_io import BLOCK_IDS

_DAYS = ("Lu", "Ma", "Mi", "Ju", "Vi")
_PERIODS = ("1-2", "3-4", "5-6", "7-8", "9-10", "11-12", "13-14")


def _apply_max_days(teacher: dict, limits: dict) -> None:
    respects_limit = str(limits.get("respeta_max_carga_semanal", True)).strip().lower()
    if respects_limit in ("0", "false", "f", "no", "n"):
        return
    value = limits.get("max_carga_semanal")
    try:
        days = int(value)
    except (TypeError, ValueError):
        return
    if days > 0:
        teacher["max_days"] = days


def _student_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    identifier = str(value).strip()
    if identifier.endswith(".0") and identifier[:-2].isdigit():
        return identifier[:-2]
    return identifier


def _block_id(day: str, period: str) -> str:
    try:
        return BLOCK_IDS[_DAYS.index(day) * 7 + _PERIODS.index(period)]
    except ValueError as error:
        raise ValueError(f"Unsupported historical time block: {day} {period}") from error


def _restriction_references(
    item: object, course: str = "", section: str = ""
) -> list[tuple[str, str, str | None]]:
    if isinstance(item, dict):
        course = str(
            item.get("codigo_asignatura")
            or item.get("id_asignatura")
            or item.get("asignatura")
            or item.get("codigo")
            or course
        )
        section = str(
            item.get("codigo_paralelo")
            or item.get("id_paralelo")
            or item.get("paralelo")
            or section
        )
        if "clases" in item or "sesiones" in item:
            return [
                reference
                for child in (item.get("clases") or item.get("sesiones") or [])
                for reference in _restriction_references(child, course, section)
            ]
        lecture = (
            item.get("clase")
            or item.get("sesion")
            or item.get("id_sesion")
            or item.get("indice")
            or item.get("numero")
        )
        if course and section:
            return [(course, section, str(int(lecture)) if lecture is not None else None)]
    elif isinstance(item, (list, tuple)) and len(item) >= 3:
        return [(str(item[0]).strip(), str(item[1]).strip(), str(int(item[2])))]
    elif isinstance(item, (int, float)) and course and section:
        return [(course, section, str(int(item)))]
    elif isinstance(item, str):
        for separator in ("|", ";", ",", "/"):
            parts = item.split(separator)
            if len(parts) >= 3:
                return [(parts[0].strip(), parts[1].strip(), str(int(parts[2])))]
    raise ValueError("Unsupported historical lecture-day restriction.")


def convert_legacy_instance(data: dict) -> dict:
    """Translate historical domain data into the English schema.

    Notes
    -----
    Preserves source order, repeated-student merging, missing-teacher creation
    and the historical empty-availability fallback. Observed section enrollment
    is ignored. MAX_CARGA_DIARIA is measured in blocks, without hour conversion.
    """
    if "eventos" not in data or "recursos" not in data:
        raise ValueError("Expected historical instance data, not an execution settings file.")
    times = data.get("tiempos", {})
    if (
        tuple(times.get("dias", _DAYS)) != _DAYS
        or tuple(times.get("bloques", _PERIODS)) != _PERIODS
    ):
        raise ValueError(
            "Historical input must use five days and seven nominal blocks in source order."
        )
    resources = data["recursos"]
    restrictions = data.get("restricciones", {})
    daily_load = restrictions.get("MAX_CARGA_DIARIA", 8)
    if type(daily_load) is not int or daily_load <= 0:
        raise ValueError("MAX_CARGA_DIARIA must be a positive integer number of blocks.")
    teacher_limits = {
        str(item["nombre"]).lower(): item for item in restrictions.get("PROFESORES", [])
    }
    teachers = []
    teachers_by_id = {}
    for record in resources.get("maestros", []):
        identifier = str(record["nombre"])
        available = []
        for day_record in record.get("disponibilidad", []) or []:
            day, values = next(iter(day_record.items()))
            for index, value in enumerate(values[:7]):
                if int(value) == 1:
                    available.append(_block_id(day, _PERIODS[index]))
        individual_daily_load = record.get("max_daily_load", daily_load)
        if type(individual_daily_load) is not int or individual_daily_load <= 0:
            raise ValueError("Teacher max_daily_load must be a positive integer number of blocks.")
        teacher = {
            "id": identifier,
            "available_blocks": available or list(BLOCK_IDS),
            "max_daily_load": individual_daily_load,
        }
        limits = teacher_limits.get(identifier.lower(), {})
        _apply_max_days(teacher, limits)
        teachers.append(teacher)
        teachers_by_id[identifier] = teacher
    courses = []
    levels = {}
    for record in data["eventos"].get("asignaturas", []):
        identifier = str(record["codigo"])
        course = {"id": identifier, "sections": []}
        level = record.get("curso") or {}
        program, semester = level.get("carrera"), level.get("semestre")
        if program and semester and str(semester).isdigit():
            level_id = f"{program}-{int(semester)}"
            levels.setdefault(level_id, {"id": level_id, "courses": []})["courses"].append(
                identifier
            )
        for section_record in record.get("paralelos", []):
            section = {"id": str(section_record["codigo"]), "lectures": []}
            for index, lecture_record in enumerate(section_record.get("clases", []), start=1):
                lecture_type = lecture_record.get("tipo")
                if lecture_type not in ("Cát", "Cat", "Ay.", "Ay", "Lab", "Tal"):
                    raise ValueError(f"Unsupported historical lecture type: {lecture_type}")
                teacher_ids = [
                    str(value)
                    for value in (
                        lecture_record.get("maestros", section_record.get("maestros", [])) or []
                    )
                ]
                for teacher_id in teacher_ids:
                    if teacher_id not in teachers_by_id:
                        teacher = {
                            "id": teacher_id,
                            "available_blocks": list(BLOCK_IDS),
                            "max_daily_load": daily_load,
                        }
                        _apply_max_days(teacher, teacher_limits.get(teacher_id.lower(), {}))
                        teachers_by_id[teacher_id] = teacher
                        teachers.append(teacher)
                lecture = {
                    "id": str(index),
                    "teachers": [] if lecture_type in ("Ay.", "Ay") else teacher_ids,
                }
                fixed = lecture_record.get("horario_predefinido")
                if fixed:
                    lecture["preassigned_block"] = _block_id(
                        str(fixed["dia"]), str(fixed["bloque"])
                    )
                section["lectures"].append(lecture)
            course["sections"].append(section)
        courses.append(course)
    for historical_name, field in (
        ("CLASES_TEORICAS_DIAS_DISTINTOS", "distinct_day"),
        ("CLASES_TEORICAS_SEPARADAS_1_DIA", "separated_day"),
    ):
        for item in restrictions.get(historical_name, []) or []:
            for course_id, section_id, lecture_id in _restriction_references(item):
                found = False
                for course in courses:
                    if course["id"] != course_id:
                        continue
                    for section in course["sections"]:
                        if section["id"] != section_id:
                            continue
                        for lecture in section["lectures"]:
                            if lecture_id is None or lecture["id"] == lecture_id:
                                lecture[field] = True
                                found = True
                if not found:
                    raise ValueError(
                        "Historical lecture-day restriction references an unknown lecture."
                    )
    students = {}
    course_ids = {course["id"] for course in courses}
    for record in resources.get("estudiantes", []):
        identifier = _student_id(record.get("id"))
        if not identifier:
            continue
        student = students.setdefault(identifier, {"id": identifier, "courses": []})
        for enrollment in record.get("asignaturas", []):
            course_id = str(enrollment.get("codigo", ""))
            if course_id in course_ids and course_id not in student["courses"]:
                student["courses"].append(course_id)
    return {
        "courses": courses,
        "teachers": teachers,
        "students": list(students.values()),
        "curricular_levels": list(levels.values()),
    }
