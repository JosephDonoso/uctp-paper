"""Generate the single-program curricular instances used by the datasets."""

import random
from collections import Counter

from .parameters import GenerationOptions
from .preassignments import _preassign_course
from .teacher_availability import _generate_teacher_availability
from .sampling import _assign_teachers, _sample_counts
from .student_demand import _generate_students

_DAYS = ("Lu", "Ma", "Mi", "Ju", "Vi")
_BLOCKS = tuple(f"{2 * index + 1}-{2 * index + 2}" for index in range(7))


def generate_instance(options: GenerationOptions, *, seed: int) -> dict:
    """Return one candidate in the historical JSON data schema.

    Parameters
    ----------
    options
        Entity counts and distribution parameters, validated before sampling.
    seed
        Integer seed for a private Python random generator. The caller's
        random state is unchanged.

    Returns
    -------
    dict
        Synthetic instance data, without metrics or solution results. Candidate
        generation alone does not establish institutional timetable feasibility;
        the CLI finalizes the shared group daily limit and applies construction
        acceptance before saving.

    Raises
    ------
    ValueError
        A parameter is invalid.
    """
    options.validate()
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    rng = random.Random(seed)
    courses = []
    for level, count in enumerate(options.courses_per_level, start=1):
        for _ in range(count):
            identifier = f"ASG{len(courses) + 1:03d}"
            courses.append(
                {
                    "codigo": identifier,
                    "nombre": f"Asignatura {identifier}",
                    "curso": {"carrera": "CAR001", "semestre": f"{level:02d}"},
                    "_level_index": level - 1,
                    "paralelos": [],
                }
            )
    lecture_sd = options.weekly_lectures_sd
    section_counts = _sample_counts(
        len(courses), 1, 4, options.section_mean, options.section_sd, rng
    )
    lecture_counts = _sample_counts(
        len(courses), 2, 4, options.weekly_lectures_mean, lecture_sd, rng
    )
    layouts = [(lectures,) * sections for sections, lectures in zip(section_counts, lecture_counts)]
    for course, layout in zip(courses, layouts, strict=True):
        for section, lectures in enumerate(layout, 1):
            course["paralelos"].append(
                {
                    "codigo": f"{section:02d}",
                    "maestros": [],
                    "restriccion_teoricas": "none",
                    "clases": [{"tipo": "Cát"} for _ in range(lectures)],
                }
            )
    teachers = []
    for index in range(1, options.teacher_count + 1):
        teacher = {"nombre": f"DOC{index:03d}"}
        teacher["disponibilidad"] = _generate_teacher_availability(
            _DAYS, options.availability_probability, rng
        )
        teachers.append(teacher)
    target_counts = _sample_counts(len(teachers), 1, 4, 2.3, 0.7, rng)
    targets = {
        teacher["nombre"]: count for teacher, count in zip(teachers, target_counts, strict=True)
    }
    _assign_teachers(courses, teachers, targets, rng)
    weekly_loads = Counter(
        teacher
        for course in courses
        for section in course["paralelos"]
        for lecture in section["clases"]
        for teacher in section["maestros"]
    )
    availability = {
        teacher["nombre"]: {
            day: blocks for item in teacher["disponibilidad"] for day, blocks in item.items()
        }
        for teacher in teachers
    }
    occupied = {teacher["nombre"]: set() for teacher in teachers}
    for course in courses:
        if rng.random() < options.preassignment_probability:
            _preassign_course(course, _DAYS, _BLOCKS, availability, occupied, rng)
    students = _generate_students(courses, options, rng)
    for course in courses:
        course.pop("_level_index")
    return {
        "restricciones": {
            "MAX_CARGA_DIARIA": max(weekly_loads.values()) // 3 + 1,
            "MAX_CARGA_BLOQUE": 15,
            "PROFESORES": [
                {
                    "nombre": teacher["nombre"],
                    "respeta_max_carga_semanal": True,
                    "max_carga_semanal": "default",
                }
                for teacher in teachers
            ],
        },
        "tiempos": {"dias": list(_DAYS), "bloques": list(_BLOCKS)},
        "eventos": {"tipo_clases": ["Cát"], "asignaturas": courses},
        "recursos": {"maestros": teachers, "estudiantes": students},
    }
