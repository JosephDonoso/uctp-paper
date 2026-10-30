"""Generate the single-program curricular instances used by the datasets."""

import random
from collections import Counter

from .parameters import GenerationOptions
from .preassignments import _preassign_course
from .teacher_availability import _generate_teacher_availability
from .sampling import _assign_section_teachers, _sample_counts
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
        the CLI applies construction acceptance before saving.

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
    teacher_names = _assign_section_teachers(
        courses,
        rng,
        load_mean=options.teacher_load_mean,
        load_sd=options.teacher_load_sd,
        max_load=options.teacher_max_load,
        co_teaching_probability=options.co_teaching_probability,
        shared_teacher_probability=options.shared_teacher_probability,
    )
    teachers = []
    for name in teacher_names:
        teacher = {"nombre": name}
        teacher["disponibilidad"] = _generate_teacher_availability(
            _DAYS, options.availability_probability, rng
        )
        teachers.append(teacher)
    availability = {
        teacher["nombre"]: {
            day: blocks for item in teacher["disponibilidad"] for day, blocks in item.items()
        }
        for teacher in teachers
    }
    occupied = {teacher["nombre"]: set() for teacher in teachers}
    weekly_loads = Counter(
        teacher
        for course in courses
        for section in course["paralelos"]
        for _ in section["clases"]
        for teacher in section["maestros"]
    )
    max_days = {
        name: len(_DAYS) - 2 if weekly_loads[name] <= 6 else len(_DAYS) - 1
        for name in teacher_names
    }
    for course in courses:
        if rng.random() < options.preassignment_probability:
            _preassign_course(
                course, _DAYS, _BLOCKS, availability, occupied, rng,
                max_daily_load=options.max_daily_load, max_days=max_days,
            )
    students = _generate_students(courses, options, rng)
    for course in courses:
        course.pop("_level_index")
    return {
        "restricciones": {
            "MAX_CARGA_DIARIA": options.max_daily_load,
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
