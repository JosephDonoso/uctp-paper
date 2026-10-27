"""Whole-course preassignment attempts with rollback on placement failure."""

import random


def _preassign_course(
    course: dict,
    days: tuple[str, ...],
    blocks: tuple[str, ...],
    availability: dict[str, dict[str, list[int]]],
    occupied: dict[str, set[tuple[str, str]]],
    rng: random.Random,
) -> bool:
    snapshot = {teacher: positions.copy() for teacher, positions in occupied.items()}
    for section in course["paralelos"]:
        section_positions = set()
        for lecture in section["clases"]:
            teachers = [
                teacher
                for teacher in lecture.get("maestros", section["maestros"])
                if teacher in availability
            ]
            candidates = []
            for day in days:
                for block_index, block in enumerate(blocks):
                    position = day, block
                    if position == ("Ju", "7-8") or position in section_positions:
                        continue
                    if all(
                        availability[teacher][day][block_index] == 1
                        and position not in occupied[teacher]
                        for teacher in teachers
                    ):
                        candidates.append(position)
            if (not teachers and "maestros" not in lecture) or not candidates:
                occupied.clear()
                occupied.update(snapshot)
                for previous_section in course["paralelos"]:
                    for previous_lecture in previous_section["clases"]:
                        previous_lecture.pop("horario_predefinido", None)
                return False
            day, block = rng.choice(candidates)
            lecture["horario_predefinido"] = {"dia": day, "bloque": block}
            section_positions.add((day, block))
            for teacher in teachers:
                occupied[teacher].add((day, block))
    return True
