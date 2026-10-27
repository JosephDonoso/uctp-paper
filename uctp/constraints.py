"""Institutional feasibility checks used during construction and TS."""

from .domain import Instance, Timetable


class InstitutionalConstraints:
    """Check X without introducing student conflicts as hard constraints.

    Partial construction uses separate checks in the original order. TS uses
    a combined traversal, as in the reference implementation.
    """

    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self._teachers_by_lecture = {}
        for teacher_index in range(len(instance.teachers)):
            for course_index, course in enumerate(instance.courses):
                for section_index, section in enumerate(course.sections):
                    for lecture_index, lecture in enumerate(section.lectures):
                        if teacher_index in lecture.teacher_indices:
                            key = course_index, section_index, lecture_index
                            self._teachers_by_lecture.setdefault(key, set()).add(teacher_index)
        self.preassignments = {
            (course_index, section_index, lecture_index, lecture.preassigned_block): 1
            for course_index, course in enumerate(instance.courses)
            for section_index, section in enumerate(course.sections)
            for lecture_index, lecture in enumerate(section.lectures)
            if lecture.preassigned_block is not None
        }

    def allows_partial_timetable(self, X: Timetable) -> bool:
        """Check construction constraints in the retained short-circuit order."""
        return (
            self._has_no_teacher_overlaps(X)
            and self._respects_availability(X)
            and self._respects_max_days(X)
            and self._respects_daily_load(X)
            and self._avoids_protected_blocks(X)
            and self._respects_block_capacity(X)
            and self._preserves_preassignments(X)
        )

    def is_complete(self, X: Timetable) -> bool:
        """Require exactly one block for each lecture (rest:all_assignments)."""
        for course_index, course in enumerate(self.instance.courses):
            for section_index, section in enumerate(course.sections):
                for lecture_index in range(len(section.lectures)):
                    assigned_count = sum(
                        X.get((course_index, section_index, lecture_index, block_index), 0)
                        for block_index in range(len(self.instance.block_ids))
                    )
                    if assigned_count != 1:
                        return False
        return True

    def is_feasible(self, X: Timetable) -> bool:
        """Check institutional constraints in one pass; completeness is separate."""
        teacher_occupancy = set()
        daily_loads = {}
        teacher_days = {}
        block_loads = {}
        distinct_days = {}
        separated_days = {}
        for (course_index, section_index, lecture_index, block_index), active in X.items():
            if active == 0:
                continue
            if block_index in self.instance.protected_blocks:
                return False
            block_loads[block_index] = block_loads.get(block_index, 0) + 1
            if block_loads[block_index] > self.instance.max_block_load:
                return False
            day_index = block_index // self.instance.blocks_per_day
            lecture_key = course_index, section_index, lecture_index
            section_key = course_index, section_index
            for restricted_lectures, used_days in (
                (self.instance.distinct_day_lectures, distinct_days),
                (self.instance.separated_day_lectures, separated_days),
            ):
                if lecture_key in restricted_lectures:
                    days = used_days.setdefault(section_key, set())
                    if day_index in days:
                        return False
                    days.add(day_index)
            for teacher_index in self._teachers_by_lecture.get(lecture_key, set()):
                teacher = self.instance.teachers[teacher_index]
                if block_index not in teacher.available_blocks:
                    return False
                occupancy_key = teacher_index, block_index
                if occupancy_key in teacher_occupancy:
                    return False
                teacher_occupancy.add(occupancy_key)
                load_key = teacher_index, day_index
                daily_loads[load_key] = daily_loads.get(load_key, 0) + 1
                if daily_loads[load_key] > teacher.max_daily_load:
                    return False
                teacher_days.setdefault(teacher_index, set()).add(day_index)
        for teacher_index, days in teacher_days.items():
            if len(days) > self.instance.teachers[teacher_index].max_days:
                return False
        for days in separated_days.values():
            ordered_days = sorted(days)
            for index in range(1, len(ordered_days)):
                if ordered_days[index] - ordered_days[index - 1] <= 1:
                    return False
        return self._preserves_preassignments(X)

    def _has_no_teacher_overlaps(self, X: Timetable) -> bool:
        occupied = set()
        for (course_index, section_index, lecture_index, block_index), active in X.items():
            if active == 1:
                for teacher_index in self._teachers_by_lecture.get(
                    (course_index, section_index, lecture_index), set()
                ):
                    key = teacher_index, block_index
                    if key in occupied:
                        return False
                    occupied.add(key)
        return True

    def _respects_availability(self, X: Timetable) -> bool:
        for (course_index, section_index, lecture_index, block_index), active in X.items():
            if active == 1:
                for teacher_index in self._teachers_by_lecture.get(
                    (course_index, section_index, lecture_index), set()
                ):
                    if block_index not in self.instance.teachers[teacher_index].available_blocks:
                        return False
        return True

    def _respects_max_days(self, X: Timetable) -> bool:
        teacher_days = {}
        for (course_index, section_index, lecture_index, block_index), active in X.items():
            if active == 1:
                for teacher_index in self._teachers_by_lecture.get(
                    (course_index, section_index, lecture_index), set()
                ):
                    days = teacher_days.setdefault(teacher_index, set())
                    days.add(block_index // self.instance.blocks_per_day)
                    if len(days) > self.instance.teachers[teacher_index].max_days:
                        return False
        return True

    def _respects_daily_load(self, X: Timetable) -> bool:
        daily_loads = {}
        for (course_index, section_index, lecture_index, block_index), active in X.items():
            if active == 1:
                for teacher_index in self._teachers_by_lecture.get(
                    (course_index, section_index, lecture_index), set()
                ):
                    key = teacher_index, block_index // self.instance.blocks_per_day
                    daily_loads[key] = daily_loads.get(key, 0) + 1
                    if daily_loads[key] > self.instance.teachers[teacher_index].max_daily_load:
                        return False
        return True

    def _avoids_protected_blocks(self, X: Timetable) -> bool:
        for (*_, block_index), active in X.items():
            if active == 1 and block_index in self.instance.protected_blocks:
                return False
        return True

    def _respects_block_capacity(self, X: Timetable) -> bool:
        block_loads = {}
        for (*_, block_index), active in X.items():
            if active == 1:
                total = block_loads.get(block_index, 0) + 1
                if total > self.instance.max_block_load:
                    return False
                block_loads[block_index] = total
        return True

    def _preserves_preassignments(self, X: Timetable) -> bool:
        for key in self.preassignments:
            if X.get(key, 0) == 0:
                return False
        return True
