"""Curricular-level affinity and construction priority (Initial Timetabling)."""

from .domain import Instance


class LevelAffinity:
    """Group levels using mean bidirectional shared-student proportions."""

    def __init__(self, instance: Instance, *, beta: float) -> None:
        self.instance = instance
        self.beta = beta
        level_count = len(instance.curricular_levels)
        self._shared_students = [[0 for _ in range(level_count)] for _ in range(level_count)]
        student_in_level = [[False for _ in range(level_count)] for _ in instance.students]
        for student_index, student in enumerate(instance.students):
            for level_index, level in enumerate(instance.curricular_levels):
                for course_index in level.course_indices:
                    if course_index in student.course_indices:
                        student_in_level[student_index][level_index] = True
                        break
        for student_index in range(len(instance.students)):
            levels = [
                index for index in range(level_count) if student_in_level[student_index][index]
            ]
            for first_level in levels:
                for second_level in levels:
                    self._shared_students[first_level][second_level] += 1

    def ordered_groups(self) -> list[list[int]]:
        """Return level groups in decreasing lecture-load order.

        Notes
        -----
        Pair ordering and stable ties retain the source traversal. A pair is
        eligible only when its mean affinity strictly exceeds beta.
        """
        pairs = []
        for first_level in range(len(self._shared_students)):
            first_total = self._shared_students[first_level][first_level]
            if first_total > 0:
                for second_level in range(first_level + 1, len(self._shared_students)):
                    second_total = self._shared_students[second_level][second_level]
                    shared = self._shared_students[first_level][second_level]
                    reverse_ratio = shared / second_total if second_total else 0
                    ratio = (shared / first_total + reverse_ratio) / 2
                    pairs.append((first_level, second_level, ratio))
            else:
                pairs.append((first_level, first_level, 0))
        pairs.sort(key=lambda pair: pair[2], reverse=True)
        seen = {}
        groups = []
        for first_level, second_level, ratio in pairs:
            if first_level in seen:
                continue
            seen[first_level] = True
            groups.append([first_level, second_level] if ratio > self.beta else [first_level])
        lecture_counts = {}
        for level_index, level in enumerate(self.instance.curricular_levels):
            total = 0
            for course_index, course in enumerate(self.instance.courses):
                if course_index in level.course_indices:
                    for section in course.sections:
                        total += len(section.lectures)
            lecture_counts[level_index] = total
        groups.sort(key=lambda group: sum(lecture_counts[index] for index in group), reverse=True)
        return groups
