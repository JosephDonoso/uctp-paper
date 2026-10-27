"""Endpoint measurements and raw student/section data, outside search timing."""

from .domain import Instance, Solution, collect_section_blocks
from .student_cost import compute_inconvenience_cost


def measure_solution(
    instance: Instance, solution: Solution, *, alpha: float, details: bool = False
) -> dict:
    """Measure a complete X,Y pair without changing assignments or RNG state.

    Returns aggregate metrics and, when requested, section enrollment and
    student attendance/cost records. Conflicts count excess attendance records
    per student and nominal block, rather than all overlapping record pairs.
    The same attendance multiplicity and cost rules as the optimizer apply.
    """
    blocks = collect_section_blocks(instance, solution.X)
    enrollment = {
        (c, p): 0 for c, course in enumerate(instance.courses) for p in range(len(course.sections))
    }
    students, total_cost, conflicts, requests = [], 0.0, 0, 0
    for index, student in enumerate(instance.students):
        attendance = []
        assigned = 0
        for course in student.course_indices:
            for section in range(len(instance.courses[course].sections)):
                if solution.Y.get((index, course, section), 0) == 1:
                    enrollment[course, section] += 1
                    attendance.extend(blocks[course, section])
                    assigned += 1
        cost = compute_inconvenience_cost(attendance, idle_tolerance=4)
        count = len(attendance) - len(set(attendance))
        total_cost += cost
        conflicts += count
        requests += len(student.course_indices)
        if details:
            students.append(
                dict(
                    student=student.student_id,
                    requested_courses=len(student.course_indices),
                    assigned_courses=assigned,
                    attendance_blocks=[instance.block_ids[b] for b in attendance],
                    conflicts=count,
                    inconvenience_cost=cost,
                )
            )
    observed, reference, parallel_courses = 0.0, 0.0, 0
    sections = []
    for course_index, course in enumerate(instance.courses):
        counts = [enrollment[course_index, p] for p in range(len(course.sections))]
        mean = sum(counts) / len(counts)
        sd = (sum((count - mean) ** 2 for count in counts) / len(counts)) ** 0.5
        if len(counts) > 1:
            observed += sd
            reference += mean * 0.1
            parallel_courses += 1
        if details:
            for p, section in enumerate(course.sections):
                sections.append(
                    dict(
                        course=course.course_id,
                        section=section.section_id,
                        enrollment=counts[p],
                        course_mean_enrollment=mean,
                        course_population_sd=sd,
                        course_reference_sd=mean * 0.1,
                    )
                )
    observed /= max(parallel_courses, 1)
    reference /= max(parallel_courses, 1)
    active = sum(bool(student.course_indices) for student in instance.students)
    dissatisfaction_reference = active * 4
    imbalance = observed / max(reference, 1e-6)
    dissatisfaction = total_cost / max(dissatisfaction_reference, 1e-6)
    metrics = dict(
        F=imbalance**alpha * dissatisfaction,
        F_ref=imbalance * dissatisfaction,
        R_imb=imbalance,
        R_dis=dissatisfaction,
        conflicts=conflicts,
        imbalance_observed=observed,
        imbalance_reference=reference,
        dissatisfaction_total=total_cost,
        dissatisfaction_reference=dissatisfaction_reference,
        active_students=active,
        students=len(instance.students),
        courses=len(instance.courses),
        parallel_courses=parallel_courses,
        requested_courses=requests,
        assigned_courses=sum(value == 1 for value in solution.Y.values()),
        scheduled_lectures=sum(value == 1 for value in solution.X.values()),
    )
    result = {"metrics": metrics}
    if details:
        result["details"] = {"students": students, "sections": sections}
    return result
