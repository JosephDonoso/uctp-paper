"""Directional enrollment projection for Compatibility Matrix construction."""

import numpy as np

type Matrix = list[list[int]]


def _coefficient_of_variation(loads: list[float]) -> float:
    mean = np.mean(loads) if np.any(loads) else 0.0
    if mean == 0:
        return 0.0
    return float(np.std(loads) / mean)


def _project_enrollment(
    matrix: Matrix, first_students: int, second_students: int, shared_students: int
) -> tuple[list[float], list[float]]:
    first_loads = []
    second_loads = []
    first_exclusive = first_students - shared_students
    second_exclusive = second_students - shared_students
    connected_rows = []
    for row in matrix:
        row_sum = 0
        for cell in row:
            row_sum += cell
        connected_rows.append(1 if row_sum > 0 else 0)
    shared_per_row = shared_students / sum(connected_rows)
    first_exclusive_per_section = first_exclusive / len(matrix)
    for row_index in range(len(matrix)):
        first_loads.append(
            first_exclusive_per_section + (shared_per_row if connected_rows[row_index] else 0)
        )
    second_exclusive_per_section = second_exclusive / len(matrix[0])
    for column_index in range(len(matrix[0])):
        second_loads.append(second_exclusive_per_section)
        for row_index in range(len(matrix)):
            if matrix[row_index][column_index] == 1:
                positive_count = 0
                for cell in matrix[row_index]:
                    positive_count += cell if cell == 1 else 0
                second_loads[column_index] += (
                    first_loads[row_index] - first_exclusive_per_section
                ) / positive_count
    return first_loads, second_loads


def _rank_connections(
    matrix: Matrix, first_students: int, second_students: int, shared_students: int
) -> list[tuple[int, int, float, float]]:
    candidates = []
    for row_index in range(len(matrix)):
        for column_index in range(len(matrix[row_index])):
            if matrix[row_index][column_index] == 0:
                matrix[row_index][column_index] = 1
                first_loads, second_loads = _project_enrollment(
                    matrix, first_students, second_students, shared_students
                )
                first_cv = _coefficient_of_variation(first_loads)
                second_cv = _coefficient_of_variation(second_loads)
                matrix[row_index][column_index] = 0
                candidates.append((row_index, column_index, first_cv, second_cv))
    candidates.sort(key=lambda candidate: candidate[2] + candidate[3])
    return candidates


def build_compatibility_matrix(
    first_section_count: int,
    first_students: int,
    second_section_count: int,
    second_students: int,
    shared_students: int,
    *,
    cv_tolerance: float,
) -> Matrix:
    """Construct preferences using alg:compatibility_construction.

    Returns
    -------
    list of list of int
        Rows belong to the already scheduled course; columns to the next one.
        Positive cells seek nonoverlap, one negative cell seeks overlap, and
        zero cells impose no preference. Ties keep row-major enumeration.
    """
    matrix = [[0 for _ in range(second_section_count)] for _ in range(first_section_count)]
    matrix[0][0] = 1
    first_loads, second_loads = _project_enrollment(
        matrix, first_students, second_students, shared_students
    )
    first_cv = _coefficient_of_variation(first_loads)
    second_cv = _coefficient_of_variation(second_loads)
    while first_cv > cv_tolerance or second_cv > cv_tolerance:
        candidates = _rank_connections(matrix, first_students, second_students, shared_students)
        row_index, column_index, first_cv, second_cv = candidates[0]
        matrix[row_index][column_index] = 1
    candidates = _rank_connections(matrix, first_students, second_students, shared_students)
    if candidates:
        # A single maximizer is discouraged; a stable max resolves ties.
        row_index, column_index, _, _ = max(
            candidates, key=lambda candidate: candidate[2] + candidate[3]
        )
        matrix[row_index][column_index] = -1
    return matrix
