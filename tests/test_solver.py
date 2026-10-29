"""Behavioral checks for the manuscript rules and solution-only interface."""

import contextlib
import io
import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from uctp.affinity import LevelAffinity
from uctp.cli import main
from uctp.compatibility import build_compatibility_matrix
from uctp.constraints import InstitutionalConstraints
from uctp.construction import ConstructionFailure, InitialStrategy, InitialTimetabling
from uctp.domain import Solution
from uctp.instance_io import load_instance, parse_instance, write_solution
from uctp.legacy_input import convert_legacy_instance
from uctp.objective import SearchObjective
from uctp.planner import Approach, SolveOptions, solve
from uctp.sectioning import StudentSectioning
from uctp.student_cost import compute_inconvenience_cost

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "small.json"


def sample_data() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def parse(data: dict):
    return parse_instance(data, max_daily_load=4, max_block_load=15)


class SolverTests(unittest.TestCase):
    def test_all_strategies_and_approaches_resolve_complete_assignments(self):
        instance = parse(sample_data())
        constraints = InstitutionalConstraints(instance)
        for strategy in InitialStrategy:
            for approach in Approach:
                with self.subTest(strategy=strategy, approach=approach):
                    options = SolveOptions(
                        strategy=strategy,
                        approach=approach,
                        seed=703,
                        iterations_per_search=3,
                        search_blocks=2 if approach == Approach.DECOUPLED else 1,
                    )
                    result = solve(instance, options)
                    self.assertTrue(constraints.is_complete(result.solution.X))
                    self.assertTrue(constraints.is_feasible(result.solution.X))
                    for student_index, student in enumerate(instance.students):
                        for course_index, course in enumerate(instance.courses):
                            assigned = sum(
                                result.solution.Y.get(
                                    (student_index, course_index, section_index), 0
                                )
                                for section_index in range(len(course.sections))
                            )
                            self.assertEqual(assigned, int(course_index in student.course_indices))

    def test_seeded_runs_are_independent_and_repeatable(self):
        instance = parse(sample_data())
        options = SolveOptions(seed=41, iterations_per_search=4)
        first = solve(instance, options)
        solve(instance, replace(options, seed=7))
        second = solve(instance, options)
        self.assertEqual(first, second)
        self.assertIsNot(first.solution.X, second.solution.X)

    @patch("uctp.sectioning.kernel_enabled", return_value=False)
    def test_ssp_visits_each_student_once_in_seeded_random_order(self, _kernel):
        # Inspects the pure-Python path; the kernel is checked for identical
        # results in tests/test_ssp_kernel.py.
        instance = parse(sample_data())
        result = solve(instance, SolveOptions(seed=41, iterations_per_search=1))
        original_students = instance.students
        objective = SearchObjective(instance, alpha=1, cv_tolerance=0.1, idle_tolerance=4)
        sectioning = StudentSectioning(instance, objective)

        def visit_order(seed=None):
            if seed is not None:
                random.seed(seed)
            with patch.object(
                sectioning, "_refine_sections", wraps=sectioning._refine_sections
            ) as refine:
                sectioning.solve(result.solution)
            return [call.args[1] for call in refine.call_args_list]

        first_order = visit_order(41)
        next_order = visit_order()
        self.assertEqual(sorted(first_order), list(range(len(instance.students))))
        self.assertEqual(sorted(next_order), sorted(first_order))
        self.assertNotEqual(first_order, list(range(len(instance.students))))
        self.assertNotEqual(first_order, next_order)
        self.assertEqual(first_order, visit_order(41))
        self.assertIs(instance.students, original_students)
        for student_index, student in enumerate(instance.students):
            for course_index in student.course_indices:
                self.assertEqual(
                    sum(
                        result.solution.Y.get((student_index, course_index, section_index), 0)
                        for section_index in range(len(instance.courses[course_index].sections))
                    ),
                    1,
                )

    def test_partial_preassignment_keeps_other_lectures(self):
        data = sample_data()
        data["courses"][0]["sections"][0]["lectures"][0]["preassigned_block"] = "Mon:1"
        instance = parse(data)
        for strategy in InitialStrategy:
            result = solve(
                instance, SolveOptions(strategy=strategy, seed=5, iterations_per_search=2)
            )
            self.assertEqual(result.solution.X[0, 0, 0, 0], 1)
            self.assertTrue(InstitutionalConstraints(instance).is_complete(result.solution.X))

    def test_protected_preassignment_is_rejected(self):
        data = sample_data()
        data["courses"][0]["sections"][0]["lectures"][0]["preassigned_block"] = "Thu:4"
        with self.assertRaises(ConstructionFailure):
            solve(parse(data), SolveOptions(seed=1, iterations_per_search=1))

    def test_retry_limit_zero_permits_one_attempt_only(self):
        data = sample_data()
        data["teachers"][0]["available_blocks"] = []
        instance = parse(data)
        construction = InitialTimetabling(
            instance,
            InstitutionalConstraints(instance),
            strategy=InitialStrategy.RANDOM,
            max_retries=0,
            beta=0.25,
            cv_tolerance=0.1,
        )
        with patch.object(
            construction, "_schedule_randomly", wraps=construction._schedule_randomly
        ) as attempt:
            with self.assertRaises(ConstructionFailure):
                construction.solve()
            self.assertEqual(attempt.call_count, 1)

    def test_retry_budget_is_initial_attempt_plus_retries(self):
        data = sample_data()
        data["teachers"][0]["available_blocks"] = []
        instance = parse(data)
        construction = InitialTimetabling(
            instance,
            InstitutionalConstraints(instance),
            strategy=InitialStrategy.RANDOM,
            max_retries=2,
            beta=0.25,
            cv_tolerance=0.1,
        )
        with patch.object(
            construction, "_schedule_randomly", wraps=construction._schedule_randomly
        ) as attempt:
            with self.assertRaises(ConstructionFailure):
                construction.solve()
            self.assertEqual(attempt.call_count, 3)

    def test_compatibility_matrix_balances_example_and_discourages_one_connection(self):
        matrix = build_compatibility_matrix(3, 90, 2, 100, 60, cv_tolerance=0.1)
        # The paper's sequence is illustrative; tied section permutations are
        # valid. Verify projected loads independently of the implementation.
        self.assertEqual(sum(cell == -1 for row in matrix for cell in row), 1)
        self.assertTrue(all(any(cell == 1 for cell in row) for row in matrix))
        first_loads = [10 + 60 / 3 for _ in matrix]
        second_loads = [
            20 + sum(20 / row.count(1) for row in matrix if row[column] == 1) for column in range(2)
        ]
        self.assertEqual(first_loads, [30, 30, 30])
        self.assertEqual(second_loads, [50, 50])

    def test_affinity_uses_both_denominators(self):
        data = sample_data()
        data["curricular_levels"] = [
            {"id": "Q1", "courses": ["C1"]},
            {"id": "Q2", "courses": ["C2"]},
        ]
        data["students"] = [{"id": str(index), "courses": ["C1"]} for index in range(10)]
        data["students"][0]["courses"].append("C2")
        groups = LevelAffinity(parse(data), beta=0.25).ordered_groups()
        self.assertEqual(groups[0], [0, 1])

    def test_inconvenience_preserves_overlaps_and_lunch(self):
        self.assertEqual(compute_inconvenience_cost([], idle_tolerance=4), 0)
        self.assertEqual(compute_inconvenience_cost([0, 1], idle_tolerance=4), 0)
        self.assertEqual(compute_inconvenience_cost([3, 4], idle_tolerance=4), 1)
        self.assertEqual(compute_inconvenience_cost([0, 0], idle_tolerance=4), 13)
        self.assertEqual(compute_inconvenience_cost([0, 0, 0], idle_tolerance=4), 104)

    def test_unavoidable_student_conflicts_do_not_drop_courses(self):
        data = sample_data()
        data["courses"] = data["courses"][:2]
        for course in data["courses"]:
            course["sections"] = course["sections"][:1]
            course["sections"][0]["lectures"] = course["sections"][0]["lectures"][:1]
        data["students"] = [{"id": "S1", "courses": ["C1", "C2"]}]
        data["curricular_levels"] = []
        instance = parse(data)
        solution = Solution({(0, 0, 0, 0): 1, (1, 0, 0, 0): 1})
        objective = SearchObjective(instance, alpha=0, cv_tolerance=0.1, idle_tolerance=4)
        random.seed(1)
        StudentSectioning(instance, objective).solve(solution)
        self.assertEqual(solution.Y, {(0, 0, 0): 1, (0, 1, 0): 1})
        self.assertTrue(InstitutionalConstraints(instance).is_feasible(solution.X))
        self.assertEqual(objective.evaluate(solution), 13 / 4)

    def test_single_section_and_empty_enrollment_have_defined_objective(self):
        data = sample_data()
        for course in data["courses"]:
            course["sections"] = course["sections"][:1]
        data["students"] = [{"id": "S1", "courses": []}]
        instance = parse(data)
        result = solve(instance, SolveOptions(seed=1, iterations_per_search=1))
        self.assertEqual(result.solution.Y, {})
        self.assertEqual(
            SearchObjective(instance, alpha=1, cv_tolerance=0.1, idle_tolerance=4).evaluate(
                result.solution
            ),
            0,
        )

    def test_teacher_capacity_and_day_restrictions(self):
        data = sample_data()
        data["teachers"][0].update(max_days=1, max_daily_load=1)
        instance = parse(data)
        constraints = InstitutionalConstraints(instance)
        self.assertFalse(constraints.is_feasible({(0, 0, 0, 0): 1, (0, 0, 1, 0): 1}))
        self.assertFalse(constraints.is_feasible({(0, 0, 0, 0): 1, (0, 0, 1, 1): 1}))
        self.assertFalse(constraints.is_feasible({(0, 0, 0, 0): 1, (0, 0, 1, 7): 1}))
        self.assertFalse(
            InstitutionalConstraints(replace(instance, max_block_load=1)).is_feasible(
                {(0, 0, 0, 0): 1, (1, 0, 0, 0): 1}
            )
        )

    def test_unknown_identifiers_and_settings_are_rejected(self):
        for mutation in (
            lambda data: data.update(alpha=2),
            lambda data: data["students"][0].update(courses=["unknown"]),
            lambda data: data["teachers"][0].update(available_blocks=["Sun:1"]),
            lambda data: data["courses"][0].update(settings={}),
            lambda data: data["courses"][0].update(sections=[None]),
        ):
            data = sample_data()
            mutation(data)
            with self.assertRaises(ValueError):
                parse(data)

    def test_legacy_defaults_and_observed_enrollment_are_handled_at_boundary(self):
        historical = {
            "restricciones": {"PROFESORES": [{"nombre": "T1", "max_carga_semanal": "default"}]},
            "eventos": {
                "asignaturas": [
                    {
                        "codigo": "C1",
                        "paralelos": [
                            {"codigo": "P1", "clases": [{"tipo": "Cát", "maestros": ["T1"]}]}
                        ],
                    }
                ]
            },
            "recursos": {
                "maestros": [{"nombre": "T1", "disponibilidad": []}],
                "estudiantes": [
                    {"id": "1.0", "asignaturas": [{"codigo": "C1", "paralelo": "P1"}]},
                    {"id": 1, "asignaturas": [{"codigo": "C1"}]},
                ],
            },
        }
        instance = parse(convert_legacy_instance(historical))
        self.assertEqual(len(instance.students), 1)
        self.assertEqual(instance.students[0].student_id, "1")
        self.assertEqual(instance.students[0].course_indices, (0,))
        self.assertEqual(instance.teachers[0].max_days, 3)
        self.assertEqual(instance.teachers[0].max_daily_load, 8)
        self.assertEqual(len(instance.teachers[0].available_blocks), 35)

        # Global instance units are blocks; stale per-teacher legacy values
        # must not silently turn the real case's four-block limit into eight.
        historical["restricciones"]["PROFESORES"][0]["max_carga_diaria"] = 8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instance.json"
            for blocks in (8, 4):
                historical["restricciones"]["MAX_CARGA_DIARIA"] = blocks
                path.write_text(json.dumps(historical), encoding="utf-8")
                loaded = load_instance(
                    path, input_format="legacy", max_daily_load=None, max_block_load=15
                )
                self.assertEqual(loaded.teachers[0].max_daily_load, blocks)
                overridden = load_instance(
                    path, input_format="legacy", max_daily_load=3, max_block_load=15
                )
                self.assertEqual(overridden.teachers[0].max_daily_load, 3)
            for invalid in (0, -1, True, "8"):
                historical["restricciones"]["MAX_CARGA_DIARIA"] = invalid
                with self.assertRaises(ValueError):
                    convert_legacy_instance(historical)

    def test_solution_contains_assignments_and_no_report(self):
        instance = parse(sample_data())
        result = solve(instance, SolveOptions(seed=1, iterations_per_search=1))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "solution.json"
            write_solution(
                destination,
                instance,
                result.solution,
                stop_reason=result.stop_reason,
                metadata={"seed": 1},
            )
            document = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(set(document), {"status", "stop_reason", "metadata", "X", "Y"})
            self.assertEqual(
                set(document["X"][0]), {"course", "section", "lecture", "block", "teachers"}
            )
            self.assertEqual(set(document["Y"][0]), {"student", "course", "section"})

    def test_cli_decoupled_and_invalid_combinations(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "solution.json"
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                code = main(
                    [
                        "--input",
                        str(EXAMPLE),
                        "--output",
                        str(output),
                        "--seed",
                        "5",
                        "--approach",
                        "Decoupled",
                        "--ssp-interval",
                        "2",
                        "--search-blocks",
                        "2",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(captured.getvalue().startswith("Solved."))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["--input", str(EXAMPLE), "--output", str(output), "--ssp-interval", "2"])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
