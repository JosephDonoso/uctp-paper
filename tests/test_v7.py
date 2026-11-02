"""Additions for the v7 experiments: explicit SSP random streams, common random
numbers, Decoupled checkpoints, the sequential baseline, warm starts, partial
local search and component traces."""

import os
import random
import unittest
from pathlib import Path

from uctp.domain import Solution
from uctp.instance_io import load_instance
from uctp.metrics import measure_solution
from uctp.objective import SearchObjective
from uctp.planner import Approach, SolveOptions, solve
from uctp.proxy import ProxyObjective
from uctp.sectioning import StudentSectioning

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "small.json"


def _instance():
    return load_instance(EXAMPLE, input_format="instance", max_daily_load=None, max_block_load=15)


def _active(solution):
    return sorted(key for key, value in solution.X.items() if value == 1)


class SectioningStreamTests(unittest.TestCase):
    def setUp(self):
        self.instance = _instance()
        self.X = solve(self.instance, SolveOptions(seed=5, iterations_per_search=3)).solution.X

    def _solve(self, rng=None, **kwargs):
        objective = SearchObjective(self.instance, alpha=1, cv_tolerance=0.1, idle_tolerance=4)
        ssp = StudentSectioning(self.instance, objective, **kwargs)
        solution = Solution(dict(self.X), {})
        value = ssp.solve(solution, rng=rng)
        return value, solution.Y

    def test_explicit_stream_is_deterministic_and_leaves_global_state(self):
        state = random.getstate()
        first = self._solve(random.Random(9))
        self.assertEqual(random.getstate(), state)
        self.assertEqual(first, self._solve(random.Random(9)))

    def test_kernel_and_python_agree_with_explicit_stream(self):
        kernel = self._solve(random.Random(3))
        os.environ["UCTP_PURE_PYTHON"] = "1"
        try:
            python = self._solve(random.Random(3))
        finally:
            del os.environ["UCTP_PURE_PYTHON"]
        self.assertEqual(kernel, python)

    def test_probability_extremes_equal_local_search_switch(self):
        self.assertEqual(
            self._solve(random.Random(4), local_search_probability=1.0),
            self._solve(random.Random(4), local_search=True),
        )
        self.assertEqual(
            self._solve(random.Random(4), local_search_probability=0.0),
            self._solve(random.Random(4), local_search=False),
        )
        partial = self._solve(random.Random(4), local_search_probability=0.5)
        self.assertEqual(partial, self._solve(random.Random(4), local_search_probability=0.5))


class SearchAdditionsTests(unittest.TestCase):
    def setUp(self):
        self.instance = _instance()

    def test_common_random_numbers_run_is_reproducible(self):
        options = SolveOptions(seed=7, iterations_per_search=15, common_random_numbers=True)
        first = solve(self.instance, options)
        second = solve(self.instance, options)
        self.assertEqual(_active(first.solution), _active(second.solution))
        self.assertEqual(first.solution.Y, second.solution.Y)

    def test_embedded_checkpoints_with_common_random_numbers(self):
        base = dict(alpha=1.0, seed=703, max_retries=50, common_random_numbers=True)
        full = solve(self.instance, SolveOptions(**base, iterations_per_search=20,
                                                 total_iteration_limit=20, checkpoints=(5, 10)))
        recorded = {item["iterations"]: item["solution"] for item in full.checkpoints}
        for budget in (5, 10):
            separate = solve(self.instance, SolveOptions(**base, iterations_per_search=budget,
                                                         total_iteration_limit=budget))
            self.assertEqual(_active(separate.solution), _active(recorded[budget]))
            self.assertEqual(separate.solution.Y, recorded[budget].Y)

    def test_decoupled_checkpoints_equal_separate_runs(self):
        base = dict(approach=Approach.DECOUPLED, alpha=0.25, seed=17, max_retries=50,
                    iterations_per_search=4, search_blocks=1000, max_stagnant_iterations=1000)
        full = solve(self.instance, SolveOptions(**base, total_iteration_limit=30))
        checked = solve(self.instance, SolveOptions(**base, total_iteration_limit=30,
                                                    checkpoints=(3, 8, 13, 21)))
        self.assertEqual(_active(full.solution), _active(checked.solution))
        self.assertEqual(full.solution.Y, checked.solution.Y)
        recorded = {item["iterations"]: item["solution"] for item in checked.checkpoints}
        for budget in (3, 8, 13, 21):
            separate = solve(self.instance, SolveOptions(**base, total_iteration_limit=budget))
            self.assertEqual(_active(separate.solution), _active(recorded[budget]))
            self.assertEqual(separate.solution.Y, recorded[budget].Y)
            self.assertEqual(
                measure_solution(self.instance, separate.solution, alpha=0.25)["metrics"],
                measure_solution(self.instance, recorded[budget], alpha=0.25)["metrics"],
            )

    def test_sequential_assigns_students_once(self):
        result = solve(self.instance, SolveOptions(approach=Approach.SEQUENTIAL, seed=3,
                                                   iterations_per_search=25),
                       collect_metrics=True)
        self.assertEqual(result.diagnostics["ssp_calls"], 2)  # final + initial metrics
        metrics = measure_solution(self.instance, result.solution, alpha=1)["metrics"]
        self.assertEqual(metrics["assigned_courses"], metrics["requested_courses"])
        proxy = ProxyObjective(self.instance)
        self.assertLessEqual(proxy.evaluate(result.solution),
                             proxy.evaluate(result.initial_solution) + 1e-9)

    def test_proxy_counts_shared_students_in_overlapping_sections(self):
        proxy = ProxyObjective(self.instance)
        X = solve(self.instance, SolveOptions(seed=1, iterations_per_search=1)).solution.X
        expected = 0.0
        blocks = {}
        for (c, p, _l, b), v in X.items():
            if v:
                blocks.setdefault((c, p), set()).add(b)
        courses = self.instance.courses
        for i in range(len(courses)):
            for j in range(i + 1, len(courses)):
                n = sum(1 for s in self.instance.students
                        if i in s.course_indices and j in s.course_indices)
                if not n:
                    continue
                pairs = [(p, q) for p in range(len(courses[i].sections))
                         for q in range(len(courses[j].sections))]
                overlap = sum(1 for p, q in pairs if blocks[i, p] & blocks[j, q])
                expected += n * overlap / len(pairs)
        self.assertAlmostEqual(proxy.evaluate(Solution(X, {})), expected)

    def test_warm_start_and_component_trace(self):
        first = solve(self.instance, SolveOptions(seed=2, alpha=0.25, iterations_per_search=10))
        start = tuple(_active(first.solution))
        warm = solve(self.instance, SolveOptions(seed=2, alpha=1.0, iterations_per_search=10,
                                                 initial_timetable=start, trace=True,
                                                 trace_components=True), collect_metrics=True)
        self.assertEqual(_active(warm.initial_solution), list(start))
        self.assertIn("best_R_dis", warm.trace[-1])
        with self.assertRaises(ValueError):
            solve(self.instance, SolveOptions(seed=2, initial_timetable=start[:-1]))

    def test_larger_neighborhood_is_allowed_up_to_200(self):
        solve(self.instance, SolveOptions(seed=2, iterations_per_search=2, neighborhood_size=200))
        with self.assertRaises(ValueError):
            solve(self.instance, SolveOptions(seed=2, neighborhood_size=201))


if __name__ == "__main__":
    unittest.main()
