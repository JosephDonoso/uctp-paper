"""Checkpoints reproduce separate runs with smaller iteration budgets exactly."""

import json
import unittest
from pathlib import Path

from uctp.instance_io import load_instance
from uctp.metrics import measure_solution
from uctp.planner import SolveOptions, solve
from uctp.reporting import execute_run

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "small.json"


def _active(solution):
    return sorted(key for key, value in solution.X.items() if value == 1)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_equals_separate_run_and_does_not_change_trajectory(self):
        instance = load_instance(EXAMPLE, input_format="instance", max_daily_load=None, max_block_load=15)
        for alpha in (0.5, 1.0, 2.0):
            base = dict(alpha=alpha, seed=703, max_retries=50)
            full = solve(
                instance,
                SolveOptions(**base, iterations_per_search=40, total_iteration_limit=40,
                             max_stagnant_iterations=40),
            )
            checked = solve(
                instance,
                SolveOptions(**base, iterations_per_search=40, total_iteration_limit=40,
                             max_stagnant_iterations=40, checkpoints=(5, 10, 20)),
            )
            self.assertEqual(_active(full.solution), _active(checked.solution))
            self.assertEqual(full.solution.Y, checked.solution.Y)
            recorded = {item["iterations"]: item["solution"] for item in checked.checkpoints}
            for budget in (5, 10, 20):
                separate = solve(
                    instance,
                    SolveOptions(**base, iterations_per_search=budget, total_iteration_limit=budget,
                                 max_stagnant_iterations=budget),
                )
                if budget in recorded:
                    self.assertEqual(_active(separate.solution), _active(recorded[budget]))
                    self.assertEqual(separate.solution.Y, recorded[budget].Y)
                    self.assertEqual(
                        measure_solution(instance, separate.solution, alpha=alpha)["metrics"],
                        measure_solution(instance, recorded[budget], alpha=alpha)["metrics"],
                    )

    def test_record_contains_checkpoint_metrics(self):
        options = SolveOptions(seed=703, max_retries=50, iterations_per_search=12,
                               total_iteration_limit=12, max_stagnant_iterations=12,
                               checkpoints=(3, 6))
        document = execute_run(EXAMPLE, options, include_solution=False)
        self.assertEqual([item["iteration_budget"] for item in document["checkpoints"]], [3, 6])
        self.assertTrue(all("F_ref" in item["metrics"] for item in document["checkpoints"]))
        json.dumps(document, allow_nan=False)

    def test_invalid_checkpoints_are_rejected(self):
        instance = load_instance(EXAMPLE, input_format="instance", max_daily_load=None, max_block_load=15)
        for points in [(10,), (5, 3), (0,)]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                solve(instance, SolveOptions(seed=1, iterations_per_search=10,
                                             total_iteration_limit=10, checkpoints=points))


if __name__ == "__main__":
    unittest.main()
