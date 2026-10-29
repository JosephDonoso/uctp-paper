"""Decoupled endpoints must use the SSP solved on the final timetable."""

import json
import random
import unittest
from pathlib import Path

from uctp.domain import Solution
from uctp.instance_io import parse_instance
from uctp.objective import SearchObjective
from uctp.planner import Approach, SolveOptions, solve
from uctp.sectioning import StudentSectioning

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "small.json"


class FinalSspTests(unittest.TestCase):
    def setUp(self):
        data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        self.instance = parse_instance(data, max_daily_load=4, max_block_load=15)

    def _options(self, **changes):
        base = dict(
            approach=Approach.DECOUPLED, iterations_per_search=10, search_blocks=3, seed=7
        )
        base.update(changes)
        return SolveOptions(**base)

    def test_decoupled_keeps_pre_ssp_solution_and_resolves_final_y(self):
        result = solve(self.instance, self._options())
        self.assertIsNotNone(result.stale_solution)
        self.assertEqual(result.solution.X, result.stale_solution.X)
        requested = sum(len(student.course_indices) for student in self.instance.students)
        self.assertEqual(sum(result.solution.Y.values()), requested)
        # Conflicts from the SSP depend only on X (no capacities), so the final
        # Y must match a fresh SSP solve in its per-student conflict count.
        objective = SearchObjective(self.instance, alpha=1, cv_tolerance=0.10, idle_tolerance=4)
        fresh = Solution(result.solution.X.copy(), {})
        random.seed(123)
        StudentSectioning(self.instance, objective).solve(fresh)
        self.assertEqual(self._conflicts(result.solution), self._conflicts(fresh))

    def test_final_ssp_can_be_disabled(self):
        result = solve(self.instance, self._options(final_ssp=False))
        self.assertIsNone(result.stale_solution)

    def test_embedded_is_unchanged(self):
        result = solve(
            self.instance, SolveOptions(approach=Approach.EMBEDDED, iterations_per_search=10, seed=7)
        )
        self.assertIsNone(result.stale_solution)

    def _conflicts(self, solution):
        from uctp.metrics import measure_solution

        return measure_solution(self.instance, solution, alpha=1)["metrics"]["conflicts"]


if __name__ == "__main__":
    unittest.main()
