"""Bounded distinct attempts, excluding fixed lectures before sampling."""

import random
import unittest
from unittest.mock import patch

from uctp.constraints import InstitutionalConstraints
from uctp.domain import Solution
from uctp.instance_io import parse_instance
from uctp.objective import SearchObjective
from uctp.sectioning import StudentSectioning
from uctp.tabu import TabuSearch, _Candidate


def fixture(movable):
    data = dict(
        courses=[
            dict(
                id=f"C{i}",
                sections=[
                    dict(
                        id="S",
                        lectures=[
                            dict(
                                id="L",
                                teachers=[],
                                **({} if i < movable else {"preassigned_block": f"Mon:{i + 1}"}),
                            )
                        ],
                    )
                ],
            )
            for i in range(3)
        ],
        teachers=[],
        students=[],
        curricular_levels=[],
    )
    instance = parse_instance(data, max_daily_load=8, max_block_load=15)
    state = Solution({(i, 0, 0, i): 1 for i in range(3)})
    objective = SearchObjective(instance, alpha=1, cv_tolerance=0.1, idle_tolerance=4)
    search = TabuSearch(
        instance,
        InstitutionalConstraints(instance),
        objective,
        StudentSectioning(instance, objective),
        embedded=False,
        neighborhood_size=50,
        max_stagnant_iterations=200,
        time_limit_seconds=None,
    )
    return state, search


class NeighborhoodSamplingTests(unittest.TestCase):
    def test_fixed_lectures_are_excluded_and_moves_are_distinct(self):
        state, search = fixture(2)
        random.seed(703)
        with patch.object(search.constraints, "is_feasible", return_value=True):
            candidates = search._generate_neighborhood(_Candidate(state.copy(), None, 0), state)
        self.assertEqual(len(candidates), 50)
        self.assertEqual(len({c.move for c in candidates}), 50)
        self.assertTrue(all(c.move[0] in (0, 1) for c in candidates))
        self.assertTrue(all(c.move[3] != c.move[4] for c in candidates))
        self.assertTrue(all(c.solution.X[(2, 0, 0, 2)] == 1 for c in candidates))

    def test_infeasible_moves_consume_the_fifty_attempt_budget(self):
        state, search = fixture(2)
        attempted = []
        original_X = state.X.copy()

        def reject(X):
            attempted.append(tuple(k for k, v in X.items() if v and not original_X.get(k)))
            return False

        with patch.object(search.constraints, "is_feasible", side_effect=reject):
            candidates = search._generate_neighborhood(_Candidate(state.copy(), None, 0), state)
        self.assertEqual(candidates, [])
        self.assertEqual(len(attempted), 50)
        self.assertEqual(len(set(attempted)), 50)

    def test_pool_smaller_than_fifty_is_exhausted_once(self):
        state, search = fixture(1)
        with patch.object(search.constraints, "is_feasible", return_value=True):
            candidates = search._generate_neighborhood(_Candidate(state.copy(), None, 0), state)
        self.assertEqual(len(candidates), 34)
        self.assertEqual(len({c.move for c in candidates}), 34)

    def test_all_fixed_lectures_produce_no_attempts(self):
        state, search = fixture(0)
        with patch.object(search.constraints, "is_feasible") as feasibility:
            self.assertEqual(
                search._generate_neighborhood(_Candidate(state.copy(), None, 0), state), []
            )
            feasibility.assert_not_called()
