"""The Numba SSP kernel must reproduce the pure-Python SSP exactly."""

import json
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from uctp import sectioning
from uctp.constraints import InstitutionalConstraints
from uctp.construction import InitialStrategy, InitialTimetabling
from uctp.domain import Solution
from uctp.instance_io import load_instance, parse_instance
from uctp.objective import SearchObjective

ROOT = Path(__file__).resolve().parents[1]


def _instances():
    data = json.loads((ROOT / "examples" / "small.json").read_text(encoding="utf-8"))
    yield "small", parse_instance(data, max_daily_load=4, max_block_load=15)
    real = ROOT / "datasets" / "Real" / "real.json"
    if real.is_file():
        yield "real", load_instance(
            real, input_format="instance", max_daily_load=None, max_block_load=15
        )


@unittest.skipIf(sectioning._ssp_kernel is None, "Numba not installed")
class KernelEquivalenceTests(unittest.TestCase):
    def test_same_assignment_objective_and_rng_state(self):
        for name, instance in _instances():
            constraints = InstitutionalConstraints(instance)
            random.seed(11)
            X = InitialTimetabling(
                instance, constraints, strategy=InitialStrategy.COMPATIBILITY_MATRIX,
                max_retries=500, beta=0.25, cv_tolerance=0.10,
            ).solve().X
            for local_search in (True, False):
                for alpha in (0.0, 0.25, 1.0, 4.0):
                    objective = SearchObjective(
                        instance, alpha=alpha, cv_tolerance=0.10, idle_tolerance=4
                    )
                    ssp = sectioning.StudentSectioning(
                        instance, objective, local_search=local_search
                    )
                    for seed in range(3):
                        with self.subTest(instance=name, ls=local_search, alpha=alpha, seed=seed):
                            random.seed(seed)
                            fast = Solution(X.copy(), {})
                            fast_value = ssp.solve(fast)
                            fast_state = random.getstate()
                            random.seed(seed)
                            slow = Solution(X.copy(), {})
                            with patch.object(sectioning, "kernel_enabled", return_value=False):
                                slow_value = ssp.solve(slow)
                            self.assertEqual(fast.Y, slow.Y)
                            self.assertEqual(fast_value, slow_value)
                            self.assertEqual(fast_state, random.getstate())
                            # Fixed-Y evaluation (Decoupled) must also match.
                            with patch.dict("os.environ", {"UCTP_PURE_PYTHON": "1"}):
                                reference = objective.evaluate(slow)
                            self.assertEqual(objective.evaluate(slow), reference)


if __name__ == "__main__":
    unittest.main()
