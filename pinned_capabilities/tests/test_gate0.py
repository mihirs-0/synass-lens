import math
import unittest

import torch

from pinned_capabilities.gate0 import deep_linear_control
from pinned_capabilities.local_stability import (
    capability_preconditioned_curvature,
    largest_preconditioned_curvature,
)


class DeepLinearControlTests(unittest.TestCase):
    def test_known_stability_boundary(self) -> None:
        result = deep_linear_control((0.9, 1.0, 1.1), target=1.0)
        self.assertEqual(result["analytic_critical_learning_rate"], 1.0)
        rows = result["rows"]
        self.assertTrue(rows[0]["locally_stable"])
        self.assertFalse(rows[1]["locally_stable"])
        self.assertFalse(rows[2]["locally_stable"])
        self.assertAlmostEqual(rows[0]["nontrivial_multiplier"], -0.8, places=10)


class CurvatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.y = torch.nn.Parameter(torch.tensor(-0.3, dtype=torch.float64))
        self.optimizer = torch.optim.AdamW([self.x, self.y], lr=0.1, eps=1e-8)
        beta2 = self.optimizer.param_groups[0]["betas"][1]
        for parameter in (self.x, self.y):
            self.optimizer.state[parameter] = {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.full_like(parameter, 1.0 - beta2),
            }

    def loss(self) -> torch.Tensor:
        return 0.5 * (2.0 * self.x.square() + 5.0 * self.y.square())

    def test_capability_direction_and_largest_mode(self) -> None:
        score = self.x
        observed = capability_preconditioned_curvature(
            self.loss(), score, (self.x, self.y), self.optimizer
        )
        expected_scale = 1.0 / (1.0 + 1e-8)
        self.assertTrue(math.isclose(observed.item(), 2.0 * expected_scale, rel_tol=1e-7))
        largest = largest_preconditioned_curvature(
            self.loss, (self.x, self.y), self.optimizer, iterations=40, seed=3
        )
        self.assertTrue(math.isclose(largest.item(), 5.0 * expected_scale, rel_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
