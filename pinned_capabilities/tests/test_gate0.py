import math
import unittest

import numpy as np
import torch

from pinned_capabilities.gate0 import deep_linear_control
from pinned_capabilities.local_stability import (
    AugmentedAdamWLinearization,
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

    def test_augmented_adam_jacobian_matches_finite_difference(self) -> None:
        linearization = AugmentedAdamWLinearization(
            self.loss(), (self.x, self.y), self.optimizer
        )
        observed = linearization.dense_jacobian()
        theta = torch.tensor([self.x.item(), self.y.item()], dtype=torch.float64)
        moment = torch.stack(
            [self.optimizer.state[p]["exp_avg"] for p in (self.x, self.y)]
        )
        second = torch.stack(
            [self.optimizer.state[p]["exp_avg_sq"] for p in (self.x, self.y)]
        )
        base = torch.cat((theta, moment, second))
        beta1, beta2 = self.optimizer.param_groups[0]["betas"]
        learning_rate = self.optimizer.param_groups[0]["lr"]
        weight_decay = self.optimizer.param_groups[0]["weight_decay"]
        eps = self.optimizer.param_groups[0]["eps"]

        def state_map(value):
            current_theta, current_m, current_v = value[:2], value[2:4], value[4:]
            gradient = torch.tensor([2.0, 5.0], dtype=torch.float64) * current_theta
            next_m = beta1 * current_m + (1 - beta1) * gradient
            next_v = beta2 * current_v + (1 - beta2) * gradient.square()
            m_hat = next_m / (1 - beta1**2)
            v_hat = next_v / (1 - beta2**2)
            next_theta = (1 - learning_rate * weight_decay) * current_theta
            next_theta = next_theta - learning_rate * m_hat / (v_hat.sqrt() + eps)
            return torch.cat((next_theta, next_m, next_v))

        epsilon = 1e-6
        numerical = []
        for index in range(6):
            direction = torch.zeros(6, dtype=torch.float64)
            direction[index] = epsilon
            numerical.append(((state_map(base + direction) - state_map(base - direction)) / (2 * epsilon)).numpy())
        numerical = torch.from_numpy(np.column_stack(numerical))
        torch.testing.assert_close(torch.from_numpy(observed), numerical, rtol=2e-5, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
