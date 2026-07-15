"""Gate 0 controls and boundary-analysis primitives."""

from __future__ import annotations

import math
from typing import Dict, Iterable

import torch


def deep_linear_jacobian(learning_rate: float, target: float = 1.0) -> torch.Tensor:
    """Jacobian of one GD step at the balanced solution of a depth-2 scalar net."""
    root = math.sqrt(target)
    point = torch.tensor([root, root], dtype=torch.float64, requires_grad=True)

    def update(weights: torch.Tensor) -> torch.Tensor:
        loss = 0.5 * (weights[0] * weights[1] - target) ** 2
        gradient = torch.autograd.grad(loss, weights, create_graph=True)[0]
        return weights - learning_rate * gradient

    return torch.autograd.functional.jacobian(update, point).detach()


def deep_linear_control(
    learning_rates: Iterable[float],
    *,
    target: float = 1.0,
) -> Dict[str, object]:
    """Calibrate the numerical local-stability pipeline against closed form."""
    rows = []
    for learning_rate in learning_rates:
        jacobian = deep_linear_jacobian(float(learning_rate), target)
        eigenvalues = torch.linalg.eigvals(jacobian).real
        nontrivial = float(eigenvalues[torch.argmin(eigenvalues)].item())
        rows.append(
            {
                "learning_rate": float(learning_rate),
                "jacobian": jacobian.tolist(),
                "eigenvalues": sorted(float(value) for value in eigenvalues.tolist()),
                "nontrivial_multiplier": nontrivial,
                "locally_stable": abs(nontrivial) < 1.0,
            }
        )
    analytic = 1.0 / target
    return {
        "system": "depth_2_scalar_linear_network",
        "target": target,
        "hessian_lambda_max": 2.0 * target,
        "analytic_critical_learning_rate": analytic,
        "rows": rows,
    }
