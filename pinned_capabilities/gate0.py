"""Gate 0 controls and boundary-analysis primitives."""

from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np
import torch

from .local_stability import AugmentedAdamWLinearization


def deep_linear_jacobian(learning_rate: float, target: float = 1.0) -> torch.Tensor:
    """Jacobian of one GD step at the balanced solution of a depth-2 scalar net."""
    root = math.sqrt(target)
    point = torch.tensor([root, root], dtype=torch.float64, requires_grad=True)

    def update(weights: torch.Tensor) -> torch.Tensor:
        loss = 0.5 * (weights[0] * weights[1] - target) ** 2
        gradient = torch.autograd.grad(loss, weights, create_graph=True)[0]
        return weights - learning_rate * gradient

    return torch.autograd.functional.jacobian(update, point).detach()


def _complex_diagnostic(value: complex) -> Dict[str, float]:
    """Return one JSON-compatible complex eigenvalue diagnostic."""

    return {
        "real": float(value.real),
        "imag": float(value.imag),
        "magnitude": float(abs(value)),
    }


def _augmented_adam_numerical_control() -> Dict[str, object]:
    """Exercise the production augmented-Adam linearization on a tiny system.

    The fixed diagonal quadratic has an explicit gradient, so its full AdamW
    state map can be finite-differenced independently of the HVP-based
    :class:`AugmentedAdamWLinearization`.  The system is deliberately only two
    parameters (six augmented coordinates), making both dense comparison and
    exact dense eigendecomposition deterministic and inexpensive.
    """

    curvature = torch.tensor([2.0, 5.0], dtype=torch.float64)
    parameters = (
        torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64)),
        torch.nn.Parameter(torch.tensor(-0.3, dtype=torch.float64)),
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    beta1, beta2 = optimizer.param_groups[0]["betas"]
    for parameter in parameters:
        optimizer.state[parameter] = {
            "step": torch.tensor(1.0, dtype=torch.float64),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.full_like(parameter, 1.0 - beta2),
        }

    loss = 0.5 * sum(
        coefficient * parameter.square()
        for coefficient, parameter in zip(curvature, parameters)
    )
    linearization = AugmentedAdamWLinearization(loss, parameters, optimizer)
    dense_hvp = linearization.dense_jacobian()

    theta = torch.stack([parameter.detach() for parameter in parameters])
    moment = torch.stack(
        [optimizer.state[parameter]["exp_avg"] for parameter in parameters]
    )
    second = torch.stack(
        [optimizer.state[parameter]["exp_avg_sq"] for parameter in parameters]
    )
    base_state = torch.cat((theta, moment, second))
    group = optimizer.param_groups[0]
    learning_rate = float(group["lr"])
    epsilon = float(group["eps"])
    weight_decay = float(group["weight_decay"])
    step_before_update = float(optimizer.state[parameters[0]]["step"].item())
    step_after_update = step_before_update + 1.0

    def state_map(state: torch.Tensor) -> torch.Tensor:
        current_theta = state[:2]
        current_moment = state[2:4]
        current_second = state[4:]
        gradient = curvature * current_theta
        next_moment = beta1 * current_moment + (1.0 - beta1) * gradient
        next_second = beta2 * current_second + (1.0 - beta2) * gradient.square()
        corrected_moment = next_moment / (1.0 - beta1**step_after_update)
        corrected_second = next_second / (1.0 - beta2**step_after_update)
        next_theta = (1.0 - learning_rate * weight_decay) * current_theta
        next_theta = next_theta - learning_rate * corrected_moment / (
            corrected_second.sqrt() + epsilon
        )
        return torch.cat((next_theta, next_moment, next_second))

    # This scale keeps truncation error below 1e-7 even in the second-moment
    # coordinates while remaining comfortably above float64 roundoff.
    finite_difference_epsilon = 3e-7
    finite_difference_columns = []
    for index in range(linearization.dimension):
        perturbation = torch.zeros_like(base_state)
        perturbation[index] = finite_difference_epsilon
        finite_difference_columns.append(
            (
                state_map(base_state + perturbation)
                - state_map(base_state - perturbation)
            )
            / (2.0 * finite_difference_epsilon)
        )
    dense_finite_difference = torch.stack(
        finite_difference_columns, dim=1
    ).detach().cpu().numpy()
    difference = dense_hvp - dense_finite_difference

    eigenvalue_count = 3
    eigenvalues, eigenvectors = linearization.dominant_eigenpairs(
        count=eigenvalue_count,
        tolerance=1e-12,
        max_iterations=200,
        seed=0,
    )
    residuals = linearization.eigenpair_residuals(eigenvalues, eigenvectors)

    finite_norm = float(np.linalg.norm(dense_finite_difference))
    relative_frobenius_error = float(
        np.linalg.norm(difference) / max(finite_norm, np.finfo(np.float64).tiny)
    )
    return {
        "system": "two_parameter_diagonal_quadratic_adamw",
        "curvature_diagonal": [float(value) for value in curvature.tolist()],
        "initial_state": {
            "theta": [float(value) for value in theta.tolist()],
            "exp_avg": [float(value) for value in moment.tolist()],
            "exp_avg_sq": [float(value) for value in second.tolist()],
            "step": step_before_update,
        },
        "optimizer": {
            "learning_rate": learning_rate,
            "betas": [float(beta1), float(beta2)],
            "epsilon": epsilon,
            "weight_decay": weight_decay,
        },
        "augmented_dimension": linearization.dimension,
        "finite_difference_epsilon": finite_difference_epsilon,
        "dense_hvp_jacobian": dense_hvp.tolist(),
        "centered_finite_difference_jacobian": dense_finite_difference.tolist(),
        "jacobian_max_absolute_error": float(np.max(np.abs(difference))),
        "jacobian_relative_frobenius_error": relative_frobenius_error,
        "dominant_eigenvalue_count": eigenvalue_count,
        "dominant_eigenvalues": [
            _complex_diagnostic(complex(value)) for value in eigenvalues
        ],
        "eigenpair_relative_residuals": [float(value) for value in residuals],
        "maximum_eigenpair_relative_residual": float(np.max(residuals)),
    }


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
        "augmented_adam_control": _augmented_adam_numerical_control(),
    }
