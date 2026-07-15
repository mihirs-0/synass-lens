"""Matrix-free local curvature diagnostics for Gate 0."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import torch


def _trainable(parameters: Iterable[torch.nn.Parameter]) -> tuple[torch.nn.Parameter, ...]:
    result = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not result:
        raise ValueError("no trainable parameters")
    return result


def adam_diagonal(
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor, ...]:
    by_parameter = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    diagonals = []
    for parameter in parameters:
        group = by_parameter[id(parameter)]
        state = optimizer.state.get(parameter, {})
        second = state.get("exp_avg_sq", torch.zeros_like(parameter))
        step = state.get("step", 0)
        step_value = float(step.item() if isinstance(step, torch.Tensor) else step)
        beta2 = group["betas"][1]
        correction = 1.0 - beta2 ** max(step_value, 1.0)
        v_hat = second / correction
        diagonals.append(1.0 / (v_hat.sqrt() + group["eps"]))
    return tuple(diagonals)


def capability_preconditioned_curvature(
    training_loss: torch.Tensor,
    capability_score: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    """Compute (g^T D H D g) / (g^T D g) without forming H."""
    params = _trainable(parameters)
    capability_gradient = torch.autograd.grad(
        capability_score, params, retain_graph=True, allow_unused=True
    )
    capability_gradient = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(params, capability_gradient)
    )
    diagonal = adam_diagonal(optimizer, params)
    direction = tuple(d * g for d, g in zip(diagonal, capability_gradient))
    loss_gradient = torch.autograd.grad(training_loss, params, create_graph=True)
    directional_derivative = sum(
        (gradient * vector).sum() for gradient, vector in zip(loss_gradient, direction)
    )
    hessian_direction = torch.autograd.grad(directional_derivative, params, retain_graph=True)
    numerator = sum(
        (vector * h_vector).sum() for vector, h_vector in zip(direction, hessian_direction)
    )
    denominator = sum(
        (gradient * vector).sum()
        for gradient, vector in zip(capability_gradient, direction)
    )
    if denominator.abs() <= torch.finfo(denominator.dtype).eps:
        raise ValueError("capability gradient is zero in the Adam metric")
    return numerator / denominator


def largest_preconditioned_curvature(
    loss_closure: Callable[[], torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    iterations: int = 30,
    seed: int = 0,
) -> torch.Tensor:
    """Power iteration on D^(1/2) H D^(1/2)."""
    params = _trainable(parameters)
    diagonal_sqrt = tuple(value.sqrt() for value in adam_diagonal(optimizer, params))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vector = tuple(
        torch.randn(parameter.shape, generator=generator, device="cpu", dtype=parameter.dtype).to(
            parameter.device
        )
        for parameter in params
    )
    norm = sum((value * value).sum() for value in vector).sqrt()
    vector = tuple(value / norm for value in vector)
    eigenvalue = torch.zeros((), device=params[0].device)
    for _ in range(iterations):
        loss = loss_closure()
        gradient = torch.autograd.grad(loss, params, create_graph=True)
        physical = tuple(scale * value for scale, value in zip(diagonal_sqrt, vector))
        directional = sum((g * value).sum() for g, value in zip(gradient, physical))
        hvp = torch.autograd.grad(directional, params)
        transformed = tuple(scale * value for scale, value in zip(diagonal_sqrt, hvp))
        eigenvalue = sum(
            (old * new).sum() for old, new in zip(vector, transformed)
        ).detach()
        norm = sum((value * value).sum() for value in transformed).sqrt().detach()
        if norm == 0:
            return torch.zeros_like(eigenvalue)
        vector = tuple((value / norm).detach() for value in transformed)
    return eigenvalue
