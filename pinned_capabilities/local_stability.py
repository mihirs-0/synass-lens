"""Matrix-free local curvature diagnostics for Gate 0."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigs


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


class AugmentedAdamWLinearization:
    """Matrix-free Jacobian of one AdamW step on ``(theta, m, v)``.

    The object retains one autograd graph for repeated Hessian-vector products.
    It is intended for a frozen checkpoint and a fixed registered training
    batch; mutating model parameters before measurement completes is invalid.
    """

    def __init__(
        self,
        training_loss: torch.Tensor,
        parameters: Iterable[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.parameters = _trainable(parameters)
        self.optimizer = optimizer
        self.gradients = torch.autograd.grad(training_loss, self.parameters, create_graph=True)
        self.shapes = tuple(parameter.shape for parameter in self.parameters)
        self.sizes = tuple(parameter.numel() for parameter in self.parameters)
        self.parameter_count = sum(self.sizes)
        self.dimension = 3 * self.parameter_count
        self.groups = {
            id(parameter): group
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        for parameter in self.parameters:
            group = self.groups[id(parameter)]
            if group.get("amsgrad", False):
                raise NotImplementedError("augmented linearization does not support AMSGrad")
            state = optimizer.state.get(parameter, {})
            if "exp_avg" not in state or "exp_avg_sq" not in state or "step" not in state:
                raise ValueError("AdamW state must be initialized before local linearization")

    def _split_block(self, block: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = []
        cursor = 0
        for shape, size, parameter in zip(self.shapes, self.sizes, self.parameters):
            values.append(block[cursor : cursor + size].reshape(shape).to(parameter.device, parameter.dtype))
            cursor += size
        return tuple(values)

    def unpack(self, vector: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], ...]:
        if vector.numel() != self.dimension:
            raise ValueError(f"expected augmented vector of length {self.dimension}")
        n = self.parameter_count
        return (
            self._split_block(vector[:n]),
            self._split_block(vector[n : 2 * n]),
            self._split_block(vector[2 * n :]),
        )

    @staticmethod
    def _flatten(values: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat([value.reshape(-1) for value in values])

    def matvec_torch(self, vector: torch.Tensor) -> torch.Tensor:
        delta_theta, delta_m, delta_v = self.unpack(vector)
        directional_gradient = sum(
            (gradient * direction).sum()
            for gradient, direction in zip(self.gradients, delta_theta)
        )
        hessian_delta = torch.autograd.grad(
            directional_gradient, self.parameters, retain_graph=True
        )
        out_theta = []
        out_m = []
        out_v = []
        for parameter, gradient, h_delta, d_theta, d_m, d_v in zip(
            self.parameters,
            self.gradients,
            hessian_delta,
            delta_theta,
            delta_m,
            delta_v,
        ):
            group = self.groups[id(parameter)]
            state = self.optimizer.state[parameter]
            beta1, beta2 = group["betas"]
            step = state["step"]
            step_value = float(step.item() if isinstance(step, torch.Tensor) else step) + 1.0
            sign = -1.0 if group.get("maximize", False) else 1.0
            base_gradient = sign * gradient
            delta_gradient = sign * h_delta
            moment = state["exp_avg"]
            second = state["exp_avg_sq"]
            next_m = beta1 * moment + (1.0 - beta1) * base_gradient
            next_v = beta2 * second + (1.0 - beta2) * base_gradient.square()
            next_delta_m = beta1 * d_m + (1.0 - beta1) * delta_gradient
            next_delta_v = beta2 * d_v + 2.0 * (1.0 - beta2) * base_gradient * delta_gradient
            correction1 = 1.0 - beta1**step_value
            correction2 = 1.0 - beta2**step_value
            m_hat = next_m / correction1
            v_hat = next_v / correction2
            delta_m_hat = next_delta_m / correction1
            delta_v_hat = next_delta_v / correction2
            root = v_hat.sqrt()
            denominator = root + group["eps"]
            delta_ratio = delta_m_hat / denominator
            zero_root = root == 0
            if torch.any(zero_root & (m_hat != 0)):
                raise ValueError("invalid AdamW state: nonzero first moment with zero second moment")
            denominator_sensitivity = torch.where(
                zero_root,
                torch.zeros_like(root),
                m_hat / (2.0 * root * denominator.square()),
            )
            delta_ratio = delta_ratio - denominator_sensitivity * delta_v_hat
            learning_rate = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            out_theta.append((1.0 - learning_rate * weight_decay) * d_theta - learning_rate * delta_ratio)
            out_m.append(next_delta_m)
            out_v.append(next_delta_v)
        return torch.cat(
            (self._flatten(out_theta), self._flatten(out_m), self._flatten(out_v))
        )

    def matvec_numpy(self, vector: np.ndarray) -> np.ndarray:
        device = self.parameters[0].device
        value = torch.from_numpy(np.asarray(vector, dtype=np.float64)).to(device)
        return self.matvec_torch(value).detach().cpu().double().numpy()

    def dense_jacobian(self) -> np.ndarray:
        basis = np.eye(self.dimension, dtype=np.float64)
        return np.column_stack([self.matvec_numpy(basis[:, index]) for index in range(self.dimension)])

    def dominant_eigenvalues(
        self,
        *,
        count: int = 3,
        tolerance: float = 1e-3,
        max_iterations: int = 100,
        seed: int = 0,
    ) -> np.ndarray:
        if count <= 0:
            raise ValueError("eigenvalue count must be positive")
        if self.dimension <= max(8, count + 2):
            values = np.linalg.eigvals(self.dense_jacobian())
            return values[np.argsort(np.abs(values))[::-1]][:count]
        operator = LinearOperator(
            (self.dimension, self.dimension), matvec=self.matvec_numpy, dtype=np.float64
        )
        rng = np.random.default_rng(seed)
        values = eigs(
            operator,
            k=min(count, self.dimension - 2),
            which="LM",
            v0=rng.standard_normal(self.dimension),
            tol=tolerance,
            maxiter=max_iterations,
            return_eigenvectors=False,
        )
        return values[np.argsort(np.abs(values))[::-1]]
