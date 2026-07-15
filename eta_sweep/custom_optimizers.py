"""
Custom optimizer variants for the AdamW vs RMSProp decomposition experiment.

Two variants implemented:
  - AdamWNoBiasCorrection: standard AdamW formula but without the
    1/(1-β₁^t) and 1/(1-β₂^t) bias corrections.  Uses raw m_t and v_t.
  - RMSPropDecoupledBiasCorrected: RMSProp formula but with (a) Adam-style
    bias correction on v_t and (b) decoupled weight decay (direct
    shrinkage rather than L2 added to the gradient).

These isolate the specific implementation details that distinguish
PyTorch's AdamW from PyTorch's RMSProp, beyond the first-moment EMA.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class AdamWNoBiasCorrection(Optimizer):
    """AdamW with bias correction REMOVED.  Otherwise identical formula:
        m_t = β1 * m_{t-1} + (1-β1) * g_t
        v_t = β2 * v_{t-1} + (1-β2) * g_t²
        θ ← (1 - lr*wd) * θ - lr * m_t / (sqrt(v_t) + ε)        [no bias correction]
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # NO bias correction
                denom = v.sqrt().add_(eps)
                # Decoupled weight decay (AdamW-style)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m, denom, value=-lr)
        return loss


class RMSPropDecoupledBiasCorrected(Optimizer):
    """RMSProp with (a) Adam-style bias correction on v_t and (b) decoupled
    weight decay (direct shrinkage, not L2 added to gradient).
        v_t = α * v_{t-1} + (1-α) * g_t²
        v̂_t = v_t / (1 - α^t)                                [bias correction]
        θ ← (1 - lr*wd) * θ - lr * g_t / sqrt(v̂_t + ε)       [decoupled wd]
        (RMSProp-style ε placement: inside the sqrt)
    """

    def __init__(self, params, lr=1e-3, alpha=0.999, eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                v = state["exp_avg_sq"]
                v.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                bc = 1.0 - alpha ** t
                v_hat = v / bc
                denom = (v_hat + eps).sqrt()
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(grad, denom, value=-lr)
        return loss


class RMSPropBiasCorrected(Optimizer):
    """RMSProp + bias correction ONLY.  L2 (coupled) weight decay preserved.

    Tests: does Adam-style bias correction alone close the AdamW vs RMSProp gap?
        g_eff = g + wd * θ                                    [L2 / coupled wd]
        v_t = α * v_{t-1} + (1-α) * g_eff²
        v̂_t = v_t / (1 - α^t)                                [bias correction]
        θ ← θ - lr * g_eff / sqrt(v̂_t + ε)
        (RMSProp-style ε placement: inside the sqrt)
    """

    def __init__(self, params, lr=1e-3, alpha=0.999, eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                # L2 (coupled) weight decay added to gradient
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                v = state["exp_avg_sq"]
                v.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                bc = 1.0 - alpha ** t
                v_hat = v / bc
                denom = (v_hat + eps).sqrt()
                p.addcdiv_(grad, denom, value=-lr)
        return loss


class RMSPropDecoupledOnly(Optimizer):
    """RMSProp + decoupled weight decay ONLY.  No bias correction.

    Tests: does decoupled weight decay alone close the gap, even without BC?
        v_t = α * v_{t-1} + (1-α) * g_t²
        θ ← (1 - lr*wd) * θ - lr * g_t / sqrt(v_t + ε)        [decoupled wd]
        (RMSProp-style ε placement: inside the sqrt; no bias correction)
    """

    def __init__(self, params, lr=1e-3, alpha=0.999, eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                v = state["exp_avg_sq"]
                v.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                # NO bias correction
                denom = (v + eps).sqrt()
                # Decoupled weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(grad, denom, value=-lr)
        return loss


class AdamCoupledWD(Optimizer):
    """Adam (NOT AdamW): coupled L2 weight decay, with bias correction and
    AdamW-style ε placement.  This is the "Adam" variant that lacks the
    decoupled weight decay distinction.
        g_eff = g + wd * θ                                    [L2 / coupled wd]
        m_t = β1*m + (1-β1)*g_eff;  v_t = β2*v + (1-β2)*g_eff²
        m̂ = m/(1-β1^t);  v̂ = v/(1-β2^t)                       [bias correction]
        θ ← θ - lr * m̂ / (sqrt(v̂) + ε)                        [AdamW-style ε]
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                # Coupled L2 weight decay
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                m_hat = m / (1 - beta1 ** t) if beta1 != 0 else m
                v_hat = v / (1 - beta2 ** t)
                denom = v_hat.sqrt().add_(eps)  # AdamW-style: ε outside sqrt
                p.addcdiv_(m_hat, denom, value=-lr)
        return loss


class AdamWRMSPropEps(Optimizer):
    """AdamW (decoupled wd, bias correction) BUT with RMSProp-style ε
    placement (ε inside the sqrt).
        m_t = β1*m + (1-β1)*g;  v_t = β2*v + (1-β2)*g²
        m̂ = m/(1-β1^t);  v̂ = v/(1-β2^t)                       [bias correction]
        θ ← (1 - lr*wd) * θ - lr * m̂ / sqrt(v̂ + ε)            [RMSProp ε placement]
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                m_hat = m / (1 - beta1 ** t) if beta1 != 0 else m
                v_hat = v / (1 - beta2 ** t)
                # RMSProp-style ε placement: inside the sqrt
                denom = (v_hat + eps).sqrt()
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m_hat, denom, value=-lr)
        return loss


class RMSPropAdamWEps(Optimizer):
    """RMSProp (no BC, L2 wd) BUT with AdamW-style ε placement (ε outside the sqrt).
        g_eff = g + wd * θ                                    [L2 / coupled wd]
        v_t = α * v_{t-1} + (1-α) * g_eff²                    [no bias correction]
        θ ← θ - lr * g_eff / (sqrt(v_t) + ε)                  [AdamW-style ε]
    """

    def __init__(self, params, lr=1e-3, alpha=0.999, eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                v = state["exp_avg_sq"]
                v.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                # AdamW-style: ε outside sqrt
                denom = v.sqrt().add_(eps)
                p.addcdiv_(grad, denom, value=-lr)
        return loss
