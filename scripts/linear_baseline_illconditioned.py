#!/usr/bin/env python
"""
Ill-conditioned linear regression baseline for inverse task.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_orthogonal_matrix(d: int, generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn(d, d, generator=generator)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q


def _build_transform(
    d: int,
    s_max: float,
    s_min: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = _make_orthogonal_matrix(d, generator)
    v = _make_orthogonal_matrix(d, generator)
    singular_values = torch.logspace(
        math.log10(s_max),
        math.log10(s_min),
        steps=d,
    )
    s = torch.diag(singular_values)
    m = u @ s @ v.T
    m_inv = v @ torch.diag(1.0 / singular_values) @ u.T
    return m, m_inv, v


def _sample_batch(
    batch_size: int,
    d: int,
    m: torch.Tensor,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, d, generator=generator)
    y = x @ m
    return x, y


def _component_row_norms(error_matrix: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    projected = v.T @ error_matrix
    return torch.sum(projected ** 2, dim=1)


def _train(
    d: int,
    batch_size: int,
    steps: int,
    learning_rate: float,
    eval_every: int,
    seed: int,
    s_max: float,
    s_min: float,
) -> Dict[str, List[float]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    m, m_inv, v = _build_transform(
        d=d,
        s_max=s_max,
        s_min=s_min,
        generator=generator,
    )

    model = nn.Linear(d, d, bias=False)
    nn.init.xavier_uniform_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    results = {
        "steps": [],
        "total_mse": [],
        "component_steps": [],
        "component_losses": {
            "fast_component": [],
            "mid_component": [],
            "slow_component": [],
        },
    }

    debug_steps = {0, 100, 1000}

    for step in range(steps + 1):
        if step > 0:
            x_train, y_train = _sample_batch(batch_size, d, m, generator)
            optimizer.zero_grad()
            preds = model(y_train)
            loss = F.mse_loss(preds, x_train)
            loss.backward()
            optimizer.step()

        x_eval, y_eval = _sample_batch(batch_size, d, m, generator)
        with torch.no_grad():
            preds = model(y_eval)
            mse = F.mse_loss(preds, x_eval).item()

        results["steps"].append(step)
        results["total_mse"].append(float(mse))

        if step % eval_every == 0:
            with torch.no_grad():
                w_eff = model.weight.T
                error_matrix = w_eff - m_inv
                results["component_steps"].append(step)
                row_norms = _component_row_norms(error_matrix, v)
                fast_val = float(row_norms[0].item())
                mid_val = float(row_norms[d // 2].item())
                slow_val = float(row_norms[-1].item())
                results["component_losses"]["fast_component"].append(fast_val)
                results["component_losses"]["mid_component"].append(mid_val)
                results["component_losses"]["slow_component"].append(slow_val)
                if step in debug_steps:
                    print(
                        "component_errors",
                        f"step={step}",
                        f"fast={fast_val:.6f}",
                        f"mid={mid_val:.6f}",
                        f"slow={slow_val:.6f}",
                    )

    results["config"] = {
        "d": d,
        "batch_size": batch_size,
        "steps": steps,
        "learning_rate": learning_rate,
        "eval_every": eval_every,
        "seed": seed,
        "condition_number": float(s_max / s_min),
        "s_max": s_max,
        "s_min": s_min,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ill-conditioned linear regression baseline")
    parser.add_argument("--d", type=int, default=100, help="Input/output dimension")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--steps", type=int, default=20000, help="Training steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--s-max", type=float, default=1.0, help="Largest singular value")
    parser.add_argument("--s-min", type=float, default=0.01, help="Smallest singular value")
    parser.add_argument("--eval-every", type=int, default=100, help="Component logging interval")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/linear_baseline/illconditioned_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    results = _train(
        d=args.d,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.lr,
        eval_every=args.eval_every,
        seed=args.seed,
        s_max=args.s_max,
        s_min=args.s_min,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
