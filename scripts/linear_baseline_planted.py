#!/usr/bin/env python
"""
Linear planted-structure baseline for selector usage.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_task(
    n_groups: int,
    k: int,
    d: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_embeddings = torch.randn(n_groups, d, generator=generator)
    selector_embeddings = torch.randn(k, d, generator=generator)
    candidates = torch.randn(n_groups, k, d, generator=generator)
    return group_embeddings, selector_embeddings, candidates


def _sample_batch(
    batch_size: int,
    group_embeddings: torch.Tensor,
    selector_embeddings: torch.Tensor,
    candidates: torch.Tensor,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_groups = group_embeddings.size(0)
    k = selector_embeddings.size(0)
    group_idx = torch.randint(0, n_groups, (batch_size,), generator=generator)
    selector_idx = torch.randint(0, k, (batch_size,), generator=generator)

    group_vecs = group_embeddings[group_idx]
    selector_vecs = selector_embeddings[selector_idx]
    inputs = torch.cat([group_vecs, selector_vecs], dim=1)
    targets = candidates[group_idx, selector_idx]

    return inputs, targets, group_idx, selector_idx


def _selector_gap(
    model: nn.Module,
    group_embeddings: torch.Tensor,
    selector_embeddings: torch.Tensor,
    candidates: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> Tuple[float, float, float]:
    inputs, targets, group_idx, selector_idx = _sample_batch(
        batch_size=batch_size,
        group_embeddings=group_embeddings,
        selector_embeddings=selector_embeddings,
        candidates=candidates,
        generator=generator,
    )
    with torch.no_grad():
        preds = model(inputs)
        mse_clean = F.mse_loss(preds, targets).item()

        perm = torch.randperm(selector_idx.size(0), generator=generator)
        shuffled_idx = selector_idx[perm]
        shuffled_selector = selector_embeddings[shuffled_idx]
        shuffled_inputs = torch.cat([group_embeddings[group_idx], shuffled_selector], dim=1)
        shuffled_preds = model(shuffled_inputs)
        mse_shuffled = F.mse_loss(shuffled_preds, targets).item()

    selector_gap = mse_shuffled - mse_clean
    return float(mse_clean), float(mse_shuffled), float(selector_gap)


def _train(
    d: int,
    k: int,
    n_groups: int,
    batch_size: int,
    steps: int,
    learning_rate: float,
    eval_every: int,
    eval_batch_size: int,
    seed: int,
) -> Dict[str, List[float]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    group_embeddings, selector_embeddings, candidates = _build_task(
        n_groups=n_groups,
        k=k,
        d=d,
        generator=generator,
    )

    model = nn.Linear(2 * d, d, bias=False)
    nn.init.xavier_uniform_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    results = {
        "steps": [],
        "total_mse": [],
        "eval_steps": [],
        "mse_clean": [],
        "mse_shuffled": [],
        "selector_gap": [],
    }

    for step in range(steps + 1):
        if step > 0:
            inputs, targets, _, _ = _sample_batch(
                batch_size=batch_size,
                group_embeddings=group_embeddings,
                selector_embeddings=selector_embeddings,
                candidates=candidates,
                generator=generator,
            )
            optimizer.zero_grad()
            preds = model(inputs)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()

        inputs, targets, _, _ = _sample_batch(
            batch_size=batch_size,
            group_embeddings=group_embeddings,
            selector_embeddings=selector_embeddings,
            candidates=candidates,
            generator=generator,
        )
        with torch.no_grad():
            preds = model(inputs)
            mse = F.mse_loss(preds, targets).item()

        results["steps"].append(step)
        results["total_mse"].append(float(mse))

        if step % eval_every == 0:
            mse_clean, mse_shuffled, selector_gap = _selector_gap(
                model=model,
                group_embeddings=group_embeddings,
                selector_embeddings=selector_embeddings,
                candidates=candidates,
                batch_size=eval_batch_size,
                generator=generator,
            )
            results["eval_steps"].append(step)
            results["mse_clean"].append(mse_clean)
            results["mse_shuffled"].append(mse_shuffled)
            results["selector_gap"].append(selector_gap)

    results["config"] = {
        "d": d,
        "k": k,
        "n_groups": n_groups,
        "batch_size": batch_size,
        "steps": steps,
        "learning_rate": learning_rate,
        "eval_every": eval_every,
        "eval_batch_size": eval_batch_size,
        "seed": seed,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear planted-structure baseline")
    parser.add_argument("--d", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--k", type=int, default=20, help="Number of candidates")
    parser.add_argument("--n-groups", type=int, default=1000, help="Number of groups")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--steps", type=int, default=20000, help="Training steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--eval-every", type=int, default=100, help="Selector logging interval")
    parser.add_argument("--eval-batch-size", type=int, default=2048, help="Eval batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/linear_baseline/planted_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    results = _train(
        d=args.d,
        k=args.k,
        n_groups=args.n_groups,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.lr,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
