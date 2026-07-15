#!/usr/bin/env python3
"""Small, reversal-matched autoregressive experiments.

This suite follows the experimental symmetry in Papadopoulos, Wenger, and
Hongler (2024): each backward-training example is the exact token reversal of
its forward counterpart (with a fresh BOS placed at the left).  Dataset split,
sequence length, token counts, model architecture, initialization, optimizer,
batch order, and training budget are paired.  Only prediction order changes.

The synthetic language is

    x_0 ... x_{m-1} <SEP> y_0 ... y_{m-1},   y = M x  (mod 2).

The backward model sees the reversed core and therefore learns
J M^{-1} J, where J reverses bit order.  Four matrix families are useful:

* identity: matched sparse forward/backward computation;
* shift: M = I + superdiagonal, whose inverse is triangular and dense;
* sparse_random: a random sparse invertible map with a denser inverse;
* dense: a control where M and M^{-1} are both dense.

The reported ``work_proxy`` is sum_t lr * ||grad_t||^2.  It is an optimizer
diagnostic, not thermodynamic heat or a Landauer cost.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-reversal-suite")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BIT0, BIT1, SEP, BOS, EOS = range(5)
VOCAB_SIZE = 5


def gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    """Invert a square binary matrix using Gauss-Jordan elimination."""
    a = np.asarray(matrix, dtype=np.uint8) % 2
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError("matrix must be square")
    aug = np.concatenate([a.copy(), np.eye(n, dtype=np.uint8)], axis=1)
    for col in range(n):
        pivots = np.flatnonzero(aug[col:, col])
        if len(pivots) == 0:
            raise ValueError("matrix is singular over GF(2)")
        pivot = col + int(pivots[0])
        if pivot != col:
            aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(n):
            if row != col and aug[row, col]:
                aug[row] ^= aug[col]
    inv = aug[:, n:]
    if not np.array_equal((a @ inv) % 2, np.eye(n, dtype=np.uint8)):
        raise AssertionError("GF(2) inverse self-check failed")
    return inv


def matrix_family(name: str, m: int, seed: int) -> np.ndarray:
    if name == "identity":
        return np.eye(m, dtype=np.uint8)
    if name == "shift":
        matrix = np.eye(m, dtype=np.uint8)
        matrix[np.arange(m - 1), np.arange(1, m)] = 1
        return matrix
    if name == "sparse_random":
        rng = np.random.default_rng(seed + 11_009)
        max_forward_nnz = max(m + 1, int(math.floor(0.35 * m * m)))
        min_inverse_nnz = int(math.ceil(0.40 * m * m))
        for _ in range(50_000):
            matrix = (rng.random((m, m)) < 0.25).astype(np.uint8)
            try:
                inv = gf2_inverse(matrix)
            except ValueError:
                continue
            forward_nnz = int(matrix.sum())
            inverse_nnz = int(inv.sum())
            if m < forward_nnz <= max_forward_nnz and inverse_nnz >= min_inverse_nnz:
                return matrix
        raise RuntimeError("failed to sample a sparse map with a dense GF(2) inverse")
    if name == "dense":
        rng = np.random.default_rng(seed + 17_003)
        for _ in range(10_000):
            matrix = rng.integers(0, 2, size=(m, m), dtype=np.uint8)
            try:
                inv = gf2_inverse(matrix)
            except ValueError:
                continue
            density = float(matrix.mean())
            inv_density = float(inv.mean())
            if 0.35 <= density <= 0.65 and 0.35 <= inv_density <= 0.65:
                return matrix
        raise RuntimeError("failed to sample a dense invertible GF(2) matrix")
    raise ValueError(f"unknown matrix family: {name}")


def all_bit_vectors(m: int) -> np.ndarray:
    values = np.arange(2**m, dtype=np.uint32)[:, None]
    shifts = np.arange(m - 1, -1, -1, dtype=np.uint32)[None, :]
    return ((values >> shifts) & 1).astype(np.uint8)


def make_sequences(x: np.ndarray, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = (x @ matrix.T) % 2
    core = np.concatenate(
        [x + BIT0, np.full((len(x), 1), SEP, dtype=np.uint8), y + BIT0], axis=1
    )
    forward = np.concatenate(
        [
            np.full((len(x), 1), BOS, dtype=np.uint8),
            core,
            np.full((len(x), 1), EOS, dtype=np.uint8),
        ],
        axis=1,
    )
    backward = np.concatenate(
        [
            np.full((len(x), 1), BOS, dtype=np.uint8),
            core[:, ::-1],
            np.full((len(x), 1), EOS, dtype=np.uint8),
        ],
        axis=1,
    )
    if not np.array_equal(forward[:, 1:-1][:, ::-1], backward[:, 1:-1]):
        raise AssertionError("backward examples are not exact core reversals")
    return forward.astype(np.int64), backward.astype(np.int64)


class TinyCausalTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_mlp: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token = nn.Embedding(VOCAB_SIZE, d_model)
        self.position = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_mlp,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1),
            persistent=False,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token(tokens) + self.position(positions)[None, :, :]
        x = self.blocks(x, mask=self.causal_mask[: tokens.size(1), : tokens.size(1)])
        return self.unembed(self.norm(x))


@dataclass
class Config:
    m: int
    train_fraction: float
    steps: int
    batch_size: int
    lr: float
    weight_decay: float
    eval_every: int
    d_model: int
    n_heads: int
    n_layers: int
    d_mlp: int
    dropout: float
    seed: int
    task_seed: int
    conditions: List[str]
    device: str


def metric_slices(logits: torch.Tensor, sequences: torch.Tensor, m: int) -> Dict[str, float]:
    pred = logits[:, :-1]
    target = sequences[:, 1:]
    token_loss = F.cross_entropy(
        pred.reshape(-1, pred.size(-1)), target.reshape(-1), reduction="none"
    ).reshape(target.shape)

    # Target indices 0..m-1 are the first random/source block.  The separator
    # is target m; second-block bits are targets m+1..2m; EOS is target 2m+1.
    first = token_loss[:, :m]
    second = token_loss[:, m + 1 : 2 * m + 1]
    second_pred = pred[:, m + 1 : 2 * m + 1].argmax(dim=-1)
    second_target = target[:, m + 1 : 2 * m + 1]
    return {
        "full_ce": float(token_loss.mean().item()),
        "first_bits_ce": float(first.mean().item()),
        "conditional_ce": float(second.mean().item()),
        "conditional_accuracy": float((second_pred == second_target).float().mean().item()),
        "sequence_accuracy": float((second_pred == second_target).all(dim=1).float().mean().item()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    sequences: torch.Tensor,
    m: int,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    weighted: Dict[str, float] = {}
    total = 0
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        metrics = metric_slices(model(batch), batch, m)
        n = len(batch)
        total += n
        for key, value in metrics.items():
            weighted[key] = weighted.get(key, 0.0) + n * value
    return {key: value / total for key, value in weighted.items()}


def train_one(
    model: nn.Module,
    train_sequences: torch.Tensor,
    eval_sequences: torch.Tensor,
    cfg: Config,
    batch_orders: torch.Tensor,
) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    history: List[Dict[str, float]] = []
    work_proxy = 0.0

    def record(step: int) -> None:
        train_metrics = evaluate(model, train_sequences, cfg.m, 512)
        eval_metrics = evaluate(model, eval_sequences, cfg.m, 512)
        row: Dict[str, float] = {"step": float(step), "work_proxy": work_proxy}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"eval_{k}": v for k, v in eval_metrics.items()})
        history.append(row)

    record(0)
    model.train()
    for step in range(1, cfg.steps + 1):
        idx = batch_orders[step - 1]
        batch = train_sequences[idx]
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1)
        )
        loss.backward()
        grad_norm_sq = sum(
            float(parameter.grad.detach().pow(2).sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        work_proxy += cfg.lr * grad_norm_sq
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            record(step)
            model.train()

    return {
        "history": history,
        "final_train": {k[6:]: v for k, v in history[-1].items() if k.startswith("train_")},
        "final_eval": {k[5:]: v for k, v in history[-1].items() if k.startswith("eval_")},
        "work_proxy": work_proxy,
    }


def first_crossing(history: Iterable[Dict[str, float]], key: str, threshold: float) -> int | None:
    for row in history:
        if row[key] >= threshold:
            return int(row["step"])
    return None


@torch.no_grad()
def random_theta_reversal_check(cfg: Config, repeats: int = 5) -> List[Dict[str, float]]:
    """Test the stated token/absolute-position reindexing at random parameters.

    The corpus is sampled from a directional nonlinear bit process, but has no
    BOS/EOS, tokenizer, or special-token asymmetry.  For each random model we
    compare standard-causal NLL on D with standard-causal NLL on rev(D), both
    with identical parameters and after the learned-position index flip stated
    in Theorem 4.5 of Sahasrabudhe (2025).  The identity vocabulary map is exact
    for bit tokens.  If that stated reindexing gave pointwise equality under
    ordinary forward causal masks, the position-flipped gaps would be numerical
    zero.  Nonzero gaps do not rule out a different, richer parameter map.
    """
    seq_len = 2 * cfg.m + 1
    corpus_generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 88_001)
    corpus = torch.zeros((128, seq_len), dtype=torch.long)
    corpus[:, :2] = torch.randint(0, 2, (128, 2), generator=corpus_generator)
    for position in range(2, seq_len):
        previous = corpus[:, position - 1]
        pre_previous = corpus[:, position - 2]
        if position % 3 == 0:
            next_bit = previous ^ pre_previous
        elif position % 3 == 1:
            next_bit = previous & (1 - pre_previous)
        else:
            next_bit = previous
        noise = (
            torch.rand(128, generator=corpus_generator) < 0.08
        ).to(torch.long)
        corpus[:, position] = next_bit ^ noise
    reversed_corpus = corpus.flip(1)
    corpus = corpus.to(cfg.device)
    reversed_corpus = reversed_corpus.to(cfg.device)

    def nll(model: nn.Module, tokens: torch.Tensor) -> float:
        logits = model(tokens)
        return float(
            F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB_SIZE), tokens[:, 1:].reshape(-1)
            ).item()
        )

    checks: List[Dict[str, float]] = []
    for repeat in range(repeats):
        torch.manual_seed(cfg.seed + 91_000 + repeat)
        forward_model = TinyCausalTransformer(
            seq_len=seq_len,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_mlp=cfg.d_mlp,
            dropout=0.0,
        ).to(cfg.device)
        # Pointwise equality is claimed for every theta, so testing a sharper
        # random readout is legitimate and makes finite-corpus gaps resolvable.
        forward_model.unembed.weight.mul_(4.0)
        same_model = copy.deepcopy(forward_model)
        flipped_model = copy.deepcopy(forward_model)
        flipped_model.position.weight.copy_(forward_model.position.weight.flip(0))
        loss_forward = nll(forward_model, corpus)
        loss_reverse_same = nll(same_model, reversed_corpus)
        loss_reverse_posflip = nll(flipped_model, reversed_corpus)
        checks.append(
            {
                "repeat": float(repeat),
                "forward_nll": loss_forward,
                "reverse_same_theta_nll": loss_reverse_same,
                "reverse_position_flipped_nll": loss_reverse_posflip,
                "same_theta_gap": loss_reverse_same - loss_forward,
                "position_flipped_gap": loss_reverse_posflip - loss_forward,
            }
        )
    return checks


def plot_results(results: Dict[str, object], output_path: Path) -> None:
    conditions = results["config"]["conditions"]
    fig, axes = plt.subplots(len(conditions), 3, figsize=(12, 3.2 * len(conditions)), squeeze=False)
    colors = {"forward": "#2563eb", "backward": "#dc2626"}
    for row_idx, condition in enumerate(conditions):
        for direction in ("forward", "backward"):
            history = results["conditions"][condition][direction]["history"]
            steps = [row["step"] for row in history]
            axes[row_idx, 0].plot(
                steps, [row["eval_full_ce"] for row in history],
                color=colors[direction], label=direction,
            )
            axes[row_idx, 1].plot(
                steps, [row["eval_conditional_ce"] for row in history],
                color=colors[direction], label=direction,
            )
            axes[row_idx, 2].plot(
                steps, [row["eval_conditional_accuracy"] for row in history],
                color=colors[direction], label=direction,
            )
        stats = results["conditions"][condition]["matrix_stats"]
        axes[row_idx, 0].set_ylabel(
            f"{condition}\nF/B nnz={stats['forward_nnz']}/{stats['backward_nnz']}"
        )
        axes[row_idx, 0].set_title("Full validation CE")
        axes[row_idx, 1].set_title("Conditional-half validation CE")
        axes[row_idx, 2].set_title("Conditional bit accuracy")
        axes[row_idx, 2].set_ylim(0.45, 1.01)
        for ax in axes[row_idx]:
            ax.set_xlabel("optimizer steps")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Reversal-matched linear languages", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_suite(cfg: Config) -> Dict[str, object]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device)

    x_all = all_bit_vectors(cfg.m)
    split_rng = np.random.default_rng(cfg.task_seed + 9_001)
    permutation = split_rng.permutation(len(x_all))
    n_train = int(round(cfg.train_fraction * len(x_all)))
    n_train = min(max(n_train, 2), len(x_all) - 1)
    train_idx, eval_idx = permutation[:n_train], permutation[n_train:]

    results: Dict[str, object] = {
        "config": asdict(cfg),
        "theory": {
            "entropy_nats_per_sequence": cfg.m * math.log(2.0),
            "note": "Forward/backward examples are exact reversals, so entropy is identical.",
            "work_proxy_note": "sum(lr * grad_norm_sq); optimizer diagnostic, not heat",
        },
        "conditions": {},
    }
    results["random_theta_reversal_check"] = random_theta_reversal_check(cfg)

    for condition_idx, condition in enumerate(cfg.conditions):
        matrix = matrix_family(condition, cfg.m, cfg.task_seed + condition_idx * 101)
        inverse = gf2_inverse(matrix)
        reversal = np.eye(cfg.m, dtype=np.uint8)[::-1]
        backward_map = (reversal @ inverse @ reversal) % 2
        if not np.array_equal((matrix @ inverse) % 2, np.eye(cfg.m, dtype=np.uint8)):
            raise AssertionError("matrix inverse check failed")

        forward_np, backward_np = make_sequences(x_all, matrix)
        if not np.array_equal(
            np.sort(forward_np[:, 1:-1].reshape(-1)),
            np.sort(backward_np[:, 1:-1].reshape(-1)),
        ):
            raise AssertionError("paired corpora do not have identical token counts")

        forward_train = torch.tensor(forward_np[train_idx], device=device)
        forward_eval = torch.tensor(forward_np[eval_idx], device=device)
        backward_train = torch.tensor(backward_np[train_idx], device=device)
        backward_eval = torch.tensor(backward_np[eval_idx], device=device)

        torch.manual_seed(cfg.seed + 31_337 + condition_idx)
        template = TinyCausalTransformer(
            seq_len=forward_train.size(1),
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_mlp=cfg.d_mlp,
            dropout=cfg.dropout,
        ).to(device)
        initial_state = copy.deepcopy(template.state_dict())

        order_generator = torch.Generator(device="cpu").manual_seed(
            cfg.seed + 72_001 + condition_idx
        )
        batch_orders = torch.randint(
            0,
            n_train,
            (cfg.steps, cfg.batch_size),
            generator=order_generator,
        ).to(device)

        condition_result: Dict[str, object] = {
            "matrix": matrix.astype(int).tolist(),
            "inverse": inverse.astype(int).tolist(),
            "backward_effective_map": backward_map.astype(int).tolist(),
            "matrix_stats": {
                "forward_nnz": int(matrix.sum()),
                "backward_nnz": int(backward_map.sum()),
                "forward_density": float(matrix.mean()),
                "backward_density": float(backward_map.mean()),
            },
            "n_train": n_train,
            "n_eval": len(x_all) - n_train,
        }
        for direction, train_sequences, eval_sequences in (
            ("forward", forward_train, forward_eval),
            ("backward", backward_train, backward_eval),
        ):
            model = TinyCausalTransformer(
                seq_len=forward_train.size(1),
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_layers=cfg.n_layers,
                d_mlp=cfg.d_mlp,
                dropout=cfg.dropout,
            ).to(device)
            model.load_state_dict(initial_state)
            condition_result[direction] = train_one(
                model, train_sequences, eval_sequences, cfg, batch_orders
            )
            condition_result[direction]["step_eval_acc_90"] = first_crossing(
                condition_result[direction]["history"], "eval_conditional_accuracy", 0.90
            )
            condition_result[direction]["step_train_acc_90"] = first_crossing(
                condition_result[direction]["history"], "train_conditional_accuracy", 0.90
            )

        f_eval = condition_result["forward"]["final_eval"]
        b_eval = condition_result["backward"]["final_eval"]
        condition_result["directional_gap"] = {
            "backward_minus_forward_full_ce": b_eval["full_ce"] - f_eval["full_ce"],
            "backward_minus_forward_conditional_ce": (
                b_eval["conditional_ce"] - f_eval["conditional_ce"]
            ),
            "forward_minus_backward_conditional_accuracy": (
                f_eval["conditional_accuracy"] - b_eval["conditional_accuracy"]
            ),
            "backward_over_forward_work_proxy": (
                condition_result["backward"]["work_proxy"]
                / max(condition_result["forward"]["work_proxy"], 1e-12)
            ),
        }
        results["conditions"][condition] = condition_result
        print(
            f"[{condition}] nnz F/B={matrix.sum()}/{backward_map.sum()} "
            f"eval conditional acc F/B="
            f"{f_eval['conditional_accuracy']:.3f}/{b_eval['conditional_accuracy']:.3f} "
            f"CE gap(B-F)={condition_result['directional_gap']['backward_minus_forward_full_ce']:+.4f}",
            flush=True,
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-mlp", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--task-seed",
        type=int,
        default=0,
        help="seed for the matrix and train/eval split, held fixed across training seeds",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["identity", "shift", "sparse_random", "dense"],
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/reversal_suite")
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run only the random-parameter reindexing diagnostic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(
        m=args.m,
        train_fraction=args.train_fraction,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_mlp=args.d_mlp,
        dropout=args.dropout,
        seed=args.seed,
        task_seed=args.task_seed,
        conditions=args.conditions,
        device=device,
    )
    if args.check_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "config": asdict(cfg),
            "scope": (
                "Diagnostic of the stated identity-vocabulary plus absolute-position "
                "flip under standard causal masks; not a no-go theorem for all maps."
            ),
            "random_theta_reversal_check": random_theta_reversal_check(cfg),
        }
        json_path = args.output_dir / f"reindexing_check_m{args.m}_seed{args.seed}.json"
        json_path.write_text(json.dumps(result, indent=2))
        print(f"saved {json_path}")
        return
    results = run_suite(cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"quick_m{args.m}_seed{args.seed}.json"
    fig_path = args.output_dir / f"quick_m{args.m}_seed{args.seed}.png"
    json_path.write_text(json.dumps(results, indent=2))
    plot_results(results, fig_path)
    print(f"saved {json_path}")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
