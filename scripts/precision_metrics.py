#!/usr/bin/env python
"""
Precision metrics for the ranking-vs-commitment dissociation.

For a chosen run, at each requested checkpoint, computes:
  M1: first-token cross-entropy (full vocabulary), CE - log K, CE / log K
  M2: p_correct on the first token (mean, std, n), and 1/K baseline
  M3: top-1 accuracy on the first token (full vocabulary argmax)
  M4: candidate-restricted entropy on the first character (renormalized over
      the K candidate first-character token IDs per fiber), in nats and as
      fraction of log K
  M5: top-1 accuracy restricted to the K candidate first-character token IDs

Outputs:
  - JSON with raw per-checkpoint values
  - CSV summary table
  - stdout summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import create_tokenizer_from_config, create_datasets_from_config  # noqa: E402
from src.model import create_model_from_config  # noqa: E402


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def _load_config(experiment_dir: Path) -> SimpleNamespace:
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        # Saddle-probe / branch dirs use config.json with cell-flat layout
        json_path = experiment_dir / "config.json"
        if not json_path.exists():
            raise FileNotFoundError(f"No config.yaml or config.json in {experiment_dir}")
        with open(json_path) as f:
            blob = json.load(f)
        cell = blob.get("cell", blob)
        # Re-shape into the YAML-style nested config the factories expect
        nested = {
            "experiment": {"name": experiment_dir.name, "seed": cell.get("seed", 42)},
            "data": {
                "n_unique_b": cell["n_unique_b"],
                "k": cell["k"],
                "task": cell.get("task", "bz_to_a"),
                "b_length": cell["b_length"],
                "a_length": cell["a_length"],
                "z_length": cell["z_length"],
                "vocab_chars": cell["vocab_chars"],
                "probe_fraction": cell.get("probe_fraction", 0.0),
                "split_by_base": cell.get("split_by_base", True),
                "enforce_unique_a_first_char_per_b": cell.get(
                    "enforce_unique_a_first_char_per_b", True
                ),
                "disambiguation_prefix_length": cell.get(
                    "disambiguation_prefix_length", 1
                ),
            },
            "tokenizer": {
                "pad_token": "<PAD>",
                "bos_token": "<BOS>",
                "eos_token": "<EOS>",
                "sep_token": "<SEP>",
            },
            "model": {
                "n_layers": cell["n_layers"],
                "n_heads": cell["n_heads"],
                "d_model": cell["d_model"],
                "d_head": cell["d_head"],
                "d_mlp": cell["d_mlp"],
                "act_fn": cell.get("act_fn", "gelu"),
            },
        }
        return _to_namespace(nested)
    with open(config_path) as f:
        return _to_namespace(yaml.safe_load(f))


def _list_examples(mapping_data) -> List[Tuple[str, str, str, int]]:
    """Flatten mapping_data.mappings into (B, z, A, candidate_index_within_fiber)."""
    out: List[Tuple[str, str, str, int]] = []
    for b, pairs in mapping_data.mappings.items():
        for idx, (z, a) in enumerate(pairs):
            out.append((b, z, a, idx))
    return out


def _candidate_first_char_token_ids(
    tokenizer, mapping_data
) -> Dict[str, List[int]]:
    """For each B, return the token IDs of the K candidate first characters."""
    out: Dict[str, List[int]] = {}
    for b, pairs in mapping_data.mappings.items():
        first_chars = [a[0] for (z, a) in pairs]
        ids = [tokenizer.token_to_id[c] for c in first_chars]
        out[b] = ids
    return out


def _resolve_checkpoint_path(
    experiment_dir: Path, step: int
) -> Path:
    # landauer_dense layout: outputs/<exp>/checkpoints/step_000NNN/model.pt
    nested = experiment_dir / "checkpoints" / f"step_{step:06d}" / "model.pt"
    if nested.exists():
        return nested
    # eta_sweep layout: <exp>/checkpoints/model_step_NNNNNNN.pt
    flat = experiment_dir / "checkpoints" / f"model_step_{step:07d}.pt"
    if flat.exists():
        return flat
    # eta_sweep with 8-digit width
    flat8 = experiment_dir / "checkpoints" / f"model_step_{step:08d}.pt"
    if flat8.exists():
        return flat8
    raise FileNotFoundError(
        f"Checkpoint for step {step} not found under {experiment_dir}/checkpoints"
    )


def _build_model(cfg, tokenizer, device: str):
    model = create_model_from_config(cfg, tokenizer)
    model.to(device)
    model.eval()
    return model


def _load_state(model, ckpt_path: Path, device: str):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model


def _evaluate_checkpoint(
    model,
    tokenizer,
    mapping_data,
    device: str,
    batch_size: int = 256,
) -> Dict[str, float]:
    """Forward pass over ALL training examples; aggregate the precision metrics."""
    examples = _list_examples(mapping_data)
    cand_ids_by_b = _candidate_first_char_token_ids(tokenizer, mapping_data)

    sum_ce = 0.0           # full-vocab first-token CE
    sum_p = 0.0            # full-vocab p_correct
    sum_p_sq = 0.0
    n_correct_top1_full = 0
    n_correct_top1_cand = 0
    sum_cand_entropy = 0.0           # H(K-restricted dist)
    sum_cand_ce = 0.0                # candidate-restricted CE on first char
    sum_cand_mass = 0.0              # total prob the model places on the K candidates
    n = 0

    # Encode + batch
    encoded = [
        tokenizer.encode_sequence(b, z, a, task="bz_to_a") for (b, z, a, _) in examples
    ]
    max_len = max(len(e["input_ids"]) for e in encoded)
    if any(len(e["input_ids"]) != max_len for e in encoded):
        raise RuntimeError("encoded sequences differ in length; expected fixed-length task")

    target_starts = [int(e["target_start_position"]) for e in encoded]
    if len(set(target_starts)) != 1:
        raise RuntimeError("target_start positions vary across examples; aborting")
    target_start = target_starts[0]
    pred_pos = target_start - 1

    for batch_start in range(0, len(examples), batch_size):
        batch_end = min(batch_start + batch_size, len(examples))
        chunk = encoded[batch_start:batch_end]
        chunk_examples = examples[batch_start:batch_end]

        input_ids = torch.stack([e["input_ids"] for e in chunk]).to(device)
        with torch.no_grad():
            logits = model(input_ids)  # (B, T, V)
        logits_first = logits[:, pred_pos, :]  # (B, V)
        log_probs_full = F.log_softmax(logits_first, dim=-1)
        probs_full = torch.exp(log_probs_full)

        for i, (b, _, a, _) in enumerate(chunk_examples):
            correct_id = tokenizer.token_to_id[a[0]]
            cand_ids = cand_ids_by_b[b]
            cand_ids_t = torch.tensor(cand_ids, device=device)

            # M1: first-token CE on the full vocab
            ce_i = -float(log_probs_full[i, correct_id].item())
            sum_ce += ce_i
            # M2: p_correct on the full vocab
            p_i = float(probs_full[i, correct_id].item())
            sum_p += p_i
            sum_p_sq += p_i * p_i
            # M3: full-vocab argmax top-1
            if int(probs_full[i].argmax().item()) == correct_id:
                n_correct_top1_full += 1
            # M4: candidate-restricted entropy and CE on first char
            cand_logits = logits_first[i, cand_ids_t]
            cand_log_probs = F.log_softmax(cand_logits, dim=-1)
            cand_probs = torch.exp(cand_log_probs)
            ent_i = -float((cand_probs * cand_log_probs).sum().item())
            sum_cand_entropy += ent_i
            # candidate-restricted CE on the correct first char (renormalized)
            correct_pos_in_fiber = cand_ids.index(correct_id)
            cand_ce_i = -float(cand_log_probs[correct_pos_in_fiber].item())
            sum_cand_ce += cand_ce_i
            # total mass model places on the K-candidate first chars
            cand_mass_i = float(probs_full[i, cand_ids_t].sum().item())
            sum_cand_mass += cand_mass_i
            # M5 (bonus): top-1 within candidate set
            if cand_ids[int(cand_probs.argmax().item())] == correct_id:
                n_correct_top1_cand += 1
            n += 1

    mean_ce = sum_ce / n
    mean_p = sum_p / n
    var_p = max(0.0, sum_p_sq / n - mean_p * mean_p)
    std_p = math.sqrt(var_p)
    top1_full = n_correct_top1_full / n
    top1_cand = n_correct_top1_cand / n
    mean_cand_entropy = sum_cand_entropy / n
    mean_cand_ce = sum_cand_ce / n
    mean_cand_mass = sum_cand_mass / n
    return {
        "n_examples": n,
        "first_token_ce_full_vocab": mean_ce,
        "candidate_restricted_first_token_ce": mean_cand_ce,
        "p_correct_full_vocab_mean": mean_p,
        "p_correct_full_vocab_std": std_p,
        "candidate_set_total_mass": mean_cand_mass,
        "top1_full_vocab": top1_full,
        "top1_candidate_set": top1_cand,
        "candidate_entropy_nats": mean_cand_entropy,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", required=True, type=Path,
                   help="Path to the run directory containing config + checkpoints/")
    p.add_argument("--steps", type=int, nargs="+", required=True,
                   help="Checkpoint steps to evaluate")
    p.add_argument("--tau", type=float, default=None,
                   help="Approximate τ for normalized step reporting")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--label", type=str, default="run",
                   help="Short label for this run in the JSON output")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--data-seed", type=int, default=None,
                   help="Override the seed used to regenerate the training "
                        "dataset (must match the seed the model was trained "
                        "on).  If omitted, uses cfg.experiment.seed.  Note: "
                        "eta_sweep runs use data_seed = 1_000_003 * "
                        "cell.seed + K, not cell.seed directly.")
    args = p.parse_args()

    device = _select_device()
    print(f"[precision_metrics] device = {device}")
    print(f"[precision_metrics] experiment_dir = {args.experiment_dir}")
    cfg = _load_config(args.experiment_dir)
    log_k = math.log(int(cfg.data.k))
    print(f"[precision_metrics] K = {cfg.data.k}, log K = {log_k:.4f}")

    # Re-create dataset deterministically using the run's seed.
    # IMPORTANT: src/data/dataset.py uses random.Random(cfg.experiment.seed)
    # for *every* RNG draw (B/z/A characters + per-fiber shuffles), so we
    # override cfg.experiment.seed itself rather than torch.manual_seed.
    data_seed = args.data_seed if args.data_seed is not None else int(cfg.experiment.seed)
    print(f"[precision_metrics] data_seed = {data_seed} "
          f"(cfg.experiment.seed before override = {cfg.experiment.seed})")
    cfg.experiment.seed = data_seed
    torch.manual_seed(data_seed)
    tokenizer = create_tokenizer_from_config(cfg)
    train_set, _, _ = create_datasets_from_config(cfg, tokenizer)
    mapping_data = train_set.mapping_data
    print(f"[precision_metrics] |B| = {len(mapping_data.mappings)}, K = {mapping_data.k}, |D| = {sum(len(v) for v in mapping_data.mappings.values())}")

    # Build model once; reload state per checkpoint.
    model = _build_model(cfg, tokenizer, device)

    rows = []
    for step in args.steps:
        ckpt_path = _resolve_checkpoint_path(args.experiment_dir, step)
        print(f"[precision_metrics] step {step}: loading {ckpt_path}")
        _load_state(model, ckpt_path, device)
        metrics = _evaluate_checkpoint(model, tokenizer, mapping_data, device,
                                       batch_size=args.batch_size)
        normalized_step = step / args.tau if args.tau else None
        K = int(cfg.data.k)
        row = {
            "label": args.label,
            "step": step,
            "step_over_tau": normalized_step,
            "log_k": log_k,
            **metrics,
            "full_ce_minus_log_k": metrics["first_token_ce_full_vocab"] - log_k,
            "full_ce_over_log_k": metrics["first_token_ce_full_vocab"] / log_k,
            "cand_ce_minus_log_k": metrics["candidate_restricted_first_token_ce"] - log_k,
            "cand_ce_over_log_k": metrics["candidate_restricted_first_token_ce"] / log_k,
            "p_correct_minus_uniform": metrics["p_correct_full_vocab_mean"] - 1.0 / K,
            "candidate_entropy_over_log_k": metrics["candidate_entropy_nats"] / log_k,
        }
        rows.append(row)
        print(f"  full-CE={row['first_token_ce_full_vocab']:.4f} ({row['full_ce_over_log_k']:.2f}x logK)  "
              f"cand-CE={row['candidate_restricted_first_token_ce']:.4f} ({row['cand_ce_over_log_k']:.2f}x logK)  "
              f"p_correct={row['p_correct_full_vocab_mean']:.4f}±{row['p_correct_full_vocab_std']:.4f}  "
              f"top1_full={row['top1_full_vocab']:.3f}  top1_cand={row['top1_candidate_set']:.3f}  "
              f"cand_H/logK={row['candidate_entropy_over_log_k']:.3f}  "
              f"cand_mass={row['candidate_set_total_mass']:.3f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "label": args.label,
            "experiment_dir": str(args.experiment_dir),
            "K": int(cfg.data.k),
            "log_k": log_k,
            "uniform_baseline": 1.0 / int(cfg.data.k),
            "rows": rows,
        }, f, indent=2)
    print(f"[precision_metrics] wrote JSON: {args.out_json}")

    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[precision_metrics] wrote CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
