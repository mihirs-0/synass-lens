#!/usr/bin/env python
"""Shared utilities for overnight experiment suites.

Includes a parallel runner that launches N independent training runs
concurrently on the same GPU, vastly improving utilization for our
tiny (~600K param) model.
"""

import sys
import json
import math
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_tokenizer_from_config, collate_fn
from src.data.dataset import MappingData, DisambiguationDataset, generate_mappings
from src.model import create_model_from_config
from src.training import train


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def make_config(
    experiment_name: str,
    task: str = "bz_to_a",
    k: int = 10,
    seed: int = 42,
    lr: float = 1e-3,
    bs: int = 128,
    max_steps: int = 50000,
    eval_every: int = 50,
    checkpoint_every: int = 5000,
    label_noise_prob: float = 0.0,
    early_stop_frac: Optional[float] = None,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    n_unique_b: int = 1000,
    z_length: int = 2,
    optimizer_type: str = "adamw",
    momentum: float = 0.0,
):
    """Create an OmegaConf config matching the Hydra config schema.

    Defaults match the baseline landauer_dense experiments:
    split_by_base=True, enforce_unique_a_first_char_per_b=True,
    warmup_steps=0, scheduler=constant.
    """
    return OmegaConf.create({
        "experiment": {"name": experiment_name, "seed": seed},
        "data": {
            "n_unique_b": n_unique_b, "k": k, "task": task,
            "b_length": 6, "a_length": 4, "z_length": z_length,
            "vocab_chars": "abcdefghijklmnopqrstuvwxyz0123456789",
            "probe_fraction": 0.0, "split_by_base": True,
            "enforce_unique_a_first_char_per_b": True,
            "disambiguation_prefix_length": 1,
            "label_noise_prob": label_noise_prob,
        },
        "tokenizer": {
            "pad_token": "<PAD>", "bos_token": "<BOS>",
            "eos_token": "<EOS>", "sep_token": "<SEP>",
        },
        "model": {
            "n_layers": 4, "n_heads": 4, "d_model": 128,
            "d_head": 32, "d_mlp": 512, "act_fn": "gelu",
        },
        "training": {
            "batch_size": bs, "learning_rate": lr,
            "weight_decay": weight_decay,
            "max_steps": max_steps, "warmup_steps": warmup_steps,
            "scheduler": "constant",
            "checkpoint_every": checkpoint_every,
            "eval_every": eval_every,
            "early_stop_convergence_frac": early_stop_frac,
            "optimizer_type": optimizer_type,
            "momentum": momentum,
        },
        "output": {"base_dir": "outputs"},
    })


# ---------------------------------------------------------------------------
# Run existence / history helpers
# ---------------------------------------------------------------------------

def run_exists(experiment_name, min_steps=0, output_dir="outputs"):
    """Check if a training run exists and reached min_steps."""
    p = Path(output_dir) / experiment_name / "training_history.json"
    if not p.exists():
        return False
    if min_steps <= 0:
        return True
    try:
        with open(p) as f:
            h = json.load(f)
        return len(h.get("steps", [])) > 0 and h["steps"][-1] >= min_steps
    except Exception:
        return False


def load_history(experiment_name, output_dir="outputs"):
    """Load training history from JSON."""
    p = Path(output_dir) / experiment_name / "training_history.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def detect_tau(history, log_k, key="candidate_loss", threshold_frac=0.5):
    """Detect transition time tau from training history."""
    if key not in history or not history[key]:
        key = "first_target_loss"
    if key not in history or not history[key]:
        return None
    threshold = threshold_frac * log_k
    for s, v in zip(history["steps"], history[key]):
        if v is not None and v < threshold:
            return s
    return None


def detect_convergence(history, threshold=0.01, key="train_loss"):
    """Detect convergence: first step where loss drops below threshold."""
    if key not in history or not history[key]:
        return None
    for s, v in zip(history["steps"], history[key]):
        if v < threshold:
            return s
    return None


# ---------------------------------------------------------------------------
# Mapping save/load
# ---------------------------------------------------------------------------

def save_mapping(mapping_data, path):
    """Save MappingData to JSON for reuse across tasks."""
    d = {
        "mappings": {b: pairs for b, pairs in mapping_data.mappings.items()},
        "examples": mapping_data.examples,
        "n_unique_b": mapping_data.n_unique_b,
        "n_unique_a": mapping_data.n_unique_a,
        "k": mapping_data.k,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f)


def load_mapping(path):
    """Load MappingData from JSON."""
    with open(path) as f:
        d = json.load(f)
    return MappingData(
        mappings={b: [tuple(p) for p in pairs] for b, pairs in d["mappings"].items()},
        examples=d["examples"],
        n_unique_b=d["n_unique_b"],
        n_unique_a=d["n_unique_a"],
        k=d["k"],
        task="bz_to_a",
    )


# ---------------------------------------------------------------------------
# Single experiment runner (runs in-process)
# ---------------------------------------------------------------------------

def run_single_experiment(cfg, mapping_data=None, output_dir="outputs", model=None):
    """
    Full training pipeline. Returns (model, history, mapping_data, tokenizer).
    """
    torch.manual_seed(cfg.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.experiment.seed)

    tokenizer = create_tokenizer_from_config(cfg)

    if mapping_data is None:
        from src.data import create_datasets_from_config
        train_ds, probe_ds, mapping_data = create_datasets_from_config(cfg, tokenizer)
    else:
        train_ds = DisambiguationDataset(
            mapping_data=mapping_data, tokenizer=tokenizer,
            split="train", probe_fraction=0.0,
            seed=cfg.experiment.seed, task=cfg.data.task,
            label_noise_prob=float(getattr(cfg.data, "label_noise_prob", 0.0)),
        )
        probe_ds = DisambiguationDataset(
            mapping_data=mapping_data, tokenizer=tokenizer,
            split="probe", probe_fraction=0.0,
            seed=cfg.experiment.seed, task=cfg.data.task,
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    probe_loader = DataLoader(
        probe_ds, batch_size=cfg.training.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    if model is None:
        model = create_model_from_config(cfg, tokenizer)
    else:
        model.train()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / cfg.experiment.name / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Resolve optimizer settings from config
    _opt_type = getattr(cfg.training, "optimizer_type", "adamw")
    _opt_kwargs = {}
    if _opt_type == "sgd":
        _mom = float(getattr(cfg.training, "momentum", 0.0))
        if _mom > 0:
            _opt_kwargs["momentum"] = _mom

    # Candidate eval only for bz_to_a
    if cfg.data.task == "bz_to_a":
        history = train(
            model=model, train_loader=train_loader, probe_loader=probe_loader,
            cfg=cfg, output_dir=out, grad_clip=1.0,
            optimizer_type=_opt_type, optimizer_kwargs=_opt_kwargs,
            mapping_data=mapping_data, tokenizer=tokenizer,
        )
    else:
        history = train(
            model=model, train_loader=train_loader, probe_loader=probe_loader,
            cfg=cfg, output_dir=out, grad_clip=1.0,
            optimizer_type=_opt_type, optimizer_kwargs=_opt_kwargs,
        )

    return model, history, mapping_data, tokenizer


# ---------------------------------------------------------------------------
# Parallel runner: process-pool based GPU sharing
# ---------------------------------------------------------------------------

def _worker_run(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for parallel execution. Each worker is a separate process
    sharing the same GPU via CUDA. The tiny model means many fit concurrently.

    job dict keys:
        cfg_dict: serialized OmegaConf config (dict)
        mapping_path: path to mapping JSON (or None to generate)
        output_dir: output directory
        name: experiment name (for logging)
    """
    import os
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

    cfg = OmegaConf.create(job["cfg_dict"])
    mapping_path = job.get("mapping_path")
    output_dir = job.get("output_dir", "outputs")
    name = job["name"]

    t0 = time.time()
    try:
        mapping = load_mapping(mapping_path) if mapping_path else None
        _, history, _, _ = run_single_experiment(
            cfg, mapping_data=mapping, output_dir=output_dir,
        )
        elapsed = time.time() - t0
        steps_done = history["steps"][-1] if history.get("steps") else 0
        early = history.get("early_stopped", False)
        tag = f" [early-stopped@{history.get('early_stopped_step', '?')}]" if early else ""
        return {"name": name, "ok": True, "elapsed": elapsed,
                "steps": steps_done, "tag": tag}
    except Exception as e:
        traceback.print_exc()
        return {"name": name, "ok": False, "elapsed": time.time() - t0,
                "error": str(e)}


def run_parallel(jobs: List[Dict[str, Any]], max_workers: int = 6,
                 label: str = "batch"):
    """
    Run a list of training jobs in parallel using ProcessPoolExecutor.

    Each job is a dict with keys: cfg_dict, mapping_path, output_dir, name.
    max_workers controls concurrency (default 6 — good for 600K-param model
    on a 24GB GPU).
    """
    if not jobs:
        print(f"  [{label}] No jobs to run.")
        return

    n = len(jobs)
    print(f"\n  [{label}] Launching {n} jobs with {max_workers} workers...")
    t0 = time.time()

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_worker_run, job): job["name"] for job in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            if r["ok"]:
                print(f"    OK  {r['name']} — {r['steps']} steps, "
                      f"{r['elapsed']:.0f}s{r.get('tag', '')}", flush=True)
            else:
                print(f"    FAIL {r['name']} — {r['error']}", flush=True)

    total = time.time() - t0
    print(f"  [{label}] {n} jobs done in {total:.0f}s "
          f"({total/60:.1f}min)", flush=True)
