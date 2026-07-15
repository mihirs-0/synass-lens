#!/usr/bin/env python
"""
GATE: forward (A,z)->B mirror vs inverse (B,z)->A, SAME harness.

The "literal mirror" of the spec: take a standard bz_to_a dataset (each A maps to a
UNIQUE B, since A-strings are globally unique), and for the forward direction swap
base<->target in the tokenizer so the sequence is [BOS, A, SEP, z, SEP, B, EOS] scoring B.
=> z is present and causally BEFORE the target (accessible), but A->B is deterministic
   so z is REDUNDANT. This is NOT --reverse (which makes z causally downstream/blocked).

Question: is the high-eta trap router-specific? At eta=6e-3 the inverse is proven trapped
(tier0_confirm). If the forward (no router to build) CONVERGES there, the trap is
direction-selective -> a measured arrow with a ruin boundary on one face. If the forward
also traps, the boundary is generic high-eta instability, not router-specific.

Classifier (forward has no Dz; classify on loss, per spec):
  converged = train_loss EMA < 0.15 (acc->~1)            [solved]
  trapped   = loss EMA flat (|slope| small) over last >=10k AND clearly nonzero AND not diverging
  diverged  = loss > 5x initial, or non-finite
Same metric works for inverse: a trapped inverse sits at a nonzero plateau (first-A-token
stuck near log K); a converged run goes to ~0.

Usage (single cell):
  python eta_sweep/gate_mirror_runner.py --direction forward --eta 6e-3 --seed 0
  python eta_sweep/gate_mirror_runner.py --direction forward --eta 6e-3 --seed 0 --preflight
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ETA_SWEEP = Path(__file__).resolve().parent
REPO = ETA_SWEEP.parent
sys.path.insert(0, str(REPO))

from eta_sweep.config import CellConfig, RESULTS_DIR  # noqa: E402
from eta_sweep.run_single import build_legacy_cfg, _select_device, _set_all_seeds  # noqa: E402
from src.data import collate_fn, create_datasets_from_config, create_tokenizer_from_config  # noqa: E402
from src.model import create_model_from_config  # noqa: E402
from src.training.trainer import compute_loss  # noqa: E402


def build(cc, direction, tok):
    """Return (train_ds, mapping_data). For 'forward', swap base<->target in encode."""
    if direction == "forward":
        _orig = tok.encode_sequence
        def _fwd(b, z, a, task="bz_to_a"):
            return _orig(b=a, z=z, a=b, task=task)   # incoming (B,z,A) -> encode base=A, target=B
        tok.encode_sequence = _fwd  # type: ignore[assignment]
    cfg = build_legacy_cfg(cc)
    train_ds, _, md = create_datasets_from_config(cfg, tok)
    return train_ds, md


def preflight(cc, tok):
    """Verify the forward mirror: sequence layout + A->B deterministic (z redundant)."""
    train_ds, md = build(cc, "forward", tok)
    # 1) layout of one tokenized example
    ex = train_ds[0]
    ids = ex["input_ids"].tolist(); labels = ex["labels"].tolist()
    scored = [i for i, l in enumerate(labels) if l != -100]
    print(f"  forward seq len={len(ids)}  scored positions={scored}")
    print(f"  z span: pos {ex.get('z_position')}..{ex.get('z_end_position')}  "
          f"target span: {ex.get('target_start_position')}..{ex.get('target_end_position')}")
    z_before_target = ex.get("z_end_position", 99) < ex.get("target_start_position", -1)
    print(f"  z causally BEFORE target? {z_before_target}  (must be True for z-accessible)")
    # 2) A->B deterministic & z redundant: each base(A) maps to a single target(B)
    from collections import defaultdict
    a2b = defaultdict(set)
    for e in md.examples:
        a2b[e["a"]].add(e["b"])   # in forward, original 'a'(=A) is the base, original 'b'(=B) is target
    multi = sum(1 for v in a2b.values() if len(v) > 1)
    print(f"  #unique A (forward base) = {len(a2b)} ; #examples = {len(md.examples)}")
    print(f"  A-values mapping to >1 B (would break determinism): {multi}")
    print(f"  => A->B deterministic & z REDUNDANT: {multi == 0 and z_before_target}")
    return multi == 0 and z_before_target


def classify(log):
    if not log:
        return "no_data", {}
    import statistics as st
    losses = [r["train_loss"] for r in log]
    steps = [r["step"] for r in log]
    final_ema = st.mean(losses[-10:]) if len(losses) >= 10 else losses[-1]
    init = st.mean(losses[:3]) if len(losses) >= 3 else losses[0]
    last_acc = log[-1].get("acc", 0.0)
    if not all(map(lambda x: x == x, losses)) or final_ema > 5 * init:
        return "diverged", {"final_ema": final_ema}
    if final_ema < 0.15 or last_acc > 0.95:
        return "converged", {"final_ema": final_ema, "acc": last_acc}
    # trapped: flat over last >=10k steps and clearly nonzero
    tail = [(s, l) for s, l in zip(steps, losses) if s >= steps[-1] - 10000]
    if len(tail) >= 5:
        xs = [s for s, _ in tail]; ys = [l for _, l in tail]
        mx = st.mean(xs); my = st.mean(ys)
        denom = sum((x - mx) ** 2 for x in xs) or 1.0
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom  # per-step
        slope10k = slope * 10000
        if abs(slope10k) < 0.1 and final_ema > 0.3:
            return "trapped", {"final_ema": final_ema, "slope_per_10k": slope10k}
        if slope10k <= -0.1:
            return "slow_descending", {"final_ema": final_ema, "slope_per_10k": slope10k}
    return "inconclusive", {"final_ema": final_ema}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["forward", "inverse"], required=True)
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-unique-b", type=int, default=1000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=25000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--output-subdir", type=str, default="gate_mirror")
    ap.add_argument("--eta-drop-at", type=int, default=None,
                    help="step at which to switch lr (rescue: drop; reverse-arm: raise)")
    ap.add_argument("--eta-after", type=float, default=None, help="lr to switch to at --eta-drop-at")
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()

    _set_all_seeds(a.seed)
    cc = CellConfig(eta=a.eta, k=a.k, seed=a.seed, n_unique_b=a.n_unique_b,
                    batch_size=a.batch_size, weight_decay=a.weight_decay)
    tok = create_tokenizer_from_config(build_legacy_cfg(cc))

    if a.preflight:
        ok = preflight(cc, tok)
        print(f"PREFLIGHT {'OK' if ok else 'FAILED'}")
        return

    device = _select_device()
    train_ds, md = build(cc, a.direction, tok)
    loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    model = create_model_from_config(build_legacy_cfg(cc), tok)
    opt = torch.optim.AdamW(model.parameters(), lr=a.eta, betas=(0.9, 0.999), weight_decay=a.weight_decay)

    # Delta_z fingerprint (inverse only: z-shuffle gap, same metric as Tier-0).
    dz_fn = None
    if a.direction == "inverse":
        from eta_sweep.run_single import compute_candidate_loss_and_delta_z
        def dz_fn(s):
            return compute_candidate_loss_and_delta_z(
                model=model, tokenizer=tok, mapping_data=md, n_examples=32,
                task="bz_to_a", device=device, seed=s)

    out = RESULTS_DIR / a.output_subdir / f"{a.direction}_eta{a.eta:g}_K{a.k}_nb{a.n_unique_b}_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.jsonl", "w", buffering=1)
    print(f"[{a.direction} eta{a.eta:g} s{a.seed}] device={device}"
          + (f"  eta->{a.eta_after:g}@{a.eta_drop_at}" if a.eta_drop_at else ""))

    cur_eta = a.eta
    step = 0; t0 = time.time(); init_loss = None; diverged = False
    while step < a.max_steps and not diverged:
        for batch in loader:
            if step >= a.max_steps:
                break
            if a.eta_drop_at is not None and a.eta_after is not None and step == a.eta_drop_at:
                for g in opt.param_groups:
                    g["lr"] = a.eta_after
                cur_eta = a.eta_after
                print(f"[{a.direction} s{a.seed}] eta {a.eta:g} -> {a.eta_after:g} at step {step}")
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            model.train(); opt.zero_grad(set_to_none=True)
            loss, acc, _ = compute_loss(model, batch)
            loss.backward(); opt.step()
            step += 1
            lv = float(loss.item())
            if init_loss is None: init_loss = lv
            if step % 100 == 0:
                rec = {"step": step, "train_loss": lv, "lr": cur_eta,
                       "acc": float(acc) if acc is not None else None, "wall_s": time.time() - t0}
                if dz_fn is not None and step % 500 == 0:
                    try:
                        # candidate_loss RETIRED: it is circular (a global-collapse predictor that
                        # ignores B and z also sits at log K).  Log only delta_z; characterize the
                        # plateau with plain full-vocab CE vs the input-blind floor instead.
                        _, _, dz = dz_fn(step); rec["delta_z"] = dz
                    except Exception:
                        pass
                logf.write(json.dumps(rec) + "\n")
            if (not (lv == lv)) or lv > 5 * (init_loss or 1.0):
                diverged = True; break
    torch.save(model.state_dict(), out / "model_final.pt")
    logf.close()
    log = [json.loads(x) for x in open(out / "log.jsonl")]
    verdict, info = classify(log)
    summary = {"direction": a.direction, "eta": a.eta, "k": a.k, "n_unique_b": a.n_unique_b,
               "seed": a.seed, "final_step": step, "wall_s": time.time() - t0,
               "eta_drop_at": a.eta_drop_at, "eta_after": a.eta_after,
               "verdict": verdict, **info}
    json.dump(summary, open(out / "status.json", "w"), indent=2)
    print(f"[{a.direction} eta{a.eta:g} s{a.seed}] verdict={verdict} {info} ({step} steps, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
