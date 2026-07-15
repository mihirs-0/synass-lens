"""
Position-resolved candidate-set CE activation patching for MBC.

For cells A, B, D at transition+post and C at post (negative control),
this script patches `resid_post` at five different *position groups*
inside the input sequence and measures the candidate-set CE recovery
each yields. The goal is to determine where, in token-position space,
the answer-selection computation lives after z has been routed.

Position groups (token layout: BOS=0, B=1..6, SEP=7, z=8..9, SEP=10,
A=11..14, EOS=15):
    B           — base-string positions
    z           — selector positions
    sep_after_z — single SEP token between z and A (the position whose
                  residual produces the first-A logit)
    target_start — target_start position itself (the slot of A[0] in
                  the input)
    prefix_all  — all positions [BOS, B, SEP, z, SEP], i.e. everything
                  that precedes A

For each cell × phase × layer × position-group, recovery is:
    recovery = (L_corrupt - L_patched) / (L_corrupt - L_clean)
where L_* are candidate-set CE values over the K candidate A
completions, scored against the (B, z_clean, A_clean) target.

Outputs:
    eta_sweep/results/mech_interp_position_patching.json
    eta_sweep/results/mech_interp_position_patching_summary.md
    eta_sweep/results/figures/mech_interp_position_patching_heatmap.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

ETA_SWEEP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ETA_SWEEP_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from eta_sweep.config import RESULTS_DIR  # noqa: E402
from eta_sweep.analysis.mech_interp_mbc import (  # noqa: E402
    CELLS,
    load_cell,
    make_model,
    select_phase_ckpts,
    _select_device,
)
from eta_sweep.analysis.mech_interp_patching import (  # noqa: E402
    _build_candidate_tensors,
    _build_pairs,
)


# ---------------------------------------------------------------------------
# Position-group resolution
# ---------------------------------------------------------------------------

def position_groups(target_start: int, z_len: int, b_len: int) -> Dict[str, slice]:
    """Return slices over the canonical token layout.

    target_start = position of A[0]; B begins at position 1 (after BOS).
    Sequence: [BOS] [B*b_len] [SEP] [z*z_len] [SEP] [A*a_len] [EOS]
    so:
        BOS    = 0
        B      = 1 .. 1 + b_len
        SEP1   = 1 + b_len
        z      = 2 + b_len .. 2 + b_len + z_len
        SEP2   = 2 + b_len + z_len  (= target_start - 1)
        A      = target_start ..
    """
    bos = 0
    b_lo = 1
    b_hi = b_lo + b_len  # exclusive
    sep1 = b_hi
    z_lo = sep1 + 1
    z_hi = z_lo + z_len
    sep2 = z_hi  # == target_start - 1
    return {
        "B": slice(b_lo, b_hi),
        "z": slice(z_lo, z_hi),
        "sep_after_z": slice(sep2, sep2 + 1),
        "target_start": slice(target_start, target_start + 1),
        "prefix_all": slice(0, target_start),
    }


# ---------------------------------------------------------------------------
# Score one set of K candidates with optional patch
# ---------------------------------------------------------------------------

def _score(model, input_ids, target_starts, target_ends, labels,
           correct_index, fwd_hooks=None):
    with torch.no_grad():
        if fwd_hooks:
            logits = model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks)
        else:
            logits = model(input_ids)
    K = input_ids.shape[0]
    seq_lp = torch.zeros(K, dtype=torch.float32, device=logits.device)
    for k in range(K):
        for pos in range(target_starts[k], target_ends[k] + 1):
            tgt = labels[k, pos].item()
            if tgt == -100:
                continue
            log_probs = F.log_softmax(logits[k, pos - 1], dim=-1)
            seq_lp[k] += log_probs[tgt].float()
    norm = seq_lp - torch.logsumexp(seq_lp, dim=0)
    cand_loss = float(-norm[correct_index].item())
    pred = int(torch.argmax(norm).item())
    sorted_idx = torch.argsort(seq_lp, descending=True)
    rank = int((sorted_idx == correct_index).nonzero(as_tuple=True)[0].item())
    return cand_loss, int(pred == correct_index), rank


# ---------------------------------------------------------------------------
# Main per-(cell, phase) routine
# ---------------------------------------------------------------------------

def run_phase(model, tokenizer, mapping_data, n_pairs, seed, device):
    n_layers = model.cfg.n_layers
    pairs = _build_pairs(mapping_data, n_pairs, seed)

    L_clean_list = []
    L_corrupt_list = []
    acc_clean_list = []
    acc_corrupt_list = []

    # Layout constants are identical across all examples (fixed lengths).
    # We resolve them once per pair.

    # per_layer_per_group: list[list[list[float]]]
    # indexed [L][group_index][pair_index]
    GROUPS = ["B", "z", "sep_after_z", "target_start", "prefix_all"]
    losses = {g: [[] for _ in range(n_layers)] for g in GROUPS}

    for p in pairs:
        ids_clean, lbls_clean, ts_clean, te_clean = _build_candidate_tensors(
            tokenizer, p["base"], p["z_clean"], p["candidates"], device)
        ids_corr, lbls_corr, ts_corr, te_corr = _build_candidate_tensors(
            tokenizer, p["base"], p["z_corrupt"], p["candidates"], device)

        z_len = len(p["z_clean"])
        b_len = len(p["base"])
        target_start = ts_clean[0]
        groups_clean = position_groups(target_start, z_len, b_len)
        # The corrupt run uses the same length layout (z_corrupt has same
        # length as z_clean by construction in MBC).
        groups_corr = groups_clean

        # baseline
        Lc, ac, _ = _score(model, ids_clean, ts_clean, te_clean,
                           lbls_clean, p["idx_clean"])
        Lk, ak, _ = _score(model, ids_corr, ts_corr, te_corr,
                           lbls_corr, p["idx_clean"])
        L_clean_list.append(Lc)
        L_corrupt_list.append(Lk)
        acc_clean_list.append(ac)
        acc_corrupt_list.append(ak)

        # Cache once per pair
        with torch.no_grad():
            _, cache_clean = model.run_with_cache(ids_clean)

        # For each (layer, group) — one patched forward pass
        for L in range(n_layers):
            for g in GROUPS:
                sl_clean = groups_clean[g]
                sl_corr = groups_corr[g]
                hook_name = f"blocks.{L}.hook_resid_post"

                def patch_hook(act, hook,
                               L_=L, sl_clean_=sl_clean, sl_corr_=sl_corr):
                    src = cache_clean[("resid_post", L_)]
                    act[:, sl_corr_, :] = src[:, sl_clean_, :]
                    return act

                Lp, _, _ = _score(model, ids_corr, ts_corr, te_corr,
                                  lbls_corr, p["idx_clean"],
                                  fwd_hooks=[(hook_name, patch_hook)])
                losses[g][L].append(Lp)

    L_clean_m = float(np.mean(L_clean_list))
    L_corrupt_m = float(np.mean(L_corrupt_list))
    denom = L_corrupt_m - L_clean_m
    denom_unreliable = abs(denom) < 0.05

    def recovery_of(losses_list):
        m = float(np.mean(losses_list))
        if abs(denom) < 1e-9:
            return float("nan"), m
        return (L_corrupt_m - m) / denom, m

    table = {}
    for g in GROUPS:
        table[g] = []
        for L in range(n_layers):
            rec, mean_loss = recovery_of(losses[g][L])
            table[g].append({
                "layer": L,
                "L_patched_candidate": mean_loss,
                "recovery": rec,
            })
    return {
        "n_pairs": int(len(pairs)),
        "L_clean_candidate": L_clean_m,
        "L_corrupt_candidate": L_corrupt_m,
        "acc_clean": float(np.mean(acc_clean_list)),
        "acc_corrupt": float(np.mean(acc_corrupt_list)),
        "denom": denom,
        "denom_unreliable": bool(denom_unreliable),
        "groups": table,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args):
    device = args.device or _select_device()
    print(f"device: {device}")

    SCHEDULE = {
        "A_stable_K20":   ["transition", "post"],
        "B_critical_K20": ["transition", "post"],
        "C_stuck_K20":    ["post"],            # negative control
        "D_K36":          ["transition", "post"],
    }
    label_to_cell = {cs.label: cs for cs in CELLS}

    out = {"by_cell": {}, "device": device,
           "n_pairs": args.n_pairs}
    json_path = RESULTS_DIR / "mech_interp_position_patching.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    for label, phases in SCHEDULE.items():
        cs = label_to_cell[label]
        print(f"\n>>> POSITION patching: {label}")
        cc, tokenizer, mapping_data, train_ds, rows = load_cell(cs, device)
        phase_map = select_phase_ckpts(cs, rows)
        cell_out = {"phases": phase_map, "phase_results": {}}
        for phase in phases:
            step = phase_map.get(phase)
            if step is None:
                continue
            try:
                model = make_model(cc, tokenizer, step, device)
            except Exception as e:
                print(f"  load fail ({phase}): {e}")
                continue
            t0 = time.time()
            res = run_phase(model, tokenizer, mapping_data,
                            n_pairs=args.n_pairs, seed=0, device=device)
            res["step"] = step
            cell_out["phase_results"][phase] = res
            del model
            if device == "mps":
                torch.mps.empty_cache()
            print(f"  {phase} step={step}  L_clean={res['L_clean_candidate']:.3f}"
                  f"  L_corrupt={res['L_corrupt_candidate']:.3f}"
                  f"  denom={res['denom']:+.3f}"
                  f"  ({time.time()-t0:.1f}s)")
        out["by_cell"][label] = cell_out
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)

    write_summary_md(out)
    make_figures(out)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

GROUP_ORDER = ["B", "z", "sep_after_z", "target_start", "prefix_all"]


def write_summary_md(out: Dict):
    path = RESULTS_DIR / "mech_interp_position_patching_summary.md"
    lines = []
    lines.append("# Position-resolved candidate-set CE activation patching\n")
    lines.append(
        "For each (cell, phase, layer, position-group), patches the "
        "`resid_post` activations at the named position group from the "
        "*clean* run into the corrupted run; CE is computed over K "
        "candidate A completions for the (B, z_clean) target. Recovery "
        "= (L_corrupt − L_patched) / (L_corrupt − L_clean). Denom < 0.05 "
        "is flagged unreliable.\n"
    )
    lines.append(
        "Position groups (token layout: BOS, B, SEP, z, SEP, A, EOS):\n"
        "- **B** — base-string positions (6 tokens).\n"
        "- **z** — selector positions (2 tokens).\n"
        "- **sep_after_z** — the SEP token between z and A; this is the "
        "position whose residual produces the first-A logit "
        "(target_start − 1).\n"
        "- **target_start** — the slot of A[0] in the input (1 token).\n"
        "- **prefix_all** — all positions [BOS, B, SEP, z, SEP]. The "
        "ceiling for any prefix-only patching: if z has been routed into "
        "the prefix residual stream, this should recover everything.\n"
    )

    for label, cell in out["by_cell"].items():
        lines.append(f"## {label}\n")
        for phase, pr in cell["phase_results"].items():
            denom = pr["denom"]
            rel = "ok" if not pr["denom_unreliable"] else "denom<0.05"
            n_layers = len(pr["groups"]["z"])
            lines.append(
                f"### phase = {phase} (step {pr['step']}); "
                f"L_clean = {pr['L_clean_candidate']:.3f}, "
                f"L_corrupt = {pr['L_corrupt_candidate']:.3f}, "
                f"acc_clean = {pr['acc_clean']:.2f}, "
                f"acc_corrupt = {pr['acc_corrupt']:.2f}, "
                f"denom = {denom:+.3f} ({rel}).\n"
            )
            rows = []
            rows.append("| layer | " + " | ".join(GROUP_ORDER) + " |")
            rows.append("|---|" + "---|" * len(GROUP_ORDER))
            for L in range(n_layers):
                cells_ = []
                for g in GROUP_ORDER:
                    rec = pr["groups"][g][L]["recovery"]
                    cells_.append(f"{rec:+.2f}")
                rows.append(f"| L{L} | " + " | ".join(cells_) + " |")
            lines.append("\n".join(rows))
            lines.append("")
            # per-phase finding
            best = None
            for g in GROUP_ORDER:
                for L in range(n_layers):
                    rec = pr["groups"][g][L]["recovery"]
                    if best is None or rec > best[0]:
                        best = (rec, g, L)
            ts_rec_best = max(pr["groups"]["target_start"][L]["recovery"]
                              for L in range(n_layers))
            ts_rec_best_L = int(np.argmax(
                [pr["groups"]["target_start"][L]["recovery"]
                 for L in range(n_layers)]))
            z_rec_best = max(pr["groups"]["z"][L]["recovery"]
                             for L in range(n_layers))
            z_rec_best_L = int(np.argmax(
                [pr["groups"]["z"][L]["recovery"]
                 for L in range(n_layers)]))
            sep_rec_best = max(pr["groups"]["sep_after_z"][L]["recovery"]
                               for L in range(n_layers))
            sep_rec_best_L = int(np.argmax(
                [pr["groups"]["sep_after_z"][L]["recovery"]
                 for L in range(n_layers)]))
            prefix_rec_best = max(pr["groups"]["prefix_all"][L]["recovery"]
                                  for L in range(n_layers))
            prefix_rec_best_L = int(np.argmax(
                [pr["groups"]["prefix_all"][L]["recovery"]
                 for L in range(n_layers)]))
            lines.append(
                f"Best non-prefix group: **{best[1]}** at L{best[2]} = "
                f"{best[0]:+.2f}. Best `target_start` = {ts_rec_best:+.2f} "
                f"at L{ts_rec_best_L}; best `sep_after_z` = "
                f"{sep_rec_best:+.2f} at L{sep_rec_best_L}; best `z` = "
                f"{z_rec_best:+.2f} at L{z_rec_best_L}; best `prefix_all` "
                f"= {prefix_rec_best:+.2f} at L{prefix_rec_best_L}. "
                f"target_start vs z: "
                f"{'**target_start ≥ z** — pre-target slot carries more causal mass' if ts_rec_best >= z_rec_best else '**z ≥ target_start** — z slot still dominates'}.\n"
            )

    # cross-cell summary
    lines.append("## Cross-cell summary\n")
    lines.append(
        "The key comparison is the maximum recovery from the `target_start`/"
        "`sep_after_z` groups versus the `z` group, across reliable phases. "
        "Higher target_start than z means the answer-selection computation "
        "lives in the residual stream at the pre-target slot rather than "
        "at the z token positions; partial recovery from `prefix_all` "
        "indicates whether all prefix slots together (input-side patching) "
        "are sufficient to restore the candidate distribution.\n"
    )
    summary_rows = ["| cell | phase | denom | best z (L) | best sep (L) | best target_start (L) | best prefix_all (L) | target_start − z |"]
    summary_rows.append("|---|---|---|---|---|---|---|---|")
    for label, cell in out["by_cell"].items():
        for phase, pr in cell["phase_results"].items():
            if pr["denom_unreliable"]:
                summary_rows.append(
                    f"| {label} | {phase} | {pr['denom']:+.3f} (unrel) | "
                    f"— | — | — | — | — |"
                )
                continue
            n_layers = len(pr["groups"]["z"])
            def best_layer(g):
                vals = [pr["groups"][g][L]["recovery"]
                        for L in range(n_layers)]
                bL = int(np.argmax(vals))
                return vals[bL], bL
            zr, zL = best_layer("z")
            sr, sL = best_layer("sep_after_z")
            tr, tL = best_layer("target_start")
            pr_, pL = best_layer("prefix_all")
            summary_rows.append(
                f"| {label} | {phase} | {pr['denom']:+.3f} | "
                f"{zr:+.2f} (L{zL}) | {sr:+.2f} (L{sL}) | "
                f"{tr:+.2f} (L{tL}) | {pr_:+.2f} (L{pL}) | "
                f"{tr - zr:+.2f} |"
            )
    lines.append("\n".join(summary_rows))
    lines.append("")

    lines.append("## Reading\n")
    lines.append(
        "If `prefix_all` recovery approaches 1.0 at some layer, all "
        "z-relevant signal lives in the prefix residual stream at that "
        "layer; the deeper layers' job is then a fixed map from prefix "
        "→ A and patching after the prefix is unnecessary. If `prefix_all` "
        "is still well below 1.0 even at the highest layer, then "
        "answer-selection computation has spilled over into positions at "
        "or beyond `target_start`, and `target_start`-only patching "
        "should pick that up.\n"
        "If `target_start` recovery exceeds `z` recovery at any layer, "
        "the pre-target slot is carrying causal mass that the z slot "
        "alone is not — a positive sign that the residual stream at the "
        "first-A-prediction position has integrated z (via L0 routing) "
        "into an answer-conditioning representation that is downstream "
        "of pure z routing.\n"
    )
    path.write_text("\n".join(lines))
    print(f"  wrote {path}")


def make_figures(out: Dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # one column per (cell, phase), rows = layers, cols = position groups.
    panels = []
    for label, cell in out["by_cell"].items():
        for phase, pr in cell["phase_results"].items():
            panels.append((label, phase, pr))
    if not panels:
        return

    n = len(panels)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.0 * rows),
                             squeeze=False)
    for i, (label, phase, pr) in enumerate(panels):
        r, c = i // cols, i % cols
        ax = axes[r][c]
        n_layers = len(pr["groups"]["z"])
        mat = np.zeros((n_layers, len(GROUP_ORDER)))
        for li in range(n_layers):
            for gi, g in enumerate(GROUP_ORDER):
                mat[li, gi] = pr["groups"][g][li]["recovery"]
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                       vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(GROUP_ORDER)))
        ax.set_xticklabels(GROUP_ORDER, rotation=30, ha="right",
                           fontsize=8)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{li}" for li in range(n_layers)])
        flag = " *unreliable" if pr["denom_unreliable"] else ""
        ax.set_title(
            f"{label} {phase} (step {pr['step']})\n"
            f"denom={pr['denom']:+.2f}{flag}",
            fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.04)
        # annotate
        for li in range(n_layers):
            for gi in range(len(GROUP_ORDER)):
                ax.text(gi, li, f"{mat[li, gi]:+.2f}",
                        ha="center", va="center",
                        color="white" if abs(mat[li, gi]) > 0.6 else "black",
                        fontsize=7)
    # hide unused subplots
    for i in range(n, rows * cols):
        r, c = i // cols, i % cols
        axes[r][c].axis("off")
    fig.suptitle(
        "Position-resolved candidate-set CE recovery — "
        "patch resid_post at named position group, clean → z-corrupted",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "mech_interp_position_patching_heatmap.png",
                dpi=140)
    plt.close(fig)
    print(f"  wrote {fig_dir / 'mech_interp_position_patching_heatmap.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=32)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
