"""
Minimal causal-intervention pass for MBC mechanistic interpretability.

Tests the two-stage pathway "L0 z routing → target_start deep readout"
via five focused interventions on cells A, B, D at transition and post.
Candidate-set CE over the original correct A is the metric throughout.

Interventions:
    A. patch_ts_L23   — only target_start residual at L2 AND L3, clean → corrupt
    B. patch_z_L0     — only z-position residual at L0,        clean → corrupt
    C. patch_both     — A + B together (3 hooks)
    D. ablate_L0_keep_ts  — zero all four L0 attention heads in the corrupted
                          run, then patch target_start L2/L3 from clean
                          (tests sufficiency: does target_start patch
                          bypass L0 routing?)
    E. damage_clean   — REVERSE direction; run on clean input but patch
                          target_start L2/L3 from a corrupted-run cache
                          (tests necessity: does target_start carry
                          information whose removal breaks clean retrieval?)

For interventions A–D the metric is recovery:
    recovery = (L_corrupt − L_patched) / (L_corrupt − L_clean)
For intervention E the metric is damage_fraction:
    damage   = (L_damaged − L_clean) / (L_corrupt − L_clean)

Outputs:
    eta_sweep/results/mech_interp_minimal_causal.json
    eta_sweep/results/mech_interp_minimal_causal_summary.md
    eta_sweep/results/figures/mech_interp_minimal_causal.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

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
    CELLS, load_cell, make_model, select_phase_ckpts, _select_device,
)
from eta_sweep.analysis.mech_interp_patching import (  # noqa: E402
    _build_candidate_tensors, _build_pairs,
)


# ---------------------------------------------------------------------------
# Scoring
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
    return {
        "L": cand_loss, "correct": int(pred == correct_index), "rank": rank,
    }


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------

def _make_resid_post_patch(cache_source, layer, src_slice, dst_slice):
    """Patch resid_post at `dst_slice` of the running stream from
    `cache_source[('resid_post', layer)][:, src_slice, :]`."""
    src_tensor = cache_source[("resid_post", layer)]

    def hf(act, hook):
        act[:, dst_slice, :] = src_tensor[:, src_slice, :]
        return act
    return (f"blocks.{layer}.hook_resid_post", hf)


def _make_l0_attn_zero():
    """Zero all heads of the L0 attention output (hook_z)."""
    def hf(act, hook):
        act[:, :, :, :] = 0.0
        return act
    return ("blocks.0.attn.hook_z", hf)


# ---------------------------------------------------------------------------
# Per-(cell, phase) driver
# ---------------------------------------------------------------------------

INTERVENTIONS = ["patch_ts_L23", "patch_z_L0", "patch_both",
                 "ablate_L0_keep_ts", "damage_clean"]


def run_phase(model, tokenizer, mapping_data, n_pairs, seed, device):
    pairs = _build_pairs(mapping_data, n_pairs, seed)

    # Per-pair accumulators
    L_clean_list, L_corr_list = [], []
    acc_clean_list, acc_corr_list = [], []
    rank_clean_list, rank_corr_list = [], []
    inter_L = {n: [] for n in INTERVENTIONS}
    inter_acc = {n: [] for n in INTERVENTIONS}
    inter_rank = {n: [] for n in INTERVENTIONS}

    for p in pairs:
        ids_clean, lbls_clean, ts_clean, te_clean = _build_candidate_tensors(
            tokenizer, p["base"], p["z_clean"], p["candidates"], device)
        ids_corr, lbls_corr, ts_corr, te_corr = _build_candidate_tensors(
            tokenizer, p["base"], p["z_corrupt"], p["candidates"], device)

        z_len = len(p["z_clean"])
        b_len = len(p["base"])
        target_start = ts_clean[0]
        z_lo = 2 + b_len
        z_hi = z_lo + z_len  # exclusive
        ts_slice = slice(target_start, target_start + 1)
        z_slice = slice(z_lo, z_hi)

        # Caches built once per pair
        with torch.no_grad():
            _, cache_clean = model.run_with_cache(ids_clean)
            _, cache_corr = model.run_with_cache(ids_corr)

        # baselines
        sc = _score(model, ids_clean, ts_clean, te_clean, lbls_clean,
                    p["idx_clean"])
        L_clean_list.append(sc["L"])
        acc_clean_list.append(sc["correct"])
        rank_clean_list.append(sc["rank"])
        sc = _score(model, ids_corr, ts_corr, te_corr, lbls_corr,
                    p["idx_clean"])
        L_corr_list.append(sc["L"])
        acc_corr_list.append(sc["correct"])
        rank_corr_list.append(sc["rank"])

        # ---- A. patch_ts_L23 ----
        hooks = [
            _make_resid_post_patch(cache_clean, 2, ts_slice, ts_slice),
            _make_resid_post_patch(cache_clean, 3, ts_slice, ts_slice),
        ]
        sc = _score(model, ids_corr, ts_corr, te_corr, lbls_corr,
                    p["idx_clean"], fwd_hooks=hooks)
        inter_L["patch_ts_L23"].append(sc["L"])
        inter_acc["patch_ts_L23"].append(sc["correct"])
        inter_rank["patch_ts_L23"].append(sc["rank"])

        # ---- B. patch_z_L0 ----
        hooks = [
            _make_resid_post_patch(cache_clean, 0, z_slice, z_slice),
        ]
        sc = _score(model, ids_corr, ts_corr, te_corr, lbls_corr,
                    p["idx_clean"], fwd_hooks=hooks)
        inter_L["patch_z_L0"].append(sc["L"])
        inter_acc["patch_z_L0"].append(sc["correct"])
        inter_rank["patch_z_L0"].append(sc["rank"])

        # ---- C. patch_both ----
        hooks = [
            _make_resid_post_patch(cache_clean, 0, z_slice, z_slice),
            _make_resid_post_patch(cache_clean, 2, ts_slice, ts_slice),
            _make_resid_post_patch(cache_clean, 3, ts_slice, ts_slice),
        ]
        sc = _score(model, ids_corr, ts_corr, te_corr, lbls_corr,
                    p["idx_clean"], fwd_hooks=hooks)
        inter_L["patch_both"].append(sc["L"])
        inter_acc["patch_both"].append(sc["correct"])
        inter_rank["patch_both"].append(sc["rank"])

        # ---- D. ablate_L0_keep_ts ----
        # Zero the L0 attention output AT ALL positions, then patch
        # target_start at L2/L3 from clean. The L0 ablation prevents
        # any z routing through L0 attention; the target_start patch
        # injects the clean target_start residual at L2/L3.
        # NOTE: the L0 ablation is applied via the hook_z hook so it
        # zeros the per-head outputs of attention, which then propagate
        # through hook_attn_out and into resid_post.
        hooks = [
            _make_l0_attn_zero(),
            _make_resid_post_patch(cache_clean, 2, ts_slice, ts_slice),
            _make_resid_post_patch(cache_clean, 3, ts_slice, ts_slice),
        ]
        sc = _score(model, ids_corr, ts_corr, te_corr, lbls_corr,
                    p["idx_clean"], fwd_hooks=hooks)
        inter_L["ablate_L0_keep_ts"].append(sc["L"])
        inter_acc["ablate_L0_keep_ts"].append(sc["correct"])
        inter_rank["ablate_L0_keep_ts"].append(sc["rank"])

        # ---- E. damage_clean (necessity) ----
        # Run on CLEAN inputs but inject CORRUPTED-run target_start
        # residual at L2/L3. Source is cache_corr; destination is the
        # clean run's target_start slot.
        hooks = [
            _make_resid_post_patch(cache_corr, 2, ts_slice, ts_slice),
            _make_resid_post_patch(cache_corr, 3, ts_slice, ts_slice),
        ]
        sc = _score(model, ids_clean, ts_clean, te_clean, lbls_clean,
                    p["idx_clean"], fwd_hooks=hooks)
        inter_L["damage_clean"].append(sc["L"])
        inter_acc["damage_clean"].append(sc["correct"])
        inter_rank["damage_clean"].append(sc["rank"])

    L_clean = float(np.mean(L_clean_list))
    L_corr = float(np.mean(L_corr_list))
    denom = L_corr - L_clean
    denom_unreliable = abs(denom) < 0.05

    def recovery(name):
        m = float(np.mean(inter_L[name]))
        if abs(denom) < 1e-9:
            return float("nan")
        return (L_corr - m) / denom

    def damage(name):
        m = float(np.mean(inter_L[name]))
        if abs(denom) < 1e-9:
            return float("nan")
        return (m - L_clean) / denom

    out = {
        "n_pairs": len(pairs),
        "L_clean": L_clean,
        "L_corrupt": L_corr,
        "denom": denom,
        "denom_unreliable": denom_unreliable,
        "acc_clean": float(np.mean(acc_clean_list)),
        "acc_corrupt": float(np.mean(acc_corr_list)),
        "rank_clean": float(np.mean(rank_clean_list)),
        "rank_corrupt": float(np.mean(rank_corr_list)),
        "interventions": {},
    }
    for name in INTERVENTIONS:
        m_L = float(np.mean(inter_L[name]))
        m_acc = float(np.mean(inter_acc[name]))
        m_rank = float(np.mean(inter_rank[name]))
        if name == "damage_clean":
            metric = damage(name)
            out["interventions"][name] = {
                "L_patched": m_L,
                "acc_patched": m_acc,
                "rank_patched": m_rank,
                "damage_fraction": metric,
            }
        else:
            metric = recovery(name)
            out["interventions"][name] = {
                "L_patched": m_L,
                "acc_patched": m_acc,
                "rank_patched": m_rank,
                "recovery": metric,
            }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args):
    device = args.device or _select_device()
    print(f"device: {device}")

    SCHEDULE = {
        "A_stable_K20":   ["transition", "post"],
        "B_critical_K20": ["transition", "post"],
        "D_K36":          ["transition", "post"],
    }
    label_to_cell = {cs.label: cs for cs in CELLS}
    out = {"by_cell": {}, "device": device, "n_pairs": args.n_pairs}
    json_path = RESULTS_DIR / "mech_interp_minimal_causal.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    for label, phases in SCHEDULE.items():
        cs = label_to_cell[label]
        print(f"\n>>> minimal-causal: {label}")
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
            print(f"  {phase} step={step}  L_clean={res['L_clean']:.3f}"
                  f"  L_corrupt={res['L_corrupt']:.3f}"
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

INT_DESCRIPTIONS = {
    "patch_ts_L23":      "patch target_start residual at L2 + L3",
    "patch_z_L0":        "patch z-position residual at L0",
    "patch_both":        "patch_ts_L23 + patch_z_L0 together",
    "ablate_L0_keep_ts": "zero L0 attention + patch_ts_L23",
    "damage_clean":      "patch target_start L2 + L3 from corrupt INTO clean",
}


def write_summary_md(out: Dict):
    path = RESULTS_DIR / "mech_interp_minimal_causal_summary.md"
    lines = []
    lines.append("# Minimal causal interventions: target_start vs L0 z-routing\n")
    lines.append(
        "Each row is a (cell, phase) checkpoint. Recovery = "
        "`(L_corrupt − L_patched) / (L_corrupt − L_clean)`. For "
        "`damage_clean` we report `damage_fraction = "
        "(L_damaged − L_clean) / (L_corrupt − L_clean)` — fraction of the "
        "corruption gap re-introduced when corrupted target_start "
        "residuals are injected into a clean run. Denom < 0.05 flagged "
        "unreliable.\n"
    )
    lines.append("Intervention legend:\n")
    for name, desc in INT_DESCRIPTIONS.items():
        lines.append(f"- **{name}**: {desc}.")
    lines.append("")

    for label, cell in out["by_cell"].items():
        for phase, pr in cell["phase_results"].items():
            denom = pr["denom"]
            rel = "ok" if not pr["denom_unreliable"] else "denom<0.05"
            lines.append(f"## {label} — {phase} (step {pr['step']})\n")
            lines.append(
                f"L_clean = {pr['L_clean']:.3f}, L_corrupt = "
                f"{pr['L_corrupt']:.3f}, denom = {denom:+.3f} ({rel}). "
                f"acc_clean = {pr['acc_clean']:.2f}, acc_corrupt = "
                f"{pr['acc_corrupt']:.2f}, rank_clean = "
                f"{pr['rank_clean']:.2f}, rank_corrupt = "
                f"{pr['rank_corrupt']:.2f}.\n"
            )
            rows = []
            rows.append(
                "| intervention | L_patched | metric | acc_patched | rank_patched |"
            )
            rows.append("|---|---|---|---|---|")
            for name in INTERVENTIONS:
                d = pr["interventions"][name]
                if name == "damage_clean":
                    metric_label = f"damage = {d['damage_fraction']:+.2f}"
                else:
                    metric_label = f"recovery = {d['recovery']:+.2f}"
                rows.append(
                    f"| {name} | {d['L_patched']:.3f} | {metric_label} | "
                    f"{d['acc_patched']:.2f} | {d['rank_patched']:.2f} |"
                )
            lines.append("\n".join(rows))
            lines.append("")

            # Per-phase synthesis
            r_ts = pr["interventions"]["patch_ts_L23"]["recovery"]
            r_z = pr["interventions"]["patch_z_L0"]["recovery"]
            r_both = pr["interventions"]["patch_both"]["recovery"]
            r_abl = pr["interventions"]["ablate_L0_keep_ts"]["recovery"]
            d_dam = pr["interventions"]["damage_clean"]["damage_fraction"]
            additivity = (
                "**redundant** (combined ≈ max of singletons)"
                if r_both <= max(r_ts, r_z) + 0.05
                else (
                    "**additive** (combined ≈ ts + z)"
                    if abs(r_both - (r_ts + r_z)) < 0.10
                    else (
                        "**superadditive** (combined > ts + z)"
                        if r_both > r_ts + r_z + 0.05
                        else "**subadditive but exceeds either singleton**"
                    )
                )
            )
            ablation_bypass = (
                "**target_start patch survives L0 ablation** — sufficient"
                if r_abl >= r_ts - 0.10
                else "**target_start patch loses ground under L0 ablation** — depends on L0"
            )
            damage_summary = (
                "**target_start L2/L3 is necessary**: corrupting it "
                f"alone re-introduces {d_dam:+.2f} of the gap"
                if d_dam >= 0.30
                else "target_start damage is partial / weak"
            )
            lines.append(
                f"Synthesis: target_start L2/L3 recovery = "
                f"{r_ts:+.2f}; z@L0 recovery = {r_z:+.2f}; combined = "
                f"{r_both:+.2f} → {additivity}. With L0 attention "
                f"ablated, target_start patch recovery = {r_abl:+.2f} "
                f"({ablation_bypass}). Reverse damage = {d_dam:+.2f} "
                f"({damage_summary}).\n"
            )

    # Cross-cell summary table
    lines.append("## Cross-cell summary\n")
    rows = ["| cell | phase | denom | recov ts | recov z | recov both | recov ablate+ts | damage |"]
    rows.append("|---|---|---|---|---|---|---|---|")
    for label, cell in out["by_cell"].items():
        for phase, pr in cell["phase_results"].items():
            denom = pr["denom"]
            r_ts = pr["interventions"]["patch_ts_L23"]["recovery"]
            r_z = pr["interventions"]["patch_z_L0"]["recovery"]
            r_both = pr["interventions"]["patch_both"]["recovery"]
            r_abl = pr["interventions"]["ablate_L0_keep_ts"]["recovery"]
            d_dam = pr["interventions"]["damage_clean"]["damage_fraction"]
            flag = "" if not pr["denom_unreliable"] else " (unrel)"
            rows.append(
                f"| {label} | {phase} | {denom:+.3f}{flag} | "
                f"{r_ts:+.2f} | {r_z:+.2f} | {r_both:+.2f} | "
                f"{r_abl:+.2f} | {d_dam:+.2f} |"
            )
    lines.append("\n".join(rows))
    lines.append("")

    lines.append("## Reading\n")
    lines.append(
        "- If `recov ts` > `recov z` and `recov both` ≤ `recov ts` + small, "
        "**most causal mass has migrated to target_start**: patching the "
        "z slot in addition to the deep target_start slot adds little.\n"
        "- If `recov ablate+ts` ≈ `recov ts`, **target_start patching "
        "bypasses L0 routing**: the deep readout is downstream-sufficient "
        "and does not need a working L0 to convert into the answer (the "
        "patched residual already encodes the answer-relevant features).\n"
        "- If `damage` is large (≥ 0.5), **target_start L2/L3 is "
        "necessary**: removing the clean signal there alone re-introduces "
        "the corruption gap.\n"
    )
    path.write_text("\n".join(lines))
    print(f"  wrote {path}")


def make_figures(out: Dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    row_labels = []
    for label, cell in out["by_cell"].items():
        for phase, pr in cell["phase_results"].items():
            row = []
            for name in INTERVENTIONS:
                d = pr["interventions"][name]
                if name == "damage_clean":
                    row.append(d["damage_fraction"])
                else:
                    row.append(d["recovery"])
            rows.append(row)
            flag = "*" if pr["denom_unreliable"] else ""
            row_labels.append(
                f"{label}\n{phase}{flag}\n(step {pr['step']}, "
                f"denom={pr['denom']:+.2f})")

    arr = np.array(rows)
    fig, ax = plt.subplots(figsize=(9, 1.0 * len(rows) + 1.5))
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.5)
    ax.set_xticks(range(len(INTERVENTIONS)))
    ax.set_xticklabels(INTERVENTIONS, rotation=20, ha="right",
                       fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:+.2f}",
                    ha="center", va="center",
                    color="white" if abs(arr[i, j]) > 0.6 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Minimal causal interventions: candidate-set CE recovery\n"
                 "(damage_clean reports damage_fraction)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "mech_interp_minimal_causal.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {fig_dir / 'mech_interp_minimal_causal.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=32)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
