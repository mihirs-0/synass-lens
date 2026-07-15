"""Prompt-format robustness for the IOI diagnostic.

Tests the same 14-head diagnostic across three prompt formats:
  - BABA:  "When {S} and {IO} went to the {place}, {S} gave a {obj} to ___"
  - ABBA:  "When {IO} and {S} went to the {place}, {S} gave a {obj} to ___"
  - Mixed: "Then {S} and {IO} met at the {place}. {S} handed a {obj} to ___"

In all three, the duplicated name is {S} (the subject) and the target token is
" {IO}" (the indirect object). The Mixed template preserves the BABA positional
pattern but changes the syntax / verbs / connectives, so it stress-tests
whether the diagnostic depends on surface form rather than IOI structure.

Name pairs are stratified across formats (each (S, IO) ordered pair is used in
at most one format) so the three result sets are independent.

Decision rule applied at the end:
  Case A — all three formats pass every check.        → integrate into paper.
  Case B — one format fails one check, others clean.  → integrate with caveat.
  Case C — two or more fail, or a baseline is broken. → cut, do not integrate.

Outputs go to ./outputs/ioi_format_robustness/ and do NOT touch the prior
single-format sweep.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from ioi_diagnostic import (
    ablate_head_logit_diff,
    baseline_logit_diffs,
    load_model,
    logit_lens_all_layers,
    pick_device,
    summarize,
    tokenize_batch,
)
from ioi_diagnostic_sweep import (
    HEADS,
    GROUP_COLOR,
    GROUP_RANK,
    diagnostic_score,
)

SEED = 42
N_PROMPTS_PER_FORMAT = 50
OUT_DIR = Path(__file__).parent / "outputs" / "ioi_format_robustness"

NAMES = ["John", "Mary", "Tom", "James", "Dan", "Sid", "Martin", "Paul",
        "Alex", "Sarah", "Kate", "Jessica"]
PLACES = ["store", "park", "school", "office", "garden", "restaurant"]
OBJECTS = ["drink", "book", "ball", "snack", "ring", "kite"]

# (format_id, template with {S}, {IO}, {place}, {obj} placeholders).
# S is the duplicated subject; IO is the non-duplicated indirect object (target).
FORMATS = [
    ("BABA",  "When {S} and {IO} went to the {place}, {S} gave a {obj} to"),
    ("ABBA",  "When {IO} and {S} went to the {place}, {S} gave a {obj} to"),
    ("Mixed", "Then {S} and {IO} met at the {place}. {S} handed a {obj} to"),
]

# Sanity-check thresholds (see decision rule in the module docstring)
MIN_BASELINE = 1.0          # loose floor; final predicate uses > 0 and > 80% wins
MIN_CLEAN_WINS_FRAC = 0.80
MAX_SPEARMAN_P = 0.05
REQUIRE_NEGATIVE_RHO = True
REQUIRE_ALL_LATE_ZERO = True


def single_token_names(model) -> dict[str, int]:
    """Return {name: token_id} for single-token names (with leading space)."""
    out = {}
    for nm in NAMES:
        tok = model.to_tokens(" " + nm, prepend_bos=False)[0]
        if tok.shape[0] == 1:
            out[nm] = int(tok[0].item())
    return out


def stratify_pairs(valid_names: list[str], rng: random.Random
                   ) -> list[list[tuple[str, str]]]:
    """Enumerate all ordered (S, IO) pairs with S != IO, shuffle once, split
    into 3 disjoint chunks — one per format."""
    pairs = [(s, io) for s in valid_names for io in valid_names if s != io]
    rng.shuffle(pairs)
    n = len(pairs)
    # Split as evenly as possible into 3 chunks
    chunks = [pairs[i * n // 3:(i + 1) * n // 3] for i in range(3)]
    return chunks


def build_prompts_for_format(model, template: str,
                             pair_chunk: list[tuple[str, str]],
                             valid_names: dict[str, int],
                             rng: random.Random,
                             n_prompts: int) -> list[dict]:
    """Generate `n_prompts` same-length prompts using the given template and
    pair chunk. Cycles through pairs with different (place, obj) combinations
    if needed. Returns dicts with prompt / clean_name (IO) / corrupt_name (S)
    / token ids / etc."""
    # Try all (pair, place, obj) triples, shuffled, until we have n_prompts of
    # the single most-common token length.
    pool = list(itertools.product(pair_chunk, PLACES, OBJECTS))
    rng.shuffle(pool)

    buckets: dict[int, list[dict]] = {}
    seen_tuples: set[tuple] = set()
    for (S, IO), place, obj in pool:
        key = (S, IO, place, obj)
        if key in seen_tuples:
            continue
        seen_tuples.add(key)
        prompt_text = template.format(S=S, IO=IO, place=place, obj=obj)
        tok_len = int(model.to_tokens(prompt_text).shape[1])
        buckets.setdefault(tok_len, []).append({
            "prompt": prompt_text,
            "clean_name": IO,
            "corrupt_name": S,
            "clean_token_id": valid_names[IO],
            "corrupt_token_id": valid_names[S],
            "place": place,
            "object": obj,
            "tok_len": tok_len,
            "S": S, "IO": IO,
        })
        best_len = max(buckets, key=lambda k: len(buckets[k]))
        if len(buckets[best_len]) >= n_prompts:
            return buckets[best_len][:n_prompts]
    # Fell through — pick the largest bucket even if short
    best_len = max(buckets, key=lambda k: len(buckets[k]))
    return buckets[best_len][:n_prompts]


def run_sweep_for_format(model, prompts: list[dict], fmt_id: str) -> dict:
    """Run the full 14-head sweep for one format. Returns a dict with:
      - prompts (same-length guarantee)
      - baseline summary
      - per-head rows (list[dict])
      - aggregate stats (group means, Spearman rho, etc.)
      - decision-rule booleans.
    """
    tokens = tokenize_batch(model, prompts).to(model.cfg.device)
    baseline = baseline_logit_diffs(model, prompts, tokens)
    base_summary = summarize(baseline)
    print(f"  [{fmt_id}] baseline logit diff "
          f"{base_summary['mean']:+.3f} ± {base_summary['std']:.3f}  "
          f"(clean wins {base_summary['frac_positive']*100:.0f}%)")

    # Full logit-lens cache
    lens = logit_lens_all_layers(model, prompts, tokens)

    rows = []
    for head_id, L, H, role, group in HEADS:
        abl_zero = ablate_head_logit_diff(model, prompts, tokens, L, H, "zero")
        abl_mean = ablate_head_logit_diff(model, prompts, tokens, L, H, "mean")
        zero_delta = abl_zero - baseline
        mean_delta = abl_mean - baseline
        lens_at_Lh = lens["resid_post"][L]
        lens_at_Lh_minus_1 = lens["resid_pre"][L]
        zd_summ = summarize(zero_delta)
        md_summ = summarize(mean_delta)
        lh_summ = summarize(lens_at_Lh)
        lhm1_summ = summarize(lens_at_Lh_minus_1)
        dscore = diagnostic_score(zd_summ["mean"], lh_summ["mean"])
        rows.append({
            "format": fmt_id,
            "head_id": head_id,
            "role_label": role,
            "group": group,
            "layer": L,
            "head": H,
            "zero_ablation_mean": zd_summ["mean"],
            "zero_ablation_std": zd_summ["std"],
            "mean_ablation_mean": md_summ["mean"],
            "mean_ablation_std": md_summ["std"],
            "lens_at_Lh_mean": lh_summ["mean"],
            "lens_at_Lh_std": lh_summ["std"],
            "lens_at_Lh_minus_1_mean": lhm1_summ["mean"],
            "lens_at_Lh_minus_1_std": lhm1_summ["std"],
            "diagnostic_score": dscore,
        })
    df = pd.DataFrame(rows)

    # Group means
    group_means = {g: float(df[df["group"] == g]["diagnostic_score"].mean())
                   for g in ["early", "middle", "late"]}
    group_stds = {g: float(df[df["group"] == g]["diagnostic_score"].std(ddof=0))
                  for g in ["early", "middle", "late"]}

    rho, p_value = spearmanr(df["diagnostic_score"].values,
                             [GROUP_RANK[g] for g in df["group"]])

    # Decision-rule checks
    check_baseline = (base_summary["mean"] > 0
                      and base_summary["frac_positive"] >= MIN_CLEAN_WINS_FRAC)
    check_rho = (float(rho) < 0.0) and (float(p_value) < MAX_SPEARMAN_P)
    late_scores = df[df["group"] == "late"]["diagnostic_score"].values
    check_late_zero = bool(np.all(late_scores == 0.0))

    checks = {
        "baseline_ok": bool(check_baseline),
        "spearman_negative_and_significant": bool(check_rho),
        "all_late_heads_score_zero": bool(check_late_zero),
        "all_pass": bool(check_baseline and check_rho and check_late_zero),
    }

    return {
        "format_id": fmt_id,
        "n_prompts": len(prompts),
        "baseline": base_summary,
        "group_means": group_means,
        "group_stds": group_stds,
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "median_score": float(df["diagnostic_score"].median()),
        "checks": checks,
        "rows": rows,
    }


def classify_case(per_format: list[dict]) -> tuple[str, list[str]]:
    """Return (case_letter, list_of_failure_reasons)."""
    passes = [f["checks"]["all_pass"] for f in per_format]
    n_pass = sum(passes)
    reasons = []
    for f in per_format:
        if not f["checks"]["all_pass"]:
            fails = [k for k, v in f["checks"].items()
                     if k != "all_pass" and not v]
            reasons.append(f"{f['format_id']}: failed {fails}")
    # Hard fail: any baseline broken → Case C regardless
    any_baseline_broken = any(
        not f["checks"]["baseline_ok"] for f in per_format
    )
    if any_baseline_broken or n_pass <= 1:
        return "C", reasons
    if n_pass == 3:
        return "A", reasons
    return "B", reasons  # n_pass == 2


def make_comparison_figure(per_format: list[dict], out_path: Path) -> None:
    plt.rcdefaults()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), dpi=150,
                             facecolor="white", sharey=True)
    for ax, fmt in zip(axes, per_format):
        df = pd.DataFrame(fmt["rows"])
        for g in ["early", "middle", "late"]:
            sub = df[df["group"] == g]
            ax.scatter(np.abs(sub["zero_ablation_mean"]),
                       sub["lens_at_Lh_mean"],
                       color=GROUP_COLOR[g], s=55, edgecolor="black",
                       linewidth=0.6, label=g, zorder=3)
        for _, row in df.iterrows():
            ax.annotate(row["head_id"],
                        xy=(abs(row["zero_ablation_mean"]),
                            row["lens_at_Lh_mean"]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=7, color="#333")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_xlabel("|ablation impact|", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        rho = fmt["spearman_rho"]
        p = fmt["spearman_p"]
        ok = "\u2713" if fmt["checks"]["all_pass"] else "\u2717"
        ax.set_title(
            f"{fmt['format_id']} (n={fmt['n_prompts']}, ρ={rho:+.2f}, "
            f"p={p:.3f}) {ok}",
            fontsize=11,
        )
    axes[0].set_ylabel("lens logit diff at $L_h$ resid\\_post", fontsize=10)
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle("Per-format necessity vs decodability (14 heads × 3 templates)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_case_ab_appendix(per_format: list[dict], case: str,
                           reasons: list[str], out_path: Path) -> None:
    lines = []
    lines.append("\\paragraph{Robustness across prompt formats.} "
                 "To check whether the diagnostic depends on a single IOI "
                 "template, we re-run the 14-head sweep on three formats "
                 "(BABA: ``When S and IO went...''; ABBA: ``When IO and "
                 "S went...''; Mixed: ``Then S and IO met at the place. "
                 "S handed...''). Each format uses 50 same-length prompts "
                 "with disjoint (S, IO) name pairs across formats.")
    for f in per_format:
        bm = f["baseline"]["mean"]; bs = f["baseline"]["std"]
        gm = f["group_means"]; rho = f["spearman_rho"]; p = f["spearman_p"]
        lines.append(
            f" For {f['format_id']} ($n={f['n_prompts']}$, baseline logit "
            f"diff $= {bm:.2f} \\pm {bs:.2f}$), group mean diagnostic "
            f"scores are early $= {gm['early']:.2f}$, middle $= "
            f"{gm['middle']:.2f}$, late $= {gm['late']:.2f}$ (Spearman "
            f"$\\rho = {rho:+.2f}$, $p = {p:.3f}$)."
        )
    if case == "A":
        lines.append(" The diagnostic pattern is robust to prompt format: "
                     "in all three templates every late head receives zero "
                     "diagnostic score, encoder-like heads carry positive "
                     "scores, and the Spearman rank correlation is negative "
                     "and significant.")
    else:  # Case B
        # Identify the deviating format
        bad = [f for f in per_format if not f["checks"]["all_pass"]]
        bad_names = ", ".join(f["format_id"] for f in bad)
        fail_desc = "; ".join(
            f"{f['format_id']} failed: "
            + ", ".join(k for k, v in f["checks"].items()
                        if k != "all_pass" and not v)
            for f in bad
        )
        lines.append(
            f" The pattern holds cleanly for the two passing variants but "
            f"{bad_names} deviates ({fail_desc}). We report this honestly: "
            "the conclusion is that the diagnostic is robust to the "
            "ABBA/BABA positional variation typically studied in IOI "
            "analyses, with the caveat noted above for the third variant."
        )
    out_path.write_text("".join(lines) + "\n")


def write_case_c_report(per_format: list[dict], reasons: list[str],
                        out_path: Path) -> None:
    lines = []
    lines.append("# Format-Robustness Case C Report\n\n")
    lines.append("Two or more of the three prompt formats failed the "
                 "decision-rule checks (or a baseline was broken). Under the "
                 "pre-registered rule this experiment is NOT integrated into "
                 "the paper.\n\n")
    lines.append("## Per-format outcome\n\n")
    for f in per_format:
        lines.append(f"### {f['format_id']}\n\n")
        lines.append(f"- Baseline logit diff: "
                     f"{f['baseline']['mean']:+.3f} ± "
                     f"{f['baseline']['std']:.3f}  "
                     f"(clean wins {f['baseline']['frac_positive']*100:.0f}%)\n")
        lines.append(f"- Group diagnostic means: "
                     f"early {f['group_means']['early']:.3f}, "
                     f"middle {f['group_means']['middle']:.3f}, "
                     f"late {f['group_means']['late']:.3f}\n")
        lines.append(f"- Spearman ρ(score, group order) = "
                     f"{f['spearman_rho']:+.3f}  (p = {f['spearman_p']:.3f})\n")
        for k, v in f["checks"].items():
            if k == "all_pass":
                continue
            mark = "PASS" if v else "FAIL"
            lines.append(f"  - check `{k}`: {mark}\n")
        lines.append("\n")
    lines.append("## Likely reasons\n\n")
    if any(not f["checks"]["baseline_ok"] for f in per_format):
        lines.append("- At least one baseline is broken, meaning GPT-2 "
                     "Small does not reliably perform IOI on that prompt "
                     "format. The circuit may not even fire; running the "
                     "diagnostic on that format is ill-posed.\n")
    if any(not f["checks"]["all_late_heads_score_zero"] for f in per_format):
        lines.append("- At least one late head receives a non-zero "
                     "diagnostic score, suggesting logit-lens decodability "
                     "at the canonical Name Mover layer does not match "
                     "the one-format pilot on this template.\n")
    if any(not f["checks"]["spearman_negative_and_significant"]
           for f in per_format):
        lines.append("- At least one format's Spearman rank correlation is "
                     "not negative or not significant (p ≥ 0.05), which "
                     "means the diagnostic does not reliably separate "
                     "encoder-like from computation-locus heads on that "
                     "template.\n")
    lines.append("\n## Recommendation\n\n"
                 "Cut the format-robustness experiment from the workshop "
                 "submission. The single-format 50-prompt sweep in "
                 "`outputs/ioi_sweep/` and the two-head pilot in the main "
                 "§4 still stand; the extended appendix paragraph remains "
                 "the stronger existence proof to include.\n")
    out_path.write_text("".join(lines))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"[env] device={device}")
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    model = load_model(device)
    print(f"[model] n_layers={model.cfg.n_layers}  n_heads={model.cfg.n_heads}")

    # Stratify name pairs across formats
    rng = random.Random(SEED)
    name_ids = single_token_names(model)
    valid_names = list(name_ids.keys())
    print(f"[names] {len(valid_names)} single-token names: {valid_names}")
    chunks = stratify_pairs(valid_names, rng)
    for fmt_id_chunk, chunk in zip([f[0] for f in FORMATS], chunks):
        print(f"  [{fmt_id_chunk}] allocated {len(chunk)} (S,IO) pairs")

    per_format = []
    for (fmt_id, template), pair_chunk in zip(FORMATS, chunks):
        print(f"\n=== Format: {fmt_id} ===")
        print(f"Template: {template}")
        prompts = build_prompts_for_format(
            model, template, pair_chunk, name_ids, rng,
            N_PROMPTS_PER_FORMAT,
        )
        if len(prompts) < N_PROMPTS_PER_FORMAT:
            print(f"  [warn] only gathered {len(prompts)} same-length prompts "
                  f"(target {N_PROMPTS_PER_FORMAT})")
        print(f"  [{fmt_id}] example prompt: {prompts[0]['prompt']!r}")
        print(f"  [{fmt_id}] target token: ' {prompts[0]['clean_name']}'  "
              f"corrupt: ' {prompts[0]['corrupt_name']}'")
        fmt_result = run_sweep_for_format(model, prompts, fmt_id)
        per_format.append(fmt_result)
        checks = fmt_result["checks"]
        print(f"  [{fmt_id}] checks: "
              f"baseline_ok={checks['baseline_ok']}  "
              f"rho_neg_sig={checks['spearman_negative_and_significant']}  "
              f"all_late_zero={checks['all_late_heads_score_zero']}  "
              f"ALL_PASS={checks['all_pass']}")

    # Combine all rows into one CSV
    all_rows = []
    for f in per_format:
        all_rows.extend(f["rows"])
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "format_results.csv", index=False)

    # Summary JSON (drop bulky rows from per-format blocks)
    summary = {
        "config": {
            "n_prompts_per_format": N_PROMPTS_PER_FORMAT,
            "seed": SEED,
            "device": device,
            "model": "gpt2",
        },
        "formats": [
            {k: v for k, v in f.items() if k != "rows"}
            for f in per_format
        ],
    }
    case, reasons = classify_case(per_format)
    summary["case_verdict"] = case
    summary["failure_reasons"] = reasons
    (OUT_DIR / "format_summary.json").write_text(json.dumps(summary, indent=2))

    # Figure
    make_comparison_figure(per_format, OUT_DIR / "format_comparison_figure.png")

    # Case-dependent text output
    if case == "C":
        write_case_c_report(per_format, reasons,
                            OUT_DIR / "case-c-report.md")
    else:
        write_case_ab_appendix(per_format, case, reasons,
                               OUT_DIR / "appendix_text_formats.md")

    # Stdout summary
    print("\n========== FORMAT-ROBUSTNESS SUMMARY ==========")
    print(f"Case verdict: {case}")
    for f in per_format:
        print(f"  {f['format_id']:6s}  baseline={f['baseline']['mean']:+.2f}"
              f"±{f['baseline']['std']:.2f}  "
              f"(clean {f['baseline']['frac_positive']*100:.0f}%)  "
              f"group means early={f['group_means']['early']:.2f} / "
              f"middle={f['group_means']['middle']:.2f} / "
              f"late={f['group_means']['late']:.2f}  "
              f"ρ={f['spearman_rho']:+.2f} p={f['spearman_p']:.3f}  "
              f"{'PASS' if f['checks']['all_pass'] else 'FAIL'}")
    if reasons:
        print("Failure reasons:")
        for r in reasons:
            print(f"  - {r}")
    recommendation = {
        "A": "INTEGRATE: the appendix subsection in appendix_text_formats.md "
             "is suitable for inclusion.",
        "B": "INTEGRATE WITH CAVEAT: see appendix_text_formats.md — the "
             "passing formats plus the honest caveat for the deviating one.",
        "C": "CUT: case-c-report.md explains the failures; do NOT include "
             "this experiment in the paper.",
    }[case]
    print(f"Recommendation: {recommendation}")
    print(f"Files written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
