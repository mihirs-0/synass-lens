#!/usr/bin/env python
"""
Phase A — Phantom scores in real models (Pythia checkpoint suites).

Per (model, revision, benchmark):
  - model score: accuracy by log-prob option scoring (sum and length-normalized)
  - constant baseline: best input-independent strategy for the format, fit on
    a calibration slice, evaluated on the rest (position prior, longest/
    shortest-option prior; majority class for yes/no; letter prior for the
    letter format)
  - model question-blind score: options scored under a MISMATCHED context
    (fixed derangement of contexts) — this checkpoint's own guessing capacity
  - input-sensitivity: mean JSD between the answer distribution under the true
    context vs the mismatched context; for the letter format also JSD under
    option-order permutation (content-blind letter machines show ~0)
  - phantom fractions: (baseline - chance) / (model - chance), same for blind

Benchmarks (all subsampled, budget-conscious):
  arc_easy      4-option continuation scoring
  arc_easy_ltr  same items, letter format ("A."... "Answer: A")
  hellaswag     4-option continuation scoring
  boolq         yes/no scoring

Disk-conscious: each revision's HF snapshot is deleted after eval.

Usage:
  python scripts/phantom_a.py --model EleutherAI/pythia-160m \
      --revisions step512 step1000 ... --n-items 800
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT_DIR = Path("results/phantom")


# ---------------------------------------------------------------------------
# benchmark loading (normalized: {context, options[list], label})
# ---------------------------------------------------------------------------

def load_benchmarks(n_items, seed=0):
    from datasets import load_dataset
    rng = random.Random(seed)
    benches = {}

    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    items = []
    for ex in arc:
        if len(ex["choices"]["text"]) != 4:
            continue
        items.append({
            "context": f"Question: {ex['question']}\nAnswer:",
            "options": [" " + t for t in ex["choices"]["text"]],
            "label": ex["choices"]["label"].index(ex["answerKey"]),
            "raw_q": ex["question"],
            "raw_opts": ex["choices"]["text"],
        })
    rng.shuffle(items)
    benches["arc_easy"] = items[:n_items]

    # letter format built from the same ARC items
    letters = ["A", "B", "C", "D"]
    ltr = []
    for ex in benches["arc_easy"]:
        opts = ex["raw_opts"]
        body = "\n".join(f"{letters[i]}. {opts[i]}" for i in range(4))
        ltr.append({
            "context": f"Question: {ex['raw_q']}\n{body}\nAnswer:",
            "options": [f" {l}" for l in letters],
            "label": ex["label"],
            "raw_q": ex["raw_q"],
            "raw_opts": opts,
        })
    benches["arc_easy_ltr"] = ltr

    hs = load_dataset("hellaswag", split="validation")
    items = []
    for ex in hs:
        items.append({
            "context": ex["ctx"],
            "options": [" " + e if not e.startswith(" ") else e for e in ex["endings"]],
            "label": int(ex["label"]),
        })
    rng.shuffle(items)
    benches["hellaswag"] = items[:n_items]

    bq = load_dataset("boolq", split="validation")
    items = []
    for ex in bq:
        items.append({
            "context": f"{ex['passage']}\nQuestion: {ex['question']}?\nAnswer:",
            "options": [" no", " yes"],
            "label": int(ex["answer"]),
        })
    rng.shuffle(items)
    benches["boolq"] = items[:n_items]
    return benches


# ---------------------------------------------------------------------------
# constant baselines (model-free, calibration -> eval)
# ---------------------------------------------------------------------------

def constant_baselines(items, calib_frac=0.3, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_cal = int(len(idx) * calib_frac)
    cal, ev = idx[:n_cal], idx[n_cal:]

    def acc(pick_fn, subset):
        return float(np.mean([pick_fn(items[i]) == items[i]["label"] for i in subset]))

    n_opt = len(items[0]["options"])
    strategies = {}
    # position prior: most common correct position on calibration
    pos_counts = np.zeros(n_opt)
    for i in cal:
        pos_counts[items[i]["label"]] += 1
    best_pos = int(pos_counts.argmax())
    strategies["position"] = lambda ex: best_pos
    # longest / shortest option
    strategies["longest"] = lambda ex: int(np.argmax([len(o) for o in ex["options"]]))
    strategies["shortest"] = lambda ex: int(np.argmin([len(o) for o in ex["options"]]))
    # most/least word-count
    strategies["most_words"] = lambda ex: int(np.argmax([len(o.split()) for o in ex["options"]]))

    cal_scores = {name: acc(fn, cal) for name, fn in strategies.items()}
    best_name = max(cal_scores, key=cal_scores.get)
    return {
        "strategies_calibration": cal_scores,
        "best_strategy": best_name,
        "best_constant_acc_eval": acc(strategies[best_name], ev),
        "eval_indices": ev,
        "chance": 1.0 / n_opt,
    }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_options(model, tok, contexts, options_list, device, batch_size=16,
                  max_ctx_tokens=384):
    """Returns [n_items, n_opt] arrays (sum-logprob, mean-logprob)."""
    jobs = []
    for i, (ctx, opts) in enumerate(zip(contexts, options_list)):
        ctx_ids = tok(ctx)["input_ids"][-max_ctx_tokens:]
        for j, opt in enumerate(opts):
            opt_ids = tok(opt)["input_ids"]
            jobs.append((i, j, ctx_ids, opt_ids))
    n_items = len(contexts)
    n_opt = len(options_list[0])
    sum_lp = np.zeros((n_items, n_opt))
    mean_lp = np.zeros((n_items, n_opt))
    jobs.sort(key=lambda t: len(t[2]) + len(t[3]))
    for s in range(0, len(jobs), batch_size):
        chunk = jobs[s:s + batch_size]
        maxlen = max(len(c) + len(o) for (_i, _j, c, o) in chunk)
        input_ids = torch.full((len(chunk), maxlen), tok.eos_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for r, (_i, _j, c, o) in enumerate(chunk):
            seq = c + o
            input_ids[r, :len(seq)] = torch.tensor(seq)
            mask[r, :len(seq)] = 1
        input_ids = input_ids.to(device); mask = mask.to(device)
        logits = model(input_ids=input_ids, attention_mask=mask).logits.float()
        logprobs = F.log_softmax(logits, dim=-1)
        for r, (i, j, c, o) in enumerate(chunk):
            lp = logprobs[r, len(c) - 1:len(c) + len(o) - 1, :]
            tgt = torch.tensor(o, device=device)
            vals = lp.gather(1, tgt[:, None]).squeeze(1)
            sum_lp[i, j] = float(vals.sum())
            mean_lp[i, j] = float(vals.mean())
    return sum_lp, mean_lp


def jsd_rows(a, b, eps=1e-12):
    def _kl(p, q):
        return (p * (np.log(p + eps) - np.log(q + eps))).sum(-1)
    m = 0.5 * (a + b)
    return 0.5 * _kl(a, m) + 0.5 * _kl(b, m)


def softmax_np(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def evaluate_checkpoint(model, tok, benches, baselines, device, seed=0):
    out = {}
    rng = random.Random(seed)
    for name, items in benches.items():
        ev = baselines[name]["eval_indices"]
        contexts = [items[i]["context"] for i in ev]
        options = [items[i]["options"] for i in ev]
        labels = np.array([items[i]["label"] for i in ev])

        # mismatched contexts: fixed derangement of the eval slice
        perm = list(range(len(ev)))
        rng.shuffle(perm)
        perm = [(p + 1) % len(ev) if p == k else p for k, p in enumerate(perm)]
        mis_contexts = [contexts[p] for p in perm]

        s_true, m_true = score_options(model, tok, contexts, options, device)
        s_mis, m_mis = score_options(model, tok, mis_contexts, options, device)

        res = {}
        for tag, tru, mis in [("sum", s_true, s_mis), ("norm", m_true, m_mis)]:
            res[f"acc_{tag}"] = float((tru.argmax(1) == labels).mean())
            res[f"acc_blind_{tag}"] = float((mis.argmax(1) == labels).mean())
            p_t, p_m = softmax_np(tru), softmax_np(mis)
            res[f"jsd_mismatch_{tag}"] = float(jsd_rows(p_t, p_m).mean())

        # letter format: option-order permutation sensitivity
        if name == "arc_easy_ltr":
            perm_items_ctx, perm_map = [], []
            for i in ev:
                ex = benches[name][i]
                order = list(range(4)); rng.shuffle(order)
                opts = [ex["raw_opts"][o] for o in order]
                body = "\n".join(f"{l}. {t}" for l, t in
                                 zip(["A", "B", "C", "D"], opts))
                perm_items_ctx.append(f"Question: {ex['raw_q']}\n{body}\nAnswer:")
                perm_map.append(order)
            s_p, m_p = score_options(model, tok, perm_items_ctx, options, device)
            # compare distribution over CONTENTS (remap letters -> contents)
            p_orig = softmax_np(m_true)
            p_perm_letters = softmax_np(m_p)
            p_perm_content = np.zeros_like(p_perm_letters)
            for r, order in enumerate(perm_map):
                for letter_slot, content_idx in enumerate(order):
                    p_perm_content[r, content_idx] = p_perm_letters[r, letter_slot]
            res["jsd_option_permutation_content"] = float(
                jsd_rows(p_orig, p_perm_content).mean())
            res["acc_permuted_content"] = float(
                (p_perm_content.argmax(1) == labels).mean())

        out[name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--revisions", nargs="+", default=[
        "step128", "step256", "step512", "step1000", "step2000", "step4000",
        "step8000", "step16000", "step32000", "step64000", "step143000"])
    ap.add_argument("--n-items", type=int, default=600)
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mname = args.model.split("/")[-1]

    benches = load_benchmarks(args.n_items)
    baselines = {name: constant_baselines(items) for name, items in benches.items()}
    with open(OUT_DIR / f"{mname}_baselines.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "eval_indices"}
                   for k, v in baselines.items()}, f, indent=1)
    for name, b in baselines.items():
        print(f"[baseline] {name}: chance={b['chance']:.3f} "
              f"best={b['best_strategy']} acc={b['best_constant_acc_eval']:.3f} "
              f"(cal: {b['strategies_calibration']})", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    results_path = OUT_DIR / f"{mname}_checkpoints.json"
    all_res = json.load(open(results_path)) if results_path.exists() else {}

    for rev in args.revisions:
        if rev in all_res:
            print(f"[skip] {rev} already done", flush=True)
            continue
        print(f"[{mname}] loading {rev} ...", flush=True)
        snap = snapshot_download(args.model, revision=rev,
                                 allow_patterns=["*.json", "*.bin", "*.safetensors",
                                                 "tokenizer*"])
        model = AutoModelForCausalLM.from_pretrained(
            snap, torch_dtype=torch.float16).to(device).eval()
        res = evaluate_checkpoint(model, tok, benches, baselines, device)
        all_res[rev] = res
        with open(results_path, "w") as f:
            json.dump(all_res, f, indent=1)
        for name, r in res.items():
            print(f"  {rev} {name}: acc={r['acc_sum']:.3f}/{r['acc_norm']:.3f} "
                  f"blind={r['acc_blind_sum']:.3f}/{r['acc_blind_norm']:.3f} "
                  f"jsd={r['jsd_mismatch_norm']:.4f}", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()
        if not args.keep_cache:
            # snapshots symlink into blobs/; nuke the whole model cache dir —
            # every revision has distinct weight blobs, config re-downloads are tiny
            model_cache = Path(snap).parents[1]
            assert "models--" in model_cache.name
            shutil.rmtree(model_cache, ignore_errors=True)
    print("done ->", results_path)


if __name__ == "__main__":
    main()
