#!/usr/bin/env python
"""
Run 6 / H6 — Blind-score hygiene (arc_easy_ltr only).

Analysis first: for continuation formats (arc_easy, hellaswag, boolq) the
options never appear in the context and the template is bench-constant
boilerplate around the question text, so the existing whole-context mismatch
IS the hygienic question-only mismatch. The two procedures differ only for
arc_easy_ltr, whose context embeds the option body: the old mismatch swaps
question AND option list; the hygienic one swaps the question text only,
keeping the recipient's own option body, labels, order, template
byte-identical.

This script re-evaluates arc_easy_ltr blind scores under both procedures for
every finished Pythia checkpoint. Rule (precommitted): material difference =
|hygienic - old| > 0.02 acc on the majority of checkpoints -> hygienic
replaces old everywhere.
"""

import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.phantom_a import (load_benchmarks, constant_baselines,
                               score_options, softmax_np, jsd_rows, OUT_DIR)
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-160m"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = random.Random(0)
    benches = load_benchmarks(600, seed=0)
    # reconstruct the eval slice exactly as phantom_a does (the saved
    # baselines json strips eval_indices): same function, same seed
    items = benches["arc_easy_ltr"]
    ev = constant_baselines(items)["eval_indices"]

    contexts = [items[i]["context"] for i in ev]
    options = [items[i]["options"] for i in ev]
    labels = np.array([items[i]["label"] for i in ev])
    letters = ["A", "B", "C", "D"]

    # old-procedure mismatch (whole context) — same derangement construction
    perm = list(range(len(ev)))
    rng.shuffle(perm)
    perm = [(p + 1) % len(ev) if p == k else p for k, p in enumerate(perm)]
    mis_old = [contexts[p] for p in perm]
    # hygienic mismatch: donor question + recipient's own option body
    mis_hyg = []
    for k, p in enumerate(perm):
        rec = items[ev[k]]
        don = items[ev[p]]
        body = "\n".join(f"{letters[i]}. {rec['raw_opts'][i]}"
                         for i in range(4))
        mis_hyg.append(f"Question: {don['raw_q']}\n{body}\nAnswer:")

    tok = AutoTokenizer.from_pretrained(MODEL)
    done = json.load(open(OUT_DIR / "pythia-160m_checkpoints.json"))
    res = {}
    for rev in done.keys():
        snap = snapshot_download(MODEL, revision=rev,
                                 allow_patterns=["*.json", "*.bin",
                                                 "*.safetensors",
                                                 "tokenizer*"])
        model = AutoModelForCausalLM.from_pretrained(
            snap, torch_dtype=torch.float16).to(device).eval()
        s_true, _ = score_options(model, tok, contexts, options, device)
        s_old, _ = score_options(model, tok, mis_old, options, device)
        s_hyg, _ = score_options(model, tok, mis_hyg, options, device)
        row = {
            "acc": float((s_true.argmax(1) == labels).mean()),
            "blind_old": float((s_old.argmax(1) == labels).mean()),
            "blind_hyg": float((s_hyg.argmax(1) == labels).mean()),
            "jsd_old": float(jsd_rows(softmax_np(s_true),
                                      softmax_np(s_old)).mean()),
            "jsd_hyg": float(jsd_rows(softmax_np(s_true),
                                      softmax_np(s_hyg)).mean()),
        }
        res[rev] = row
        print(f"{rev}: acc={row['acc']:.3f} blind old={row['blind_old']:.3f} "
              f"hyg={row['blind_hyg']:.3f} jsd old={row['jsd_old']:.4f} "
              f"hyg={row['jsd_hyg']:.4f}", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()
        model_cache = Path(snap).parents[1]
        if "models--" in model_cache.name:
            shutil.rmtree(model_cache, ignore_errors=True)

    diffs = [abs(r["blind_hyg"] - r["blind_old"]) for r in res.values()]
    material = sum(dd > 0.02 for dd in diffs) > len(diffs) / 2
    res["verdict"] = {"material": bool(material),
                      "mean_abs_diff": float(np.mean(diffs)),
                      "rule": "hygienic replaces old everywhere" if material
                              else "old numbers stand"}
    with open(Path("results/adversarial") / "h6_hygiene.json", "w") as f:
        json.dump(res, f, indent=1)
    print("H6 verdict:", json.dumps(res["verdict"]))


if __name__ == "__main__":
    main()
