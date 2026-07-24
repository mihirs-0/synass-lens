#!/usr/bin/env python
"""FORECLOSURE TEST — does gradient descent induce a z-indexing (binding) circuit when memorization
is foreclosed by combination-holdout?

Task (structured pointer / indexing): a GLOBAL ordered position-rule per index, rule[z] = a 4-tuple of
positions drawn from B's 6 tokens. Target A = B's tokens at rule[z]. "Emit the z-th candidate of B."
  - systematic & generalizable: learn the K rules -> predict ANY (B,z), including unseen combinations
  - requires BOTH inputs: B alone -> 1/K candidates; z alone -> can't produce B's tokens -> chance
Negative control = RANDOM A per (B,z): unpredictable, so held-out combinations MUST be chance (leak test).

Foreclosure = combination-holdout: for each B, keep round(p*K) of its z-indices in train, hold out the
rest. Every z is trained for some B's and held out for others; no whole-B / whole-z removed. Train fraction
p is the foreclosure axis. Memorization can fit train but cannot reach held-out (B,z); only a model that
learned the per-z rule generalizes.

Sequence (vocab + lengths identical to the cardinality task; z is one token reusing the content range):
  [BOS, B(6), SEP, z, SEP, A(4), EOS]  -> T=15, first A at TS=10, z at ZPOS=8.

Sweeps: p (foreclosure) x K (candidates) x eta (LR) x seeds.  Headline = train_acc vs test_acc(p); is it LR-gated?
"""
import argparse, math, random, json, os
import torch, torch.nn as nn, torch.nn.functional as F

PAD, BOS, SEP, EOS = 0, 1, 2, 3; NSPEC = 4
ALPHA, LB, LA = 26, 6, 4
T = 1 + LB + 1 + 1 + 1 + LA + 1     # [BOS B SEP z SEP A EOS] = 15
TS = 1 + LB + 1 + 1 + 1            # first answer token index = 10
ZPOS = 1 + LB + 1                  # z token position = 8
device = "mps" if torch.backends.mps.is_available() else "cpu"


def make_rules(K):
    """K distinct GLOBAL ordered 4-of-6 position tuples (the index rules). Same across seeds & B's."""
    rng = random.Random(1234); seen = set(); rules = []
    while len(rules) < K:
        r = tuple(rng.sample(range(LB), LA))
        if r not in seen: seen.add(r); rules.append(r)
    return rules


def build_pairs(N_B, K, mode="pointer", seed=0):
    """All N_B*K (B,z,A) triples. mode=pointer: A=select(B,rule[z]); mode=random: A arbitrary (control)."""
    rng = random.Random(seed); content = list(range(NSPEC, NSPEC + ALPHA))
    rules = make_rules(K)
    usedB = set(); Bs = []
    for _ in range(N_B):
        while True:
            b = tuple(rng.sample(content, LB))          # distinct tokens -> K candidates always distinct
            if b not in usedB: usedB.add(b); break
        Bs.append(b)
    pairs = []
    for b in Bs:
        for z in range(K):
            a = tuple(rng.choice(content) for _ in range(LA)) if mode == "random" else tuple(b[p] for p in rules[z])
            pairs.append((b, z, a))
    return pairs


def to_seq(b, z, a):
    return [BOS, *b, SEP, NSPEC + z, SEP, *a, EOS]


def split_holdout(pairs, K, p, seed=0):
    """Per-B keep ntr=round(p*K) z-indices for train (clamped to [1,K-1] when p<1); rest test."""
    rng = random.Random(7000 + seed); byB = {}
    for i, (b, z, a) in enumerate(pairs): byB.setdefault(b, []).append(i)
    ntr = K if p >= 1.0 else max(1, min(K - 1, round(p * K)))
    train, test = [], []
    for b, idxs in byB.items():
        zl = list(range(K)); rng.shuffle(zl); tr = set(zl[:ntr])
        for i in idxs:
            (train if pairs[i][1] in tr else test).append(i)
    return train, test, ntr / K


def seqs_of(pairs, idxs):
    return torch.tensor([to_seq(*pairs[i]) for i in idxs])


class TinyGPT(nn.Module):
    def __init__(self, vocab, d=64, h=2, L=2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(T, d)
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(2 * L)])
        self.at = nn.ModuleList([nn.MultiheadAttention(d, h, batch_first=True) for _ in range(L)])
        self.ml = nn.ModuleList([nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(L)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, vocab); self.L = L
    def forward(self, x):
        m = torch.triu(torch.full((x.size(1),) * 2, float("-inf"), device=x.device), 1)
        z = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))
        for i in range(self.L):
            a = self.ln[2*i](z); z = z + self.at[i](a, a, a, attn_mask=m, need_weights=False)[0]
            z = z + self.ml[i](self.ln[2*i+1](z))
        return self.head(self.lnf(z))


def lm_loss(model, seq):
    lg = model(seq); return F.cross_entropy(lg[:, TS-1:-1].reshape(-1, lg.size(-1)), seq[:, TS:].reshape(-1))


@torch.no_grad()
def acc(model, seqs, perturb=None, bs=2048):
    """Exact-match accuracy over all LA answer tokens. perturb in {None,'zshuf','zmask','bmask'}."""
    model.eval(); correct = 0; n = seqs.size(0)
    for i in range(0, n, bs):
        s = seqs[i:i+bs].clone().to(device)
        if perturb == "zshuf":  s[:, ZPOS] = s[torch.randperm(s.size(0)), ZPOS]
        elif perturb == "zmask": s[:, ZPOS] = NSPEC                       # constant z -> only B info
        elif perturb == "bmask": s[:, 1:LB+1] = PAD                       # remove B -> only z info
        pred = model(s)[:, TS-1:TS-1+LA].argmax(-1); tgt = s[:, TS:TS+LA]
        correct += int((pred == tgt).all(1).sum())
    return correct / n


def train(train_seqs, eta, seed, steps, vocab):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    model = TinyGPT(vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=eta, betas=(0.9, 0.999), weight_decay=0.01)
    n = train_seqs.size(0)
    for _ in range(steps):
        idx = torch.randint(0, n, (256,), generator=g)
        loss = lm_loss(model, train_seqs[idx].to(device))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model


def run_cell(N_B, K, p, eta, seed, steps, mode="pointer"):
    pairs = build_pairs(N_B, K, mode=mode, seed=seed)
    tr, te, effp = split_holdout(pairs, K, p, seed=seed)
    trs = seqs_of(pairs, tr); tes = seqs_of(pairs, te) if te else trs[:0]
    m = train(trs, eta, seed, steps, NSPEC + ALPHA)
    out = {"K": K, "p": p, "eff_p": round(effp, 3), "eta": eta, "seed": seed, "mode": mode,
           "n_train": len(tr), "n_test": len(te), "chance_1overK": round(1.0 / K, 4),
           "train_acc": round(acc(m, trs[:4000]), 4),
           "test_acc": round(acc(m, tes), 4) if len(te) else None,
           "test_acc_zshuf": round(acc(m, tes, "zshuf"), 4) if len(te) else None,
           "test_acc_zmask": round(acc(m, tes, "zmask"), 4) if len(te) else None,
           "test_acc_bmask": round(acc(m, tes, "bmask"), 4) if len(te) else None}
    out["delta_z_test"] = round(out["test_acc"] - out["test_acc_zshuf"], 4) if len(te) else None
    return out


def fmt(o):
    t = f"{o['test_acc']:.3f}" if o["test_acc"] is not None else "  -  "
    return (f"K={o['K']:>2} p={o['eff_p']:.2f} eta={o['eta']:.4f} s{o['seed']} [{o['mode'][:3]}] | "
            f"train={o['train_acc']:.3f} test={t} (chance {o['chance_1overK']:.3f}) "
            f"Δz_test={o['delta_z_test']}")


def pilot(steps):
    print(f"=== PILOT (device={device}, steps={steps}) — validate task + 4 controls before the grid ===\n")
    K, eta = 8, 0.01; N_B = 1000
    print("[control 1] rule learnable at all?  p=1.0 (no holdout) must reach high train acc.")
    o = run_cell(N_B, K, 1.0, eta, 0, steps); print("   " + fmt(o))
    print(f"   -> {'PASS' if o['train_acc'] > 0.9 else 'FAIL — shrink K before interpreting anything'}\n")

    print("[main + controls 2,4] foreclosed cell p=0.5, K=8:")
    o = run_cell(N_B, K, 0.5, eta, 0, steps); print("   " + fmt(o))
    print(f"   test_acc (held-out combos) ...... {o['test_acc']}   (chance {o['chance_1overK']})")
    print(f"   z-necessity  test|z-shuffled .... {o['test_acc_zshuf']}  (should fall to ~chance)")
    print(f"   z-necessity  test|z-masked ...... {o['test_acc_zmask']}  (B-only -> ~1/K)")
    print(f"   B-necessity  test|B-masked ...... {o['test_acc_bmask']}  (z-only -> ~0)")
    gen = o["test_acc"] and o["test_acc"] > 1.5 * o["chance_1overK"]
    print(f"   -> generalizes beyond chance? {'YES' if gen else 'no'}\n")

    print("[control 3] NEGATIVE control: random-A (the old memorization task), same holdout p=0.5.")
    print("   held-out test MUST be ~chance; if it 'generalizes', the test set leaks and results are void.")
    o = run_cell(N_B, K, 0.5, eta, 0, steps, mode="random"); print("   " + fmt(o))
    leak = o["test_acc"] and o["test_acc"] > 3 * o["chance_1overK"]
    print(f"   -> {'LEAK — abort' if leak else 'clean (random-A stays at chance)'}\n")
    print("pilot done.")


def grid(steps, Ks, ps, etas, seeds):
    N_B = 1000; os.makedirs("card_ckpts", exist_ok=True)
    path = "card_ckpts/foreclosure.json"
    res = json.load(open(path)) if os.path.exists(path) else {"meta": {"N_B": N_B, "steps": steps}, "cells": []}
    done = {(c["K"], c["p"], c["eta"], c["seed"], c["mode"]) for c in res["cells"]}
    total = len(Ks)*len(ps)*len(etas)*len(seeds); i = 0
    print(f"=== GRID: {total} pointer cells + negative controls, steps={steps} ===")
    for K in Ks:
        for p in ps:
            for eta in etas:
                for s in seeds:
                    i += 1
                    if (K, p, eta, s, "pointer") in done: continue
                    o = run_cell(N_B, K, p, eta, s, steps, mode="pointer")
                    res["cells"].append(o); print(f"[{i}/{total}] " + fmt(o))
                    json.dump(res, open(path, "w"), indent=2)
    # negative controls: random-A at every (K,p), one seed, mid eta
    mideta = etas[len(etas)//2]
    for K in Ks:
        for p in ps:
            if (K, p, mideta, 0, "random") in done: continue
            o = run_cell(1000, K, p, mideta, 0, steps, mode="random")
            res["cells"].append(o); print("[ctrl] " + fmt(o)); json.dump(res, open(path, "w"), indent=2)
    print(f"\nsaved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--Ks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.9, 0.7, 0.5, 0.3])
    ap.add_argument("--etas", type=float, nargs="+", default=[3e-3, 7e-3, 1.5e-2, 3e-2, 6e-2])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    if a.pilot: pilot(a.steps)
    elif a.grid: grid(a.steps, a.Ks, a.ps, a.etas, list(range(a.seeds)))
    else: pilot(a.steps)


if __name__ == "__main__":
    main()
