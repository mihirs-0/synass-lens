#!/usr/bin/env python
"""FORECLOSURE TEST — RULE B (context-dependent binding / two-level indexing).

Rule A made z a GLOBAL position-rule: z=k meant the same transform for every B. That tests composition
of a small fixed rule set, NOT binding. Rule B makes z's referent depend on B:

  B = [scheme_token, 5 content tokens].  s = scheme(B[0]).  z indexes WITHIN scheme s.
  A = content tokens of B at positions scheme_rules[s][z].
  -> scheme_rules[s][z] differ across s, so z=2 under scheme 0 selects different positions than z=2
     under scheme 1. The SAME surface z means different things under different B. That is binding.

Systematic & learnable: scheme_rules[s] is a fixed family of ordered position-tuples shared by all B that
carry scheme s, so held-out (B,z) are predictable in principle from other pairs (rule seen on other B's).
Foreclosure = combination-holdout per B (keep round(p*K) of its z's). No whole scheme / z / B removed.

Solving a held-out (B,z) requires BOTH B[0] (which scheme) and z (which index) jointly -> z-necessity on
test (-> 1/K) and scheme-necessity (-> 1/S) are the crux controls. Random-A = negative (must be chance).

Sequence (identical lengths/vocab to Rule A & the cardinality task):
  [BOS, B(6)=scheme+5content, SEP, z, SEP, A(4), EOS] -> T=15, A at TS=10, z at ZPOS=8, scheme at seq pos 1.
"""
import argparse, math, random, json, os
import torch, torch.nn as nn, torch.nn.functional as F

PAD, BOS, SEP, EOS = 0, 1, 2, 3; NSPEC = 4
ALPHA, LB, LA = 26, 6, 4
T = 1 + LB + 1 + 1 + 1 + LA + 1     # 15
TS = 1 + LB + 1 + 1 + 1            # first A index = 10
ZPOS = 1 + LB + 1                  # z token seq position = 8
SCHEMEPOS = 1                      # B[0] scheme token seq position = 1
CONTENT_IDX = [1, 2, 3, 4, 5]      # B-array indices of the 5 content tokens (B[0] is the scheme)
device = "mps" if torch.backends.mps.is_available() else "cpu"


def make_scheme_rules(S, K):
    """S*K distinct ordered 4-of-5 content-position tuples, partitioned into S schemes of K.
    Disjoint partition => scheme_rules[s][z] != scheme_rules[s'][z]: z's meaning is scheme-dependent."""
    assert S * K <= 5 * 4 * 3 * 2, "S*K exceeds distinct ordered 4-of-5 position tuples (120)"
    rng = random.Random(2024); seen = set(); allr = []
    while len(allr) < S * K:
        r = tuple(rng.sample(CONTENT_IDX, LA))
        if r not in seen: seen.add(r); allr.append(r)
    return [allr[s*K:(s+1)*K] for s in range(S)]


def build_pairs(N_B, S, K, mode="bind", seed=0):
    rng = random.Random(seed); content = list(range(NSPEC, NSPEC + ALPHA))
    scheme_rules = make_scheme_rules(S, K)
    usedB = set(); Bs = []
    for i in range(N_B):
        s = i % S                                   # balanced scheme assignment
        while True:
            c = tuple(rng.sample(content, LB - 1))   # 5 distinct content tokens
            b = (NSPEC + s, *c)
            if b not in usedB: usedB.add(b); break
        Bs.append((b, s))
    pairs = []
    for b, s in Bs:
        for z in range(K):
            a = tuple(rng.choice(content) for _ in range(LA)) if mode == "random" \
                else tuple(b[p] for p in scheme_rules[s][z])
            pairs.append((b, z, a))
    return pairs


def to_seq(b, z, a):
    return [BOS, *b, SEP, NSPEC + z, SEP, *a, EOS]


def split_holdout(pairs, K, p, seed=0):
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
    model.eval(); correct = 0; n = seqs.size(0)
    if n == 0: return None
    for i in range(0, n, bs):
        s = seqs[i:i+bs].clone().to(device)
        if perturb == "zshuf":     s[:, ZPOS] = s[torch.randperm(s.size(0)), ZPOS]
        elif perturb == "zmask":   s[:, ZPOS] = NSPEC                # remove index -> 1/K within scheme
        elif perturb == "schemask": s[:, SCHEMEPOS] = NSPEC          # remove scheme -> 1/S (binding crux)
        elif perturb == "bmask":   s[:, 1:LB+1] = PAD                # remove all B -> ~0
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


def run_cell(N_B, S, K, p, eta, seed, steps, mode="bind"):
    pairs = build_pairs(N_B, S, K, mode=mode, seed=seed)
    tr, te, effp = split_holdout(pairs, K, p, seed=seed)
    trs = seqs_of(pairs, tr); tes = seqs_of(pairs, te) if te else trs[:0]
    m = train(trs, eta, seed, steps, NSPEC + ALPHA)
    def r(x): return round(x, 4) if x is not None else None
    o = {"S": S, "K": K, "p": p, "eff_p": round(effp, 3), "eta": eta, "seed": seed, "mode": mode,
         "n_train": len(tr), "n_test": len(te), "chance_1overK": round(1.0/K, 4), "chance_1overS": round(1.0/S, 4),
         "train_acc": r(acc(m, trs[:4000])),
         "test_acc": r(acc(m, tes)),
         "test_zshuf": r(acc(m, tes, "zshuf")),
         "test_zmask": r(acc(m, tes, "zmask")),
         "test_schemask": r(acc(m, tes, "schemask")),
         "test_bmask": r(acc(m, tes, "bmask"))}
    o["delta_z_test"] = r((o["test_acc"] - o["test_zshuf"])) if o["test_acc"] is not None else None
    return o


def fmt(o):
    t = f"{o['test_acc']:.3f}" if o["test_acc"] is not None else "  -  "
    return (f"S={o['S']} K={o['K']:>2} p={o['eff_p']:.2f} eta={o['eta']:.4f} s{o['seed']} [{o['mode'][:4]}] | "
            f"train={o['train_acc']:.3f} test={t} (1/K={o['chance_1overK']:.3f})")


def pilot(steps, S=2, K=8, eta=0.01, N_B=1000):
    print(f"=== RULE B PILOT (device={device}, steps={steps}, S={S}, K={K}, eta={eta}) ===")
    print("   binding = z's referent depends on B[0] (scheme). Crux control = z-necessity on test -> 1/K.\n")

    print("[control 1] rule learnable?  p=1.0 (no holdout) must reach high TRAIN acc.")
    o = run_cell(N_B, S, K, 1.0, eta, 0, steps); print("   " + fmt(o))
    print(f"   -> {'PASS' if o['train_acc'] > 0.9 else 'FAIL — shrink K/S before interpreting anything'}\n")

    print("[capacity] random-A p=1.0 must also reach high TRAIN acc (memorization reachable ->")
    print("           any test failure later is a generalization failure, not a capacity failure).")
    o = run_cell(N_B, S, K, 1.0, eta, 0, steps, mode="random"); print("   " + fmt(o))
    print(f"   -> {'capacity ok' if o['train_acc'] > 0.85 else 'CAPACITY-LIMITED — increase d before trusting test failures'}\n")

    print("[main + controls 2,3,4] foreclosed binding cell p=0.5:")
    o = run_cell(N_B, S, K, 0.5, eta, 0, steps); print("   " + fmt(o))
    print(f"   >>> z-necessity   test|z-shuffled .. {o['test_zshuf']}   (CRUX: must fall to ~1/K={o['chance_1overK']})")
    print(f"   >>> z-necessity   test|z-masked .... {o['test_zmask']}   (~1/K)")
    print(f"   >>> scheme-necess test|scheme-masked {o['test_schemask']}   (binding: must fall to ~1/S={o['chance_1overS']})")
    print(f"       B-necessity   test|B-masked .... {o['test_bmask']}   (~0)")
    print(f"       test_acc (held-out combos) ..... {o['test_acc']}   Δz_test={o['delta_z_test']}")
    gen = o["test_acc"] and o["test_acc"] > 1.5 * o["chance_1overK"]
    znec = (o["test_zshuf"] is not None) and o["test_zshuf"] < 2 * o["chance_1overK"]
    snec = (o["test_schemask"] is not None) and o["test_schemask"] < 2 * o["chance_1overS"]
    print(f"   -> generalizes? {'YES' if gen else 'no'} | z-necessary? {'yes' if znec else 'NO'} | "
          f"scheme-necessary? {'yes' if snec else 'NO'}\n")

    print("[control 3] negative: random-A, same holdout p=0.5 -> held-out test MUST be ~chance (leak test).")
    o = run_cell(N_B, S, K, 0.5, eta, 0, steps, mode="random"); print("   " + fmt(o))
    leak = o["test_acc"] and o["test_acc"] > 3 * o["chance_1overK"]
    print(f"   -> {'LEAK — abort' if leak else 'clean (random-A at chance)'}\n")
    print("pilot done — HOLD for grid go/no-go.")


def grid(steps, Ss, Ks, ps, etas, seeds):
    N_B = 1000; os.makedirs("card_ckpts", exist_ok=True)
    path = "card_ckpts/foreclosure_b.json"
    res = json.load(open(path)) if os.path.exists(path) else {"meta": {"N_B": N_B, "steps": steps}, "cells": []}
    done = {(c["S"], c["K"], c["p"], c["eta"], c["seed"], c["mode"]) for c in res["cells"]}
    cells = [(S, K, p, e, s) for S in Ss for K in Ks for p in ps for e in etas for s in seeds]
    print(f"=== RULE B GRID: {len(cells)} binding cells + controls, steps={steps} ===")
    for i, (S, K, p, e, s) in enumerate(cells, 1):
        if (S, K, p, e, s, "bind") in done: continue
        o = run_cell(N_B, S, K, p, e, s, steps, mode="bind"); res["cells"].append(o)
        print(f"[{i}/{len(cells)}] " + fmt(o)); json.dump(res, open(path, "w"), indent=2)
    mideta = etas[len(etas)//2]
    for S in Ss:
        for K in Ks:
            for p in ps:
                if (S, K, p, mideta, 0, "random") in done: continue
                o = run_cell(N_B, S, K, p, mideta, 0, steps, mode="random"); res["cells"].append(o)
                print("[ctrl] " + fmt(o)); json.dump(res, open(path, "w"), indent=2)
    print(f"\nsaved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true"); ap.add_argument("--grid", action="store_true")
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--S", type=int, default=2); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--Ss", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--Ks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.9, 0.7, 0.5, 0.3])
    ap.add_argument("--etas", type=float, nargs="+", default=[3e-3, 7e-3, 1.5e-2, 3e-2, 6e-2])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    if a.grid: grid(a.steps, a.Ss, a.Ks, a.ps, a.etas, list(range(a.seeds)))
    else: pilot(a.steps, S=a.S, K=a.K)


if __name__ == "__main__":
    main()
