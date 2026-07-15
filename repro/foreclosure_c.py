#!/usr/bin/env python
"""FORECLOSURE TEST — RULE C (computed-context binding; the hardened, non-discountable version).

Rule B let the model READ the scheme from a label token (b0) -> a skeptic calls it a 16-entry
(scheme,z) lookup + copy. Rule C removes the label: the binding context must be COMPUTED from B's
structure.

  B = 6 content tokens (no label; all six are content and copyable).
  o = bucket_S( argmax_position(B) )           # WHICH token is largest, bucketed into S schemes.
  A = B[ scheme_rules[o][z] ]                   # z indexes within the computed scheme o.
  scheme_rules disjoint across o  ->  z=2 under o=0 selects different positions than z=2 under o=1.

Why this is not discountable:
  - there is NO scheme token to read; o is a non-local relation over all six tokens (argmax).
  - the ONLY way to beat 1/S on held-out (B,z) is to actually compute o -> high test == proof of
    computed binding (no token can leak it). Confirmed causally: move the argmax into another bucket,
    hold z fixed, and the correct answer changes; a binding model's output follows.

Foreclosure = combination-holdout per B (keep round(p*K) of its z's). No whole scheme/z/B removed.
Sequence (lengths/vocab identical to the cardinality task): [BOS, B(6), SEP, z, SEP, A(4), EOS].
"""
import argparse, math, random, json, os
import torch, torch.nn as nn, torch.nn.functional as F

PAD, BOS, SEP, EOS = 0, 1, 2, 3; NSPEC = 4
ALPHA, LB, LA = 26, 6, 4
MAXTOK = NSPEC + ALPHA - 1          # 29 = largest content token
T = 1 + LB + 1 + 1 + 1 + LA + 1     # 15
TS = 1 + LB + 1 + 1 + 1            # first A index = 10
ZPOS = 1 + LB + 1                  # z token seq position = 8
BPOS = list(range(1, 1 + LB))      # B token seq positions 1..6
device = "mps" if torch.backends.mps.is_available() else "cpu"


def bucket(pos, S):
    return pos * S // LB


def make_scheme_rules(S, K):
    assert S * K <= 6 * 5 * 4 * 3, "S*K exceeds distinct ordered 4-of-6 tuples (360)"
    rng = random.Random(2024); seen = set(); allr = []
    while len(allr) < S * K:
        r = tuple(rng.sample(range(LB), LA))           # ordered 4 of the 6 B positions
        if r not in seen: seen.add(r); allr.append(r)
    return [allr[s*K:(s+1)*K] for s in range(S)]


def scheme_of(b, S):
    return bucket(max(range(LB), key=lambda i: b[i]), S)   # bucket of the argmax position


def build_pairs(N_B, S, K, mode="bind", seed=0):
    rng = random.Random(seed); content = list(range(NSPEC, NSPEC + ALPHA))
    rules = make_scheme_rules(S, K); usedB = set(); Bs = []
    for _ in range(N_B):
        while True:
            b = tuple(rng.sample(content, LB))          # 6 distinct -> unique argmax
            if b not in usedB: usedB.add(b); break
        Bs.append(b)
    pairs = []
    for b in Bs:
        o = scheme_of(b, S)
        for z in range(K):
            a = tuple(rng.choice(content) for _ in range(LA)) if mode == "random" else tuple(b[p] for p in rules[o][z])
            pairs.append((b, z, a))
    return pairs, rules


def to_seq(b, z, a):
    return [BOS, *b, SEP, NSPEC + z, SEP, *a, EOS]


def split_holdout(pairs, K, p, seed=0):
    rng = random.Random(7000 + seed); byB = {}
    for i, (b, z, a) in enumerate(pairs): byB.setdefault(b, []).append(i)
    ntr = K if p >= 1.0 else max(1, min(K - 1, round(p * K)))
    train, test = [], []
    for b, idxs in byB.items():
        zl = list(range(K)); rng.shuffle(zl); tr = set(zl[:ntr])
        for i in idxs: (train if pairs[i][1] in tr else test).append(i)
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
        if perturb == "zshuf":   s[:, ZPOS] = s[torch.randperm(s.size(0)), ZPOS]
        elif perturb == "zmask": s[:, ZPOS] = NSPEC
        elif perturb == "bmask": s[:, 1:LB+1] = PAD
        pred = model(s)[:, TS-1:TS-1+LA].argmax(-1); tgt = s[:, TS:TS+LA]
        correct += int((pred == tgt).all(1).sum())
    return correct / n


@torch.no_grad()
def causal_scheme_test(model, pairs, te_idx, S, K, rules, nmax=2000):
    """Move the argmax into a different bucket (hold z fixed) -> correct answer changes.
    follow_rate = fraction where the model's output switches to the NEW scheme's rule (causal binding)."""
    model.eval(); rng = random.Random(123); follow = total = 0
    for i in te_idx[:nmax]:
        b, z, _ = pairs[i]; b = list(b); o = scheme_of(b, S)
        o2 = (o + 1) % S
        cands = [pos for pos in range(LB) if bucket(pos, S) == o2]
        if not cands: continue
        j = rng.choice(cands)
        b2 = b[:]
        if b2[j] != MAXTOK and MAXTOK not in [b2[k] for k in range(LB) if k != j]:
            b2[j] = MAXTOK                                  # make j the unique global max
        else:
            b2 = [min(t, MAXTOK - 1) for t in b2]; b2[j] = MAXTOK
        if scheme_of(b2, S) != o2: continue                 # ensure the flip landed
        a2 = tuple(b2[p] for p in rules[o2][z])             # target under the NEW computed scheme
        seq = torch.tensor([to_seq(tuple(b2), z, a2)]).to(device)
        pred = model(seq)[:, TS-1:TS-1+LA].argmax(-1)[0]
        follow += int((pred == seq[0, TS:TS+LA]).all()); total += 1
    return (follow / total) if total else None, total


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


def run_cell(N_B, S, K, p, eta, seed, steps, mode="bind", causal=False):
    pairs, rules = build_pairs(N_B, S, K, mode=mode, seed=seed)
    tr, te, effp = split_holdout(pairs, K, p, seed=seed)
    trs = seqs_of(pairs, tr); tes = seqs_of(pairs, te) if te else trs[:0]
    m = train(trs, eta, seed, steps, NSPEC + ALPHA)
    def r(x): return round(x, 4) if x is not None else None
    o = {"S": S, "K": K, "p": p, "eff_p": round(effp, 3), "eta": eta, "seed": seed, "mode": mode,
         "n_train": len(tr), "n_test": len(te), "chance_1overK": round(1.0/K, 4), "chance_1overS": round(1.0/S, 4),
         "train_acc": r(acc(m, trs[:4000])), "test_acc": r(acc(m, tes)),
         "test_zshuf": r(acc(m, tes, "zshuf")), "test_zmask": r(acc(m, tes, "zmask")),
         "test_bmask": r(acc(m, tes, "bmask"))}
    o["delta_z_test"] = r(o["test_acc"] - o["test_zshuf"]) if o["test_acc"] is not None else None
    if causal and len(te):
        fr, ntot = causal_scheme_test(m, pairs, te, S, K, rules); o["causal_follow"] = r(fr); o["causal_n"] = ntot
    return o


def fmt(o):
    t = f"{o['test_acc']:.3f}" if o["test_acc"] is not None else "  -  "
    return (f"S={o['S']} K={o['K']:>2} p={o['eff_p']:.2f} eta={o['eta']:.4f} s{o['seed']} [{o['mode'][:4]}] | "
            f"train={o['train_acc']:.3f} test={t} (1/K={o['chance_1overK']:.3f} 1/S={o['chance_1overS']:.3f})")


def pilot(steps, S=2, K=8, eta=0.01, N_B=1000):
    print(f"=== RULE C PILOT — computed-context binding (device={device}, steps={steps}, S={S}, K={K}, eta={eta}) ===")
    print("   scheme o = bucket(argmax position of B). No label token. High test (> 1/S) == computed binding.\n")

    print("[control 1 + capacity] rule learnable?  p=1.0 -> high TRAIN acc (also the capacity check for the rule).")
    o = run_cell(N_B, S, K, 1.0, eta, 0, steps); print("   " + fmt(o))
    print(f"   -> {'PASS' if o['train_acc'] > 0.9 else 'FAIL — task not learned in budget; more steps or smaller S/K'}\n")

    print("[main + controls + causal] foreclosed cell p=0.5:")
    o = run_cell(N_B, S, K, 0.5, eta, 0, steps, causal=True); print("   " + fmt(o))
    print(f"   test_acc (held-out combos) ........ {o['test_acc']}   (must exceed 1/S={o['chance_1overS']} to prove computed binding)")
    print(f"   >>> z-necessity  test|z-shuffled .. {o['test_zshuf']}   (must fall to ~1/K={o['chance_1overK']})")
    print(f"       z-necessity  test|z-masked .... {o['test_zmask']}   (~1/K)")
    print(f"       B-necessity  test|B-masked .... {o['test_bmask']}   (~0)")
    print(f"   >>> CAUSAL scheme-follow .......... {o.get('causal_follow')}  on n={o.get('causal_n')}  "
          f"(move argmax to new bucket, hold z -> output follows NEW scheme? high == computed binding)")
    comp = o["test_acc"] and o["test_acc"] > 2 * o["chance_1overS"]
    znec = (o["test_zshuf"] is not None) and o["test_zshuf"] < 2 * o["chance_1overK"]
    caus = (o.get("causal_follow") is not None) and o["causal_follow"] > 0.5
    print(f"   -> computed binding? {'YES' if comp else 'no'} | z-necessary? {'yes' if znec else 'NO'} | "
          f"causal? {'yes' if caus else 'NO'}\n")

    print("[control 3] negative: random-A, same holdout p=0.5 -> held-out test MUST be ~chance (leak test).")
    o = run_cell(N_B, S, K, 0.5, eta, 0, steps, mode="random"); print("   " + fmt(o))
    leak = o["test_acc"] and o["test_acc"] > 3 * o["chance_1overK"]
    print(f"   -> {'LEAK — abort' if leak else 'clean (random-A at chance)'}\n")
    print("pilot done — HOLD for grid go/no-go.")


def grid(steps, Ss, Ks, ps, etas, seeds):
    N_B = 1000; os.makedirs("card_ckpts", exist_ok=True)
    path = "card_ckpts/foreclosure_c.json"
    res = json.load(open(path)) if os.path.exists(path) else {"meta": {"N_B": N_B, "steps": steps}, "cells": []}
    done = {(c["S"], c["K"], c["p"], c["eta"], c["seed"], c["mode"]) for c in res["cells"]}
    cells = [(S, K, p, e, s) for S in Ss for K in Ks for p in ps for e in etas for s in seeds]
    print(f"=== RULE C GRID: {len(cells)} cells + controls, steps={steps} ===")
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
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--S", type=int, default=2); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--Ss", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--Ks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.9, 0.7, 0.5, 0.3])
    ap.add_argument("--etas", type=float, nargs="+", default=[3e-3, 7e-3, 1.5e-2, 3e-2, 6e-2])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    if a.grid: grid(a.steps, a.Ss, a.Ks, a.ps, a.etas, list(range(a.seeds)))
    else: pilot(a.steps, S=a.S, K=a.K)


if __name__ == "__main__":
    main()
