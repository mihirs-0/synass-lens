#!/usr/bin/env python
"""eta*(N): critical LR vs output cardinality. Surjective M->N memorization, no z, no binding.
Hold M (examples), string lengths, vocab FIXED; vary only N (distinct outputs).
Phase A: per-N eta sweep -> fraction trapped -> eta*(N); plot log eta* vs log N, fit alpha.
Phase B: top Hessian eigenvalue at the memorized solution; check 2/lambda_max ~ eta*(N).
"""
import argparse, math, random
import torch, torch.nn as nn, torch.nn.functional as F

PAD, BOS, SEP, EOS = 0, 1, 2, 3; NSPEC = 4
ALPHA, LB, LA = 26, 6, 4
T = 1 + LB + 1 + LA + 1            # [BOS B(6) SEP A(4) EOS] = 13
TS = 1 + LB + 1                    # first answer token index = 8
device = "mps" if torch.backends.mps.is_available() else "cpu"


def build_task(M, N, seed=0):
    rng = random.Random(seed); content = list(range(NSPEC, NSPEC + ALPHA))
    usedA = set(); outs = []
    while len(outs) < N:
        s = tuple(rng.choice(content) for _ in range(LA))
        if s not in usedA: usedA.add(s); outs.append(s)
    assign = [outs[i % N] for i in range(M)]; rng.shuffle(assign)   # surjective: every output used
    usedB = set(); seqs = []
    for i in range(M):
        while True:
            b = tuple(rng.choice(content) for _ in range(LB))
            if b not in usedB: usedB.add(b); break
        seqs.append([BOS, *b, SEP, *assign[i], EOS])
    return torch.tensor(seqs)


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


def train(X, eta, seed, steps, vocab, d=64):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    model = TinyGPT(vocab, d=d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=eta, betas=(0.9, 0.999), weight_decay=0.01)
    n = X.size(0)
    for _ in range(steps):
        idx = torch.randint(0, n, (256,), generator=g)
        loss = lm_loss(model, X[idx].to(device))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        ev = X[:min(2000, n)].to(device); fl = float(lm_loss(model, ev))
        sw = ev.clone(); sw[:, 1:LB+1] = torch.roll(ev[:, 1:LB+1], 1, 0)   # input-swap
        p = F.log_softmax(model(ev)[:, TS-1], -1); q = F.log_softmax(model(sw)[:, TS-1], -1)
        dki = float((p.exp() * (p - q)).sum(-1).mean())                    # input-dependence KL
    return model, fl, dki


def classify(fl, dki):
    if fl < 0.5: return "mem"
    if fl > 1.8 and dki < 0.3: return "trap"
    return "partial"


def main():
    import json, math, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8000); ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    vocab = NSPEC + ALPHA
    Ns = [500, 1000, 2000, 4000, 8000]
    GRID = [3e-3, 7e-3, 1.5e-2, 3e-2, 6e-2]
    out = {"M": a.M, "steps": a.steps, "Ns": Ns, "grid": GRID, "data": {}}
    os.makedirs("card_ckpts", exist_ok=True)
    print(f"M={a.M} fixed, steps={a.steps}, seeds={a.seeds}.  classify per (N,eta): frac trapped over seeds.")
    for N in Ns:
        X = build_task(a.M, N, seed=0)
        print(f"\n--- N={N} ---  eta: frac_trap (mem/trap/partial)")
        row = {}
        for eta in GRID:
            cs = []; mem_model = None
            for s in range(a.seeds):
                model, fl, dki = train(X, eta, s, a.steps, vocab)
                cs.append(classify(fl, dki))
                if cs[-1] == "mem" and mem_model is None and eta <= 7e-3:
                    torch.save(model.state_dict(), f"card_ckpts/mem_N{N}.pt"); mem_model = 1
            ntr = cs.count("trap"); nme = cs.count("mem"); npa = cs.count("partial")
            frac = (ntr + 0.5 * npa) / len(cs)
            row[eta] = frac
            print(f"  {eta:>7.4f}: {frac:.2f}  ({nme}/{ntr}/{npa})")
        # eta* = log-interpolated 50% crossing
        es = sorted(row); star = None
        for i in range(len(es) - 1):
            f0, f1 = row[es[i]], row[es[i+1]]
            if f0 < 0.5 <= f1:
                t = (0.5 - f0) / (f1 - f0 + 1e-9)
                star = math.exp(math.log(es[i]) + t * (math.log(es[i+1]) - math.log(es[i]))); break
        out["data"][N] = {"frac": row, "eta_star": star}
        print(f"  -> eta*(N={N}) = {star:.4f}" if star else "  -> eta* outside grid")
    json.dump(out, open("card_ckpts/sweep.json", "w"), indent=2)

    # power-law fit log eta* vs log N
    pts = [(N, d["eta_star"]) for N, d in out["data"].items() if d["eta_star"]]
    if len(pts) >= 3:
        import statistics as st
        xs = [math.log(N) for N, _ in pts]; ys = [math.log(e) for _, e in pts]
        mx, my = st.mean(xs), st.mean(ys)
        sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        alpha = -sxy/sxx; b = my - (-alpha)*mx
        ss_res = sum((y-(b-alpha*x))**2 for x, y in zip(xs, ys)); ss_tot = sum((y-my)**2 for y in ys)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
        print(f"\nPOWER LAW: eta* ∝ N^(-alpha),  alpha = {alpha:.3f}   R² = {r2:.3f}")
        print("  eta*(N): " + ", ".join(f"N{N}:{e:.4f}" for N, e in pts))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.loglog([N for N, _ in pts], [e for _, e in pts], "o-", color="#7B1FA2", ms=8)
        if len(pts) >= 3:
            import numpy as np
            xx = np.array([N for N, _ in pts], float)
            ax.loglog(xx, math.exp(b)*xx**(-alpha), "--", color="#888", label=f"α={alpha:.2f}, R²={r2:.2f}")
            ax.legend(frameon=False)
        ax.set_xlabel("output cardinality N"); ax.set_ylabel("critical LR η*(N)")
        ax.set_title("η*(N): critical learning rate vs output cardinality")
        fig.tight_layout(); fig.savefig("../lesswrong_figures/cardinality_law.png", dpi=140, bbox_inches="tight", facecolor="white")
        print("saved lesswrong_figures/cardinality_law.png")
    except Exception as e:
        print("plot skipped:", e)


main()
