#!/usr/bin/env python
"""Phase B mechanism check: top Hessian eigenvalue lambda_max at the memorized solution per N.
Edge-of-stability says GD is unstable for eta > 2/lambda_max. If 2/lambda_max tracks eta*(N),
the cardinality law is curvature-driven (Papyan: Hessian outliers scale with #classes -> critical LR).
Self-contained (does not import the sweep, which runs on import)."""
import json, math, random
import torch, torch.nn as nn, torch.nn.functional as F

PAD, BOS, SEP, EOS = 0, 1, 2, 3; NSPEC = 4
ALPHA, LB, LA = 26, 6, 4
T = 1 + LB + 1 + LA + 1; TS = 1 + LB + 1
device = "mps" if torch.backends.mps.is_available() else "cpu"


def build_task(M, N, seed=0):
    rng = random.Random(seed); content = list(range(NSPEC, NSPEC + ALPHA))
    usedA = set(); outs = []
    while len(outs) < N:
        s = tuple(rng.choice(content) for _ in range(LA))
        if s not in usedA: usedA.add(s); outs.append(s)
    assign = [outs[i % N] for i in range(M)]; rng.shuffle(assign)
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


def top_eig(model, batch, iters=30):
    ps = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p) for p in ps]
    nrm = lambda vs: math.sqrt(sum(float((x*x).sum()) for x in vs)) + 1e-12
    s = nrm(v); v = [x/s for x in v]; eig = 0.0
    for _ in range(iters):
        g = torch.autograd.grad(lm_loss(model, batch), ps, create_graph=True)
        gv = sum((gi*vi).sum() for gi, vi in zip(g, v))
        Hv = torch.autograd.grad(gv, ps, retain_graph=False)
        eig = sum(float((hv*vi).sum()) for hv, vi in zip(Hv, v))   # Rayleigh v^T H v
        s = nrm(Hv); v = [hv.detach()/s for hv in Hv]
    return eig


def main():
    sweep = json.load(open("card_ckpts/sweep.json")); M = sweep["M"]; vocab = NSPEC + ALPHA
    print(f"{'N':>6} {'eta*':>8} {'lambda_max':>11} {'2/lmax':>8} {'ratio eta*/(2/lmax)':>20}")
    rows = []
    for N in sweep["Ns"]:
        es = sweep["data"][str(N)]["eta_star"] if str(N) in sweep["data"] else sweep["data"][N]["eta_star"]
        import os
        ck = f"card_ckpts/mem_N{N}.pt"
        if not os.path.exists(ck) or es is None:
            print(f"{N:>6}  (missing ckpt or eta*)"); continue
        m = TinyGPT(vocab).to(device); m.load_state_dict(torch.load(ck, map_location=device)); m.eval()
        X = build_task(M, N, 0)[:1024].to(device)
        lmax = top_eig(m, X)
        twol = 2.0 / lmax if lmax > 0 else float("nan")
        print(f"{N:>6} {es:>8.4f} {lmax:>11.2f} {twol:>8.4f} {es/twol:>20.2f}")
        rows.append((N, es, lmax, twol))
    if len(rows) >= 3:
        import statistics as st
        # does 2/lmax track eta*? fit log eta* vs log(2/lmax)
        xs = [math.log(t) for _, _, _, t in rows]; ys = [math.log(e) for _, e, _, _ in rows]
        mx, my = st.mean(xs), st.mean(ys); sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        slope = sxy/sxx; r2 = 1 - sum((y-(my+slope*(x-mx)))**2 for x, y in zip(xs, ys))/(sum((y-my)**2 for y in ys)+1e-12)
        print(f"\nlog eta* vs log(2/lambda_max): slope={slope:.2f} (1.0 = perfect EoS tracking), R²={r2:.2f}")
        print("slope≈1 & high R² -> curvature-driven (edge of stability); else scaling is elsewhere.")


if __name__ == "__main__":
    main()
