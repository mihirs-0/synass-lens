#!/usr/bin/env python
"""Positively test gradient dilution behind eta*(N) ~ N^-0.5. Per-example gradient SNR vs N,
measured EARLY (step 0 + step 200), fixed LR=1e-3, over the FULL M=8000 (covers all N's outputs).
Defs: (1) coherence ||mean g||/RMS||g||,  (2) McCandlish ||E[g]||^2/tr(Cov).  Same data/model as the sweep."""
import argparse, math, random, json, os
import torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call, vmap, grad

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


def snr_stats(model, X, chunk=256):
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    names = list(params)
    def loss1(p, x):
        lg = functional_call(model, (p, buffers), (x.unsqueeze(0),))
        return F.cross_entropy(lg[0, TS-1:-1], x[TS:])
    gfn = vmap(grad(loss1), in_dims=(None, 0))
    n = 0; sum_g = None; sum_n2 = 0.0
    for i in range(0, X.size(0), chunk):
        xb = X[i:i+chunk]
        gd = gfn(params, xb)
        flat = torch.cat([gd[k].reshape(xb.size(0), -1) for k in names], dim=1)  # (b, P)
        sg = flat.sum(0)
        sum_g = sg if sum_g is None else sum_g + sg
        sum_n2 += float((flat * flat).sum()); n += xb.size(0)
    Eg = sum_g / n; eg2 = float((Eg * Eg).sum()); En2 = sum_n2 / n
    snr1 = math.sqrt(eg2 / En2)                  # ||mean||/RMS (coherence)
    trcov = En2 - eg2
    snr2 = eg2 / trcov if trcov > 0 else float("nan")
    return snr1, snr2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--n", type=int, default=8000); a = ap.parse_args()
    vocab = NSPEC + ALPHA
    if a.test:
        import time
        X = build_task(8000, 500, 0).to(device)
        torch.manual_seed(0); m = TinyGPT(vocab).to(device)
        t = time.time(); s1, s2 = snr_stats(m, X[:a.n])
        print(f"TEST N=500 step0 n={a.n}: SNR1(coh)={s1:.4f} SNR2(McC)={s2:.4f}  ({time.time()-t:.0f}s)  device={device}")
        return
    Ns = [500, 1000, 2000, 4000, 8000]
    g = torch.Generator()
    res = {ck: {d: {} for d in ("snr1", "snr2")} for ck in ("step0", "step200")}
    for N in Ns:
        X = build_task(8000, N, 0).to(device)
        for ck, warm in [("step0", 0), ("step200", 200)]:
            s1s, s2s = [], []
            for seed in range(a.seeds):
                torch.manual_seed(seed); m = TinyGPT(vocab).to(device)
                if warm:
                    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
                    gg = torch.Generator().manual_seed(seed)
                    for _ in range(warm):
                        idx = torch.randint(0, X.size(0), (256,), generator=gg)
                        loss = lm_loss(m, X[idx]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                s1, s2 = snr_stats(m, X[:a.n])
                s1s.append(s1); s2s.append(s2)
            res[ck]["snr1"][N] = sum(s1s)/len(s1s); res[ck]["snr2"][N] = sum(s2s)/len(s2s)
            print(f"N={N:>5} {ck}: SNR1={res[ck]['snr1'][N]:.4f}  SNR2={res[ck]['snr2'][N]:.4f}")
    json.dump(res, open("card_ckpts/snr.json", "w"), indent=2)

    sweep = json.load(open("card_ckpts/sweep.json"))
    etastar = {int(k): v["eta_star"] for k, v in sweep["data"].items()}
    def slope(d):
        xs = [math.log(N) for N in Ns]; ys = [math.log(d[N]) for N in Ns]
        import statistics as st; mx, my = st.mean(xs), st.mean(ys)
        sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        sl = sxy/sxx; r2 = 1 - sum((y-(my+sl*(x-mx)))**2 for x, y in zip(xs, ys))/(sum((y-my)**2 for y in ys)+1e-12)
        return sl, r2
    print("\nlog SNR vs log N slopes (target -0.5):")
    for ck in ("step0", "step200"):
        for d in ("snr1", "snr2"):
            sl, r2 = slope(res[ck][d]); print(f"  {ck} {d}: slope={sl:+.3f} R²={r2:.2f}")
    # eta* vs SNR (step200, snr1)
    xs = [math.log(res["step200"]["snr1"][N]) for N in Ns]; ys = [math.log(etastar[N]) for N in Ns]
    import statistics as st; mx, my = st.mean(xs), st.mean(ys)
    sl = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sum((x-mx)**2 for x in xs)
    print(f"\nlog eta* vs log SNR1(step200): slope={sl:+.2f}  (~1 => eta* governed by gradient signal)")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for ck, mk in [("step0", "s--"), ("step200", "o-")]:
            ax.loglog(Ns, [res[ck]["snr1"][N] for N in Ns], mk, label=f"SNR1 coherence ({ck})")
            ax.loglog(Ns, [res[ck]["snr2"][N] for N in Ns], mk, alpha=0.5, label=f"SNR2 McCandlish ({ck})")
        x = np.array(Ns, float); ref = res["step200"]["snr1"][Ns[0]] * (x/Ns[0])**(-0.5)
        ax.loglog(x, ref, "k:", label="N^(-0.5) ref")
        ax.set_xlabel("output cardinality N"); ax.set_ylabel("gradient SNR"); ax.legend(fontsize=8, frameon=False)
        ax.set_title("Gradient SNR vs N (dilution test)")
        fig.tight_layout(); fig.savefig("../lesswrong_figures/snr_dilution.png", dpi=140, bbox_inches="tight", facecolor="white")
        print("saved lesswrong_figures/snr_dilution.png")
    except Exception as e:
        print("plot skipped:", e)


main()
