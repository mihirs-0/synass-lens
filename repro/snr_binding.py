#!/usr/bin/env python
"""Earn or falsify the dilution mechanism for the Rule C BINDING task — the same bar as the -0.48 case.

Measured at the decision point (fixed LR=1e-3, full p=0.5 train set, K in {4,8,16}, both S, seeds):
  - McCandlish gradient SNR2 = ||E[g]||^2 / tr(Cov)   (+ coherence SNR1)   at step 200 (before the fork)
  - curvature lambda_max (power iteration)             at step 200
Binding-collapse eta* was ~K^-0.25 and S-INDEPENDENT. Verdict:
  SNR2 ~ K^-0.25 & S-indep & eta* ∝ SNR  -> dilution confirmed for binding; gentle exponent = gentle SNR decay.
  SNR2 ~ K^-0.5 (steeper than eta*)       -> dilution does NOT explain the -0.25; memorization-specific.
  SNR2 flat/noisy                          -> mechanism unresolved; cap the claim.
  2/lambda_max slope != -0.25 (esp. flat)  -> curvature falsified for binding too (as for memorization).
"""
import math, json
import torch, torch.nn.functional as F
from torch.func import functional_call, vmap, grad
import foreclosure_c as fc

device = fc.device; TS = fc.TS
ETA_STAR = {(2,4):0.0245,(2,8):0.0221,(2,16):0.0173,(3,4):0.0245,(3,8):0.0221,(3,16):0.0186}  # from the grid


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
        xb = X[i:i+chunk].to(device)
        gd = gfn(params, xb)
        flat = torch.cat([gd[k].reshape(xb.size(0), -1) for k in names], dim=1)
        sg = flat.sum(0); sum_g = sg if sum_g is None else sum_g + sg
        sum_n2 += float((flat*flat).sum()); n += xb.size(0)
    Eg = sum_g/n; eg2 = float((Eg*Eg).sum()); En2 = sum_n2/n
    snr1 = math.sqrt(eg2/En2); trcov = En2 - eg2
    return snr1, (eg2/trcov if trcov > 0 else float("nan"))


def top_eig(model, batch, iters=25):
    ps = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p) for p in ps]
    nrm = lambda vs: math.sqrt(sum(float((x*x).sum()) for x in vs)) + 1e-12
    s = nrm(v); v = [x/s for x in v]; eig = 0.0
    for _ in range(iters):
        g = torch.autograd.grad(fc.lm_loss(model, batch), ps, create_graph=True)
        gv = sum((gi*vi).sum() for gi, vi in zip(g, v))
        Hv = torch.autograd.grad(gv, ps, retain_graph=False)
        eig = sum(float((hv*vi).sum()) for hv, vi in zip(Hv, v))
        s = nrm(Hv); v = [hv.detach()/s for hv in Hv]
    return eig


def warm(X, steps, seed, lr=1e-3):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = fc.TinyGPT(fc.NSPEC + fc.ALPHA).to(device)
    if steps > 0:
        opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        for _ in range(steps):
            idx = torch.randint(0, X.size(0), (256,), generator=g)
            loss = fc.lm_loss(m, X[idx].to(device)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return m


def slope(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sl = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sum((x-mx)**2 for x in xs)
    r2 = 1 - sum((y-(my+sl*(x-mx)))**2 for x, y in zip(xs, ys))/(sum((y-my)**2 for y in ys)+1e-12)
    return sl, r2


def main():
    Ss = [2, 3]; Ks = [4, 8, 16, 32]; LR = 1e-3
    res = {"step0": {}, "step200": {}}
    print(f"device={device}  LR={LR}  full p=0.5 train set per (S,K).  binding eta* ~ K^-0.25, S-independent.")
    print(f"{'S':>2}{'K':>4}{'ck':>8} {'snr1':>9}{'snr2(McC)':>11}{'lmax':>9}")
    for S in Ss:
        for K in Ks:
            pairs, _ = fc.build_pairs(1000, S, K, mode="bind", seed=0)
            tr, te, _ = fc.split_holdout(pairs, K, 0.5, seed=0)
            X = fc.seqs_of(pairs, tr)
            for ck, steps, seeds in [("step0", 0, [0, 1]), ("step200", 200, list(range(8)))]:
                s1s, s2s, lms = [], [], []
                for seed in seeds:
                    gsub = torch.Generator().manual_seed(100 + seed)
                    Xs = X[torch.randperm(X.size(0), generator=gsub)[:2000]]    # FIXED n=2000, RESAMPLED per seed
                    m = warm(X, steps, seed, LR); m.eval()
                    s1, s2 = snr_stats(m, Xs); s2 = s2 - 1.0 / Xs.size(0)        # bias-correct (eg2 inflated by trcov/n)
                    s1 = math.sqrt(max(s2, 1e-9) / (1 + max(s2, 1e-9)))          # coherence from corrected s2
                    lm = top_eig(m, X[:1024].to(device))
                    s1s.append(s1); s2s.append(s2); lms.append(lm)
                a = lambda L: sum(L)/len(L)
                res[ck][(S, K)] = {"snr1": a(s1s), "snr2": a(s2s), "lmax": a(lms)}
                print(f"{S:>2}{K:>4}{ck:>8} {a(s1s):>9.4f}{a(s2s):>11.5f}{a(lms):>9.2f}")
            json.dump({f"{ck}|{S}_{K}": res[ck][(S, K)] for ck in res for (S, K) in res[ck]},
                      open("card_ckpts/snr_binding.json", "w"), indent=2)

    print("\n=== step200 scaling vs K  (binding eta* slope was -0.25) ===")
    for S in Ss:
        sl, r2 = slope([math.log(K) for K in Ks], [math.log(res["step200"][(S, K)]["snr2"]) for K in Ks])
        print(f"  SNR2  S={S}: slope={sl:+.3f} R2={r2:.2f}")
    for S in Ss:
        sl, r2 = slope([math.log(K) for K in Ks], [math.log(2.0/res["step200"][(S, K)]["lmax"]) for K in Ks])
        print(f"  2/lmax S={S}: slope={sl:+.3f} R2={r2:.2f}   (curvature; flat or !=-0.25 -> falsified)")
    print("\n=== eta* vs SNR2 (slope ~1 => eta* governed by gradient signal) ===")
    keys = [(S, K) for S in Ss for K in Ks if (S, K) in ETA_STAR]
    sl, r2 = slope([math.log(res["step200"][k]["snr2"]) for k in keys], [math.log(ETA_STAR[k]) for k in keys])
    print(f"  log eta* vs log SNR2: slope={sl:+.2f} R2={r2:.2f}  (memorization was ~1.1; uses K={sorted({k for _,k in keys})})")
    print("\n=== S-independence of SNR2 (step200) — should mirror eta*'s S-independence ===")
    for K in Ks:
        print(f"  K={K:>2}: S2={res['step200'][(2,K)]['snr2']:.5f}  S3={res['step200'][(3,K)]['snr2']:.5f}")
    print("\n=== step0 (init) SNR2 — should be ~flat/uninformative, confirming step200 is task-driven ===")
    sl, r2 = slope([math.log(K) for K in Ks], [math.log(res["step0"][(2, K)]["snr2"]) for K in Ks])
    print(f"  SNR2 S=2 step0: slope={sl:+.3f} R2={r2:.2f}")


if __name__ == "__main__":
    main()
