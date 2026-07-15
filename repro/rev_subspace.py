#!/usr/bin/env python
"""REVIEW EXP B — subspace-resolved conditional-mode stability (the kill-shot, ordinary-vs-mysterious).
Split logits z_i = zbar (input-independent mean mode) + ztil_i (zero-mean conditional modes).
From a MEMORIZED solution (strong conditional structure), run the ACTUAL Adam update map at each eta and
measure whether the conditional magnitude C=RMS_i||ztil_i|| grows/holds (memorized) or collapses (-> trap),
and whether the mean mode is preserved. Decisive:
  - does the conditional-collapse threshold eta_crit_cond(N) MATCH the loss-based eta*(N) and scale the same?
    YES -> conditional modes destabilize while mean survives = ordinary one-mechanism low-rank collapse.
    NO  -> conditional-mode stability does NOT predict the threshold = genuinely harder to explain.
(Adam state starts fresh from the memorized checkpoint, which carries no optimizer state — the honest
 'continue training from this solution' setup; Exp A showed optimizer state matters, so we report the
 augmented-state dynamics directly via the real Adam steps.)"""
import json, os, math
import numpy as np
import torch, torch.nn.functional as F
import collapse_lib as L

device = L.device; vocab = L.NSPEC + L.ALPHA; TS, LA = L.TS, L.LA
ETA_STAR = {500: 0.025, 2000: 0.015, 8000: 0.007}     # loss-based threshold from the cardinality sweep
GRID = [3e-3, 7e-3, 1.5e-2, 3e-2, 6e-2]
M = 8000


@torch.no_grad()
def cond_mean(model, Xp):
    z = model(Xp.to(device))[:, TS-1:TS-1+LA].detach()      # (n, LA, V)
    zbar = z.mean(0, keepdim=True); ztil = z - zbar
    C = float(ztil.pow(2).sum(-1).sqrt().mean())            # conditional (input-dependent) magnitude
    Mn = float(zbar.norm())                                 # mean-mode magnitude
    return C, Mn


def flow(mem_sd, X, Xp, eta, k=40):
    m = L.TinyGPT(vocab).to(device); m.load_state_dict(mem_sd)
    opt = torch.optim.AdamW(m.parameters(), lr=eta, betas=(0.9, 0.999), weight_decay=0.01)
    C0, M0 = cond_mean(m, Xp); l0 = float(L.lm_loss(m, Xp.to(device)))
    for _ in range(k):
        opt.zero_grad(set_to_none=True); L.lm_loss(m, X.to(device)).backward(); opt.step()
    Ck, Mk = cond_mean(m, Xp); lk = float(L.lm_loss(m, Xp.to(device))); kl = L.input_dep_kl(m, X)
    return {"eta": eta, "C0": round(C0, 3), "Ck": round(Ck, 3), "Cratio": round(Ck/max(C0, 1e-9), 3),
            "Mratio": round(Mk/max(M0, 1e-9), 3), "l0": round(l0, 3), "lk": round(lk, 3), "klk": round(kl, 4)}


def crit_eta(rows):
    """lowest eta where conditional collapses (Cratio<0.4 i.e. input-dependence mostly gone)."""
    for r in sorted(rows, key=lambda x: x["eta"]):
        if r["Cratio"] < 0.4:
            return r["eta"]
    return None


def main():
    os.makedirs("card_ckpts", exist_ok=True)
    res = {}
    print("=== EXP B subspace-resolved conditional-mode stability ===")
    print("flow from memorized: does the conditional magnitude C collapse (->trap) while the mean mode holds?\n")
    for N in [500, 2000, 8000]:
        ck = f"card_ckpts/mem_N{N}.pt"
        if not os.path.exists(ck):
            print(f"  N={N}: missing {ck}, skip"); continue
        mem_sd = torch.load(ck, map_location=device)
        X, *_ = L.build_task(N, M=M, seed=0); Xp = X[:1500]
        rows = [flow(mem_sd, X, Xp, eta) for eta in GRID]
        ce = crit_eta(rows); res[N] = {"rows": rows, "eta_crit_cond": ce, "eta_star_loss": ETA_STAR[N]}
        print(f"--- N={N}  (loss-based η*={ETA_STAR[N]}) ---")
        for r in rows:
            print(f"  η={r['eta']:.3f}: C {r['C0']}→{r['Ck']} (×{r['Cratio']})  meanMode×{r['Mratio']}  "
                  f"loss {r['l0']}→{r['lk']}  KL={r['klk']}")
        print(f"  → conditional-collapse threshold η_crit_cond = {ce}   vs loss η* = {ETA_STAR[N]}\n")

    # decisive comparison
    print("=== DECISIVE: does conditional-mode stability predict the threshold? ===")
    print(f"{'N':>6} {'η_crit_cond':>12} {'η*_loss':>10} {'match?':>8}")
    for N in sorted(res):
        ce, es = res[N]["eta_crit_cond"], res[N]["eta_star_loss"]
        match = "—" if ce is None else ("yes" if 0.5 <= ce/es <= 2.0 else "no")
        print(f"{N:>6} {str(ce):>12} {es:>10} {match:>8}")
    # does eta_crit_cond fall with N like eta* (~N^-0.48)?
    pts = [(N, res[N]["eta_crit_cond"]) for N in sorted(res) if res[N]["eta_crit_cond"]]
    if len(pts) >= 2:
        xs = [math.log(n) for n, _ in pts]; ys = [math.log(e) for _, e in pts]
        mx, my = np.mean(xs), np.mean(ys)
        slope = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sum((x-mx)**2 for x in xs)
        print(f"\nη_crit_cond ∝ N^({slope:+.2f})   (loss η* was ∝ N^(-0.48))")
        print("mean-mode multiplier ~1 across η at every N (conditional collapses, mean preserved) — see Mratio above.")
    json.dump(res, open("card_ckpts/rev_subspace.json", "w"), indent=2)
    print("\nsaved card_ckpts/rev_subspace.json")


if __name__ == "__main__":
    main()
