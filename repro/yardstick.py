#!/usr/bin/env python
"""Express the collapse threshold eta* in MODEL-INTERNAL units. Memorization cardinality sweep
(same d=64 model/task/N as cardinality_sweep). Two intrinsic yardsticks:

  Y1: eta* / eta_opt          (eta_opt = LR with fewest steps to clean memorization, train loss<0.05)
  Y2: eta* / (2/lambda_max)   (classical edge-of-stability), lambda_max measured at init/step200/memorized,
                               RAW Hessian AND Adam-preconditioned (D^{-1/2} H D^{-1/2}, D=sqrt(v_hat)+eps).

The Adam-preconditioned step200 bound is the RIGHT denominator: we train AdamW (the raw 2/lmax GD bound is
only a rough reference under Adam), and the collapse decision happens early, not at the converged sharp min.
"""
import json, math, copy
import torch
import hessian_check as H   # import-safe (guarded); provides build_task, TinyGPT, lm_loss, top_eig, device

device = H.device
ETA_GRID_SWEEP = [3e-3, 7e-3, 1.5e-2, 3e-2, 6e-2]                          # the coarse grid eta* came from
OPT_GRID = [1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2]             # finer grid for eta_opt search


def train_steps_to_threshold(X, eta, seed, vocab, budget=4000, thr=0.05, evaly=50):
    """Steps to reach clean memorization (train loss < thr). Returns (steps or None, final_loss)."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = H.TinyGPT(vocab).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=eta, betas=(0.9, 0.999), weight_decay=0.01)
    n = X.size(0); ev = X[:min(2000, n)].to(device)
    for step in range(1, budget + 1):
        idx = torch.randint(0, n, (256,), generator=g)
        loss = H.lm_loss(m, X[idx].to(device)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % evaly == 0:
            with torch.no_grad():
                fl = float(H.lm_loss(m, ev))
            if fl < thr:
                return step, fl
    with torch.no_grad():
        return None, float(H.lm_loss(m, ev))


def warm_capture_v(X, steps, seed, vocab, lr=1e-3):
    """Train AdamW `steps` at fixed small LR; return model + bias-corrected v_hat (Adam 2nd moment)."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = H.TinyGPT(vocab).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    n = X.size(0)
    for _ in range(steps):
        idx = torch.randint(0, n, (256,), generator=g)
        loss = H.lm_loss(m, X[idx].to(device)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    ps = [p for p in m.parameters() if p.requires_grad]
    vhat = []
    for p in ps:
        st = opt.state[p]
        if "exp_avg_sq" in st and "step" in st:
            t = float(st["step"]); v = st["exp_avg_sq"] / (1 - 0.999 ** t)
        else:
            v = torch.zeros_like(p)
        vhat.append(v.detach())
    return m, vhat


def top_eig_precond(model, batch, vhat, iters=25, eps=1e-8, floor_frac=1e-3):
    """lambda_max of the Adam-preconditioned Hessian D^{-1/2} H D^{-1/2}, D=diag(sqrt(vhat)+eps).
    Governs Adam stability: step δ -= η D^{-1} H δ  => stable iff η·λ_max(D^{-1}H) < (Adam-crit, ~38 not 2).
    Relative floor on sqrt(vhat): dead-parameter directions (tiny vhat early) otherwise blow up D^{-1/2}."""
    ps = [p for p in model.parameters() if p.requires_grad]
    smax = max(float(torch.sqrt(v).max()) for v in vhat) + 1e-12
    floor = floor_frac * smax
    dih = [1.0 / torch.sqrt(torch.clamp(torch.sqrt(v), min=floor) + eps) for v in vhat]   # D^{-1/2}, floored
    v = [torch.randn_like(p) for p in ps]
    nrm = lambda vs: math.sqrt(sum(float((x * x).sum()) for x in vs)) + 1e-12
    s = nrm(v); v = [x / s for x in v]; eig = 0.0
    for _ in range(iters):
        u = [d * vi for d, vi in zip(dih, v)]                              # D^{-1/2} v
        g = torch.autograd.grad(H.lm_loss(model, batch), ps, create_graph=True)
        gu = sum((gi * ui).sum() for gi, ui in zip(g, u))
        Hu = torch.autograd.grad(gu, ps, retain_graph=False)
        w = [d * hu for d, hu in zip(dih, Hu)]                             # D^{-1/2} H D^{-1/2} v
        eig = sum(float((wi * vi).sum()) for wi, vi in zip(w, v))
        s = nrm(w); v = [wi.detach() / s for wi in w]
    return eig


def main():
    sweep = json.load(open("card_ckpts/sweep.json")); M = sweep["M"]; vocab = H.NSPEC + H.ALPHA
    etastar = {int(k): v["eta_star"] for k, v in sweep["data"].items()}
    Ns = sweep["Ns"]
    print(f"=== eta* in model-internal units (memorization sweep, d=64, M={M}) ===")
    print(f"eta grid eta* came from: {ETA_GRID_SWEEP} (coarse; ~factor-2 rungs -> eta* +/- one rung)")
    print(f"eta_opt search grid: {OPT_GRID}\n")
    rows = {}
    for N in Ns:
        X = H.build_task(M, N, 0)
        es = etastar[N]
        # --- Y1: eta_opt ---
        best = None
        for eta in OPT_GRID:
            steps, fl = train_steps_to_threshold(X, eta, 0, vocab)
            if steps is not None and (best is None or steps < best[1]):
                best = (eta, steps)
        eta_opt = best[0] if best else None
        # --- Y2: lambda_max raw at init / step200 / memorized ---
        torch.manual_seed(0); m0 = H.TinyGPT(vocab).to(device)
        lmax_init = H.top_eig(m0, X[:1024].to(device))
        m200, vhat = warm_capture_v(X, 200, 0, vocab, lr=1e-3)
        lmax_200 = H.top_eig(m200, X[:1024].to(device))
        lmax_adam200 = top_eig_precond(m200, X[:1024].to(device), vhat)
        ck = f"card_ckpts/mem_N{N}.pt"
        mm = H.TinyGPT(vocab).to(device); mm.load_state_dict(torch.load(ck, map_location=device)); mm.eval()
        lmax_mem = H.top_eig(mm, X[:1024].to(device))
        rows[N] = dict(es=es, eta_opt=eta_opt, lmax_init=lmax_init, lmax_200=lmax_200,
                       lmax_adam200=lmax_adam200, lmax_mem=lmax_mem)
        r = rows[N]
        print(f"N={N:>5}: eta*={es:.4f}  eta_opt={eta_opt}  "
              f"lmax[init/200/mem]={lmax_init:.2f}/{lmax_200:.2f}/{lmax_mem:.2f}  lmax_adam200={lmax_adam200:.3f}")

    # ---- table ----
    print("\n" + "=" * 118)
    print(f"{'N':>5} | {'eta*':>7} {'eta_opt':>8} {'Y1 e*/eopt':>11} | "
          f"{'2/lmax_mem':>10} {'r mem':>6} | {'2/lmax_200':>10} {'r 200':>6} | {'2/lmaxAdam':>11} {'r Adam':>7}")
    print("-" * 118)
    for N in Ns:
        r = rows[N]; es = r["es"]
        y1 = es / r["eta_opt"] if r["eta_opt"] else float("nan")
        tl_mem = 2 / r["lmax_mem"]; tl_200 = 2 / r["lmax_200"]; tl_ad = 2 / r["lmax_adam200"]
        print(f"{N:>5} | {es:>7.4f} {r['eta_opt']:>8.4f} {y1:>10.1f}x | "
              f"{tl_mem:>10.4f} {es/tl_mem:>6.2f} | {tl_200:>10.4f} {es/tl_200:>6.2f} | "
              f"{tl_ad:>11.3f} {es/tl_ad:>7.3f}")
    print("=" * 118)
    print("Y1 buckets: 3-5x high-but-normal | 10-20x clearly-high | 50x+ extreme")
    print("r mem/200 = raw-Hessian GD edge ratio (reference only under Adam). r Adam = Adam-preconditioned (correct).")
    json.dump({str(N): rows[N] for N in Ns}, open("card_ckpts/yardstick.json", "w"), indent=2)
    print("\nsaved card_ckpts/yardstick.json")


if __name__ == "__main__":
    main()
