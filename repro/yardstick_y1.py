#!/usr/bin/env python
"""Yardstick 1 only: eta_opt(N) = fastest clean convergence to memorization (fewest steps to train loss<0.5,
the project's memorization criterion). Then ratio eta*/eta_opt. Early-stops on convergence AND on collapse."""
import json, math
import torch
import hessian_check as H

device = H.device
OPT_GRID = [1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2]
THR = 0.5; BUDGET = 6000


def steps_to_mem(X, eta, seed, vocab):
    """Return (steps_to_loss<THR or None, final_loss). Early-stop on convergence or collapse."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = H.TinyGPT(vocab).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=eta, betas=(0.9, 0.999), weight_decay=0.01)
    n = X.size(0); ev = X[:min(2000, n)].to(device); fl = 9.9
    for step in range(1, BUDGET + 1):
        idx = torch.randint(0, n, (256,), generator=g)
        loss = H.lm_loss(m, X[idx].to(device)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 50 == 0:
            with torch.no_grad():
                fl = float(H.lm_loss(m, ev))
            if fl < THR:
                return step, fl
            if step >= 2500 and fl > 2.0:                 # collapsed to marginal -> abandon
                return None, fl
    return None, fl


def main():
    sweep = json.load(open("card_ckpts/sweep.json")); M = sweep["M"]; vocab = H.NSPEC + H.ALPHA
    etastar = {int(k): v["eta_star"] for k, v in sweep["data"].items()}
    Ns = sweep["Ns"]
    print(f"Yardstick 1: eta_opt = fastest LR to train loss<{THR} within {BUDGET} steps. grid={OPT_GRID}")
    out = {}
    for N in Ns:
        X = H.build_task(M, N, 0); best = None; trace = []
        for eta in OPT_GRID:
            s, fl = steps_to_mem(X, eta, 0, vocab)
            trace.append((eta, s, round(fl, 3)))
            if s is not None and (best is None or s < best[1]):
                best = (eta, s)
        eta_opt = best[0] if best else None
        y1 = etastar[N] / eta_opt if eta_opt else None
        out[N] = {"eta_star": etastar[N], "eta_opt": eta_opt, "Y1": y1, "trace": trace}
        print(f"N={N:>5}: eta*={etastar[N]:.4f} eta_opt={eta_opt} Y1={'%.2f' % y1 if y1 else 'n/a'}x")
        print(f"        trace (eta: steps,loss): " + " | ".join(f"{e:.3f}:{s},{fl}" for e, s, fl in trace))
    json.dump({str(N): out[N] for N in Ns}, open("card_ckpts/yardstick_y1.json", "w"), indent=2)
    print("\nsaved card_ckpts/yardstick_y1.json")
    print("\n=== Y1 summary (eta*/eta_opt): 3-5x normal | 10-20x high | 50x+ extreme ===")
    for N in Ns:
        y1 = out[N]["Y1"]
        bucket = "n/a" if not y1 else ("near-optimal (<3x)" if y1 < 3 else "high-but-normal" if y1 < 10 else "clearly-high" if y1 < 50 else "extreme")
        print(f"  N={N:>5}: {('%.2f'%y1+'x' if y1 else 'n/a'):>8}  {bucket}")


if __name__ == "__main__":
    main()
