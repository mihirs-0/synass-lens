#!/usr/bin/env python
"""REVIEW EXP D — sum-vs-mean loss (hygiene, not a kill-shot).
L_sum = (B*LA) * L_mean. For SGD, eta_sum = eta_mean/(B*LA) gives identical updates. Under Adam, the loss
SCALE cancels (Adam normalizes by sqrt(v)), so sum and mean should give ~identical eta* at the SAME eta.
This confirms the collapse isn't an artifact of the loss-reduction convention. Run at N=2000."""
import json, os
import torch, torch.nn.functional as F
import collapse_lib as L

device = L.device; vocab = L.NSPEC + L.ALPHA; TS = L.TS
GRID = [3e-3, 7e-3, 1.5e-2, 3e-2, 6e-2]; STEPS, SEEDS, N, M = 5000, 2, 2000, 8000


def lm_loss_sum(model, seq):
    lg = model(seq)
    return F.cross_entropy(lg[:, TS-1:-1].reshape(-1, lg.size(-1)), seq[:, TS:].reshape(-1), reduction="sum")


def train_sum(X, eta, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = L.TinyGPT(vocab).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=eta, betas=(0.9, 0.999), weight_decay=0.01)
    for _ in range(STEPS):
        idx = torch.randint(0, X.size(0), (256,), generator=g)
        opt.zero_grad(set_to_none=True); lm_loss_sum(m, X[idx].to(device)).backward(); opt.step()
    with torch.no_grad():
        fl = float(L.lm_loss(m, X[:2000].to(device)))
    return fl, L.input_dep_kl(m, X)


def main():
    os.makedirs("card_ckpts", exist_ok=True)
    X, *_ = L.build_task(N, M=M, seed=0)
    print(f"=== EXP D sum-vs-mean (N={N}); Adam is scale-invariant → expect η* unchanged ===\n")
    out = {}
    for label, trainer in [("mean (standard)", lambda e, s: L.train(X, e, s, STEPS, vocab)[1:]),
                           ("sum", lambda e, s: train_sum(X, e, s))]:
        frac = {}
        for eta in GRID:
            fr = [L.classify(*trainer(eta, s)) for s in range(SEEDS)]
            frac[eta] = (fr.count("trap") + 0.5 * fr.count("partial")) / SEEDS
        es = L.eta_star(frac); out[label] = {"eta_star": es, "frac": frac}
        print(f"  {label:>16}: η*={('%.4f'%es) if es else 'n/a':>8}  " + " ".join(f"{e:.3f}:{frac[e]:.2f}" for e in GRID))
    a, b = out["mean (standard)"]["eta_star"], out["sum"]["eta_star"]
    print(f"\n  η*(mean)={a}  η*(sum)={b}  →  {'IDENTICAL (Adam scale-invariance; collapse not a reduction artifact)' if a and b and abs(a-b)<1e-6 else 'differ (optimizer-specific, not a fundamental law)'}")
    json.dump(out, open("card_ckpts/rev_sumvsmean.json", "w"), indent=2)
    print("saved card_ckpts/rev_sumvsmean.json")


if __name__ == "__main__":
    main()
