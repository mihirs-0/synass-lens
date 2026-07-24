#!/usr/bin/env python
"""EXP 4 — is the collapse really to the target prior P(A), or just input-independent?
Under teacher forcing, 'ignores B' != 'predicts P(A)': the model could ignore B while still using the
generated prefix A_<t, sitting at the autoregressive prior P(A_t|A_<t), not the flat marginal P(A).
On a trapped model, at EVERY answer position measure:
  - B-dependence:      swap B, keep true prefix -> KL. trapped => ~0 at every position (input-independent).
  - prefix-dependence: shuffle the true prefix A_<t, keep B -> KL. ~0 => flat per-position marginal P(A);
                       >0 => autoregressive prior P(A_t|A_<t). Distinguishes the two claims.
Bonus: impose skewed target frequencies (mapping fixed), train trapped, check the trapped first-token
marginal follows the IMPOSED prior (KL to imposed << KL to uniform) -> collapse is frequency-driven.
(No saved trapped ckpts exist, so trapped models are trained inline at high eta.)"""
import json, os, math
import torch, torch.nn.functional as F
import collapse_lib as L

vocab = L.NSPEC + L.ALPHA; STEPS = 5000; HI_ETA = 0.06
TS, LA, LB = L.TS, L.LA, L.LB
path = "card_ckpts/exp4_prior.json"
res = {}


@torch.no_grad()
def per_position(model, X, n=2000):
    """B-swap KL and prefix-shuffle KL at each of the LA answer positions (teacher-forced true prefix)."""
    model.eval(); ev = X[:min(n, X.size(0))].to(device := L.device)
    out = []
    base = F.log_softmax(model(ev), -1)                       # (B,T,V)
    # B-swap
    swB = ev.clone(); swB[:, 1:LB+1] = torch.roll(ev[:, 1:LB+1], 1, 0)
    lgB = F.log_softmax(model(swB), -1)
    # prefix-shuffle: permute the answer tokens already in the sequence (positions TS..TS+LA-1) across batch
    swP = ev.clone(); perm = torch.randperm(ev.size(0))
    swP[:, TS:TS+LA] = ev[perm, TS:TS+LA]
    lgP = F.log_softmax(model(swP), -1)
    for t in range(LA):
        pos = TS - 1 + t                                      # logits that predict answer token t
        p = base[:, pos]
        klB = float((p.exp() * (p - lgB[:, pos])).sum(-1).mean())
        klP = float((p.exp() * (p - lgP[:, pos])).sum(-1).mean())
        out.append({"pos": t, "B_swap_kl": round(klB, 4), "prefix_shuffle_kl": round(klP, 4)})
    return out


@torch.no_grad()
def first_tok_marginal(model, X, n=4000):
    ev = X[:min(n, X.size(0))].to(L.device)
    p = F.softmax(model(ev)[:, TS-1], -1).mean(0)             # avg predicted dist at first answer pos
    return p


def imposed_first_tok_prior(outs, probs):
    """Empirical P(first answer token) under the target prior."""
    v = vocab; q = torch.zeros(v)
    for o, pr in zip(outs, probs): q[o[0]] += pr
    return q / q.sum()


def kl(p, q):
    p = p / p.sum(); q = q / q.sum()
    return float((p * (torch.log(p + 1e-9) - torch.log(q + 1e-9))).sum())


def main():
    print(f"EXP4 prior-vs-input-independent. steps={STEPS} hi_eta={HI_ETA}\n")

    # Part 1: per-position B- and prefix-dependence on a trapped model (uniform prior, N=2000)
    print("--- Part 1: trapped model (N=2000 uniform), per-position dependence ---")
    X, assign, outs, probs = L.build_task(2000, M=8000, prior="uniform", seed=0)
    m, fl, klB = L.train(X, HI_ETA, 0, STEPS, vocab)
    print(f"  trapped check: loss={fl:.3f} first-pos B-swap kl={klB:.4f} -> {L.classify(fl, klB)}")
    pp = per_position(m, X)
    res["part1"] = {"loss": round(fl, 3), "per_position": pp}
    print(f"  {'pos':>3} {'B_swap_kl':>10} {'prefix_shuffle_kl':>18}")
    for r in pp: print(f"  {r['pos']:>3} {r['B_swap_kl']:>10} {r['prefix_shuffle_kl']:>18}")
    bmax = max(r["B_swap_kl"] for r in pp); pmax = max(r["prefix_shuffle_kl"] for r in pp)
    verdict = ("flat marginal P(A)" if pmax < 0.1 else "autoregressive prior P(A_t|A_<t)") if bmax < 0.1 \
              else "still input-dependent (not fully trapped)"
    res["part1"]["verdict"] = verdict
    print(f"  -> B-indep at all pos? {bmax<0.1} (max {bmax}); prefix used? {pmax>=0.1} (max {pmax}) => {verdict}\n")

    # Part 2: does the trapped marginal follow an IMPOSED skewed prior?
    print("--- Part 2: impose skewed target frequencies (mapping fixed), does trapped marginal follow? ---")
    res["part2"] = []
    for skew in [0.0, 8.0]:
        prior = "uniform" if skew == 0 else "skew"
        Xs, a, o, pr = L.build_task(1000, M=8000, prior=prior, skew=skew, seed=1)
        ms, fls, klBs = L.train(Xs, HI_ETA, 0, STEPS, vocab)
        model_marg = first_tok_marginal(ms, Xs).cpu()
        imposed = imposed_first_tok_prior(o, pr)
        uniform = imposed_first_tok_prior(o, [1.0/len(o)]*len(o))
        kimp, kuni = kl(model_marg, imposed), kl(model_marg, uniform)
        H = -sum(p*math.log(p+1e-12) for p in pr if p>0)/math.log(len(o))
        res["part2"].append({"skew": skew, "H_ratio": round(H,3), "loss": round(fls,3), "first_pos_Bkl": round(klBs,4),
                             "kl_model_to_imposed": round(kimp,4), "kl_model_to_uniform": round(kuni,4)})
        print(f"  skew={skew:>4} H/Hmax={H:.3f}: trapped loss={fls:.3f} Bkl={klBs:.4f} | "
              f"KL(model||imposed)={kimp:.4f}  KL(model||uniform)={kuni:.4f} -> "
              f"{'follows imposed prior' if kimp < kuni*0.6 else 'closer to uniform' if kuni<kimp*0.6 else 'between'}")
    json.dump(res, open(path, "w"), indent=2); print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
