#!/usr/bin/env python
"""EXP 4b — does the collapse go to the DATA marginal P(A) or to UNIFORM? (fixed Exp4 Part2)
Original Part2 skewed at the OUTPUT level but measured at the first-token level, where 1000 outputs over 26
first-token symbols washed the skew out. Fix: make the FIRST answer token itself follow a Zipf(s) over the 26
content symbols, so the data's first-token marginal is genuinely non-uniform. Train a trapped model (high eta),
measure its first-token output marginal, compare KL(model||data-marginal) vs KL(model||uniform).
  KL_data << KL_uniform -> collapses to the DATA marginal P(A) (frequency-driven). Bulletproofs the claim.
  KL_uniform << KL_data -> collapses to UNIFORM regardless of frequencies. Reword the claim.
Mapping is still a valid lookup B->A (a0 skewed, rest random), so it traps at high eta."""
import json, math, random
import torch, torch.nn.functional as F
import collapse_lib as L

vocab = L.NSPEC + L.ALPHA
CONTENT = list(range(L.NSPEC, L.NSPEC + L.ALPHA))     # 26 symbols
STEPS, HI_ETA, M, SEEDS = 5000, 0.06, 8000, 2


def build_skewed_firsttok(s, seed=0):
    """First answer token ~ Zipf(s) over the 26 content symbols; remaining 3 tokens uniform-random."""
    rng = random.Random(seed)
    w = [1.0 / ((i + 1) ** s) for i in range(L.ALPHA)]; sw = sum(w); probs = [x / sw for x in w]
    cum = []; acc = 0.0
    for p in probs: acc += p; cum.append(acc)
    def draw():
        u = rng.random()
        for i, c in enumerate(cum):
            if u <= c: return CONTENT[i]
        return CONTENT[-1]
    usedB = set(); seqs = []; cnt = [0] * vocab
    for _ in range(M):
        while True:
            b = tuple(rng.choice(CONTENT) for _ in range(L.LB))
            if b not in usedB: usedB.add(b); break
        a0 = draw(); a = [a0] + [rng.choice(CONTENT) for _ in range(L.LA - 1)]; cnt[a0] += 1
        seqs.append([L.BOS, *b, L.SEP, *a, L.EOS])
    tot = sum(cnt); marg = torch.tensor([c / tot for c in cnt])
    return torch.tensor(seqs), marg


@torch.no_grad()
def first_tok_marginal(model, X, n=4000):
    ev = X[:min(n, X.size(0))].to(L.device)
    return F.softmax(model(ev)[:, L.TS - 1], -1).mean(0).cpu()


def kl_on_content(p, q):
    """KL restricted+renormalized to the 26 content tokens (avoids special-token noise)."""
    idx = torch.tensor(CONTENT)
    pp = p[idx].clamp_min(1e-9); pp = pp / pp.sum()
    qq = q[idx].clamp_min(1e-9); qq = qq / qq.sum()
    return float((pp * (pp.log() - qq.log())).sum())


def ent_ratio(marg):
    idx = torch.tensor(CONTENT); m = marg[idx].clamp_min(1e-12); m = m / m.sum()
    return float(-(m * m.log()).sum() / math.log(L.ALPHA))


def main():
    print(f"EXP4b first-token-level skew. steps={STEPS} hi_eta={HI_ETA} M={M} seeds={SEEDS}")
    print("Decisive: KL(model||data) << KL(model||uniform) => collapse to DATA marginal P(A).\n")
    uniform = torch.zeros(vocab);
    for c in CONTENT: uniform[c] = 1.0 / L.ALPHA
    res = []
    for s in [0.0, 1.0, 1.5]:
        kds, kus, ments = [], [], []
        for seed in range(SEEDS):
            X, data_marg = build_skewed_firsttok(s, seed=seed)
            m, fl, klB = L.train(X, HI_ETA, seed, STEPS, vocab)
            mm = first_tok_marginal(m, X)
            kds.append(kl_on_content(mm, data_marg))
            kus.append(kl_on_content(mm, uniform))
            ments.append(ent_ratio(mm))
            dent = ent_ratio(data_marg)
        a = lambda L_: sum(L_) / len(L_)
        kd, ku, me = a(kds), a(kus), a(ments)
        verdict = ("DATA marginal P(A)" if kd < ku * 0.5 else
                   "UNIFORM (not frequency-driven)" if ku < kd * 0.5 else "between/ambiguous")
        row = {"skew_s": s, "data_H_ratio": round(dent, 3), "trap_loss": round(fl, 3), "trap_Bkl": round(klB, 4),
               "model_H_ratio": round(me, 3), "KL_model_to_data": round(kd, 4), "KL_model_to_uniform": round(ku, 4),
               "verdict": verdict}
        res.append(row)
        print(f"  s={s:<4} data H/Hmax={dent:.3f} | trapped(loss={fl:.2f},Bkl={klB:.3f}) "
              f"model H/Hmax={me:.3f} | KL->data={kd:.4f}  KL->uniform={ku:.4f}  => {verdict}")
    json.dump(res, open("card_ckpts/exp4b_firsttok.json", "w"), indent=2)
    print("\nSanity: s=0 (uniform) -> both KLs ~equal & small. Decisive row = s=1.5 (strong skew).")
    print("saved card_ckpts/exp4b_firsttok.json")


if __name__ == "__main__":
    main()
