#!/usr/bin/env python
"""Shared lib for the collapse confound experiments (weight decay / factorial / batch / prior).
Memorization task (NO z): [BOS, B(6), SEP, A(4), EOS]. Flexible build_task: control N (distinct
outputs), M (examples) / epp (examples per target), and target prior (uniform vs skewed). Matches
hessian_check / cardinality_sweep model & sequence exactly so eta* is comparable to the existing sweep."""
import math, random
import torch, torch.nn as nn, torch.nn.functional as F

PAD, BOS, SEP, EOS = 0, 1, 2, 3; NSPEC = 4
ALPHA, LB, LA = 26, 6, 4
T = 1 + LB + 1 + LA + 1            # 13
TS = 1 + LB + 1                    # first answer index = 8
device = "mps" if torch.backends.mps.is_available() else "cpu"
ETA_GRID = [3e-3, 7e-3, 1.5e-2, 3e-2, 6e-2]   # same grid eta* was measured on


def build_task(N, epp=None, M=None, prior="uniform", skew=1.0, seed=0):
    """Returns (X seqs, target_index per example, outs list, prior_probs over N).
    prior='uniform' -> balanced round-robin (each output ~M/N times).
    prior='skew'    -> target index ~ geometric-ish weights w_i ∝ exp(-skew*i/N); first N forced to
                       cover all outputs (keeps mapping surjective), remainder sampled."""
    rng = random.Random(seed); content = list(range(NSPEC, NSPEC + ALPHA))
    if M is None: M = (epp if epp else 8) * N
    usedA = set(); outs = []
    while len(outs) < N:
        s = tuple(rng.choice(content) for _ in range(LA))
        if s not in usedA: usedA.add(s); outs.append(s)
    if prior == "uniform":
        assign = [i % N for i in range(M)]
        probs = [1.0 / N] * N
    else:
        w = [math.exp(-skew * i / N) for i in range(N)]; sw = sum(w); probs = [x / sw for x in w]
        cum = []; acc = 0.0
        for p in probs: acc += p; cum.append(acc)
        def draw():
            u = rng.random()
            for i, c in enumerate(cum):
                if u <= c: return i
            return N - 1
        assign = list(range(N)) + [draw() for _ in range(max(0, M - N))]  # cover all, then sample
    rng.shuffle(assign)
    usedB = set(); seqs = []
    for i in range(M):
        while True:
            b = tuple(rng.choice(content) for _ in range(LB))
            if b not in usedB: usedB.add(b); break
        seqs.append([BOS, *b, SEP, *outs[assign[i]], EOS])
    return torch.tensor(seqs), torch.tensor(assign), outs, probs


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


@torch.no_grad()
def input_dep_kl(model, X, n=2000):
    """B-swap KL at the first answer position (input-dependence)."""
    model.eval(); ev = X[:min(n, X.size(0))].to(device)
    sw = ev.clone(); sw[:, 1:LB+1] = torch.roll(ev[:, 1:LB+1], 1, 0)
    p = F.log_softmax(model(ev)[:, TS-1], -1); q = F.log_softmax(model(sw)[:, TS-1], -1)
    return float((p.exp() * (p - q)).sum(-1).mean())


@torch.no_grad()
def norms(model):
    """Param / output-bias / input-pathway norms — the 'cheap prior vs expensive lookup' picture."""
    total = math.sqrt(sum(float((p * p).sum()) for p in model.parameters()))
    obias = float(model.head.bias.norm())
    owt = float(model.head.weight.norm())
    inpath = float(model.tok.weight.norm())                 # token embeddings = how B enters
    return dict(total=round(total, 3), head_bias=round(obias, 3), head_w=round(owt, 3), tok_emb=round(inpath, 3))


def train(X, eta, seed, steps, vocab, wd=0.01, batch=256, track_norms=False, lam_per_step=None):
    """Train AdamW; return (model, final_loss, kl, [norm_trace]). lam_per_step overrides wd per step
    (for iso-cumulative-decay condition). track_norms logs norms every steps//12."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    model = TinyGPT(vocab).to(device)
    wd0 = 0.0 if wd is None else wd        # lam_per_step overrides each step; AdamW rejects None
    opt = torch.optim.AdamW(model.parameters(), lr=eta, betas=(0.9, 0.999), weight_decay=wd0)
    n = X.size(0); trace = []; every = max(1, steps // 12)
    bs = n if batch in (None, "full") else min(batch, n)
    for step in range(1, steps + 1):
        if lam_per_step is not None:
            for grp in opt.param_groups: grp["weight_decay"] = lam_per_step
        if bs >= n:
            xb = X.to(device)
        else:
            idx = torch.randint(0, n, (bs,), generator=g); xb = X[idx].to(device)
        loss = lm_loss(model, xb); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if track_norms and (step % every == 0 or step == 1):
            with torch.no_grad():
                trace.append(dict(step=step, loss=round(float(loss), 4), **norms(model)))
    with torch.no_grad():
        ev = X[:min(2000, n)].to(device); fl = float(lm_loss(model, ev))
    kl = input_dep_kl(model, X)
    out = (model, fl, kl)
    return out + (trace,) if track_norms else out


def classify(loss, kl):
    """Prior-robust: trap == input-independent (kl<0.3); mem == low loss; else partial."""
    if loss < 0.5: return "mem"
    if kl < 0.3:  return "trap"
    return "partial"


def eta_star(frac_by_eta):
    """log-interpolated 50% collapse crossing over an eta->frac_trap map."""
    es = sorted(frac_by_eta)
    for i in range(len(es) - 1):
        f0, f1 = frac_by_eta[es[i]], frac_by_eta[es[i+1]]
        if f0 < 0.5 <= f1:
            t = (0.5 - f0) / (f1 - f0 + 1e-9)
            return math.exp(math.log(es[i]) + t * (math.log(es[i+1]) - math.log(es[i])))
    return None
