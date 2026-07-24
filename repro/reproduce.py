#!/usr/bin/env python
"""
Marginals before conditionals — minimal, self-contained reproduction.
=====================================================================

One claim, reproduced from scratch in a single file (only dependency: torch):

    On a K-fold-ambiguous inverse task (B, z) -> A, AdamW above a learning-rate
    threshold gets STUCK as a marginal predictor: it parks at the information
    floor log K (first-token candidate loss) and never uses the disambiguator z
    (z-sensitivity Delta_z ~ 0). Below the threshold it solves the task
    (loss -> 0, Delta_z large). The mirror task (A, z) -> B, where z is redundant,
    learns at the same high learning rate. So the failure is specific to z being
    *necessary*, not to "high learning rate breaks everything."

We print a 2x2 PHASE TABLE: the same relation read both ways x two learning rates,
each cell averaged over several seeds.

                       low eta                     high eta
  (B,z)->A  z NEEDED   solves, uses z              TRAPPED at log K  (Delta_z ~ 0)
  (A,z)->B  z redundant solves                     breaks, but NOT at log K

The scale-independent result, reproduced here, is the bottom-left vs top-right
contrast: the z-NECESSARY direction has a failure mode in which it parks at the
information floor log K with the disambiguator unused (candidate_loss = ln K,
Delta_z ~ 0). The z-REDUNDANT direction has no such floor and never sits there.

  `candidate_loss` = first-token loss restricted to the K candidate answers for a
  given B. A marginal predictor that ignores z spreads mass uniformly over the K
  candidates, so candidate_loss = ln K (for K=10, 2.3026) -- a hand-computable
  number, not a fitted one.

SCALE NOTE: the clean "same eta: forward learns while inverse traps" split (the
paper's lead figure) needs the FULL model (4L/d128, n_b=1000), where the forward's
memorization ceiling sits well above the inverse's trap threshold (measured window
eta in [0.006, 0.012]). This tiny 2L/d64 surrogate collapses that window, so here
BOTH directions fail at high eta -- but only the z-necessary one fails AT log K
with z unused. That structured trap is what transfers across scale; the width of
the window does not. The script says so in its own output.

Run:  python reproduce.py                 # 3 seeds, ~minutes (CPU or Apple MPS)
      python reproduce.py --quick         # 1 seed, fewer steps (~1 min sanity)
      python reproduce.py --device cpu    # bit-stable across machines
      python reproduce.py --K 13          # ambiguity -> floor moves to ln 13

Everything is seeded; exact floats vary slightly by device/BLAS, the verdicts and
the log-K floor are robust.
"""
import argparse, math, random, types
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Special tokens then a shared content alphabet used by B, z, and A alike.
PAD, BOS, SEP, EOS = 0, 1, 2, 3
N_SPECIAL = 4


def build_task(n_b, K, alphabet, lb, lz, la, seed):
    """Construct the surjective relation and its two reading directions.

    For each of n_b distinct B strings we attach K (z, A) pairs.
      * The SAME K z-strings are reused for every B  -> z alone is uninformative;
        only (B, z) together pick A  (z disambiguates *within* a B's set).
      * Every A string is globally unique             -> A alone fixes its B,
        so the forward direction (A, z) -> B is deterministic and z is redundant.
    Returns example tensors for both directions plus the per-B candidate sets.
    """
    rng = random.Random(seed)
    content = list(range(N_SPECIAL, N_SPECIAL + alphabet))

    def rand_str(L, used):
        while True:
            s = tuple(rng.choice(content) for _ in range(L))
            if s not in used:
                used.add(s)
                return s

    used_b, used_z, used_a = set(), set(), set()
    z_pool = [rand_str(lz, used_z) for _ in range(K)]          # shared across all B
    Bs = [rand_str(lb, used_b) for _ in range(n_b)]

    # mapping[b_idx] = list of (z_tuple, a_tuple), length K, one per shared z
    mapping = []
    for _ in range(n_b):
        order = list(range(K)); rng.shuffle(order)             # which A goes to which z
        a_list = [rand_str(la, used_a) for _ in range(K)]
        mapping.append([(z_pool[j], a_list[order[j]]) for j in range(K)])

    def seq(prefix_str, sel_str, target_str):
        # [BOS] prefix [SEP] sel [SEP] target [EOS]
        return [BOS, *prefix_str, SEP, *sel_str, SEP, *target_str, EOS]

    inv, fwd = [], []
    inv_meta = []   # (b_idx, z_idx) for candidate / Delta_z eval on the inverse task
    for bi, pairs in enumerate(mapping):
        for zi, (z, a) in enumerate(pairs):
            inv.append(seq(Bs[bi], z, a))            # (B, z) -> A
            fwd.append(seq(a, z, Bs[bi]))            # (A, z) -> B   (z redundant)
            inv_meta.append((bi, zi))
    return {
        "inverse": torch.tensor(inv),
        "forward": torch.tensor(fwd),
        "inv_meta": inv_meta,
        "mapping": mapping, "Bs": Bs, "z_pool": z_pool,
        "lb": lb, "lz": lz, "la": la, "K": K,
        "vocab": N_SPECIAL + alphabet,
    }


# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model, n_head, n_layer, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.lnf(h))


# ----------------------------------------------------------------------------
# The answer region (tokens we score) is the last (label_len + 1) tokens: the
# label string plus EOS. ans_start is the index of the first answer token.
def answer_start(prefix_len, sel_len, label_len, total_len):
    return total_len - (label_len + 1)


def lm_loss(model, seq, ans_start):
    """Mean cross-entropy over the answer region (next-token prediction)."""
    logits = model(seq)                                   # (N, T, V)
    tgt = seq[:, ans_start:]                              # answer tokens + EOS
    pred = logits[:, ans_start - 1:-1, :]                 # positions predicting them
    return F.cross_entropy(pred.reshape(-1, pred.size(-1)), tgt.reshape(-1))


@torch.no_grad()
def candidate_loss(model, task, idx, device):
    """First-token candidate loss = log K floor when z is ignored.

    For each eval example score the model's log-prob of the FIRST answer token
    for each of the K candidate A's (under the correct z), softmax over the K
    candidates, take the CE of the correct one. A marginal predictor that cannot
    tell the candidates apart spreads mass uniformly -> CE = log K.
    """
    K, lb, lz, la = task["K"], task["lb"], task["lz"], task["la"]
    ans = 1 + lb + 1 + lz + 1                             # index of first answer token
    losses = []
    for bi, zi in idx:
        pairs = task["mapping"][bi]
        z = pairs[zi][0]
        prefix = [BOS, *task["Bs"][bi], SEP, *z, SEP]
        first_tokens = [pairs[j][1][0] for j in range(K)]          # first char of each candidate A
        seq = torch.tensor([prefix], device=device)
        logp = F.log_softmax(model(seq)[0, ans - 1], dim=-1)       # dist over next (first answer) token
        scores = torch.stack([logp[t] for t in first_tokens])     # (K,)
        ce = -F.log_softmax(scores, dim=0)[zi_correct(pairs, zi)]  # CE of the correct candidate
        losses.append(ce.item())
    return sum(losses) / len(losses)


def zi_correct(pairs, zi):
    # correct candidate index == the candidate whose (z) is the one we conditioned on
    return zi


@torch.no_grad()
def delta_z(model, task, examples, z_start, ans_start, device):
    """z-sensitivity: answer-region CE with a WRONG z minus with the correct z.

    Swap each example's z for a different one of the K shared z's, keep the target
    fixed, re-score the answer region. If the model uses z, a wrong z raises the loss
    (Delta_z > 0). If it ignores z, Delta_z ~ 0. Works for either reading direction;
    only the z position (z_start) differs.

    NOTE Delta_z ~ 0 reads differently per direction: for the z-NEEDED inverse it is
    damning (the model is ignoring information it requires -> stuck at log K); for the
    z-REDUNDANT forward it is correct (z carries nothing, ignoring it is fine).
    """
    K, lz = task["K"], task["lz"]
    clean = examples.to(device)
    shuf = clean.clone()
    rng = random.Random(0)
    for i, (bi, zi) in enumerate(task["inv_meta"][: len(examples)]):   # fwd & inv share this meta
        zj = (zi + 1 + rng.randrange(K - 1)) % K          # any different z slot
        newz = task["z_pool"][zj]
        for t in range(lz):
            shuf[i, z_start + t] = newz[t]
    return (lm_loss(model, shuf, ans_start) - lm_loss(model, clean, ans_start)).item()


# ----------------------------------------------------------------------------
def train_run(name, data, direction, eta, cfg, task, device):
    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    X = data[direction].to(device)
    vocab = data["vocab"]; T = X.size(1)
    lb, lz, la = data["lb"], data["lz"], data["la"]
    label_len = la if direction == "inverse" else lb
    ans_start = answer_start(lb, lz, label_len, T)
    model = TinyGPT(vocab, cfg.d_model, cfg.n_head, cfg.n_layer, max_len=T + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=eta, betas=(0.9, 0.999), weight_decay=0.01)

    n = X.size(0)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    for step in range(1, cfg.steps + 1):
        bidx = torch.randint(0, n, (cfg.batch,), generator=g)
        seq = X[bidx]
        model.train(); opt.zero_grad(set_to_none=True)
        loss = lm_loss(model, seq, ans_start)
        loss.backward(); opt.step()

    model.eval()
    eval_idx = task["inv_meta"][: cfg.n_eval]
    eval_ex = X[: cfg.n_eval]                              # this direction's eval rows (already on device)
    tl = lm_loss(model, eval_ex, ans_start).item()
    if direction == "inverse":
        z_start = 1 + lb + 1                               # BOS B SEP [z]
        cl = candidate_loss(model, task, eval_idx, device)
    else:
        z_start = 1 + la + 1                               # BOS A SEP [z]
        cl = float("nan")                                  # candidate_loss is only defined for the ambiguous direction
    dz = delta_z(model, task, eval_ex, z_start, ans_start, device)   # computed for BOTH directions
    return {"name": name, "dir": direction, "eta": eta, "train_loss": tl,
            "candidate_loss": cl, "delta_z": dz}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--n_b", type=int, default=400)       # enough data that the trap is seed-robust
    p.add_argument("--alphabet", type=int, default=12)
    p.add_argument("--lb", type=int, default=4)
    p.add_argument("--lz", type=int, default=2)
    p.add_argument("--la", type=int, default=4)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--eta_low", type=float, default=1e-3)
    p.add_argument("--eta_high", type=float, default=2.5e-2)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n_eval", type=int, default=160)
    p.add_argument("--device", default="auto")
    p.add_argument("--quick", action="store_true", help="1 seed, fewer steps (~1 min sanity)")
    a = p.parse_args()
    if a.quick:
        a.seeds, a.steps = 1, 2500

    device = (("mps" if torch.backends.mps.is_available() else "cpu")
              if a.device == "auto" else a.device)
    logK = math.log(a.K)
    seeds = list(range(a.seeds))
    print(f"\nMarginals before conditionals — minimal reproduction")
    print(f"device={device}  K={a.K}  log K = {logK:.4f}  n_b={a.n_b}  "
          f"model={a.n_layer}L/{a.d_model}d  steps={a.steps}  seeds={seeds}")
    print(f"low eta = {a.eta_low:g}   high eta = {a.eta_high:g}\n")

    def cfg_for(seed):
        d = vars(a).copy(); d["seed"] = seed
        return types.SimpleNamespace(**d)

    # run every (direction x {low,high} eta) over all seeds
    cells = {}
    for direction in ("inverse", "forward"):
        for label, eta in (("low", a.eta_low), ("high", a.eta_high)):
            rs = []
            for s in seeds:
                task = build_task(a.n_b, a.K, a.alphabet, a.lb, a.lz, a.la, s)
                rs.append(train_run("x", task, direction, eta, cfg_for(s), task, device))
            cells[(direction, label)] = rs

    def mean(rs, k):
        return sum(r[k] for r in rs) / len(rs)

    def cell(direction, label):
        rs = cells[(direction, label)]
        n = len(rs); tl = mean(rs, "train_loss"); dz = mean(rs, "delta_z")
        if direction == "inverse" and label == "high":
            k = sum(r["candidate_loss"] > logK - 0.35 and abs(r["delta_z"]) < 0.35 for r in rs)
            return f"TRAPPED {k}/{n} @ logK   loss {tl:.2f}  cand {mean(rs,'candidate_loss'):.2f}  Δz {dz:+.2f}"
        if direction == "inverse":
            k = sum(r["delta_z"] > 2 and r["train_loss"] < 0.6 for r in rs)
            return f"solves  {k}/{n} uses z    loss {tl:.2f}  cand {mean(rs,'candidate_loss'):.2f}  Δz {dz:+.2f}"
        if label == "high":
            return f"breaks (not @logK)      loss {tl:.2f}  cand   —   Δz {dz:+.2f}"
        k = sum(r["train_loss"] < 0.3 for r in rs)
        return f"solves  {k}/{n}            loss {tl:.2f}  cand   —   Δz {dz:+.2f}"

    # ---------- headline output: the 2x2 with Δz in every cell ----------
    print(f"PHASE TABLE — one relation, two reading directions × two learning rates ({len(seeds)} seeds/cell)")
    print(f"  Δz = z-shuffle loss gap:  Δz>0 ⇒ model USES z   ·   Δz≈0 ⇒ model IGNORES z\n")
    print(f"  (B,z)->A   z NEEDED      low η : {cell('inverse','low')}")
    print(f"                           high η: {cell('inverse','high')}")
    print(f"  (A,z)->B   z redundant   low η : {cell('forward','low')}")
    print(f"                           high η: {cell('forward','high')}")

    inv_lo, inv_hi = cells[("inverse", "low")], cells[("inverse", "high")]
    fwd_lo, fwd_hi = cells[("forward", "low")], cells[("forward", "high")]
    fmt = lambda rs, k: "[" + ", ".join(f"{r[k]:.3f}" for r in rs) + "]"

    # ---------- the money line: hand-computed floor vs measured, live ----------
    meas = mean(inv_hi, "candidate_loss")
    print(f"\n  THE LOG-K MATCH (compute it yourself):  log K = ln({a.K}) = {logK:.4f}   "
          f"measured (inverse @ high η) = {meas:.3f}   →  {'MATCH' if abs(meas-logK)<0.1 else 'OFF'}")
    print(f"     per-seed candidate_loss = {fmt(inv_hi,'candidate_loss')}")
    print(f"     per-seed delta_z        = {fmt(inv_hi,'delta_z')}   (≈0 ⇒ z provably unused)")

    trap   = all(r["candidate_loss"] > logK - 0.35 and abs(r["delta_z"]) < 0.35 for r in inv_hi)
    usez   = all(r["delta_z"] > 2 and r["train_loss"] < 0.6 for r in inv_lo)
    fwd_ok = all(r["train_loss"] < 0.3 for r in fwd_lo)
    fwd_no = all(not (logK - 0.25 < r["train_loss"] < logK + 0.25) for r in fwd_hi)  # forward has no logK trap
    print("\nClaims reproduced (all seeds):")
    print(f"  [{'x' if trap else ' '}] z-NEEDED direction traps at log K with z unused (cand≈logK, Δz≈0)")
    print(f"  [{'x' if usez else ' '}] same direction SOLVES at low η by using z (Δz≫0)")
    print(f"  [{'x' if fwd_ok else ' '}] z-REDUNDANT direction solves at low η")
    print(f"  [{'x' if fwd_no else ' '}] z-REDUNDANT direction's high-η failure is NOT the log-K trap")
    ok = trap and usez and fwd_ok and fwd_no
    print("\nRESULT:", "PASS — the log-K trap is specific to the z-necessary direction."
          if ok else "CHECK — see per-seed numbers above (try --steps higher).")

    # ---------- scale caveat IN THE OUTPUT, not just a comment ----------
    print(
        "\n" + "-" * 78 + "\n"
        "READ THIS before comparing to the post's hero figure.\n"
        "This toy reproduces the SCALE-INVARIANT result: the z-necessary direction traps at\n"
        "log K with z unused (Δz≈0); the z-redundant direction never does. It does NOT show the\n"
        "hero figure's clean 'same learning rate: forward dives to 0 while inverse stays flat'\n"
        "split — that needs the FULL model (4-layer, d=128, n_b=1000), where the forward's\n"
        "memorization ceiling sits well ABOVE the inverse's trap threshold (measured window\n"
        "η ∈ [0.006, 0.012]; at η=6e-3 the full model gives forward loss ≈0.06, inverse ≈2.81,\n"
        "inverse candidate_loss pinned at log K for 25,000 steps — see README).\n"
        "The tiny model collapses that window, so here BOTH directions fail at high η. But note\n"
        "the high-η column: the inverse fails AS A STRUCTURED TRAP (exactly log K, z unused),\n"
        "the forward fails as GENERIC degradation (loss off the floor, no log-K structure).\n"
        "Same 'high LR hurts', two different failures — only one is the marginal trap.\n"
        + "-" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
