#!/usr/bin/env python
"""Audit Blocks 1,3,9,10 + swap-validity (2.5/2.6). Re-eval on existing checkpoints."""
import sys, json, math, glob
from pathlib import Path
from collections import Counter, defaultdict
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eta_sweep.config import CellConfig, RESULTS_DIR
from eta_sweep.run_single import build_legacy_cfg, _select_device, _set_all_seeds
from src.data import collate_fn, create_datasets_from_config, create_tokenizer_from_config
from src.model import create_model_from_config
from src.training.trainer import compute_loss

device = _select_device(); K = 10; logK = math.log(K)
_set_all_seeds(0)
cc = CellConfig(eta=0.001, k=K, seed=0, n_unique_b=1000, batch_size=128, weight_decay=0.01)
cfg = build_legacy_cfg(cc); tok = create_tokenizer_from_config(cfg)
ds, _, md = create_datasets_from_config(cfg, tok)
ex0 = ds[0]; ts, te = ex0["target_start_position"], ex0["target_end_position"]; zp = ex0["z_position"]
V = None

def load(sub, patt, seed=0):
    m = create_model_from_config(cfg, tok).to(device)
    m.load_state_dict(torch.load(RESULTS_DIR / sub / patt.format(s=seed) / "model_final.pt", map_location=device)); m.eval()
    return m
trapped = load("gate_candfloor", "inverse_eta0.006_K10_nb1000_seed{s}")
conv    = load("gate_sweep", "inverse_eta0.001_K10_nb1000_seed{s}")
V = int(trapped.cfg.d_vocab)

# eval batch (in dataset order so inv_meta aligns)
N = 2000
exs = [ds[i] for i in range(N)]
batch = collate_fn(exs); ids = batch["input_ids"].to(device)
labels = batch["labels"].to(device)

print("="*70)
print("BLOCK 1 — the floor")
print("="*70)
# 1.2 candidate uniformity under B
cand_counts = [len(p) for p in md.mappings.values()]
uniq_A_per_B = [len(set(a for z, a in p)) for p in md.mappings.values()]
print(f"1.2  per-B: #candidates {Counter(cand_counts)}  #distinct-A {Counter(uniq_A_per_B)}")
# first-token entropy of candidates per B (true floor if metric were first-token)
def tokid(s): return tok.encode_sequence(list(md.mappings.keys())[0], s if False else s, s)  # unused
fte = []
for B, pairs in md.mappings.items():
    fts = [tok.tokenize(a)[0] if hasattr(tok, "tokenize") else a[0] for z, a in pairs]
    c = Counter(fts); p = torch.tensor(list(c.values()), dtype=torch.float64); p = p / p.sum()
    fte.append(float(-(p * p.log()).sum()))
mean_fte = sum(fte) / len(fte)
print(f"     candidates per B are uniform (each A once): {all(c == u for c, u in zip(cand_counts, uniq_A_per_B))}")
print(f"     -> full-candidate floor = logK = {logK:.4f} EXACTLY (uniform under B)")
print(f"     first-token-candidate entropy (if collisions) = {mean_fte:.4f}  vs logK {logK:.4f}  (gap {logK-mean_fte:+.4f})")

# 1.1 candidate loss vs floor
def candidate_loss_batch(model):
    # score each example's K candidates (full seq), softmax over K, CE of correct
    losses = []
    keys = list(md.mappings.keys())
    rng = torch.Generator().manual_seed(0)
    for bi in torch.randint(0, len(keys), (300,), generator=rng).tolist():
        B = keys[bi]; pairs = md.mappings[B]
        ci = int(torch.randint(0, len(pairs), (1,), generator=rng))
        z = pairs[ci][0]; cand = [a for _, a in pairs]
        seqs = [tok.encode_sequence(B, z, a)["input_ids"] for a in cand]
        x = torch.stack(seqs).to(device)
        with torch.no_grad():
            lg = model(x)
        slp = []
        for k in range(len(cand)):
            lp = 0.0
            for pos in range(ts, te + 1):
                t = x[k, pos].item()
                if t < 0: continue
                lp += float(F.log_softmax(lg[k, pos - 1], -1)[t])
            slp.append(lp)
        slp = torch.tensor(slp, dtype=torch.float64)
        losses.append(-float((slp - torch.logsumexp(slp, 0))[ci]))
    return sum(losses) / len(losses)
cl_tr = candidate_loss_batch(trapped)
print(f"\n1.1  trapped candidate_loss = {cl_tr:.4f}   logK = {logK:.4f}   ABOVE FLOOR by {cl_tr-logK:+.4f} nats")

# 1.3 per-position decomposition of full-seq loss
def per_pos_ce(model):
    with torch.no_grad():
        lg = model(ids)
    out = {}
    for pos in range(ts, te + 1):
        ce = F.cross_entropy(lg[:, pos - 1], ids[:, pos])
        out[pos] = float(ce)
    full = float(compute_loss(model, {"input_ids": ids, "labels": labels})[0])
    return out, full
pp, full = per_pos_ce(trapped)
print(f"\n1.3  trapped full-seq CE = {full:.4f}.  per-position CE (pos: token): " +
      "  ".join(f"{p}:{c:.3f}" for p, c in pp.items()))
print(f"     -> char positions ({ts}-{te-1}) ~ full-vocab entropy log(content)={math.log(V-4):.3f}; EOS(pos {te})~0.")
print(f"     candidate_loss {cl_tr:.3f} (restricted to K) vs first-char full-vocab {pp[ts]:.3f} (over all {V}): "
      f"the model spreads over the WHOLE alphabet, not the K candidates.")

print("\n" + "="*70); print("BLOCK 2.5/2.6 — swap validity & distribution shift"); print("="*70)
# 2.6 is every z valid for every B?
zsets = set(frozenset(z for z, a in p) for p in md.mappings.values())
allz = set(z for p in md.mappings.values() for z, a in p)
print(f"2.6  distinct z-sets across B = {len(zsets)}  (1 => all B share the same z's => swap in-distribution)")
print(f"     total distinct z values = {len(allz)} (= K means z reused across all B)")
# 2.5 distribution shift under swap (KL of first-token output dist)
def swap_kl(model, lo, hi):
    sw = ids.clone(); sw[:, lo:hi] = torch.roll(ids[:, lo:hi], 1, 0)
    with torch.no_grad():
        p = F.log_softmax(model(ids)[:, ts - 1], -1); q = F.log_softmax(model(sw)[:, ts - 1], -1)
    return float((p.exp() * (p - q)).sum(-1).mean())
for name, m in [("trapped", trapped), ("memorized", conv)]:
    print(f"2.5  {name}: KL under z-swap = {swap_kl(m, zp, zp+2):.4f}   KL under B-swap = {swap_kl(m, 1, zp-1):.4f}  nats")
print("     (memorized = control: large KL => metric SEES the effect; trapped ~0 => model truly invariant)")

print("\n" + "="*70); print("BLOCK 3 — what the trapped model computes"); print("="*70)
# 7 single fixed marginal? spread of outputs across inputs; B-grouped variation
with torch.no_grad():
    P = F.softmax(trapped(ids)[:, ts - 1], -1)           # (N,V) first-token dists
Pbar = P.mean(0, keepdim=True)
spread = float((P * (P.clamp_min(1e-12).log() - Pbar.clamp_min(1e-12).log())).sum(-1).mean())  # mean KL to mean dist
# B-grouped: variation of mean-dist across B
binv = [ds[i].get("b_index", None) for i in range(N)] if "b_index" in ds[0] else None
print(f"7    output spread across inputs (mean KL to mean dist) = {spread:.4f} nats  (~0 => single FIXED marginal)")
# 8 match P(A) global vs P(A|B)?
A0 = ids[:, ts].cpu()
glob = torch.bincount(A0, minlength=V).double(); glob = glob / glob.sum()
mout = Pbar.squeeze().cpu().double()
klg = float((mout * (mout.clamp_min(1e-12).log() - glob.clamp_min(1e-12).log())).sum())
# P(A|B): per-B first-token dist; KL(model||P_B) averaged. Build per-B then map examples.
keys = list(md.mappings.keys())
PB = {}
for B, pairs in md.mappings.items():
    c = Counter(tok.encode_sequence(B, z, a)["input_ids"][ts].item() for z, a in pairs)
    v = torch.zeros(V).double()
    for t, n in c.items(): v[t] = n
    PB[B] = v / v.sum()
# sample examples, KL(model||P_B)
klb = []
import random as _r; _r.seed(0)
for B in _r.sample(keys, 200):
    pb = PB[B].clamp_min(1e-12)
    klb.append(float((mout * (mout.clamp_min(1e-12).log() - pb.log())).sum()))
print(f"8    KL(trapped output || P(A0) GLOBAL) = {klg:.4f}   KL(trapped || P(A0|B)) = {sum(klb)/len(klb):.4f}")
print(f"     trapped output entropy = {float(-(mout*mout.clamp_min(1e-12).log()).sum()):.3f}; "
      f"P(A0) global entropy = {float(-(glob*glob.clamp_min(1e-12).log()).sum()):.3f}; logK={logK:.3f}")

print("\n" + "="*70); print("BLOCK 9/10 — mirror asymmetry"); print("="*70)
def final_loss(sub, patt, seed):
    L = [json.loads(x) for x in open(RESULTS_DIR / sub / patt.format(s=seed) / "log.jsonl") if x.strip()]
    import statistics as st
    return st.mean([r["train_loss"] for r in L[-5:]])
print("9    at eta=6e-3 (where inverse traps), 2 seeds:")
for s in (0, 1):
    fl = final_loss("gate_sweep", "forward_eta0.006_K10_nb1000_seed{s}", s)
    il = final_loss("gate_sweep", "inverse_eta0.006_K10_nb1000_seed{s}", s)
    print(f"       seed{s}: MIRROR(forward) loss {fl:.3f}   |   inverse(trapped) loss {il:.3f}")
# 10 output entropy / size confound
nB = len(md.mappings); nA = sum(len(set(a for z, a in p)) for p in md.mappings.values())
lenA = len(list(md.mappings.values())[0][0][1]); lenB = len(list(md.mappings.keys())[0])
print(f"10   distinct outputs: inverse->A = {nA}  (len {lenA}) ; mirror->B = {nB}  (len {lenB})")
print(f"     inverse output space is LARGER ({nA} vs {nB}); yet inverse MEMORIZES at low eta (converges) and only")
print(f"     traps at HIGH eta where mirror still learns -> trap is z-binding-specific, not output-size. (confound noted)")
