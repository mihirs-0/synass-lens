"""Matched-noise experiment: is minibatch noise the causal maintainer of the
collapsed plateau? Five arms from ONE identical collapsed state (weights +
AdamW buffers), same lr/wd/step-count. Pre-registered in
pinned_capabilities/MATCHED_NOISE_PREREG.md.

  A1 full        : g = g_full                                  (deterministic; escapes ~2250)
  A2 minibatch   : g = g_B (sequential)                        (reference; traps)
  A3 full+emp    : g = g_full + (g_B1 - g_B2)/sqrt(2)          (SURGICAL: full signal + empirical-covariance noise)
  A4 full+iso    : g = g_full + eps, ||eps||=||noise_A3||      (norm-only control)
  A5 reset+emp   : A3 but AdamW m,v,step reset to 0 at fork    (optimizer-memory control)

All arms: same T steps, same update count, full-data gradient where applicable.
Escape criterion (registered, matches earlier escape-time analysis): first
step with full_vocab_ce < 3.0 sustained for the next 500 logged steps; else
right-censored at T. Reported by optimizer steps AND examples processed.
"""
import json, sys, math
from pathlib import Path
sys.path.insert(0, "/Users/mihir/synass-lens/synass-lens")
import torch; torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
from dataclasses import replace
from pinned_capabilities.config import ProtocolConfig
from pinned_capabilities.experiment import MBCExperiment
from pinned_capabilities.gate0e_escape import load_snapshot_fresh_stream
from pinned_capabilities.parameter_groups import set_learning_rates
from src.data import collate_fn
from src.training.trainer import compute_loss

ARM = sys.argv[1]              # A1|A2|A3|A4|A5
STREAM = int(sys.argv[2])
T = 3500
EVAL_EVERY = 50
LR = 0.0125
CHUNK = 2500                   # full grad = 4 chunks of 2500 = all 10000
MB = 128
REPO = Path("/Users/mihir/synass-lens/synass-lens")
COLLAPSED = REPO / "pinned_capabilities/results/gate0e_dev_curve_seed100/eta_0p0125/stream_00/checkpoints/slot_0.pt"
OUT = REPO / f"pinned_capabilities/results/gate0e_matched_noise/{ARM}_stream{STREAM:02d}"
OUT.mkdir(parents=True, exist_ok=True)

cfg = ProtocolConfig()
exp = MBCExperiment(replace(cfg.experiment, seed=100, device="cpu"), cfg.metric)
load_snapshot_fresh_stream(COLLAPSED, model=exp.model, optimizer=exp.optimizer, map_location="cpu")
set_learning_rates(exp.optimizer, LR)
if ARM == "A5":                # optimizer-memory control: wipe Adam buffers
    for st in exp.optimizer.state.values():
        if "exp_avg" in st: st["exp_avg"].zero_()
        if "exp_avg_sq" in st: st["exp_avg_sq"].zero_()
        if "step" in st: st["step"] = torch.zeros_like(st["step"]) if torch.is_tensor(st["step"]) else 0

params = [p for grp in exp.optimizer.param_groups for p in grp["params"]]
n = len(exp.dataset)
full_chunks = [collate_fn([exp.dataset[i] for i in range(s, s+CHUNK)]) for s in range(0, n, CHUNK)]
gen = torch.Generator().manual_seed(90_000 + 1000*ord(ARM[1]) + STREAM)

def mb_batch():
    idx = torch.randint(n, (MB,), generator=gen)
    return collate_fn([exp.dataset[int(i)] for i in idx])

def grad_of(batches, weight):
    """Accumulate mean gradient over a list of batches; return cloned grads."""
    exp.optimizer.zero_grad(set_to_none=True)
    for b in batches:
        loss, _, _ = compute_loss(exp.model, b)
        (loss * weight).backward()
    return [(p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for p in params]

def full_grad():
    return grad_of(full_chunks, 1.0/len(full_chunks))

def set_and_step(g):
    for p, gi in zip(params, g):
        p.grad = gi
    exp.optimizer.step()

rows = []
writer = (OUT / "metrics.jsonl").open("w")
for step in range(1, T+1):
    exp.model.train()
    if ARM == "A1":
        g = full_grad()
    elif ARM == "A2":
        g = grad_of([mb_batch()], 1.0)
    elif ARM in ("A3", "A5"):
        gf = full_grad()
        g1 = grad_of([mb_batch()], 1.0); g2 = grad_of([mb_batch()], 1.0)
        g = [f + (a - b)/math.sqrt(2) for f, a, b in zip(gf, g1, g2)]
    elif ARM == "A4":
        gf = full_grad()
        g1 = grad_of([mb_batch()], 1.0); g2 = grad_of([mb_batch()], 1.0)
        noise = [(a - b)/math.sqrt(2) for a, b in zip(g1, g2)]
        nn = math.sqrt(sum(float(x.pow(2).sum()) for x in noise))
        iso = [torch.randn(x.shape, generator=gen) for x in noise]
        ni = math.sqrt(sum(float(x.pow(2).sum()) for x in iso))
        g = [f + e*(nn/ni) for f, e in zip(gf, iso)]
    else:
        raise ValueError(ARM)
    set_and_step(g)
    exp.step += 1
    if step % EVAL_EVERY == 0:
        ev = exp.evaluate()
        row = {"step": step, "examples": step*(n if ARM in ("A1","A3","A4","A5") else MB),
               "full_vocab_ce": ev["full_vocab_ce"], "c_int": ev["c_int"], "exact_match": ev["exact_match"]}
        rows.append(row); writer.write(json.dumps(row)+"\n"); writer.flush()
writer.close()
def escape(rows):
    for i, r in enumerate(rows):
        if r["full_vocab_ce"] < 3.0:
            fut = [x for x in rows[i:] if x["step"] <= r["step"]+500]
            if fut and all(x["full_vocab_ce"] < 3.0 for x in fut):
                return r["step"]
    return None
esc = escape(rows)
summ = {"arm": ARM, "stream": STREAM, "T": T, "escape_step": esc,
        "escape_examples": (esc*n if esc and ARM!="A2" else (esc*MB if esc else None)),
        "censored": esc is None, "final_ce": rows[-1]["full_vocab_ce"], "final_c_int": rows[-1]["c_int"]}
(OUT / "summary.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ))
