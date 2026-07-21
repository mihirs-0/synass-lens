"""Reacquisition 2x2 (init x noise-channel) + full-jitter N replication.

Registered amendment 2026-07-21 (REACQ_2X2_PREREG.md). Fills the 2x2 the
capability-persistence thesis needs, all at eta=0.0125, full-gradient:

              clean(C0)         v-noise(V)        both-moments(N)
  collapsed   800 (have)        4600 (have)       THIS: 3 seeds, cap 16k
  fresh       THIS: 1, cap 10k  THIS: 1, cap 10k  (n/a)

Noise = per-step MATCHED isotropic xi, ||xi||=||(g_B1-g_B2)/sqrt2||, S=1.0.
  C0: m<-g,      v<-g^2
  V : m<-g,      v<-(g+xi)^2
  N : m<-g+xi,   v<-(g+xi)^2
INIT collapsed loads s0 (theta0,m0,v0,t0=22400); fresh = random init at
FRESH_MODEL_SEED (m=v=0, t0=0), same table+weights for its clean & v cells.

Endpoints (frozen): tau_onset = first step CE<3.0 sustained >=500; tau_solve =
first step 100% (>=0.999) exact retrieval confirmed next eval. PRE-COMMITTED
CENSORING: run to CAP; if no tau_onset by CAP -> censored=onset, report as a
bound (escape>CAP). NO ad-hoc extension. Resumable; fp32 optimizer state.

usage: reacq_2x2_run.py ARM INIT SEED CAP [device] [threads]
  ARM in {C0,V,N}   INIT in {collapsed,fresh}
"""
import json, sys, math
from pathlib import Path
sys.path.insert(0, "/Users/mihir/synass-lens/synass-lens")
import torch
from dataclasses import replace

ARM = sys.argv[1]
INIT = sys.argv[2]
SEED = int(sys.argv[3])
CAP = int(sys.argv[4])
DEVICE = sys.argv[5] if len(sys.argv) > 5 else "cpu"
torch.set_num_threads(int(sys.argv[6]) if len(sys.argv) > 6 else 4)
assert ARM in ("C0", "V", "N") and INIT in ("collapsed", "fresh")

from pinned_capabilities.config import ProtocolConfig
from pinned_capabilities.experiment import MBCExperiment
from pinned_capabilities.gate0e_escape import load_snapshot_fresh_stream
from src.data import collate_fn
from src.training.trainer import compute_loss

S_SCALE = 1.0
LR, WD, B1, B2, EPS = 0.0125, 0.01, 0.9, 0.999, 1e-8
CE_EVERY, ACC_EVERY, ACC_EVERY_NEAR, CKPT_EVERY, CHUNK = 200, 1000, 100, 1000, 2500
NEAR_CE = 2.5
ONSET_CE, ONSET_SUSTAIN, ONSET_TAIL = 3.0, 500, 200
FRESH_MODEL_SEED = 300           # fresh init: same weights+table for its clean & v cells
COLLAPSED_SEED = 100
REPO = Path("/Users/mihir/synass-lens/synass-lens")
COLLAPSED = REPO / "pinned_capabilities/results/gate0e_dev_curve_seed100/eta_0p0125/stream_00/checkpoints/slot_0.pt"
OUT = REPO / f"pinned_capabilities/results/reacq_2x2/{ARM}_{INIT}_seed{SEED}"


def to_dev(b):
    if torch.is_tensor(b): return b.to(DEVICE)
    if isinstance(b, dict): return {k: to_dev(v) for k, v in b.items()}
    if isinstance(b, (list, tuple)): return type(b)(to_dev(v) for v in b)
    return b


def build():
    cfg = ProtocolConfig()
    if INIT == "collapsed":
        exp = MBCExperiment(replace(cfg.experiment, seed=COLLAPSED_SEED, device="cpu"), cfg.metric)
        load_snapshot_fresh_stream(COLLAPSED, model=exp.model, optimizer=exp.optimizer, map_location="cpu")
        if DEVICE != "cpu": exp.model.to(DEVICE)
        params = [p for grp in exp.optimizer.param_groups for p in grp["params"]]
        St = exp.optimizer.state
        m = [St[p]["exp_avg"].detach().clone().to(DEVICE).float() for p in params]
        v = [St[p]["exp_avg_sq"].detach().clone().to(DEVICE).float() for p in params]
        step0 = St[params[0]]["step"]
        t0 = int(step0.item()) if torch.is_tensor(step0) else int(step0)
    else:  # fresh random init, empty optimizer state
        exp = MBCExperiment(replace(cfg.experiment, seed=FRESH_MODEL_SEED, device="cpu"), cfg.metric)
        if DEVICE != "cpu": exp.model.to(DEVICE)
        params = [p for grp in exp.optimizer.param_groups for p in grp["params"]]
        m = [torch.zeros_like(p).float() for p in params]
        v = [torch.zeros_like(p).float() for p in params]
        t0 = 0
    n = len(exp.dataset)
    chunks = [to_dev(collate_fn([exp.dataset[i] for i in range(s, min(s + CHUNK, n))])) for s in range(0, n, CHUNK)]
    return exp, params, m, v, t0, n, chunks


def full_grad(exp, params, chunks):
    for p in params: p.grad = None
    for b in chunks:
        loss, _, _ = compute_loss(exp.model, b)
        (loss / len(chunks)).backward()
    return [p.grad.detach().clone().float() for p in params]


def flat_norm(lst): return math.sqrt(sum(float(x.pow(2).sum()) for x in lst))


def ce_full(exp, chunks):
    exp.model.eval(); tot = 0.0; nex = 0
    with torch.no_grad():
        for b in chunks:
            logits = exp.model(b["input_ids"]); sl = logits[:, :-1]; st = b["labels"][:, 1:]
            fi = (st != -100).float().argmax(dim=1); ar = torch.arange(st.size(0), device=st.device)
            tot += float(torch.nn.functional.cross_entropy(sl[ar, fi], st[ar, fi], reduction="sum")); nex += st.size(0)
    exp.model.train(); return tot / nex


def retrieval_acc(exp, chunks):
    exp.model.eval(); correct = total = 0
    with torch.no_grad():
        for b in chunks:
            pred = exp.model(b["input_ids"])[:, :-1].argmax(-1); tgt = b["labels"][:, 1:]; mask = tgt != -100
            ok = ((pred == tgt) | ~mask).all(dim=1); correct += int(ok.sum()); total += int(ok.numel())
    exp.model.train(); return correct / total


def bias_corrected_update(m, v, t_eff):
    c1, c2 = 1 - B1 ** t_eff, 1 - B2 ** t_eff
    return [-LR * (m[i] / c1) / ((v[i] / c2).sqrt() + EPS) for i in range(len(m))]


def run():
    exp, params, m, v, t0, n, chunks = build()
    ngen = torch.Generator().manual_seed(1_000_000 + 7919 * (ord(ARM[0]) + len(INIT)) + SEED)
    mbgen = torch.Generator().manual_seed(2_000_000 + 7919 * (ord(ARM[0]) + len(INIT)) + SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    def mb_grad():
        idx = torch.randint(n, (128,), generator=mbgen)
        batch = to_dev(collate_fn([exp.dataset[int(i)] for i in idx]))
        for p in params: p.grad = None
        loss, _, _ = compute_loss(exp.model, batch); loss.backward()
        return [p.grad.detach().clone().float() for p in params]

    start = 1; ce_hist = []; tau_onset = None; tau_solve = None; solve_candidate = None
    ckpt = OUT / "ckpt.pt"
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu")
        for i, p in enumerate(params): p.data.copy_(st["theta"][i].to(DEVICE))
        m = [x.to(DEVICE) for x in st["m"]]; v = [x.to(DEVICE) for x in st["v"]]
        start = st["step"] + 1; ce_hist = st["ce_hist"]
        tau_onset = st["tau_onset"]; tau_solve = st["tau_solve"]; solve_candidate = st["solve_candidate"]
        ngen.set_state(st["ngen"]); mbgen.set_state(st["mbgen"])
        writer = (OUT / "metrics.jsonl").open("a")
    else:
        writer = (OUT / "metrics.jsonl").open("w")

    stop_at = CAP
    for step in range(start, CAP + 1):
        exp.model.train()
        g = full_grad(exp, params, chunks)
        if ARM == "C0":
            gpx = g                                                # clean; v<-g^2
        else:
            ref = [(a - b) / math.sqrt(2) for a, b in zip(mb_grad(), mb_grad())]
            iso = [torch.randn(p.shape, generator=ngen).to(DEVICE) for p in params]
            scale = S_SCALE * flat_norm(ref) / (flat_norm(iso) + 1e-30)
            gpx = [a + scale * e for a, e in zip(g, iso)]          # g + xi
        t_eff = t0 + step
        for i in range(len(m)):
            m[i].mul_(B1).add_(gpx[i] if ARM == "N" else g[i], alpha=1 - B1)   # N: noisy m; C0/V: clean m
            v[i].mul_(B2).add_((g[i] * g[i]) if ARM == "C0" else (gpx[i] * gpx[i]), alpha=1 - B2)  # C0: clean v; V/N: noisy v
        u = bias_corrected_update(m, v, t_eff)
        for i, p in enumerate(params):
            p.data.mul_(1 - LR * WD).add_(u[i])

        row = {"step": step, "arm": ARM, "init": INIT, "seed": SEED}
        near = ce_hist and ce_hist[-1][1] < NEAR_CE
        acc_now = ACC_EVERY_NEAR if near else ACC_EVERY
        if step % CE_EVERY == 0 or step % acc_now == 0:
            ce = ce_full(exp, chunks); row["full_vocab_ce"] = ce; ce_hist.append((step, ce))
            if tau_onset is None:
                for j, (sj, cj) in enumerate(ce_hist):
                    if cj < ONSET_CE and all(ck < ONSET_CE for sk, ck in ce_hist[j:] if sk <= sj + ONSET_SUSTAIN) \
                       and ce_hist[-1][0] >= sj + ONSET_SUSTAIN:
                        tau_onset = sj; break
                if tau_onset is not None:
                    stop_at = min(CAP, step + ONSET_TAIL)
        if step % acc_now == 0:
            acc = retrieval_acc(exp, chunks); row["exact_acc"] = acc
            if acc >= 0.999:
                if solve_candidate is None: solve_candidate = step
                elif tau_solve is None: tau_solve = solve_candidate; stop_at = min(CAP, tau_solve + 200)
            else: solve_candidate = None
        if len(row) > 4:
            row["tau_onset"] = tau_onset; row["tau_solve"] = tau_solve
            writer.write(json.dumps(row) + "\n"); writer.flush()

        if step % CKPT_EVERY == 0 or step == stop_at:
            torch.save({"step": step, "theta": [p.data.detach().cpu() for p in params],
                        "m": [x.detach().cpu() for x in m], "v": [x.detach().cpu() for x in v],
                        "ce_hist": ce_hist, "tau_onset": tau_onset, "tau_solve": tau_solve,
                        "solve_candidate": solve_candidate, "ngen": ngen.get_state(), "mbgen": mbgen.get_state()}, ckpt)
        if step >= stop_at: break

    writer.close()
    summ = {"arm": ARM, "init": INIT, "seed": SEED, "cap": CAP, "last_step": step,
            "tau_onset": tau_onset, "tau_solve": tau_solve,
            "censored_onset": tau_onset is None, "censored_solve": tau_solve is None,
            "final_ce": ce_hist[-1][1] if ce_hist else None, "device": DEVICE}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    run()
