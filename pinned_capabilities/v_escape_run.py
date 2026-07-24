"""Weak-noise V-arm escape runs (registered amendment 2026-07-20).

Measures tau_onset / tau_solve vs injected-noise scale s, to build an
escape-time-vs-noise trend and extrapolate the s=1.0 (full-strength) escape.

Identical to the registered V arm EXCEPT the injected-noise std is x s:
  m <- g (clean full gradient) ;  v <- (g + s*xi)^2
  xi = per-step MATCHED isotropic noise, ||xi|| = ||(g_B1 - g_B2)/sqrt(2)||
  theta <- theta*(1 - lr*wd) + u  (bias-corrected AdamW, t_eff = t0 + step)

Endpoints:
  tau_onset = first step with full_vocab_ce < 3.0 sustained >= 500 steps
  tau_solve = first step with 100% exact retrieval on all 10,000 keys,
              confirmed at the next accuracy eval
Early stop at tau_solve + 200. Resumable from (theta,m,v,step) ckpt every 1000.
Optimizer state (m,v) and gradients kept fp32 throughout.

usage: v_escape_run.py S SEED CAP [device=cpu|mps] [threads]
"""
import json, sys, math
from pathlib import Path
sys.path.insert(0, "/Users/mihir/synass-lens/synass-lens")
import torch
from dataclasses import replace

S_SCALE = float(sys.argv[1])
SEED = int(sys.argv[2])
CAP = int(sys.argv[3])
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "cpu"
torch.set_num_threads(int(sys.argv[5]) if len(sys.argv) > 5 else 4)

from pinned_capabilities.config import ProtocolConfig
from pinned_capabilities.experiment import MBCExperiment
from pinned_capabilities.gate0e_escape import load_snapshot_fresh_stream
from src.data import collate_fn
from src.training.trainer import compute_loss

LR, WD, B1, B2, EPS = 0.0125, 0.01, 0.9, 0.999, 1e-8
CE_EVERY, ACC_EVERY, ACC_EVERY_NEAR, CKPT_EVERY, CHUNK = 200, 1000, 100, 1000, 2500
NEAR_CE = 2.5          # tighten accuracy cadence once CE dips below this
ONSET_CE, ONSET_SUSTAIN = 3.0, 500
ONSET_TAIL = 200          # stop this many steps after tau_onset confirms (fast iteration)
REPO = Path("/Users/mihir/synass-lens/synass-lens")
COLLAPSED = REPO / "pinned_capabilities/results/gate0e_dev_curve_seed100/eta_0p0125/stream_00/checkpoints/slot_0.pt"
OUT = REPO / f"pinned_capabilities/results/v_escape/s{S_SCALE:.2f}_seed{SEED}"


def to_dev(b):
    if torch.is_tensor(b): return b.to(DEVICE)
    if isinstance(b, dict): return {k: to_dev(v) for k, v in b.items()}
    if isinstance(b, (list, tuple)): return type(b)(to_dev(v) for v in b)
    return b


def build():
    cfg = ProtocolConfig()
    exp = MBCExperiment(replace(cfg.experiment, seed=100, device="cpu"), cfg.metric)
    load_snapshot_fresh_stream(COLLAPSED, model=exp.model, optimizer=exp.optimizer, map_location="cpu")
    if DEVICE != "cpu":
        exp.model.to(DEVICE)
    # optimizer order throughout so params[i], m[i], v[i], g[i] all align
    params = [p for grp in exp.optimizer.param_groups for p in grp["params"]]
    S = exp.optimizer.state
    m = [S[p]["exp_avg"].detach().clone().to(DEVICE).float() for p in params]
    v = [S[p]["exp_avg_sq"].detach().clone().to(DEVICE).float() for p in params]
    step0 = S[params[0]]["step"]
    t0 = int(step0.item()) if torch.is_tensor(step0) else int(step0)
    n = len(exp.dataset)
    chunks = [to_dev(collate_fn([exp.dataset[i] for i in range(s, min(s + CHUNK, n))])) for s in range(0, n, CHUNK)]
    return exp, params, m, v, t0, n, chunks


def full_grad(exp, params, chunks):
    for p in params: p.grad = None
    for b in chunks:
        loss, _, _ = compute_loss(exp.model, b)
        (loss / len(chunks)).backward()
    return [p.grad.detach().clone().float() for p in params]


def flat_norm(lst):
    return math.sqrt(sum(float(x.pow(2).sum()) for x in lst))


def ce_full(exp, chunks):
    # FIRST-target-token full-vocab CE over ALL 10k keys (device-resident;
    # avoids the CPU probe-tensor path in exp.evaluate that breaks on MPS).
    # Matches the registered probe full_vocab_ce (first_token_only) -> ln36 at
    # collapse; validated == 3.585 at the collapsed state.
    exp.model.eval()
    tot = 0.0; nex = 0
    with torch.no_grad():
        for b in chunks:
            logits = exp.model(b["input_ids"])
            sl = logits[:, :-1]
            st = b["labels"][:, 1:]
            fi = (st != -100).float().argmax(dim=1)       # first target position per key
            ar = torch.arange(st.size(0), device=st.device)
            ce = torch.nn.functional.cross_entropy(sl[ar, fi], st[ar, fi], reduction="sum")
            tot += float(ce); nex += st.size(0)
    exp.model.train()
    return tot / nex


def retrieval_acc(exp, chunks):
    exp.model.eval()
    correct = total = 0
    with torch.no_grad():
        for b in chunks:
            logits = exp.model(b["input_ids"])
            pred = logits[:, :-1].argmax(-1)
            tgt = b["labels"][:, 1:]
            mask = tgt != -100
            ok = ((pred == tgt) | ~mask).all(dim=1)
            correct += int(ok.sum()); total += int(ok.numel())
    exp.model.train()
    return correct / total


def bias_corrected_update(m, v, t_eff):
    c1, c2 = 1 - B1 ** t_eff, 1 - B2 ** t_eff
    return [-LR * (m[i] / c1) / ((v[i] / c2).sqrt() + EPS) for i in range(len(m))]


def run():
    exp, params, m, v, t0, n, chunks = build()
    ngen = torch.Generator().manual_seed(1_000_000 + SEED)          # cpu noise gen (deterministic)
    mbgen = torch.Generator().manual_seed(2_000_000 + SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    def mb_grad():
        idx = torch.randint(n, (128,), generator=mbgen)
        batch = to_dev(collate_fn([exp.dataset[int(i)] for i in idx]))
        for p in params: p.grad = None
        loss, _, _ = compute_loss(exp.model, batch)
        loss.backward()
        return [p.grad.detach().clone().float() for p in params]

    # resume if a checkpoint exists
    start = 1
    ce_hist = []            # (step, ce)
    tau_onset = None; tau_solve = None; solve_candidate = None
    ckpt = OUT / "ckpt.pt"
    if ckpt.exists():
        # load to CPU (RNG-state ByteTensors must stay on CPU for set_state),
        # then move only theta/m/v to the device.
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
        # per-step matched isotropic noise, scaled by s
        ref = [(a - b) / math.sqrt(2) for a, b in zip(mb_grad(), mb_grad())]
        ref_norm = flat_norm(ref)
        iso = [torch.randn(p.shape, generator=ngen).to(DEVICE) for p in params]
        scale = S_SCALE * ref_norm / (flat_norm(iso) + 1e-30)
        xi = [scale * e for e in iso]
        gpx = [a + b for a, b in zip(g, xi)]                       # g + s*xi
        t_eff = t0 + step
        for i in range(len(m)):
            m[i].mul_(B1).add_(g[i], alpha=1 - B1)                 # V: clean m
            v[i].mul_(B2).add_(gpx[i] * gpx[i], alpha=1 - B2)      # V: noisy v
        u = bias_corrected_update(m, v, t_eff)
        for i, p in enumerate(params):
            p.data.mul_(1 - LR * WD).add_(u[i])

        row = {"step": step, "s": S_SCALE, "seed": SEED}
        near = ce_hist and ce_hist[-1][1] < NEAR_CE
        acc_now = ACC_EVERY_NEAR if near else ACC_EVERY
        if step % CE_EVERY == 0 or step % acc_now == 0:
            ce = ce_full(exp, chunks)
            row["full_vocab_ce"] = ce
            ce_hist.append((step, ce))
            # tau_onset: CE<3.0 sustained >=500 steps
            if tau_onset is None:
                for j, (sj, cj) in enumerate(ce_hist):
                    if cj < ONSET_CE and all(ck < ONSET_CE for sk, ck in ce_hist[j:] if sk <= sj + ONSET_SUSTAIN) \
                       and ce_hist[-1][0] >= sj + ONSET_SUSTAIN:
                        tau_onset = sj; break
                if tau_onset is not None:
                    # onset captured (the trend-fit metric). Escape is oscillatory
                    # (kick-backs); solve is noise-dominated -> fast-stop, don't chase it.
                    stop_at = min(CAP, step + ONSET_TAIL)
        if step % acc_now == 0:
            acc = retrieval_acc(exp, chunks)
            row["exact_acc"] = acc
            if acc >= 0.999:                                   # effective solve (last 0.1% is noise-jitter)
                if solve_candidate is None:
                    solve_candidate = step
                elif tau_solve is None:
                    tau_solve = solve_candidate                    # confirmed at next acc eval
                    stop_at = min(CAP, tau_solve + 200)
            else:
                solve_candidate = None
        if len(row) > 3:
            row["tau_onset"] = tau_onset; row["tau_solve"] = tau_solve
            writer.write(json.dumps(row) + "\n"); writer.flush()

        if step % CKPT_EVERY == 0 or step == stop_at:
            torch.save({"step": step, "theta": [p.data.detach().cpu() for p in params],
                        "m": [x.detach().cpu() for x in m], "v": [x.detach().cpu() for x in v],
                        "ce_hist": ce_hist, "tau_onset": tau_onset, "tau_solve": tau_solve,
                        "solve_candidate": solve_candidate,
                        "ngen": ngen.get_state(), "mbgen": mbgen.get_state()}, ckpt)
        if step >= stop_at:
            break

    writer.close()
    capped = tau_solve is None
    summ = {"s": S_SCALE, "seed": SEED, "cap": CAP, "last_step": step,
            "tau_onset": tau_onset, "tau_solve": tau_solve, "capped": capped,
            "final_ce": ce_hist[-1][1] if ce_hist else None,
            "noise_seed": 1_000_000 + SEED, "mb_seed": 2_000_000 + SEED, "device": DEVICE}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    run()
