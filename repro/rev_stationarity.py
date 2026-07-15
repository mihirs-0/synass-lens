#!/usr/bin/env python
"""REVIEW EXP A — stationarity / limit-cycle diagnostic (run FIRST).
Is the trap a frozen predictor or a high-LR orbit? Treat (theta, Adam state) as the real state.
At the trap measure: full-batch ||grad L||, per-module grad norms, parameter-update norms ||theta_{t+1}-theta_t||,
and lag-1/lag-2 autocorrelation of update directions and logits. Memorized checkpoint = control.
Also: optimizer-state-reset variant; gauge-fixed (class-centered) output-bias norm; weight-decay-on-bias check.

Verdict logic: loss+output flat but ||grad|| or ||dtheta|| NOT small  -> ORBIT, not frozen ("sits there" is wrong).
"""
import json, math, os
import numpy as np
import torch, torch.nn.functional as F
import collapse_lib as L

device = L.device
vocab = L.NSPEC + L.ALPHA
N, ETA_TRAP, M = 2000, 0.06, 8000
TS, LA = L.TS, L.LA


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

def module_groups(model):
    g = {"tok": [model.tok.weight], "pos": [model.pos.weight],
         "attn": [p for a in model.at for p in a.parameters()],
         "mlp": [p for m in model.ml for p in m.parameters()],
         "head_w": [model.head.weight], "head_bias": [model.head.bias],
         "ln": [p for ln in model.ln for p in ln.parameters()] + list(model.lnf.parameters())}
    return g

@torch.no_grad()
def probe_logits(model, X):
    return model(X.to(device))[:, TS-1:TS-1+LA].reshape(X.size(0), -1).detach().clone()  # (n, LA*V)

def full_batch_grad(model, X):
    model.zero_grad(set_to_none=True)
    loss = L.lm_loss(model, X.to(device)); loss.backward()
    gn = math.sqrt(sum(float((p.grad**2).sum()) for p in model.parameters() if p.grad is not None))
    per = {}
    for name, ps in module_groups(model).items():
        per[name] = round(math.sqrt(sum(float((p.grad**2).sum()) for p in ps if p.grad is not None)), 5)
    return float(loss), gn, per

def cos(a, b):
    na, nb = a.norm(), b.norm()
    return float((a @ b) / (na * nb + 1e-12))

def autocorr_dirs(seq):
    """lag-1 & lag-2 cosine of consecutive increments (drift +1 / cycle alternating / random ~0)."""
    incs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
    incs = [d for d in incs if d.norm() > 1e-9]
    if len(incs) < 3: return None, None
    l1 = np.mean([cos(incs[i], incs[i+1]) for i in range(len(incs)-1)])
    l2 = np.mean([cos(incs[i], incs[i+2]) for i in range(len(incs)-2)])
    return round(float(l1), 3), round(float(l2), 3)

def centered_bias_norm(model):
    b = model.head.bias.detach()
    return float(b.norm()), float((b - b.mean()).norm())   # raw, gauge-fixed (class-centered)


def continue_run(model, opt, X, steps, full_batch, label):
    Xp = X[:1500]
    theta_seq, logit_seq, dthetas, gnorms, kls = [], [], [], [], []
    z0 = probe_logits(model, Xp)
    base_out = F.softmax(z0.reshape(Xp.size(0), LA, vocab)[:, 0], -1).mean(0)  # time-0 mean first-tok dist
    g = torch.Generator().manual_seed(0)
    for t in range(steps):
        xb = X if full_batch else X[torch.randint(0, X.size(0), (256,), generator=g)]
        before = flat_params(model)
        opt.zero_grad(set_to_none=True); loss = L.lm_loss(model, xb.to(device)); loss.backward()
        gn = math.sqrt(sum(float((p.grad**2).sum()) for p in model.parameters() if p.grad is not None))
        opt.step()
        after = flat_params(model)
        dthetas.append(float((after - before).norm())); gnorms.append(gn)
        if t % 4 == 0:
            theta_seq.append(after.cpu());
            z = probe_logits(model, Xp); logit_seq.append(z.reshape(-1).cpu())
            out = F.softmax(z.reshape(Xp.size(0), LA, vocab)[:, 0], -1).mean(0)
            kls.append(float((out * (out.add(1e-9).log() - base_out.add(1e-9).log())).sum()))
    tl1, tl2 = autocorr_dirs(theta_seq); zl1, zl2 = autocorr_dirs(logit_seq)
    return {"label": label, "full_batch": full_batch,
            "dtheta_mean": round(float(np.mean(dthetas)), 5), "dtheta_last": round(dthetas[-1], 5),
            "gradnorm_mean": round(float(np.mean(gnorms)), 5), "gradnorm_last": round(gnorms[-1], 5),
            "theta_inc_cos_lag1": tl1, "theta_inc_cos_lag2": tl2,
            "logit_inc_cos_lag1": zl1, "logit_inc_cos_lag2": zl2,
            "output_KL_to_t0_max": round(max(kls), 5)}


def main():
    os.makedirs("card_ckpts", exist_ok=True)
    X, *_ = L.build_task(N, M=M, seed=0)
    print(f"=== EXP A stationarity (N={N}, eta_trap={ETA_TRAP}) ===\n")

    # train to the trap, KEEP optimizer (Adam) state
    torch.manual_seed(0); g = torch.Generator().manual_seed(0)
    trap = L.TinyGPT(vocab).to(device)
    opt = torch.optim.AdamW(trap.parameters(), lr=ETA_TRAP, betas=(0.9, 0.999), weight_decay=0.01)
    for _ in range(4000):
        xb = X[torch.randint(0, X.size(0), (256,), generator=g)]
        opt.zero_grad(set_to_none=True); L.lm_loss(trap, xb.to(device)).backward(); opt.step()
    fl, kl = float(L.lm_loss(trap, X[:2000].to(device))), L.input_dep_kl(trap, X)
    print(f"trapped: loss={fl:.3f} input-dep KL={kl:.4f} -> {L.classify(fl, kl)}")
    torch.save({"model": trap.state_dict(), "opt": opt.state_dict()}, "card_ckpts/trap_N2000.pt")

    # memorized control
    mem = L.TinyGPT(vocab).to(device)
    if os.path.exists("card_ckpts/mem_N2000.pt"):
        mem.load_state_dict(torch.load("card_ckpts/mem_N2000.pt", map_location=device))
    flm = float(L.lm_loss(mem, X[:2000].to(device))); klm = L.input_dep_kl(mem, X)
    print(f"memorized control: loss={flm:.3f} input-dep KL={klm:.4f}\n")

    # --- full-batch gradient norm at each state ---
    print("=== full-batch ||grad L|| and per-module (is it a stationary point?) ===")
    for name, mdl in [("TRAP", trap), ("MEMORIZED", mem)]:
        lo, gn, per = full_batch_grad(mdl, X)
        print(f"  {name:>10}: loss={lo:.3f}  ||grad||={gn:.4f}")
        print(f"             per-module: {per}")
    print()

    # --- continue dynamics: is output frozen while theta moves? ---
    print("=== continue at trap (does output stay flat while parameters move?) ===")
    runs = []
    # reload fresh copies so each run starts at the trap
    def fresh_trap(reset_opt=False):
        m = L.TinyGPT(vocab).to(device); ck = torch.load("card_ckpts/trap_N2000.pt", map_location=device)
        m.load_state_dict(ck["model"])
        o = torch.optim.AdamW(m.parameters(), lr=ETA_TRAP, betas=(0.9, 0.999), weight_decay=0.01)
        if not reset_opt: o.load_state_dict(ck["opt"])
        return m, o
    m, o = fresh_trap(); runs.append(continue_run(m, o, X, 200, True, "trap full-batch, Adam-state kept"))
    m, o = fresh_trap(); runs.append(continue_run(m, o, X, 200, False, "trap minibatch, Adam-state kept"))
    m, o = fresh_trap(reset_opt=True); runs.append(continue_run(m, o, X, 200, True, "trap full-batch, Adam-state RESET"))
    m2 = L.TinyGPT(vocab).to(device); m2.load_state_dict(mem.state_dict())
    o2 = torch.optim.AdamW(m2.parameters(), lr=ETA_TRAP*0.0+1e-3, betas=(0.9,0.999), weight_decay=0.01)  # mem at low eta
    runs.append(continue_run(m2, o2, X, 200, True, "memorized full-batch @ eta=1e-3 (control)"))
    for r in runs:
        print(f"\n  [{r['label']}]")
        print(f"    ||dtheta|| mean={r['dtheta_mean']} last={r['dtheta_last']}   "
              f"||grad|| mean={r['gradnorm_mean']} last={r['gradnorm_last']}")
        print(f"    output KL to t0 (max) = {r['output_KL_to_t0_max']}  (small => output stays flat)")
        print(f"    theta-increment cos: lag1={r['theta_inc_cos_lag1']} lag2={r['theta_inc_cos_lag2']}  "
              f"logit-increment cos: lag1={r['logit_inc_cos_lag1']} lag2={r['logit_inc_cos_lag2']}")
        print(f"    (drift: lag1~+1 | limit-cycle: lag1<0 | random walk: lag1~0)")

    # --- gauge-fixed bias norm ---
    print("\n=== output-bias norm: raw vs gauge-fixed (class-centered) ===")
    rb_t, cb_t = centered_bias_norm(trap); rb_m, cb_m = centered_bias_norm(mem)
    print(f"  TRAP:      raw={rb_t:.3f}  centered={cb_t:.3f}")
    print(f"  MEMORIZED: raw={rb_m:.3f}  centered={cb_m:.3f}")
    print(f"  ratio raw={rb_t/max(rb_m,1e-6):.1f}x   ratio centered={cb_t/max(cb_m,1e-6):.1f}x")

    # --- weight-decay-on-bias implementation check ---
    print("\n=== weight-decay-on-bias check ===")
    o = torch.optim.AdamW(trap.parameters(), lr=1e-3, weight_decay=0.01)
    bias_in_decay = any(trap.head.bias is p for grp in o.param_groups for p in grp["params"] if grp["weight_decay"] > 0)
    print(f"  AdamW(model.parameters(), weight_decay=0.01): bias IS subject to decay = {bias_in_decay}")
    print(f"  (single param group, no bias exclusion -> decay applies to biases; the 'decay penalizes bias' argument holds)")

    json.dump({"runs": runs, "bias": {"trap_raw": rb_t, "trap_centered": cb_t, "mem_raw": rb_m, "mem_centered": cb_m},
               "fullbatch_grad": {"trap": full_batch_grad(trap, X)[1], "mem": full_batch_grad(mem, X)[1]}},
              open("card_ckpts/rev_stationarity.json", "w"), indent=2)
    print("\nsaved card_ckpts/rev_stationarity.json")


if __name__ == "__main__":
    main()
