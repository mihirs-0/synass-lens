"""Routing/correctness unit tests for the reacq harness (amendment N_PM, M_T).
Covers: per-channel noise routing, scaled-sigma injection, manual AdamW==torch.
Run: python3 pinned_capabilities/test_reacq_arms.py  (expects ALL PASS)."""
import sys
sys.path.insert(0, "/Users/mihir/synass-lens/synass-lens")
sys.argv = ["x", "C0", "collapsed", "0", "5", "cpu", "1"]   # valid argv so module imports
import torch
import pinned_capabilities.reacq_2x2_run as R

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond: fails.append(name)

torch.manual_seed(0)
g   = [torch.randn(6), torch.randn(3, 2)]
g2  = [x * x for x in g]
gpx = [x + 0.5 for x in g]           # stand-in for g + scale*xi
gpx2 = [x * x for x in gpx]

# --- per-channel routing (the core of the amendment's two new paths) ---
ms, vi = R.route_noisy("V", g, g2, gpx, gpx2)
check("V: m clean (==g), v noisy (==gpx^2)", all(a.equal(b) for a, b in zip(ms, g)) and all(a.equal(b) for a, b in zip(vi, gpx2)))
ms, vi = R.route_noisy("N", g, g2, gpx, gpx2)
check("N: m noisy, v noisy", all(a.equal(b) for a, b in zip(ms, gpx)) and all(a.equal(b) for a, b in zip(vi, gpx2)))
ms, vi = R.route_noisy("N_PM", g, g2, gpx, gpx2)
check("N_PM: same routing as N (scale handled upstream)", all(a.equal(b) for a, b in zip(ms, gpx)) and all(a.equal(b) for a, b in zip(vi, gpx2)))
ms, vi = R.route_noisy("M_T", g, g2, gpx, gpx2)
check("M_T: m noisy (==gpx), v CLEAN (==g^2)", all(a.equal(b) for a, b in zip(ms, gpx)) and all(a.equal(b) for a, b in zip(vi, g2)))

# --- scaled-sigma injection: ||injected|| == SCALE * ||ref|| ---
ref = [torch.randn(50), torch.randn(20)]
iso = [torch.randn(50), torch.randn(20)]
for SCALE in (0.5, 0.6, 0.7, 1.0):
    nsc = SCALE * R.flat_norm(ref) / (R.flat_norm(iso) + 1e-30)
    inj = [nsc * e for e in iso]
    check(f"scaled injection ||xi||=={SCALE}*||ref|| (c={SCALE})", abs(R.flat_norm(inj) - SCALE * R.flat_norm(ref)) < 1e-4)
# monotonic: smaller c -> smaller injected power (the N_PM knob)
def injnorm(c):
    nsc = c * R.flat_norm(ref) / (R.flat_norm(iso) + 1e-30)
    return R.flat_norm([nsc * e for e in iso])
check("N_PM knob monotone: inj(0.5) < inj(0.7) < inj(1.0)", injnorm(0.5) < injnorm(0.7) < injnorm(1.0))

# --- manual AdamW == torch.optim.AdamW (3 clean steps, decoupled WD) ---
torch.manual_seed(1)
p0 = torch.randn(40, dtype=torch.float64)
grads = [torch.randn(40, dtype=torch.float64) for _ in range(3)]
# torch reference
pt = p0.clone().requires_grad_(True)
opt = torch.optim.AdamW([pt], lr=R.LR, betas=(R.B1, R.B2), eps=R.EPS, weight_decay=R.WD)
for gg in grads:
    opt.zero_grad(); pt.grad = gg.clone(); opt.step()
# manual (matches run(): theta<-theta*(1-lr*wd)+u ; u bias-corrected)
pm = p0.clone(); m = torch.zeros(40, dtype=torch.float64); v = torch.zeros(40, dtype=torch.float64)
for t, gg in enumerate(grads, start=1):
    m = R.B1 * m + (1 - R.B1) * gg
    v = R.B2 * v + (1 - R.B2) * gg * gg
    u = R.bias_corrected_update([m], [v], t)[0]
    pm = pm * (1 - R.LR * R.WD) + u
rel = float((pm - pt.detach()).norm() / (pt.detach().norm() + 1e-30))
check(f"manual AdamW == torch (rel {rel:.2e} < 1e-6)", rel < 1e-6)

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
