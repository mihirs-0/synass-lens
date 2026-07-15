# Novelty audit: what this suite must add to the field

**Frozen 2026-07-15, before Gate 0 or Gate 1 outcomes**

## Verdict

The project remains worth running, but **hysteresis or metastability by itself
is not the contribution**. Ersoy and Wiesner now give an explicit first-order
phase-transition account of grokking in deep linear networks, demonstrate
hysteresis, deliberately trap low-accuracy metastable states, and recover
Arrhenius noise-driven escape. Their paper is the closest prior work and moves
the novelty bar substantially
([arXiv:2606.17120](https://arxiv.org/abs/2606.17120)).

The publishable question is now:

> In a nonlinear conditional-binding system trained with AdamW, is capability
> absence a history-dependent, dynamically maintained state that cannot be
> reduced to local optimizer stability; where is that state remembered; and
> can a short, module-specific intervention reliably move training across its
> basin boundary?

That is a finite experimental question. It is broader than the particular
lookup task because it asks what the complete training process remembers and
whether that memory can be used for control.

## Claims already owned by prior work

The suite must not claim novelty for any of the following:

1. **Plateaus followed by sharp learning transitions.** Deep linear networks
   have exhibited analytically solved plateaus and rapid mode acquisition since
   Saxe, McClelland, and Ganguli
   ([arXiv:1312.6120](https://arxiv.org/abs/1312.6120)). Grokking supplies the
   modern nonlinear benchmark
   ([Power et al.](https://arxiv.org/abs/2201.02177)).
2. **Hysteresis, metastability, or noise-driven escape in neural training.**
   Ersoy and Wiesner demonstrate all three in regularized deep linear networks,
   including escape-time scaling over two orders of magnitude
   ([arXiv:2606.17120](https://arxiv.org/abs/2606.17120)).
3. **Learning-rate phase diagrams and stability bifurcations.** These are
   established for neural training under SGD and deep matrix factorization
   ([Kalra and Barkeshli](https://openreview.net/forum?id=Al9yglQGKj),
   [Ghosh et al.](https://openreview.net/forum?id=J4Dvxv7WnG)).
4. **Edge-of-stability explanations.** Both ordinary and adaptive optimizers
   have dedicated accounts. In particular, Adam's preconditioned Hessian has a
   momentum-dependent adaptive stability threshold
   ([Cohen et al.](https://openreview.net/forum?id=dHGNgkUcGd)).
5. **Cyclic late-training instability.** Slingshot dynamics already connect
   adaptive optimizers, weight norms, and repeated loss spikes
   ([Thilak et al.](https://arxiv.org/abs/2206.04817)). The suite's fixed-rule
   holds and norm exclusions are therefore mandatory.
6. **Hidden progress before behavioral emergence.** Circuit-derived progress
   measures already predict modular-addition grokking retrospectively
   ([Nanda et al.](https://openreview.net/forum?id=9XFSbDPmdW)).
7. **Training history matters.** Critical-period work shows that temporary
   early deficits can have lasting skill effects
   ([Achille et al.](https://openreview.net/forum?id=BkeStsCcKQ)).
8. **A capability can exist before ordinary behavior reveals it.** Latent
   interventions already expose hidden concept manipulation before naive
   prompting elicits it, including in synthetic generative systems
   ([Park et al.](https://arxiv.org/abs/2406.19370)). `C_int` is therefore a
   deliberately cleaner state variable, not a claim that hidden capability is
   new.
9. **Layer- or module-specific training rules.** Different learning rates per
   layer and selective module updating are established optimization tools
   ([LeRaC](https://openreview.net/forum?id=AdK9_GTEvG),
   [Modular Adaptive Training](https://openreview.net/forum?id=dWDEBW2raJ)).
   Merely finding that an MLP-only pulse helps is not a contribution.

## The surviving novelty package

### 1. Behavioral bistability beyond an analytically engineered linear mode

The primary system is a nonlinear transformer learning conditional binding,
and state is defined by a counterfactual interaction assay rather than train or
test accuracy alone. A positive result would extend metastability from known
singular modes and L2-controlled linear transitions to a conditional behavior
whose relevant parameter direction is not supplied analytically.

This extension is meaningful only if the fixed-rule state survives the
long-dwell, second-cycle, full-batch, and slingshot exclusions. Otherwise the
result reduces to known transient or stochastic escape behavior.

### 2. Complete training-state memory localization

The 2x2 crossing of weights and complete update-process state is not a generic
optimizer reset. It asks whether future behavioral fate is carried by weights,
Adam moments and counters, their interaction, or neither. RNG and data cursor
travel with optimizer state, so the intervention crosses actual dynamical
states rather than vaguely similar checkpoints.

A clean factorial result would connect dynamical-systems language to a concrete
training intervention. This is stronger than observing history dependence and
more actionable than assigning a phase label.

### 3. Deterministic versus noise-maintained pinning

The closest 2026 metastability result predicts noise-activated escape with
effective temperature proportional to learning rate over batch size. MBC
pilots suggest high-rate suppression can survive very large batches. A finite
band under full-batch AdamW, after conditioning on the changing preconditioner,
would identify a different mechanism: geometry- or optimizer-maintained
pinning rather than noise-necessary trapping. If the band instead obeys the
noise-scaling prediction, that is replication and transfer, not the headline.

### 4. From diagnosis to control

The strongest possible result is not the phase diagram. It is that a short
behavioral response assay chooses a module-specific pulse that crosses the
basin boundary faster and more reliably than cheap baselines, without being
told the winning module. This converts a state description into a training-time
control instrument. The novelty is the closed causal chain -- behaviorally
defined state, prospective intervention selection, and durable post-pulse fate
change -- not the use of per-module learning rates.

The claim dies if global learning rate, gradient norm, a 50-step branch, or a
development-set default performs as well. It also dies if the assay merely
predicts the largest immediate metric improvement: the post-restoration hold
must show basin crossing. It is validation rather than blind localization if
the MLP wins almost everywhere.

## Publication threshold

- **Gate 0 reduction succeeds:** publish only a capability-resolved adaptive
  stability note; stop the larger phase claim.
- **Hysteresis alone:** insufficient after Ersoy and Wiesner.
- **Gate 1 with clean memory surgery and deterministic/noise classification:**
  potentially publishable as nonlinear optimizer-state memory and reversible
  capability control.
- **Gate 2 beats all cheap baselines out of sample:** strongest paper; the
  diagnostic has earned its cost by changing training fate. Module-specific
  optimization alone is explicitly not the claim.
- **Gate 3 alone:** insufficient. Early-warning statistics are mature in
  dynamical systems and count only if they improve control or forecasting here.

The paper promised to stakeholders is therefore not “we found a strange small
model.” It is: **we identified when behavioral absence is a maintained training
state, located the memory that maintains it, and used that knowledge to control
what training learns.**
