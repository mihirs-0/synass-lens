#!/usr/bin/env python
"""
Run 8 shared machinery: dense state saving (weights + Adam moments) and
optimizer-state restoration through the existing trainer's callbacks.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import TrainingCallbacks


class StateSaver:
    def __init__(self, out_dir, every, also_at=()):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.also_at = set(also_at)

    def __call__(self, model=None, optimizer=None, step=None, **kw):
        if step % self.every == 0 or step in self.also_at:
            torch.save({"step": step,
                        "model": {k: v.detach().cpu() for k, v in
                                  model.state_dict().items()},
                        "opt": optimizer.state_dict()},
                       self.dir / f"state_{step:06d}.pt")


class StateRestorer:
    """Loads a saved optimizer state before the FIRST optimizer step of a
    train() call (via on_after_backward, which fires pre-step)."""

    def __init__(self, opt_state):
        self.opt_state = opt_state
        self.done = False

    def __call__(self, optimizer=None, **kw):
        if not self.done and self.opt_state is not None:
            optimizer.load_state_dict(self.opt_state)
            self.done = True


def make_callbacks(saver=None, restorer=None):
    return TrainingCallbacks(
        on_after_step=saver,
        on_after_backward=restorer,
    )


def load_state(path, model=None, device="cpu"):
    d = torch.load(path, map_location=device, weights_only=False)
    if model is not None:
        model.load_state_dict(d["model"])
    return d


def flat_params(model):
    return torch.cat([p.detach().flatten().cpu()
                      for p in model.parameters()])


def flat_from_state(sd, ref_model):
    return torch.cat([sd[k].flatten().float()
                      for k, _ in ref_model.named_parameters()])


def flat_opt_moment(opt_state, which, ref_model):
    """Flatten exp_avg or exp_avg_sq in parameter order."""
    out = []
    ids = opt_state["param_groups"][0]["params"]
    st = opt_state["state"]
    for i, (_n, p) in enumerate(zip(ids, ref_model.parameters())):
        s = st.get(i, st.get(str(i)))
        if s is None or which not in s:
            out.append(torch.zeros(p.numel()))
        else:
            out.append(s[which].flatten().float().cpu())
    return torch.cat(out)
