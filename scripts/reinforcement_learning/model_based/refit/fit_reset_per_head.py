#!/usr/bin/env python3
"""Run fit_world_model_go2_pilot_segments.py with the B3 fix (doubt register B3, 13 Oct 2026).

ETH's SystemDynamicsEnsemble.compute_loss resets the GRU memory once per batch. Head 0 then starts
from zero memory, but head k>0 starts from the memory head k-1 left behind (on a different bootstrap
resample). At scoring and in imagination every head starts from zero memory. This wrapper resets the
state and auxiliary trunk memory before every head's loss, so training matches use.

The fitter file itself is not changed (running fits keep importing the same code). All arguments are
passed through unchanged. Every checkpoint written under the --output folder during this run gets
provenance["b3_reset_per_head"] = True.

Usage: python refit/fit_reset_per_head.py <same args as fit_world_model_go2_pilot_segments.py>
"""
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FITTER = os.path.join(os.path.dirname(HERE), "fit_world_model_go2_pilot_segments.py")


def apply_patch():
    """Patch SystemDynamicsEnsemble so each head's state/aux loss starts from zero trunk memory."""
    from rsl_rl.modules import SystemDynamicsEnsemble as SDE
    if getattr(SDE, "_b3_reset_per_head", False):
        return SDE
    orig_state, orig_aux = SDE.compute_state_loss, SDE.compute_auxiliary_loss

    def compute_state_loss(self, head, state_batch, action_batch):
        self.state_base.reset()
        return orig_state(self, head, state_batch, action_batch)

    def compute_auxiliary_loss(self, head, *args, **kwargs):
        self.auxiliary_base.reset()
        return orig_aux(self, head, *args, **kwargs)

    SDE.compute_state_loss = compute_state_loss
    SDE.compute_auxiliary_loss = compute_auxiliary_loss
    SDE._b3_reset_per_head = True
    SDE._b3_originals = (orig_state, orig_aux)
    return SDE


def remove_patch():
    from rsl_rl.modules import SystemDynamicsEnsemble as SDE
    if getattr(SDE, "_b3_reset_per_head", False):
        SDE.compute_state_loss, SDE.compute_auxiliary_loss = SDE._b3_originals
        SDE._b3_reset_per_head = False


def mark_checkpoints(out_dir, since):
    import torch
    n = 0
    for f in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, f)
        if f.endswith(".pt") and os.path.getmtime(p) >= since:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(ck, dict):
                ck.setdefault("provenance", {})["b3_reset_per_head"] = True
                torch.save(ck, p)
                n += 1
    print(f"[b3] marked {n} checkpoint(s) in {out_dir} with b3_reset_per_head=True")


def main():
    argv = sys.argv[1:]
    out = argv[argv.index("--output") + 1] if "--output" in argv else None
    t0 = time.time() - 1.0
    apply_patch()
    print("[b3] SystemDynamicsEnsemble patched: trunk memory reset before every head")
    sys.argv = [FITTER] + argv
    sys.path.insert(0, os.path.dirname(FITTER))
    runpy.run_path(FITTER, run_name="__main__")
    if out is not None:
        mark_checkpoints(os.path.dirname(os.path.abspath(out)), t0)


if __name__ == "__main__":
    main()
