#!/usr/bin/env python3
"""Plan v6 amendment A1 (19 Nov 2026): the frozen F1 calibration, s(d) for every training depth and one gain g.

Rule (10 Research Direction/plan-amendment-a1-f1-pilot.ai.md §2, fixed before any F1 policy):
* model: the world model the pilot trains with (D's model, the E1-V refit);
* data: a pre-reset trace that model was never fitted on (A1: test_reference.npz, envs 52-63);
* windows: every origin t at a 16-row spacing whose rows t-31 .. t+H-1 carry no episode end (one episode,
  no fall, no time-out), H = the imagination episode length (256); `--windows` (512) drawn with seed 0;
* imagination: each head on its own path (member trajectories), recorded actions, cold start (one real row,
  zeroed memory), as every episode after the first in paper-mode training;
* m(d) = sum over dims of the unbiased std over members of their normalized states (as envs/base.py F1);
  e(d) = mean over members of sum over dims |member state - true state| (normalized);
  u(d) = mean over the followed paths of the summed head disagreement (the standard penalty);
* s(d) = mean_w e(d) / mean_w m(d), d = 0 .. H-1;
* g = mean_{w,d} u(d) / mean_{w,d} s(d) m(d): F1's mean penalty equals the standard one on these windows.
Writes f1_calibration.json (read by e1_train.py --f1-calibration) and a text report. Run once; the JSON is
frozen and its SHA-256 recorded before training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c0_battery as c0  # noqa: E402
from pre_reset_transitions import gather, sha256, validate_trace  # noqa: E402

SPACING, CONTEXT_ROWS = 16, 32


def calibration_origins(done, horizon, n_windows, seed=0, spacing=SPACING):
    """(t, n) with rows t-31 .. t+horizon-1 free of episode ends; a seeded draw of n_windows, sorted."""
    T, N = done.shape
    cands = []
    for n in range(N):
        ends = np.flatnonzero(done[:, n])
        for t in range(CONTEXT_ROWS - 1, T - horizon + 1, spacing):
            lo, hi = t - (CONTEXT_ROWS - 1), t + horizon - 1
            if not ((ends >= lo) & (ends <= hi)).any():
                cands.append((t, n))
    cands = np.asarray(cands, dtype=np.int64).reshape(-1, 2)
    if len(cands) < n_windows:
        raise ValueError(f"only {len(cands)} clean windows of {horizon} steps; need {n_windows}")
    pick = np.sort(np.random.default_rng(seed).choice(len(cands), n_windows, replace=False))
    return cands[pick], len(cands)


def calibrate(model, contract, trace, horizon=256, n_windows=512, seed=0):
    origins, n_cands = calibration_origins(trace["done"], horizon, n_windows, seed)
    with torch.no_grad():
        imag = c0.imagined_pass(model, contract, trace, origins, horizon, context=1, feedback="head", seed=seed)
    stats = contract.statistics
    own = np.asarray(imag["own"], np.float64)                                  # [P, H, W, 45] normalized
    target = ((torch.from_numpy(gather(trace["next_state"], origins, 0, horizon)).float()
               - stats["state_mean"]) / stats["state_std"]).numpy().transpose(1, 0, 2)   # [H, W, 45]
    m = own.std(axis=0, ddof=1).sum(-1)                                        # [H, W]
    e = np.abs(own - target[None]).sum(-1).mean(0)                             # [H, W]
    u = np.asarray(imag["spread"], np.float64).sum(-1).mean(0)                 # [H, W]
    if (m.mean(1) <= 0).any():
        raise ValueError("zero member spread at some depth: s(d) undefined")
    s = e.mean(1) / m.mean(1)                                                  # [H]
    f1_unscaled = s[:, None] * m
    g = float(u.mean() / f1_unscaled.mean())
    return dict(origins=origins, candidates=n_cands, scale=s, gain=g, m=m, e=e, u=u,
                check=dict(f1_mean=float((g * f1_unscaled).mean()), standard_mean=float(u.mean())))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", required=True, help="the pilot's world model (D's model)")
    p.add_argument("--trace", type=Path, required=True, help="calibration trace the model was never fitted on")
    p.add_argument("--out", type=Path, required=True, help="fresh output folder")
    p.add_argument("--horizon", type=int, default=256, help="imagination episode length (max_episode_length)")
    p.add_argument("--windows", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.out.exists():
        p.error("output folder exists")
    torch.set_num_threads(8)
    model, contract, paths = c0.load_models(a.wm)
    with np.load(a.trace, allow_pickle=False) as z:
        meta = validate_trace(z)
        trace = {k: z[k] for k in z.files if k != "metadata"}
    r = calibrate(model, contract, trace, a.horizon, a.windows, a.seed)
    a.out.mkdir(parents=True)
    depths = (0, 1, 8, 16, 25, 50, 100, 150, 200, 255)
    depths = tuple(d for d in depths if d < a.horizon)
    payload = dict(
        protocol="plan v6 amendment A1 (10 Research Direction/plan-amendment-a1-f1-pilot.ai.md)",
        members=int(model.ensemble_size), horizon=a.horizon, context_rows=1, feedback="head", actions="recorded",
        wm=[str(x) for x in paths], wm_sha256=[sha256(x) for x in paths],
        normalizer=contract.metadata(),
        calibration_trace=str(a.trace), calibration_trace_sha256=sha256(a.trace),
        calibration_task=meta.get("task"), calibration_checkpoint=meta.get("checkpoint"),
        calibration_envs=meta.get("e1_test_envs"), windows=int(len(r["origins"])), candidates=int(r["candidates"]),
        spacing=SPACING, seed=a.seed, origins_sha256=__import__("hashlib").sha256(r["origins"].tobytes()).hexdigest(),
        scale=[float(x) for x in r["scale"]], gain=r["gain"], check=r["check"],
        by_depth={f"d{d}": dict(s=float(r["scale"][d]), m=float(r["m"][d].mean()), e=float(r["e"][d].mean()),
                                u=float(r["u"][d].mean()), f1_penalty=float(r["gain"] * r["scale"][d] * r["m"][d].mean()))
                  for d in depths},
        analysis_sha256=sha256(__file__))
    (a.out / "f1_calibration.json").write_text(json.dumps(payload, indent=1, allow_nan=False))
    lines = ["Plan v6 amendment A1: frozen F1 calibration",
             f"model {Path(paths[0]).name} ({payload['wm_sha256'][0][:12]}); trace {a.trace.name} "
             f"({payload['calibration_trace_sha256'][:12]}), envs {payload['calibration_envs']}",
             f"windows {payload['windows']} of {payload['candidates']} clean candidates; horizon {a.horizon}; cold start; "
             f"{payload['members']} members",
             f"gain g = {r['gain']:.6g}; check: mean F1 penalty {r['check']['f1_mean']:.4f} = mean standard penalty "
             f"{r['check']['standard_mean']:.4f}",
             "depth | s(d) | member spread m | error e | standard u | F1 penalty g s m"]
    for d, v in payload["by_depth"].items():
        lines.append(f"{d:5s} | {v['s']:.4f} | {v['m']:.4f} | {v['e']:.4f} | {v['u']:.4f} | {v['f1_penalty']:.4f}")
    text = "\n".join(lines) + "\n"
    (a.out / "f1_calibration.txt").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
