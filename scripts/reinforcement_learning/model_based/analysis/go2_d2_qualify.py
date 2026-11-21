#!/usr/bin/env python3
"""A4 D2 qualification (written 22 Nov 2026 before it reads the D2 model). Read-only, CPU; trains nothing.

Data never used in the D2 fit: the E22 withheld nominal s2 episodes (walking) and the D1 probe (0.30 friction,
seed 911: falls and surviving episodes). Stop rule as imagination: a head stops when sigmoid(logit) > 0.5.
Rules (A4 §1 and §3, all (C)); heads averaged unless stated:
  Q1  walking (withheld nominal episodes):
      Q1a stop share with 32 recorded steps of memory <= 0.01;
      reset type: cold passes if the stop share with 1 step (cold reset) <= 0.01; warm passes if Q1a passes.
          D3 uses cold if cold passes, else warm if warm passes; if neither, Q1 fails;
      Q1c warm-start rollout survival to step 100 (recorded s2 actions) >= 0.95.
  Q1b rollout fidelity (same warm rollouts, raw units, pooled over starts and heads): base-velocity xy RMSE
      <= 0.25 m/s at step 25 and <= 0.40 m/s at step 50; tilt RMSE <= 5 deg at step 50.
  Q2  falls (probe): a head detects a fall if p > 0.5 at the terminal row or in the 10 rows before (memory from the
      episode start); detected share >= 0.5.
  Q3  surviving episodes (probe): a time-out episode is a false stop for a head if p > 0.5 on any row (memory from
      the episode start); false-stop share <= 0.10.
D2 passes if Q1, Q1b, Q2 and Q3 all pass. Push recovery (D1b) is reported separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MB = HERE.parent
for p in (str(HERE), str(MB)):
    if p not in sys.path:
        sys.path.insert(0, p)

import go2_g0_replay as rp  # noqa: E402
from pre_reset_transitions import sha256, validate_trace  # noqa: E402

VXY, GRAV = slice(0, 2), slice(6, 9)


def tilt_deg(g):
    g = np.asarray(g, np.float64)
    return np.degrees(np.arccos(np.clip(-g[..., 2] / np.maximum(np.linalg.norm(g, axis=-1), 1e-9), -1.0, 1.0)))


def fidelity(pred_norm, starts, eps_raw, step, norm):
    """Pooled RMSE of base-velocity xy (m/s) and tilt (deg) at a rollout step; pred_norm [n, E, 45]."""
    pred = pred_norm * norm["state_std"] + norm["state_mean"]
    true = np.stack([eps_raw[i][j + step - 1] for i, j in starts])[:, None, :]
    v = np.sqrt(np.mean(np.sum((pred[..., VXY] - true[..., VXY]) ** 2, -1)))
    t = np.sqrt(np.mean((tilt_deg(pred[..., GRAV]) - tilt_deg(true[..., GRAV])) ** 2))
    return float(v), float(t)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        meta = validate_trace(z)
        return {k: z[k] for k in z.files if k != "metadata"}, meta


def probe_episodes(trace, norm):
    import go2_round0_prepare as prep
    out = []
    for n, s, e, finished in prep.episodes(trace["done"]):
        if not finished:
            continue
        kind = "fall" if (trace["terminated"][e, n] and not trace["time_out"][e, n]) else "time_out"
        out.append(rp.Episode(trace, dict(env=n, first=s, last=e, kind=kind), norm))
    return out


def decide(r):
    """Apply the rules to the measured numbers (pure)."""
    m = lambda v: float(np.mean(v))
    k1, k32 = m(r["walking"]["k1"]["stop_share"]), m(r["walking"]["k32"]["stop_share"])
    cold_ok, warm_ok = k1 <= 0.01, k32 <= 0.01
    surv = m(r["walking"]["warm_rollouts"]["survival"]["100"]) if "100" in r["walking"]["warm_rollouts"]["survival"] else float("nan")
    q1 = dict(k32=k32, k1=k1, q1a=warm_ok, warm_survival_100=surv, q1c=bool(surv >= 0.95),
              reset_type="cold" if cold_ok else ("warm" if warm_ok else None))
    q1["pass"] = bool(q1["q1a"] and q1["q1c"] and q1["reset_type"] is not None)
    f = r["walking"]["fidelity"]
    q1b = dict(**f, pass_=bool(f["vxy_rmse_25"] <= 0.25 and f["vxy_rmse_50"] <= 0.40 and f["tilt_rmse_50"] <= 5.0))
    q2 = dict(detected_share=m(r["probe"]["falls_detected_share"]), falls=r["probe"]["falls"])
    q2["pass"] = bool(q2["detected_share"] >= 0.5)
    q3 = dict(false_stop_share=m(r["probe"]["surviving_false_stop_share"]), surviving=r["probe"]["surviving"])
    q3["pass"] = bool(q3["false_stop_share"] <= 0.10)
    q1b["pass"] = q1b.pop("pass_")
    return dict(Q1=q1, Q1b=q1b, Q2=q2, Q3=q3, D2_pass=bool(q1["pass"] and q1b["pass"] and q2["pass"] and q3["pass"]))


def run(wm, norm_path, round0, r0_withheld, probe, out, K=100, start_every=50, device="cpu"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    torch.set_num_threads(8)
    sd, H, norm, diff = rp.load_model(wm, norm_path, device)
    tr0, _ = load(round0)
    held = rp.heldout(tr0, r0_withheld, 0.1, 101)
    walk = [rp.Episode(tr0, h, norm) for h in held if h["kind"] != "fall"]
    rows = [(i, j) for i, ep in enumerate(walk) for j in range(H, ep.L)]
    P1, E1 = rp.windowed(sd, walk, rows, 1, device)
    P32, E32 = rp.windowed(sd, walk, rows, H, device)
    starts = [(i, j) for i, ep in enumerate(walk) for j in range(H, ep.L - K + 1, start_every)]
    Pw, Ew, kept = rp.rollouts(sd, walk, starts, K, H, "warm", device, keep=(25, 50))
    raw = [((ep.S * norm["state_std"]) + norm["state_mean"]) for ep in walk]
    nan2 = (float("nan"), float("nan"))
    v25, _ = fidelity(kept[25], starts, raw, 25, norm) if len(kept.get(25, [])) else nan2
    v50, t50 = fidelity(kept[50], starts, raw, 50, norm) if len(kept.get(50, [])) else nan2
    trp, meta_p = load(probe)
    peps = probe_episodes(trp, norm)
    det, fs, nf, ns = [], [], 0, 0
    for ep in peps:
        P, _ = rp.episode_context(sd, ep, device)
        if ep.kind == "fall":
            nf += 1
            det.append(np.nan_to_num(P[:, max(1, ep.L - 11):ep.L], nan=0.0).max(1) > 0.5)
        else:
            ns += 1
            fs.append(np.nan_to_num(P[:, 1:], nan=0.0).max(1) > 0.5)
    E = len(sd.state_heads)
    r = dict(walking=dict(episodes=len(walk), rows=len(rows), k1=rp.head_stats(P1, E1), k32=rp.head_stats(P32, E32),
                          warm_rollouts=dict(starts=len(starts), survival=rp.survival(Pw > 0.5, rp.SURVIVAL_AT),
                                             mse_by_step=rp.by_bin(Ew, K)),
                          fidelity=dict(vxy_rmse_25=v25, vxy_rmse_50=v50, tilt_rmse_50=t50)),
             probe=dict(episodes=len(peps), falls=nf, surviving=ns,
                        falls_detected_share=[float(x) for x in (np.mean(det, 0) if det else np.full(E, np.nan))],
                        surviving_false_stop_share=[float(x) for x in (np.mean(fs, 0) if fs else np.full(E, np.nan))],
                        friction=(meta_p.get("friction_shift") or {}).get("factor")))
    r["rules"] = decide(r)
    report = dict(protocol="A4 D2 qualification (rules fixed before the model was read)", wm=str(wm), wm_sha256=sha256(wm),
                  normalizer_sha256=sha256(norm_path), normalizer_max_diff_to_checkpoint=diff, round0_sha256=sha256(round0),
                  probe_sha256=sha256(probe), results=r, wall_s=time.time() - t0, analysis_sha256=sha256(__file__))
    (out / "d2_qualify.json").write_text(json.dumps(report, indent=1, default=float))
    q = r["rules"]
    lines = ["A4 D2 qualification (read-only; rules fixed before the model was read)",
             f"model {Path(wm).name} ({report['wm_sha256'][:12]}); walking: {len(walk)} held-out nominal episodes, {len(rows)} rows; "
             f"probe: {nf} falls, {ns} surviving episodes (friction x{r['probe']['friction']})",
             f"Q1  stop share k32 {q['Q1']['k32']:.4f}, k1 (cold) {q['Q1']['k1']:.4f}; warm survival to 100 "
             f"{q['Q1']['warm_survival_100']:.3f}; reset type {q['Q1']['reset_type']} -> {'PASS' if q['Q1']['pass'] else 'FAIL'}",
             f"Q1b vxy RMSE {q['Q1b']['vxy_rmse_25']:.3f} (step 25), {q['Q1b']['vxy_rmse_50']:.3f} m/s (step 50); tilt RMSE "
             f"{q['Q1b']['tilt_rmse_50']:.2f} deg (step 50) -> {'PASS' if q['Q1b']['pass'] else 'FAIL'}",
             f"Q2  falls detected (p > 0.5 in the last 11 rows) {q['Q2']['detected_share']:.3f} -> {'PASS' if q['Q2']['pass'] else 'FAIL'}",
             f"Q3  surviving episodes with a false stop {q['Q3']['false_stop_share']:.3f} -> {'PASS' if q['Q3']['pass'] else 'FAIL'}",
             f"D2 {'PASS' if q['D2_pass'] else 'FAIL'}"]
    text = "\n".join(lines) + "\n"
    (out / "d2_qualify.txt").write_text(text)
    print(text, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", type=Path, required=True)
    p.add_argument("--norm", type=Path, required=True, help="the D2 pooled normalizer (data/go2_norm.npz)")
    p.add_argument("--round0", type=Path, required=True)
    p.add_argument("--r0-withheld", type=Path, required=True, help="data/r0/withheld.json")
    p.add_argument("--probe", type=Path, required=True, help="the D1 probe (never fitted)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    run(a.wm, a.norm, a.round0, a.r0_withheld, a.probe, a.out, device=a.device)


if __name__ == "__main__":
    main()
