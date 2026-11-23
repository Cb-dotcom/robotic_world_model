#!/usr/bin/env python3
"""A4 D6 follow-up (23 Nov 2026): does the stale torque in the reset row matter? Read-only, CPU.
Fixed before it was run (run-log 22 Nov: every reset row after an earlier episode carries that episode's last torque).

Same episodes, starts and arms as the Q4 startup check (go2_d6_startup_check: from the physical reset, zero memory,
recorded actions, free and recorded arms). Two runs per source:
  A  reset rows as recorded
  B  the reset row's torque block (state dims 33..44) set to 0 N m (raw), everything else unchanged
Reported per source: Q4's statistics for A and B (velocity and tilt RMSE at steps 10 and 25, false stops, startup falls
caught), the mean absolute difference between A's and B's free-arm predictions of base velocity xy (m/s) and tilt (deg)
at steps 1, 10 and 25, and the largest stop-probability difference.
Reading (fixed): the stale torque MATTERS if, on a Q4 rule source, the step-25 velocity RMSE changes by more than
0.025 m/s or the tilt RMSE by more than 0.5 deg (10 % of each Q4 bar) between A and B, or a Q4 rule outcome changes;
otherwise it is NEGLIGIBLE.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MB = HERE.parent
for p in (str(HERE), str(MB)):
    if p not in sys.path:
        sys.path.insert(0, p)

import go2_d6_startup_check as sc  # noqa: E402
from go2_d2_qualify import GRAV, VXY, load, tilt_deg  # noqa: E402
from pre_reset_transitions import sha256  # noqa: E402

TORQUE = slice(33, 45)
STEPS = (1, 10, 25)
LIMITS = dict(vxy=0.025, tilt=0.5)


def zero_torque(eps, norm):
    out = copy.deepcopy(eps)
    z = (0.0 - norm["state_mean"][TORQUE]) / norm["state_std"][TORQUE]
    for e in out:
        e.R = e.R.copy()
        e.R[TORQUE] = z
    return out


def prediction_diff(pred_a, pred_b, eps, norm):
    res = {}
    for k in STEPS:
        idx = [i for i, e in enumerate(eps) if e.L >= k]
        if not idx:
            res[f"step{k}"] = dict(n=0)
            continue
        a = pred_a[idx, :, k - 1] * norm["state_std"] + norm["state_mean"]
        b = pred_b[idx, :, k - 1] * norm["state_std"] + norm["state_mean"]
        res[f"step{k}"] = dict(n=len(idx), vxy=float(np.mean(np.linalg.norm(a[..., VXY] - b[..., VXY], axis=-1))),
                               tilt=float(np.mean(np.abs(tilt_deg(a[..., GRAV]) - tilt_deg(b[..., GRAV])))))
    return res


def compare(sd, eps, norm, device="cpu"):
    eb = zero_torque(eps, norm)
    pa, ra, preda = sc.startup_rollouts(sd, eps, device)
    pb, rb, predb = sc.startup_rollouts(sd, eb, device)
    side = lambda p, r, pred, e: dict(
        fidelity_step25=sc.fidelity(pred, e, sc.FID_STEP, norm), fidelity_step10=sc.fidelity(pred, e, 10, norm),
        false_stop_free=sc.false_stops(p, e), startup_falls_free=sc.fall_scores(p, e, 0, sc.FID_STEP),
        startup_falls_recorded=sc.fall_scores(r, e, 0, sc.FID_STEP))
    a, b = side(pa, ra, preda, eps), side(pb, rb, predb, eb)
    for s in (a, b):
        for k in ("startup_falls_free", "startup_falls_recorded"):
            s[k].pop("leads", None)
    return dict(A_recorded=a, B_zero_torque=b, prediction_diff_free=prediction_diff(preda, predb, eps, norm),
                max_stop_prob_diff_free=float(np.max(np.abs(pa - pb))) if len(eps) else float("nan"))


def reading(res, rule):
    hits = []
    for name in rule:
        a, b = res[name]["A_recorded"], res[name]["B_zero_torque"]
        dv = abs(a["fidelity_step25"]["vxy"] - b["fidelity_step25"]["vxy"])
        dt = abs(a["fidelity_step25"]["tilt"] - b["fidelity_step25"]["tilt"])
        va = sc.verdict(dict(fidelity_step25=a["fidelity_step25"], false_stop_free=a["false_stop_free"]))
        vb = sc.verdict(dict(fidelity_step25=b["fidelity_step25"], false_stop_free=b["false_stop_free"]))
        flip = any(va[k] != vb[k] for k in ("vxy", "tilt", "false_stop"))
        if dv > LIMITS["vxy"] or dt > LIMITS["tilt"] or flip:
            hits.append(dict(source=name, d_vxy=dv, d_tilt=dt, outcome_changed=flip))
    return dict(matters=bool(hits), hits=hits)


def run(wm, norm_path, rule_sources, descriptive, out, device="cpu"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    sd, _, norm, diff = sc.rp.load_model(wm, norm_path, device)
    res = {}
    for name, (path, only) in list(rule_sources.items()) + list(descriptive.items()):
        tr, _ = load(path)
        res[name] = compare(sd, sc.start_episodes(tr, norm, only), norm, device)
    rep = dict(protocol="A4 D6 stale-torque sensitivity (reading fixed before the run)", wm=str(wm), wm_sha256=sha256(wm),
               normalizer_sha256=sha256(norm_path), results=res, reading=reading(res, list(rule_sources)),
               limits=LIMITS, analysis_sha256=sha256(__file__))
    (out / "torque_sensitivity.json").write_text(json.dumps(rep, indent=1, default=float))
    L = [f"A4 D6 stale-torque sensitivity (A: reset rows as recorded; B: reset-row torque 0); model {rep['wm_sha256'][:12]}"]
    for name, r in res.items():
        a, b, d = r["A_recorded"], r["B_zero_torque"], r["prediction_diff_free"]
        L.append(f"[{name}] step 25 vxy RMSE A {a['fidelity_step25']['vxy']:.3f} / B {b['fidelity_step25']['vxy']:.3f} m/s, "
                 f"tilt A {a['fidelity_step25']['tilt']:.2f} / B {b['fidelity_step25']['tilt']:.2f} deg; false stops A "
                 f"{a['false_stop_free']['share']:.3f} / B {b['false_stop_free']['share']:.3f}; startup falls caught (free) A "
                 f"{a['startup_falls_free']['caught']:.3f} / B {b['startup_falls_free']['caught']:.3f}")
        L.append("  |A - B| free predictions: " + "; ".join(
            f"step {k}: vxy {d[f'step{k}'].get('vxy', float('nan')):.4f} m/s, tilt {d[f'step{k}'].get('tilt', float('nan')):.3f} deg"
            for k in STEPS) + f"; max stop-probability difference {r['max_stop_prob_diff_free']:.4f}")
    L.append(f"STALE TORQUE: {'MATTERS' if rep['reading']['matters'] else 'NEGLIGIBLE'}")
    (out / "torque_sensitivity.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", type=Path, required=True)
    p.add_argument("--norm", type=Path, required=True)
    p.add_argument("--probe", action="append", default=[])
    p.add_argument("--withheld", action="append", default=[])
    p.add_argument("--descriptive", action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    rule = {s.split("=", 1)[0]: (Path(s.split("=", 1)[1]), None) for s in a.probe}
    for s in a.withheld:
        name, rest = s.split("=", 1)
        trace, held = rest.split(":", 1)
        rule[name] = (Path(trace), {(h["env"], h["first"]) for h in json.loads(Path(held).read_text())})
    desc = {s.split("=", 1)[0]: (Path(s.split("=", 1)[1]), None) for s in a.descriptive}
    run(a.wm, a.norm, rule, desc, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
