# SPDX-License-Identifier: BSD-3-Clause
"""T11 / H2 actuator check: our policy's logged actions through the sim actuator model.

Input: one bag with /lowcmd (the joint targets actually sent) and /lowstate
(measured q, dq, tau_est). Both are put in sim joint order.

Sim actuator (Isaac Lab UNITREE_GO2_CFG, DCMotor; defaults below, override with flags):
  tau_pd  = kp * (q_target - q) + kd * (0 - dq)
  DCMotor clip: v = clip(dq, +-v_lim * (1 + effort_lim / sat))
                top = min(sat * (1 - v / v_lim),  effort_lim)
                bot = max(sat * (-1 - v / v_lim), -effort_lim)
                tau_sim = clip(tau_pd, bot, top)
At each /lowstate sample, the target in force is the newest /lowcmd received at
least `lag` earlier. Per joint, lag is scanned (0 .. --lag-max-ms) and the lag with
the lowest torque RMSE is kept.

Per joint, at that lag:
  lag_ms            timing lag (torque fit)
  pos_lag_ms        lag that best aligns q with q_target (cross-check)
  gain, offset_nm   least squares tau_est = gain * tau_sim + offset
  rmse_nm           RMSE(tau_est - tau_sim)          (no fit; the headline error)
  rmse_fit_nm       RMSE after gain/offset
  bias_nm           mean(tau_est - tau_sim)
  pos_err_rad       mean(q_target - q)  (position offset); pos_rmse_rad
  dc_clip_rows      rows where the DCMotor clip changed the torque
  target_clip_rows  /lowcmd rows with |q_target| >= 2.5 rad - 1e-4 (deploy clip, SDK order)
Correlation is reported for reference only (the ticket does not accept it alone).

Gate (ticket T11): only logs from OUR policy at gains 25 / 0.5 count. The script
stops (exit 2) when the /lowcmd gains are not 25 / 0.5. --allow-other-gains runs it
anyway and marks the result "DOES NOT COUNT". Without --policy-sha256 the result is
also marked "DOES NOT COUNT" (policy unknown).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import go2_bag  # noqa: E402

KP_SIM, KD_SIM = 25.0, 0.5
EFFORT_LIM, SAT_EFFORT, VEL_LIM = 23.5, 23.5, 30.0
TARGET_CLIP_SDK = 2.5


def dc_motor(tau_pd, dq, effort_lim=EFFORT_LIM, sat=SAT_EFFORT, v_lim=VEL_LIM):
    v = np.clip(dq, -v_lim * (1 + effort_lim / sat), v_lim * (1 + effort_lim / sat))
    top = np.minimum(sat * (1.0 - v / v_lim), effort_lim)
    bot = np.maximum(sat * (-1.0 - v / v_lim), -effort_lim)
    return np.clip(tau_pd, bot, top)


def held_index(t_cmd, t_state, lag_s):
    """Index of the newest command received at or before t_state - lag (-1 = none)."""
    return np.searchsorted(t_cmd, t_state - lag_s, side="right") - 1


def joint_metrics(t_cmd, q_tgt, kp, kd, t_st, q, dq, tau, lags_s, model):
    best = None
    for lag in lags_s:
        i = held_index(t_cmd, t_st, lag)
        ok = i >= 0
        pd = kp[i[ok]] * (q_tgt[i[ok]] - q[ok]) - kd[i[ok]] * dq[ok]
        sim = dc_motor(pd, dq[ok], **model)
        rmse = float(np.sqrt(np.mean((tau[ok] - sim) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, lag, ok, pd, sim, i)
    rmse, lag, ok, pd, sim, i = best
    y = tau[ok]
    A = np.c_[sim, np.ones_like(sim)]
    (gain, off), *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = gain * sim + off
    # position lag: align q with the held target
    pos_best = None
    for plag in lags_s:
        j = held_index(t_cmd, t_st, plag)
        okp = j >= 0
        e = float(np.sqrt(np.mean((q_tgt[j[okp]] - q[okp]) ** 2)))
        if pos_best is None or e < pos_best[0]:
            pos_best = (e, plag)
    perr = q_tgt[i[ok]] - q[ok]
    corr = float(np.corrcoef(sim, y)[0, 1]) if np.std(sim) > 0 and np.std(y) > 0 else float("nan")
    return {
        "lag_ms": lag * 1e3,
        "pos_lag_ms": pos_best[1] * 1e3,
        "gain": float(gain),
        "offset_nm": float(off),
        "rmse_nm": rmse,
        "rmse_fit_nm": float(np.sqrt(np.mean((y - fit) ** 2))),
        "bias_nm": float(np.mean(y - sim)),
        "tau_sim_std_nm": float(np.std(sim)),
        "pos_err_rad": float(np.mean(perr)),
        "pos_rmse_rad": float(np.sqrt(np.mean(perr ** 2))),
        "dc_clip_rows": int((np.abs(pd - sim) > 1e-6).sum()),
        "rows": int(ok.sum()),
        "corr_reference_only": corr,
    }


def gains_check(kp_all, kd_all, tol_kp, tol_kd):
    good = (np.abs(kp_all - KP_SIM) <= tol_kp) & (np.abs(kd_all - KD_SIM) <= tol_kd)
    return float(good.mean())


def load_bag(bag, t_start, t_end):
    lc = go2_bag.read_topic(bag, "/lowcmd")
    lo = go2_bag.read_topic(bag, "/lowstate")
    t0 = min(lc.t[0], lo.t[0])
    tc, ts = lc.t - t0, lo.t - t0
    sc = (tc >= t_start) & (tc <= t_end)
    ss = (ts >= t_start) & (ts <= t_end)
    m = go2_bag.SIM_FROM_SDK
    return {
        "t_cmd": tc[sc],
        "q_tgt_sdk": lc.cols["q"][sc],
        "q_tgt": lc.cols["q"][sc][:, m], "kp": lc.cols["kp"][sc][:, m], "kd": lc.cols["kd"][sc][:, m],
        "mode": lc.cols["mode"][sc][:, m],
        "t_st": ts[ss], "q": lo.cols["q"][ss][:, m], "dq": lo.cols["dq"][ss][:, m],
        "tau": lo.cols["tau_est"][ss][:, m],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy-sha256", default=None, help="sha256 (first 12 chars ok) of the policy that ran")
    ap.add_argument("--t-start", type=float, default=0.0, help="s from bag start; use the policy-control window")
    ap.add_argument("--t-end", type=float, default=float("inf"))
    ap.add_argument("--lag-max-ms", type=float, default=60.0)
    ap.add_argument("--lag-step-ms", type=float, default=2.0)
    ap.add_argument("--tol-kp", type=float, default=0.5)
    ap.add_argument("--tol-kd", type=float, default=0.05)
    ap.add_argument("--min-gain-share", type=float, default=0.99)
    ap.add_argument("--allow-other-gains", action="store_true")
    ap.add_argument("--effort-limit", type=float, default=EFFORT_LIM)
    ap.add_argument("--saturation-effort", type=float, default=SAT_EFFORT)
    ap.add_argument("--velocity-limit", type=float, default=VEL_LIM)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load_bag(args.bag, args.t_start, args.t_end)
    if len(d["t_cmd"]) < 10 or len(d["t_st"]) < 10:
        print("H2: too few /lowcmd or /lowstate rows in the window.")
        return 1
    share = gains_check(d["kp"], d["kd"], args.tol_kp, args.tol_kd)
    counts = True
    notes = []
    if share < args.min_gain_share:
        kp_med, kd_med = float(np.median(d["kp"])), float(np.median(d["kd"]))
        msg = (f"NOT ACTION DATA: only {100 * share:.1f} % of joint-commands at kp {KP_SIM:g} / kd {KD_SIM:g} "
               f"(median kp {kp_med:g}, kd {kd_med:g}). Unitree's controller runs 40 / 1.")
        if not args.allow_other_gains:
            print(msg + " Stop (use --allow-other-gains for a diagnostic that does not count).")
            (out / "h2_actuator.txt").write_text(msg + "\n")
            return 2
        counts = False
        notes.append(msg)
    if not args.policy_sha256:
        counts = False
        notes.append("policy sha256 not given: policy unknown.")

    model = {"effort_lim": args.effort_limit, "sat": args.saturation_effort, "v_lim": args.velocity_limit}
    lags = np.arange(0.0, args.lag_max_ms + 1e-9, args.lag_step_ms) * 1e-3
    tclip =(np.abs(d["q_tgt_sdk"]) >= TARGET_CLIP_SDK - 1e-4)[:, go2_bag.SIM_FROM_SDK]
    res = {}
    for j, name in enumerate(go2_bag.SIM_JOINT_NAMES):
        r = joint_metrics(d["t_cmd"], d["q_tgt"][:, j], d["kp"][:, j], d["kd"][:, j],
                          d["t_st"], d["q"][:, j], d["dq"][:, j], d["tau"][:, j], lags, model)
        r["target_clip_rows"] = int(tclip[:, j].sum())
        res[name] = r

    head = (f"H2 actuator check: {args.bag}\nwindow {args.t_start:g}..{args.t_end:g} s, "
            f"{len(d['t_cmd'])} /lowcmd rows, {len(d['t_st'])} /lowstate rows, "
            f"gains 25/0.5 share {100 * share:.1f} %, policy {args.policy_sha256 or 'UNKNOWN'}\n"
            f"sim model: DCMotor kp {KP_SIM:g} kd {KD_SIM:g} effort {args.effort_limit:g} "
            f"sat {args.saturation_effort:g} vlim {args.velocity_limit:g}\n"
            + ("COUNTS for H2.\n" if counts else "DOES NOT COUNT: " + " ".join(notes) + "\n"))
    lines = [head, f"{'joint':10s} {'lag ms':>6s} {'posLag':>6s} {'gain':>6s} {'off Nm':>7s} {'RMSE Nm':>8s} "
                   f"{'fitRMSE':>8s} {'bias':>6s} {'posErr':>7s} {'dcClip':>6s} {'tgtClip':>7s} {'corr*':>6s}"]
    for n, r in res.items():
        lines.append(f"{n:10s} {r['lag_ms']:6.0f} {r['pos_lag_ms']:6.0f} {r['gain']:6.3f} {r['offset_nm']:7.3f} "
                     f"{r['rmse_nm']:8.3f} {r['rmse_fit_nm']:8.3f} {r['bias_nm']:6.2f} {r['pos_err_rad']:7.4f} "
                     f"{r['dc_clip_rows']:6d} {r['target_clip_rows']:7d} {r['corr_reference_only']:6.3f}")
    lines.append("* correlation: reference only, not a result.")
    text = "\n".join(lines) + "\n"
    (out / "h2_actuator.txt").write_text(text)
    (out / "h2_actuator.json").write_text(json.dumps({"counts": counts, "notes": notes, "gain_share": share,
                                                      "policy_sha256": args.policy_sha256, "joints": res,
                                                      "settings": vars(args)}, indent=2, default=float))
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
