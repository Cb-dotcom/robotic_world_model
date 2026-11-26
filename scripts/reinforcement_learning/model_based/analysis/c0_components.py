#!/usr/bin/env python3
"""Disagreement vs error by component (draft rules 24 Nov 2026, "draft-cpu-rules-c0m-coverage" §2). Retrospective,
descriptive, CPU only. Where the heads agree, are they right? Do models fitted with failure data keep more disagreement
near falls, and do their heads miss together less often?

Windows (one set per robot, the same for every model of that robot):
  ANYmal  C0 windows (c0_battery.fall_windows + matched non-falls, seed 0) on the given traces (Batches A + C). A trace
          is used for a model only if its sha256 does not occur in that model's fit manifest (checked first; a model
          without a manifest gets every trace and is flagged). The default is the intersection over models, so every
          model reads the same windows.
  Go2     go2_c0_check windows (target row T = terminal row - K, 25 imagined steps before T; K = 10 and K = 0) on
          probes 911 + 912; matched surviving windows.
Arms: recorded context (memory restarted per target, u_ctx), carried recorded (memory carried, u_car, C0-M) and
imagined (each head drives its own path; per-window values are the mean over driving heads).
Depths: ANYmal d = 0, 8, 16, 25; Go2 the endpoint T of each K (d = 25) and d = 0.
Per state group (base linear velocity 3, angular velocity 3, gravity 3, joint position 12, joint velocity 12,
torque 12) and for the sum of the 45 dims:
  e         RMS error of the ensemble mean against the recorded next state over the group's dims, normalized units;
            physical RMS (state_std x normalized; m/s, rad/s, rad, N m) and, for gravity, the tilt error in degrees
  u         head spread (std over heads, ddof 1) summed over the group's dims
  coverage  share of the group's dims with |error| <= that dim's spread (1 sigma)
  joint miss  the recorded value lies outside the heads' min-max range in at least half of the group's dims AND the
            group error is above the same model's non-fall p90 at d = 0 (same arm)
  agree and wrong  u below the same model's non-fall median at that depth (same arm) AND e above the non-fall p90
            at that depth (same arm)
Predictions (written before the run; each reported held / not held; no verdict):
  P1  fall windows, d = 25, imagined arm: joint-miss share (sum) lower for models fitted with failure data:
      E1-V and E1-R2 below ETH; D6b below G0
  P2  for those models the imagined / recorded-context ratio of geometric-mean spread on falls is higher
  P3  for those models the C0 ratio H_fall / H_nonfall is lower (point estimates on these windows)
  P4  for ETH and G0, gravity has the largest joint-miss share of the six groups on fall windows (imagined, d = 25)
Common scales: normalized-unit comparisons between two models are on a common scale only if both use the same
normalizer (sha256 of the statistics); physical units and degrees are always common. The report lists both.
Intervals (T04, 26 Nov): every fall and non-fall value above, and fall minus non-fall, gets a 95 % interval from the
registered matched-pair episode-cluster bootstrap stratified by source (c0_battery.pair_bootstrap, the draws
c0m_memory.decompose uses; seed 0, 10,000 draws). The thresholds (non-fall p90 / median) stay at their point values.
With two or more sources each table is repeated per source (thresholds from that source's non-falls); "repeats" lists the fall minus non-fall values whose
interval excludes 0 with the same sign in every source (ANYmal: batch A and batch C).
Part B (C0 interval): per window h = -(mean over driving heads of log u_imag - log u_ctx) at the last depth (sum);
H fall, H non-fall, ratio (finite draws, as the registered rule) and fall minus non-fall with intervals, the pairs
used and excluded, beside ETH's registered values. Descriptive, no verdict.
Part C (head by head, components_heads_<robot>.{json,txt}, kept apart): recorded context (every head reads the same
32 recorded rows and actions) at each depth; per head and dim the prediction, observed value and signed error
(physical units), the ensemble spread; per window the share of the group's dims where every head misses in the
direction of the ensemble mean (same direction), and per head the advantage |mean - y| - |head - y| (normalized,
mean over the group's dims; > 0: the head is closer than the ensemble mean). Repeat rule, fixed before reading: a
head advantage counts only if the same head shows it with the same sign (interval excluding 0) in every source;
otherwise it is noise (the heads are interchangeable copies). Stop (termination) predictions are reported apart:
per head the stop probability by depth and, on falls at the last depth, stop predicted / tilt (>= 30 deg) predicted
but stop missed / tilt missed.
Review of the 26 Nov run (added; the earlier keys are unchanged):
  zero spread  log u of a zero spread is NaN (never a clamp); that pair is left out of every log-u mean and interval
            and counted (zero_spread_windows, zero_spread_pairs_excluded). Same direction needs a nonzero ensemble error.
  unstable denominator  as c0m_memory: when the non-fall H interval includes 0 or a bootstrap denominator is <= 0, the
            ratio prints UNSTABLE DENOMINATOR and the fall - non-fall difference leads; P3 (written on the ratio) is
            then not computable (None) for that pair; P3_difference is reported apart, descriptive.
  which side  (which_side, text "SIDE") mean log u_ctx and log u_imag at d16 / d25, fall and non-fall, their
            differences and e_phys of each arm, pooled and per source.
  leave one head out  (heads.<model>.<scope>.leave_one_out, text "LOO") the ensemble without each head in turn.
  The head file starts with the COUNT lines and the per-source advantage at the last depth.
Limits: the models differ in more than failure data (data volume, recipe, seed; G0 has no friction data); ETH's
training data are unknown; the windows were seen before; associations, not causes. Eligibility is by trace sha256:
a trace in a model's fit manifest is excluded whole (envs held out of a fit are not recovered).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MB = HERE.parent
for p in (str(HERE), str(MB)):
    if p not in sys.path:
        sys.path.insert(0, p)

import c0_battery as c0  # noqa: E402
from pre_reset_transitions import sha256, validate_trace  # noqa: E402

GROUPS = {"lin_vel": (0, 3), "ang_vel": (3, 6), "gravity": (6, 9), "joint_pos": (9, 21), "joint_vel": (21, 33),
          "torque": (33, 45), "sum": (0, 45)}
SIX = ("lin_vel", "ang_vel", "gravity", "joint_pos", "joint_vel", "torque")
ANYMAL_DEPTHS = (0, 8, 16, 25)
ARMS = ("ctx", "car", "imag")
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
METRICS = ("e", "e_phys", "log_u", "coverage", "joint_miss", "agree_wrong")
TILT_DEG = 30.0
# registered C0 (c0_report.json 89f9d691, rerun 2, 18 Nov): pooled and per source. With the same trace names the ETH
# rows of this script read the same windows and pair draws, so its per-source rows should reproduce these.
REGISTERED_ETH = dict(record="c0_report.json 89f9d691 (rerun 2, 18 Nov)", by_source={
    "pooled (batchA + batchC + round0_1701)": dict(pairs=206, H_fall=1.2415, H_nonfall=0.4757, ratio=2.61, ratio_ci=[2.14, 3.28]),
    "batchA": dict(pairs=80, H_fall=1.7183, H_nonfall=0.5390, ratio=3.19, ratio_ci=[2.48, 4.59]),
    "batchC": dict(pairs=53, H_fall=0.7958, H_nonfall=0.6766, ratio=1.18, ratio_ci=[0.80, 1.86]),
    "round0_1701": dict(pairs=73, H_fall=1.0425, H_nonfall=0.2606, ratio=4.00, ratio_ci=[2.73, 6.51])})


# ---------------------------------------------------------------- windows in one layout for both robots
def anymal_windows(trace, contract, seed=0, name="trace"):
    """Fall and matched non-fall windows of one trace, as C0: Xs/Xa [W, 31 + H, .] normalized (index i is row t-31+i),
    Y [W, H, 45] normalized next states, clusters [W] ("name/env/episode", as c0_battery), strata [W] (= name)."""
    data = {k: trace[k] for k in ("state", "action", "next_state", "command", "done", "terminated", "time_out")}
    falls, _ = c0.fall_windows(data)
    nonfalls, _ = c0.match_nonfalls(data, falls, c0.nonfall_candidates(data["done"]), seed)
    st = contract.statistics
    ep = c0.episode_index(data["done"])
    out = {}
    for kind, o in (("fall", falls), ("nonfall", nonfalls)):
        xs = (torch.from_numpy(c0.gather(data["state"], o, -31, c0.HORIZON)).float() - st["state_mean"]) / st["state_std"]
        xa = (torch.from_numpy(c0.gather(data["action"], o, -31, c0.HORIZON)).float() - st["action_mean"]) / st["action_std"]
        y = (torch.from_numpy(c0.gather(data["next_state"], o, 0, c0.HORIZON)).float() - st["state_mean"]) / st["state_std"]
        out[kind] = dict(xs=xs, xa=xa, y=y, origins=o, clusters=np.array([f"{name}/{n}/{ep[t, n]}" for t, n in o]),
                         strata=np.array([name] * len(o)))
    return out


def go2_windows(eps, k, depth=25, seed=0):
    """go2_c0_check windows in the same layout: start j = T - depth; Xs = S[j-32 .. j+depth-1], Xa = A[j-31 .. j+depth],
    Y = S[j .. T] (index 31 + m of Xs is the recorded state fed at step m + 1)."""
    import go2_c0_check as g
    fw, sw, counts = g.windows(eps, k, depth=depth, seed=seed)
    out = {}
    for kind, rows in (("fall", fw), ("nonfall", sw)):
        js = [(i, T - depth) for i, T in rows]
        xs = torch.from_numpy(np.stack([eps[i].S[j - 32:j + depth] for i, j in js]))
        xa = torch.from_numpy(np.stack([eps[i].A[j - 31:j + depth + 1] for i, j in js]))
        y = torch.from_numpy(np.stack([eps[i].S[j:j + depth + 1] for i, j in js]))
        out[kind] = dict(xs=xs, xa=xa, y=y, origins=np.array(rows), clusters=np.array([eps[i].cluster for i, _ in rows]),
                         strata=np.array([eps[i].probe for i, _ in rows]))
    return out, counts


# ---------------------------------------------------------------- the three arms with every head's mean
@torch.no_grad()
def arm_means(model, xs, xa, horizon, depths):
    """{arm: [P, len(depths), heads, W, 45]} head means at the requested depths. P = 1 for ctx and car, heads for imag.
    Also "ctx_stop" [len(depths), heads, W]: each head's termination probability on recorded context."""
    W = xs.shape[0]
    res, stop = {"ctx": [], "car": [], "imag": []}, []
    for k in depths:                                           # ctx: memory restarted for each target
        model.reset()
        means, _, terms, _ = c0.forward_all(model, xs[:, k:k + 32], xa[:, k:k + 32])
        res["ctx"].append(means.numpy())
        stop.append(torch.sigmoid(terms).squeeze(-1).numpy())
    res["ctx"] = np.stack(res["ctx"])[None]
    model.reset()
    x, a, car = xs[:, :32], xa[:, :32], []
    for k in range(horizon):
        means, _, _, _ = c0.forward_all(model, x, a)
        if k in depths:
            car.append(means.numpy())
        if k + 1 < horizon:
            x, a = xs[:, 32 + k:33 + k], xa[:, 32 + k:33 + k]     # the recorded row the imagined arm would feed
    res["car"] = np.stack(car)[None]
    paths = []
    for p in range(model.ensemble_size):
        model.reset()
        x, a, rec = xs[:, :32], xa[:, :32], []
        for k in range(horizon):
            means, _, _, _ = c0.forward_all(model, x, a)
            if k in depths:
                rec.append(means.numpy())
            if k + 1 < horizon:
                x, a = means[p][:, None], xa[:, 32 + k:32 + k + 1]
        paths.append(np.stack(rec))
    model.reset()
    res["imag"] = np.stack(paths)
    assert all(v.shape[3] == W for v in res.values())
    res["ctx_stop"] = np.stack(stop)
    return res


# ---------------------------------------------------------------- metrics
def tilt_error_deg(pred, target, stats):
    g = lambda x: x[..., 6:9] * stats["state_std"][6:9] + stats["state_mean"][6:9]
    t = lambda v: np.degrees(np.arctan2(np.linalg.norm(v[..., :2], axis=-1), -v[..., 2]))
    return np.abs(t(g(pred)) - t(g(target)))


def window_metrics(means, target, stats):
    """means [heads, W, 45], target [W, 45] (normalized; stats as numpy) -> per group arrays [W]:
    e (RMS normalized), e_phys (RMS physical), u (summed spread), coverage, outside (share of dims outside the heads'
    min-max), and tilt_deg for gravity."""
    mean = means.mean(0)
    spread = means.std(0, ddof=1)
    spread[(means == means[:1]).all(0)] = 0                # identical heads: exactly 0, not float noise (~1e-15)
    err = mean - target
    lo, hi = means.min(0), means.max(0)
    outside = (target < lo) | (target > hi)
    phys = err * stats["state_std"]
    out = {}
    for name, (a, b) in GROUPS.items():
        out[name] = dict(e=np.sqrt((err[:, a:b] ** 2).mean(-1)), e_phys=np.sqrt((phys[:, a:b] ** 2).mean(-1)),
                         u=spread[:, a:b].sum(-1), coverage=(np.abs(err[:, a:b]) <= spread[:, a:b]).mean(-1),
                         outside=outside[:, a:b].mean(-1))
    out["gravity"]["tilt_deg"] = tilt_error_deg(mean, target, stats)
    return out


def arm_metrics(arm, target, stats):
    """arm [P, D, heads, W, 45] -> {depth index: {group: {metric: [P, W]}}}; target [W, D, 45]."""
    res = {}
    for di in range(arm.shape[1]):
        per = [window_metrics(arm[p, di], target[:, di], stats) for p in range(arm.shape[0])]
        res[di] = {g: {m: np.stack([x[g][m] for x in per]) for m in per[0][g]} for g in per[0]}
    return res


def log_spread(u):
    """log of a summed spread. A zero spread (identical heads) gives NaN, never a clamp or -inf: the window is left out
    of every log-u mean with its matched pair and counted (zero_spread_*). Coverage and agree-and-wrong need no log: a
    zero-spread dim is covered only by an exact hit, and u = 0 is below any median."""
    u = np.asarray(u)
    out = np.log(np.where(u > 0, u, 1))
    out[u <= 0] = np.nan
    return out


def _pair_boot_finite(f, n, boot):
    """c0.pair_bootstrap on the pairs whose fall and non-fall values are both finite: the same draws as the plain call
    when every pair is kept. Returns (fall draws, non-fall draws, kept mask); NaN draws when no pair is kept."""
    strata, cf, cn, n_boot, seed = boot
    ok = np.isfinite(f) & np.isfinite(n)
    if not ok.any():
        return np.full(n_boot, np.nan), np.full(n_boot, np.nan), ok
    bf, bn = c0.pair_bootstrap(f[ok], n[ok], np.asarray(strata)[ok], np.asarray(cf)[ok], np.asarray(cn)[ok], n_boot, seed)
    return bf, bn, ok


def _columns(x, p90_e0, med_u, p90_e):
    """Per-window values [W, K] of METRICS (+ tilt_deg): mean over paths of [P, W] metrics (log u NaN at zero spread)."""
    cols = [x["e"], x["e_phys"], log_spread(x["u"]), x["coverage"],
            (x["outside"] >= 0.5) & (x["e"] > p90_e0), (x["u"] < med_u) & (x["e"] > p90_e)]
    if "tilt_deg" in x:
        cols.append(x["tilt_deg"])
    return np.stack([np.asarray(c, float).mean(0) for c in cols], axis=1)


def shares(fall, nonfall, d0_nonfall, boot=None):
    """Joint-miss and agree-and-wrong shares per group from per-window metrics ([P, W] arrays; mean over paths).
    boot = (strata, fall clusters, non-fall clusters, n_boot, seed) adds 95 % intervals ("ci") from the registered
    matched-pair bootstrap: each metric for falls and non-falls, and fall minus non-fall ("<metric>_diff")."""
    out = {}
    for gname in GROUPS:
        f, n, z = fall[gname], nonfall[gname], d0_nonfall[gname]
        p90_e0 = np.percentile(z["e"], 90)
        med_u, p90_e = np.median(n["u"]), np.percentile(n["e"], 90)
        cf, cn = _columns(f, p90_e0, med_u, p90_e), _columns(n, p90_e0, med_u, p90_e)
        okf, okn = np.isfinite(cf[:, 2]), np.isfinite(cn[:, 2])            # log u: windows without a zero spread
        if boot is not None:                                                # matched pairs: both members leave
            okf = okn = okf & okn
        pf, pn = np.nan_to_num(cf).mean(0), np.nan_to_num(cn).mean(0)
        pf[2] = cf[okf, 2].mean() if okf.any() else np.nan
        pn[2] = cn[okn, 2].mean() if okn.any() else np.nan
        out[gname] = row = dict(joint_miss_fall=float(pf[4]), joint_miss_nonfall=float(pn[4]),
                                agree_wrong_fall=float(pf[5]), agree_wrong_nonfall=float(pn[5]),
                                e_fall=float(pf[0]), e_nonfall=float(pn[0]),
                                e_phys_fall=float(pf[1]), e_phys_nonfall=float(pn[1]),
                                u_geomean_fall=float(np.exp(pf[2])), u_geomean_nonfall=float(np.exp(pn[2])),
                                coverage_fall=float(pf[3]), coverage_nonfall=float(pn[3]),
                                thresholds=dict(nonfall_p90_e_d0=float(p90_e0), nonfall_median_u=float(med_u),
                                                nonfall_p90_e=float(p90_e)),
                                zero_spread_windows=dict(fall=int((~np.isfinite(cf[:, 2])).sum()),
                                                         nonfall=int((~np.isfinite(cn[:, 2])).sum())),
                                zero_spread_pairs_excluded=int((~okf).sum()) if boot is not None else None)
        if "tilt_deg" in f:
            row["tilt_err_deg_fall"], row["tilt_err_deg_nonfall"] = float(pf[6]), float(pn[6])
        if boot is not None:
            bf, bn = c0.pair_bootstrap(np.nan_to_num(cf), np.nan_to_num(cn), *boot)
            if not okf.all():                                               # log u draws from the kept pairs only
                bf[:, 2], bn[:, 2], _ = _pair_boot_finite(cf[:, 2], cn[:, 2], boot)
            names = METRICS + ("tilt_deg",) * (cf.shape[1] > len(METRICS))
            row["ci"] = {f"{m}_{kind}": c0.ci(b[:, k]) for k, m in enumerate(names) for kind, b in (("fall", bf), ("nonfall", bn))}
            row["ci"].update({f"{m}_diff": c0.ci(bf[:, k] - bn[:, k]) for k, m in enumerate(names)})
            for kind in ("fall", "nonfall"):
                row["ci"][f"u_geomean_{kind}"] = [float(np.exp(v)) for v in row["ci"][f"log_u_{kind}"]]
    return out


def window_h(u_ctx, u_imag):
    """Per-window C0 deficit h = -(mean over driving heads of log u_imag - log u_ctx); u_ctx [W], u_imag [P, W].
    NaN where a spread is 0 (log_spread)."""
    return -(log_spread(u_imag).mean(0) - log_spread(u_ctx))


def h_ratio(u_ctx_f, u_imag_f, u_ctx_n, u_imag_n):
    """C0 point estimate on these windows: H = mean over windows of window_h, on the pairs without a zero spread."""
    hf, hn = window_h(u_ctx_f, u_imag_f), window_h(u_ctx_n, u_imag_n)
    ok = np.isfinite(hf) & np.isfinite(hn)
    hf, hn = (float(hf[ok].mean()), float(hn[ok].mean())) if ok.any() else (float("nan"),) * 2
    return dict(H_fall=hf, H_nonfall=hn, ratio=hf / hn if hn != 0 else float("nan"),
                zero_spread_pairs_excluded=int((~ok).sum()))


def c0_interval(h_fall, h_nonfall, strata, clusters_fall, clusters_nonfall, n_boot=10_000, seed=0):
    """Part B. Per-window C0 deficit h [W] of falls and their matched non-falls -> H fall, H non-fall, ratio and fall
    minus non-fall with 95 % intervals from the registered matched-pair bootstrap (c0_battery.pair_bootstrap, the
    draws of c0m_memory.decompose). Ratio interval as the registered rule: the finite draws of H_fall / H_nonfall.
    Pairs with a zero spread (h NaN) are left out and counted. unstable_denominator (c0m_memory convention, review
    26 Nov): the non-fall H interval includes 0, or some draw of the denominator is <= 0; then the ratio is not read
    (text: UNSTABLE DENOMINATOR) and the fall - non-fall difference is the number to read."""
    h_fall, h_nonfall = np.asarray(h_fall), np.asarray(h_nonfall)
    bf, bn, ok = _pair_boot_finite(h_fall, h_nonfall, (strata, clusters_fall, clusters_nonfall, n_boot, seed))
    hf, hn = (float(np.mean(h_fall[ok])), float(np.mean(h_nonfall[ok]))) if ok.any() else (float("nan"),) * 2
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = bf / bn
    fin = np.isfinite(rb)
    lo, hi = c0.ci(bn)
    nonpos = int((bn <= 0).sum())
    comps = c0.pair_components(np.asarray(clusters_fall)[ok], np.asarray(clusters_nonfall)[ok]) if ok.any() else []
    return dict(H_fall=hf, H_fall_ci=c0.ci(bf), H_nonfall=hn, H_nonfall_ci=[lo, hi],
                ratio=hf / hn if hn != 0 else float("nan"),
                ratio_ci=c0.ci(rb[fin]) if fin.any() else [float("nan")] * 2, nonfinite_ratio_draws=int((~fin).sum()),
                difference=hf - hn, difference_ci=c0.ci(bf - bn), nonfall_ci_includes_0=bool(lo <= 0 <= hi),
                unstable_denominator=bool(lo <= 0 <= hi or nonpos > 0 or not ok.any()),
                nonpositive_denominator_draws=nonpos, zero_spread_pairs_excluded=int((~ok).sum()),
                pairs=int(ok.sum()), pair_components=int(len(np.unique(comps))), n_boot=n_boot, seed=seed)


def _tilt(g):
    return np.degrees(np.arctan2(np.linalg.norm(g[..., :2], axis=-1), -g[..., 2]))


def head_readout(means, stop, target, stats, depths, boot):
    """Part C on recorded context. means {kind: [D, heads, W, 45]} and target {kind: [W, D, 45]} normalized; stop {kind:
    [D, heads, W]} termination probability; boot as in shares. Per depth and group (see the module docstring)."""
    std, mu = stats["state_std"], stats["state_mean"]
    out = {}
    for di, d in enumerate(depths):
        row = {}
        for g, (a, b) in GROUPS.items():
            cols, desc = {}, {}
            for kind in ("fall", "nonfall"):
                p = means[kind][di][..., a:b].astype(np.float64)                     # [heads, W, dims]
                y = target[kind][:, di, a:b].astype(np.float64)                      # [W, dims]
                err_m, err_h = p.mean(0) - y, p - y[None]
                # an exact hit of the ensemble mean is no miss (identical heads would otherwise count sign 0 == 0)
                same = ((np.sign(err_h) == np.sign(err_m)[None]).all(0) & (err_m != 0)).mean(-1)   # [W]
                adv = (np.abs(err_m)[None] - np.abs(err_h)).mean(-1).T               # [W, heads]
                cols[kind] = np.column_stack([adv, same])
                ph, yh = p * std[a:b] + mu[a:b], y * std[a:b] + mu[a:b]
                desc[kind] = dict(observed=yh.mean(0).tolist(), prediction=ph.mean(1).tolist(),
                                  signed_error=(ph - yh[None]).mean(1).tolist(),
                                  ensemble_signed_error=(ph.mean(0) - yh).mean(0).tolist(),
                                  spread=float(p.std(0, ddof=1).sum(-1).mean()), same_direction=float(same.mean()),
                                  advantage=adv.mean(0).tolist())
            bf, bn = c0.pair_bootstrap(cols["fall"], cols["nonfall"], *boot)
            heads = cols["fall"].shape[1] - 1
            for kind, bb in (("fall", bf), ("nonfall", bn)):
                desc[kind]["advantage_ci"] = [c0.ci(bb[:, h]) for h in range(heads)]
                desc[kind]["same_direction_ci"] = c0.ci(bb[:, heads])
            row[g] = desc
        row["stop"] = {k: dict(mean_p=stop[k][di].mean(1).tolist(), share_above_half=(stop[k][di] > 0.5).mean(1).tolist())
                       for k in ("fall", "nonfall")}
        out[f"d{d}"] = row
    last = len(depths) - 1
    p, sl = stop["fall"][last], slice(6, 9)
    tilt_h = _tilt(means["fall"][last][..., sl] * std[sl] + mu[sl])                      # [heads, W]
    out["fall_last_depth"] = dict(
        depth=int(depths[last]), observed_tilt_share=float((_tilt(target["fall"][:, last, sl] * std[sl] + mu[sl]) >= TILT_DEG).mean()),
        stop_predicted=(p > 0.5).mean(1).tolist(),
        tilt_predicted_stop_missed=((tilt_h >= TILT_DEG) & (p <= 0.5)).mean(1).tolist(),
        tilt_missed=((tilt_h < TILT_DEG) & (p <= 0.5)).mean(1).tolist())
    return out


def leave_one_out(arms, target, stats, depths, boot, warn_depths=(16, 25)):
    """Part C, leave one head out (review 26 Nov). arms {kind: {"ctx": [1, D, heads, W, 45], "imag": [heads, D, heads,
    W, 45]}}, target {kind: [W, D, 45]} normalized. Dropping head h is exact in both arms: on recorded context every
    head reads the same rows, and imagined path p is driven by head p alone, so the 4-head ensemble keeps paths p != h
    and heads != h. Per set ("all" = every head, "without_h<i>"): ensemble-mean physical RMS error on recorded context
    (per group and depth, fall and non-fall); warning quality at d16 and d25 = Part B's fall - non-fall H with its
    interval (c0_interval), and the AUROC of the summed spread u_ctx for falls vs non-falls (go2_c0_check.auroc,
    average ranks for ties). A better ensemble mean can still give a worse warning."""
    from go2_c0_check import auroc
    n = arms["fall"]["ctx"].shape[2]
    out = {}
    for drop in [None] + list(range(n)):
        keep = [i for i in range(n) if i != drop]
        m = {k: dict(ctx=arm_metrics(a["ctx"][:, :, keep], target[k], stats),
                     imag=arm_metrics(a["imag"][keep][:, :, keep], target[k], stats)) for k, a in arms.items()}
        row = dict(e_phys={f"d{d}": {g: {k: float(m[k]["ctx"][di][g]["e_phys"].mean()) for k in m} for g in GROUPS}
                           for di, d in enumerate(depths)})
        for di, d in enumerate(depths):
            if d in warn_depths:
                u = {k: m[k]["ctx"][di]["sum"]["u"][0] for k in m}
                b = c0_interval(*(window_h(u[k], m[k]["imag"][di]["sum"]["u"]) for k in ("fall", "nonfall")), *boot)
                row[f"warning_d{d}"] = dict({k: b[k] for k in ("difference", "difference_ci", "H_fall", "H_nonfall",
                                                                "pairs", "zero_spread_pairs_excluded")},
                                            auroc_u_ctx=auroc(u["fall"], u["nonfall"]))
        out["all" if drop is None else f"without_h{drop}"] = row
    return out


def which_side(table, depths, at=(16, 25)):
    """Which side moved (review 26 Nov), sum group: per depth, for falls and non-falls, mean log u on recorded context
    (ctx) and in imagination (imag), each with its fall - non-fall difference, H = log u_ctx - log u_imag (points),
    and the ensemble-mean physical RMS error of each arm on the same windows. Read from the component tables: the
    same pair draws, no new bootstrap."""
    out = {}
    for d in [d for d in at if d in depths]:
        row = {}
        for arm in ("ctx", "imag"):
            r = table[f"{arm}/d{d}"]["sum"]
            c = r.get("ci", {})
            row[arm] = {f"{m}_{k}": float(np.log(r[f"u_geomean_{k}"])) if m == "log_u" else r[f"{m}_{k}"]
                        for m in ("log_u", "e_phys") for k in ("fall", "nonfall")}
            for m in ("log_u", "e_phys"):
                row[arm][f"{m}_diff"] = row[arm][f"{m}_fall"] - row[arm][f"{m}_nonfall"]
                row[arm].update({f"{m}_{k}_ci": c[f"{m}_{k}"] for k in ("fall", "nonfall", "diff") if f"{m}_{k}" in c})
            row[arm].update(u_geomean_fall=r["u_geomean_fall"], u_geomean_nonfall=r["u_geomean_nonfall"],
                            zero_spread_pairs_excluded=r["zero_spread_pairs_excluded"])
        for k in ("fall", "nonfall"):
            row[f"H_{k}"] = row["ctx"][f"log_u_{k}"] - row["imag"][f"log_u_{k}"]
        out[f"d{d}"] = row
    return out


def _sign(interval):
    return 1 if interval[0] > 0 else (-1 if interval[1] < 0 else 0)


def _get(table, path):
    for k in path:
        table = table[k]
    return table


def repeats(per_source, items):
    """items {tag: path to a 95 % interval}. A tag repeats if its interval excludes 0 with the same sign in every source
    (two or more); a sign in some sources only is listed apart (noise by the repeat rule)."""
    out = dict(sources=list(per_source), repeated={}, some_sources_only=[])
    if len(per_source) < 2:
        return out
    for tag, path in items.items():
        s = [_sign(_get(t, path)) for t in per_source.values()]
        if s[0] and all(x == s[0] for x in s):
            out["repeated"][tag] = "+" if s[0] > 0 else "-"
        elif any(s):
            out["some_sources_only"].append(tag)
    return out


def _take(x, sel):
    return {k: _take(v, sel) for k, v in x.items()} if isinstance(x, dict) else x[..., sel]


def _tables(m, windows, depths, sel, n_boot, seed):
    """Every table of one model on the windows selected by sel (a mask over pairs, or slice(None))."""
    m = _take(m, sel)
    boot = (windows["fall"]["strata"][sel], windows["fall"]["clusters"][sel], windows["nonfall"]["clusters"][sel],
            n_boot, seed)
    table = {}
    for arm in ARMS:
        for di, d in enumerate(depths):
            table[f"{arm}/d{d}"] = shares(m["fall"][arm][di], m["nonfall"][arm][di], m["nonfall"][arm][0], boot)
    last = len(depths) - 1
    u = lambda kind, arm: m[kind][arm][last]["sum"]["u"]
    table["c0_point"] = h_ratio(u("fall", "ctx")[0], u("fall", "imag"), u("nonfall", "ctx")[0], u("nonfall", "imag"))
    h = lambda kind: window_h(u(kind, "ctx")[0], u(kind, "imag"))
    table["c0_interval"] = dict(c0_interval(h("fall"), h("nonfall"), *boot), depth=int(depths[last]))
    table["imag_over_ctx_fall"] = float(np.exp(-np.nanmean(h("fall"))))      # fall windows without a zero spread
    table["windows"] = {k: int(len(windows[k]["clusters"][sel])) for k in ("fall", "nonfall")}
    table["which_side"] = which_side(table, depths)
    return table, boot


def model_table(model, stats_np, windows, horizon, depths, n_boot=10_000, seed=0):
    """All metrics of one model on one window set; with two or more sources also per source and the repeats.
    The Part C readout is under "heads" (written to its own file)."""
    m, ctx, stop, tgt, loo = {}, {}, {}, {}, {}
    for kind in ("fall", "nonfall"):
        w = windows[kind]
        arms = arm_means(model, w["xs"], w["xa"], horizon, depths)
        tgt[kind] = w["y"].numpy()[:, list(depths)]
        m[kind] = {arm: arm_metrics(arms[arm], tgt[kind], stats_np) for arm in ARMS}
        ctx[kind], stop[kind] = arms["ctx"][0], arms["ctx_stop"]
        loo[kind] = {a: arms[a] for a in ("ctx", "imag")}

    def scope(sel):
        table, boot = _tables(m, windows, depths, sel, n_boot, seed)
        t_sel = {k: v[sel] for k, v in tgt.items()}
        heads = head_readout({k: v[:, :, sel] for k, v in ctx.items()}, {k: v[..., sel] for k, v in stop.items()},
                             t_sel, stats_np, depths, boot)
        heads["leave_one_out"] = leave_one_out({k: {a: v[..., sel, :] for a, v in x.items()} for k, x in loo.items()},
                                               t_sel, stats_np, depths, boot)
        return table, heads

    table, heads = scope(slice(None))
    heads = {"pooled": heads}
    strata = windows["fall"]["strata"]
    sources = list(dict.fromkeys(strata.tolist()))
    if len(sources) > 1:
        per = {s: scope(strata == s) for s in sources}
        table["by_source"] = {s: v[0] for s, v in per.items()}
        heads["by_source"] = {s: v[1] for s, v in per.items()}
        table["repeats"] = repeats(table["by_source"], {
            f"{k}/{g}/{mt}": (k, g, "ci", f"{mt}_diff") for k in table if "/d" in k for g in GROUPS
            for mt in ("e_phys", "log_u", "coverage", "joint_miss", "agree_wrong")})
        heads["repeats"] = repeats(heads["by_source"], {
            f"d{d}/{g}/{kind}/head{h}": (f"d{d}", g, kind, "advantage_ci", h) for d in depths for g in GROUPS
            for kind in ("fall", "nonfall") for h in range(ctx["fall"].shape[1])})
    table["heads"] = heads
    return table


def predictions(tables, anymal_pairs=(("E1V", "ETH"), ("E1R2", "ETH")), go2_pairs=(("D6b", "G0"),), endpoint="imag/d25"):
    """P1-P4 held / not held (None when a model is absent)."""
    def jm(name, group="sum"):
        return tables[name][endpoint][group]["joint_miss_fall"] if name in tables else None
    out = {}
    pairs = [p for p in anymal_pairs + go2_pairs if p[0] in tables and p[1] in tables]
    out["P1"] = {f"{a}<{b}": jm(a) < jm(b) for a, b in pairs}
    out["P2"] = {f"{a}>{b}": tables[a]["imag_over_ctx_fall"] > tables[b]["imag_over_ctx_fall"] for a, b in pairs}
    # P3 was written on the ratio before the run, so it is not re-cast on the difference after seeing the data: with an
    # unstable denominator in either model it is not computable (None). The difference is reported apart (P3_difference).
    unstable = lambda n: tables[n]["c0_interval"]["unstable_denominator"]
    out["P3"] = {f"{a}<{b}": None if unstable(a) or unstable(b) else
                 tables[a]["c0_point"]["ratio"] < tables[b]["c0_point"]["ratio"] for a, b in pairs}
    out["P3_difference"] = {f"{a}<{b}": tables[a]["c0_interval"]["difference"] < tables[b]["c0_interval"]["difference"]
                            for a, b in pairs}
    out["P4"] = {}
    for name in ("ETH", "G0"):
        if name in tables:
            s = {g: jm(name, g) for g in SIX}
            out["P4"][name] = dict(largest=max(s, key=s.get), held=max(s, key=s.get) == "gravity", shares=s)
    for k in ("P1", "P2", "P3", "P3_difference"):
        v = list(out[k].values())
        out[k] = dict(per_pair=out[k], held=None if not v or None in v else all(v))
    out["P3"]["note"] = "None: not computable (unstable denominator); P3_difference is descriptive, added 26 Nov after the run"
    out["P4"]["held"] = all(v["held"] for v in out["P4"].values() if isinstance(v, dict)) if out["P4"] else None
    return out


def normalizer_id(stats_np):
    h = hashlib.sha256()
    for k in ("state_mean", "state_std", "action_mean", "action_std"):
        h.update(np.ascontiguousarray(stats_np[k], np.float32).tobytes())
    return h.hexdigest()


def manifest_hashes(path):
    """Every sha256-looking string anywhere in a fit manifest (JSON or text)."""
    return set(HEX64.findall(Path(path).read_text())) if path else set()


def common_scale(norm_ids):
    names = sorted(norm_ids)
    return {f"{a}~{b}": norm_ids[a] == norm_ids[b] for i, a in enumerate(names) for b in names[i + 1:]}


# ---------------------------------------------------------------- drivers
def run_anymal(models, traces, out, seed=0, use_intersection=True, n_boot=10_000):
    """models: [(name, wm path, fit manifest or None)]; traces: [(name, path)]."""
    from analyse_transition_fidelity import load_model
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    tsha = {n: sha256(p) for n, p in traces}
    data, pairs = {}, {}
    for tn, tp in traces:
        with np.load(tp, allow_pickle=False) as z:
            validate_trace(z)
            data[tn] = {k: z[k] for k in z.files if k != "metadata"}
        pairs[tn] = int(len(c0.fall_windows(data[tn])[0]))          # one matched non-fall per fall window
    eligible, flags = {}, {}
    for name, wm, man in models:
        used = manifest_hashes(man)
        eligible[name] = [n for n, _ in traces if tsha[n] not in used]
        excl = [n for n, _ in traces if tsha[n] in used]
        flags[name] = dict(manifest=str(man) if man else None, excluded=excl, excluded_pairs=sum(pairs[n] for n in excl),
                           no_manifest=man is None)
    common = [n for n, _ in traces if all(n in e for e in eligible.values())]
    tables, heads, norm_ids = {}, {}, {}
    for name, wm, _ in models:
        use = common if use_intersection else eligible[name]
        if not use:
            if use_intersection:
                raise ValueError(f"{name}: no eligible trace")
            flags[name]["skipped"] = "no eligible trace"
            continue
        model, contract = load_model(Path(wm))
        stats_np = {k: v.numpy() for k, v in contract.statistics.items()}
        norm_ids[name] = normalizer_id(stats_np)
        merged = {"fall": [], "nonfall": []}
        for tn in use:
            w = anymal_windows(data[tn], contract, seed, tn)
            for kind in merged:
                merged[kind].append(w[kind])
        win = {k: dict(xs=torch.cat([x["xs"] for x in v]), xa=torch.cat([x["xa"] for x in v]),
                       y=torch.cat([x["y"] for x in v]), clusters=np.concatenate([x["clusters"] for x in v]),
                       strata=np.concatenate([x["strata"] for x in v]))
               for k, v in merged.items()}
        tables[name] = model_table(model, stats_np, win, c0.HORIZON, ANYMAL_DEPTHS, n_boot, seed)
        heads[name] = tables[name].pop("heads")
        tables[name]["traces_used"] = use
        tables[name]["pairs_excluded"] = {n: pairs[n] for n, _ in traces if n not in use}
        tables[name]["wm_sha256"] = sha256(wm)
    same = use_intersection or len({tuple(tables[n]["traces_used"]) for n in tables}) <= 1
    inputs = dict(traces={n: tsha[n] for n, _ in traces}, common=common, pairs_per_trace=pairs,
                  pairs_excluded_from_common={n: pairs[n] for n, _ in traces if n not in common},
                  same_windows_for_every_model=same, n_boot=n_boot, seed=seed)
    return _write(out, "anymal", tables, norm_ids, flags, inputs, heads=heads, registered=REGISTERED_ETH)


def run_go2(models, norm_path, probes, out, seed=0, ks=(10, 0), n_boot=10_000):
    """models: [(name, wm path, norm path or None (then norm_path))]; probes: [(name, path)]."""
    import go2_c0_check as g
    import go2_g0_replay as rp
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    tables, heads, norm_ids = {}, {}, {}
    for name, wm, npath in models:
        npath = npath or norm_path
        sd, _, norm, _ = rp.load_model(wm, npath, "cpu")
        stats_np = {k: np.asarray(norm[k], np.float32) for k in ("state_mean", "state_std", "action_mean", "action_std")}
        norm_ids[name] = normalizer_id(stats_np)
        eps = []
        for pn, pp in probes:
            with np.load(pp, allow_pickle=False) as z:
                validate_trace(z)
                tr = {k: z[k] for k in z.files if k != "metadata"}
            eps += g.episodes_of(tr, norm, pn)[0]
        tables[name], heads[name] = {"wm_sha256": sha256(wm), "norm_sha256": sha256(npath)}, {}
        for k in ks:
            win, counts = go2_windows(eps, k, seed=seed)
            t = model_table(sd, stats_np, win, 26, (0, 25), n_boot, seed)
            heads[name][f"K{k}"] = t.pop("heads")
            tables[name][f"K{k}"] = dict(t, counts=counts)
    flat = {n: tables[n]["K10"] for n in tables}
    return _write(out, "go2", tables, norm_ids, {}, dict(probes={n: sha256(p) for n, p in probes}, n_boot=n_boot, seed=seed),
                  flat_for_pred=flat, heads=heads)


def _ci(x):
    return f"[{x[0]:.3f}, {x[1]:.3f}]"


def _c0_text(b):
    """Part B numbers, the difference first; the ratio only with a stable denominator (c0m_memory convention)."""
    ratio = (f"ratio UNSTABLE DENOMINATOR (non-fall H CI includes 0 or {b['nonpositive_denominator_draws']} draws <= 0; "
             "not read)" if b["unstable_denominator"] else f"ratio {b['ratio']:.2f} {_ci(b['ratio_ci'])}")
    z = b["zero_spread_pairs_excluded"]
    return (f"fall - non-fall {b['difference']:.3f} {_ci(b['difference_ci'])}; H fall {b['H_fall']:.3f} "
            f"{_ci(b['H_fall_ci'])}, non-fall {b['H_nonfall']:.3f} {_ci(b['H_nonfall_ci'])}; {ratio}"
            + (f"; zero-spread pairs left out {z}" if z else ""))


def _side_lines(name, src, ws):
    lines = []
    for d, r in ws.items():
        u = lambda a, k: f"{r[a][f'log_u_{k}']:.3f} {_ci(r[a].get(f'log_u_{k}_ci', [np.nan] * 2))}"
        e = lambda a, k: f"{r[a][f'e_phys_{k}']:.3f} {_ci(r[a].get(f'e_phys_{k}_ci', [np.nan] * 2))}"
        lines.append(f"  SIDE {name} {src} {d} log u: ctx fall {u('ctx', 'fall')} non-fall {u('ctx', 'nonfall')} diff "
                     f"{u('ctx', 'diff')} | imag fall {u('imag', 'fall')} non-fall {u('imag', 'nonfall')} diff "
                     f"{u('imag', 'diff')} | geomean u ctx {r['ctx']['u_geomean_fall']:.2f}/{r['ctx']['u_geomean_nonfall']:.2f}"
                     f" imag {r['imag']['u_geomean_fall']:.2f}/{r['imag']['u_geomean_nonfall']:.2f} | H fall {r['H_fall']:.3f}"
                     f" non-fall {r['H_nonfall']:.3f}")
        lines.append(f"  SIDE {name} {src} {d} e_phys sum: ctx fall {e('ctx', 'fall')} non-fall {e('ctx', 'nonfall')} diff "
                     f"{e('ctx', 'diff')} | imag fall {e('imag', 'fall')} non-fall {e('imag', 'nonfall')} diff {e('imag', 'diff')}")
    return lines


def _table_lines(name, t):
    """Text view of one model's tables on one window set (pooled or one source)."""
    b = t["c0_interval"]
    lines = [f"  C0 at d{b['depth']} ({b['pairs']} pairs, {b['pair_components']} episode components): {_c0_text(b)}",
             "  arm/depth group      e_phys fall [CI] | non-fall    u geomean fall [CI] | non-fall   coverage fall [CI] | "
             "non-fall   joint miss fall [CI] | non-fall   agree&wrong fall [CI] | non-fall"]
    for key in [k for k in t if "/d" in k]:
        for g in GROUPS:
            r, c = t[key][g], t[key][g]["ci"]
            cell = lambda m, pt: f"{r[pt + '_fall']:.3f} {_ci(c[m + '_fall'])} | {r[pt + '_nonfall']:.3f}"
            lines.append(f"  {key:9s} {g:9s}  {cell('e_phys', 'e_phys')}   {cell('u_geomean', 'u_geomean')}   "
                         f"{cell('coverage', 'coverage')}   {cell('joint_miss', 'joint_miss')}   {cell('agree_wrong', 'agree_wrong')}"
                         + (f"   tilt err deg {r['tilt_err_deg_fall']:.2f} | {r['tilt_err_deg_nonfall']:.2f}" if g == "gravity" else ""))
    return lines


def _head_lines(h):
    lines = []
    for d in [k for k in h if k.startswith("d")]:
        for g in GROUPS:
            for kind in ("fall", "nonfall"):
                x = h[d][g][kind]
                adv = " ".join(f"h{i} {a:+.3f} {_ci(c)}" for i, (a, c) in enumerate(zip(x["advantage"], x["advantage_ci"])))
                lines.append(f"  {d:4s} {g:9s} {kind:7s} spread {x['spread']:.3f}; same direction {x['same_direction']:.2f} "
                             f"{_ci(x['same_direction_ci'])}; advantage {adv}")
        st = h[d]["stop"]
        lines.append(f"  {d:4s} stop p>0.5 per head: fall {[round(v, 3) for v in st['fall']['share_above_half']]}, "
                     f"non-fall {[round(v, 3) for v in st['nonfall']['share_above_half']]}")
    f = h["fall_last_depth"]
    lines.append(f"  falls at d{f['depth']} (observed tilt >= {TILT_DEG:g} deg: {f['observed_tilt_share']:.2f}); per head: "
                 f"stop predicted {[round(v, 3) for v in f['stop_predicted']]}; tilt predicted, stop missed "
                 f"{[round(v, 3) for v in f['tilt_predicted_stop_missed']]}; tilt missed {[round(v, 3) for v in f['tilt_missed']]}")
    return lines


def _loo_lines(name, src, loo):
    lines = []
    for s, r in loo.items():
        e = r["e_phys"]
        last = list(e)[-1]
        ef = "/".join(f"{e[d]['sum']['fall']:.3f}" for d in e)
        en = "/".join(f"{e[d]['sum']['nonfall']:.3f}" for d in e)
        grp = " ".join(f"{g} {e[last][g]['fall']:.3f}|{e[last][g]['nonfall']:.3f}" for g in SIX)
        warn = "; ".join(f"{k[8:]} dH {w['difference']:.3f} {_ci(w['difference_ci'])} AUROC {w['auroc_u_ctx']:.3f}"
                         for k, w in r.items() if k.startswith("warning_"))
        lines.append(f"  LOO {name} {src} {s}: e_phys sum ({'/'.join(e)}) fall {ef} | non-fall {en}; {last} {grp}; "
                     f"warning {warn}")
    return lines


def _write(out, robot, tables, norm_ids, flags, inputs, flat_for_pred=None, heads=None, registered=None):
    pred = predictions(flat_for_pred or tables)
    limits = ["models differ in more than failure data (volume, recipe, seed)", "ETH's training data are unknown",
              "the windows were seen before", "associations, not causes",
              "thresholds fixed at their point values inside the bootstrap"]
    rep = dict(protocol=f"disagreement vs error by component ({robot}); draft rules 24 Nov 2026 §2; descriptive",
               inputs=inputs, eligibility=flags, normalizers=norm_ids, common_scale_normalized=common_scale(norm_ids),
               tables=tables, predictions=pred, registered_eth_c0=registered, analysis_sha256=sha256(__file__),
               limits=limits)
    (out / f"components_{robot}.json").write_text(json.dumps(c0.json_safe(rep), indent=1))
    lines = [f"Disagreement vs error by component ({robot}); descriptive; windows seen before; 95 % intervals: matched-pair "
             f"episode-cluster bootstrap stratified by source ({inputs.get('n_boot')} draws, seed {inputs.get('seed')}); "
             "per-source tables in the JSON (tables.<model>.by_source)"]
    if robot == "anymal":
        used = {n: t["traces_used"] for n, t in tables.items()}
        lines.append(f"  windows: SAME FOR EVERY MODEL: {inputs['same_windows_for_every_model']}; traces used {used}; "
                     f"pairs per trace {inputs['pairs_per_trace']}; excluded from the common set {inputs['pairs_excluded_from_common']}")
        for n, f in flags.items():
            lines.append(f"  eligibility {n}: manifest {f['manifest']}; excluded traces {f['excluded']} ({f['excluded_pairs']} pairs)"
                         f"{'; NO MANIFEST (every trace used)' if f['no_manifest'] else ''}{'; SKIPPED: ' + f['skipped'] if f.get('skipped') else ''}")
    for k, v in rep["common_scale_normalized"].items():
        lines.append(f"  normalized units common for {k}: {v}")
    for k in ("P1", "P2", "P3"):
        lines.append(f"  {k}: held {pred[k]['held']} {pred[k]['per_pair']}")
    lines.append(f"  P3 note: {pred['P3']['note']}; P3_difference (fall - non-fall H lower): held "
                 f"{pred['P3_difference']['held']} {pred['P3_difference']['per_pair']}")
    lines.append(f"  P4: held {pred['P4'].get('held')} " + str({n: v['largest'] for n, v in pred['P4'].items() if isinstance(v, dict)}))
    scopes = {n: t for n, t in tables.items()} if robot == "anymal" else \
        {f"{n} {k}": t[k] for n, t in tables.items() for k in t if k.startswith("K")}
    lines.append("PART B: C0 deficit on these windows (descriptive, no verdict)")
    if registered:
        for src, r in registered["by_source"].items():
            lines.append(f"  ETH registered {src} ({r['pairs']} pairs; {registered['record']}): H fall {r['H_fall']}, "
                         f"non-fall {r['H_nonfall']}, ratio {r['ratio']} {r['ratio_ci']}")
        t = tables.get("ETH", {})
        for src, r in registered["by_source"].items():      # reproduction check, printed whatever the denominator
            b = t.get("by_source", {}).get(src, {}).get("c0_interval") or \
                (t["c0_interval"] if t.get("traces_used") == [src] else None)
            if b:
                lines.append(f"  READ GATE ETH {src}: ratio {b['ratio']:.2f} {_ci(b['ratio_ci'])}, H fall {b['H_fall']:.3f}, "
                             f"non-fall {b['H_nonfall']:.3f} (registered {r['ratio']} {r['ratio_ci']}, {r['H_fall']} / "
                             f"{r['H_nonfall']}); a reproduction check, not a reading")
    for n, t in scopes.items():
        for src, tt in [("pooled", t)] + list(t.get("by_source", {}).items()):
            b = tt["c0_interval"]
            lines.append(f"  {n:8s} {src:12s} {b['pairs']:4d} pairs: {_c0_text(b)}"
                         + (f"; pairs excluded {t['pairs_excluded']}" if src == "pooled" and "pairs_excluded" in t else ""))
    lines.append("WHICH SIDE MOVED (sum group; mean log u per arm, fall | non-fall | fall - non-fall, 95 % intervals; "
                 "H = log u_ctx - log u_imag; e_phys = ensemble-mean physical RMS error on the same windows)")
    for n, t in scopes.items():
        for src, tt in [("pooled", t)] + list(t.get("by_source", {}).items()):
            lines += _side_lines(n, src, tt["which_side"])
    for n, t in scopes.items():
        lines.append(f"MODEL {n}: components (physical RMS error; summed spread, normalized; 1-sigma coverage; shares)")
        lines += _table_lines(n, t)
        if "repeats" in t:
            r = t["repeats"]
            lines.append(f"  fall - non-fall repeated in every source {r['sources']} (CI excludes 0, same sign): "
                         + (", ".join(f"{k} {v}" for k, v in r["repeated"].items()) or "none"))
            lines.append(f"  in some sources only (not repeated): {len(r['some_sources_only'])}")
    (out / f"components_{robot}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    if heads is not None:
        hrep = dict(protocol=f"Part C head-by-head readout on recorded context ({robot}); descriptive; repeat rule: same head, "
                             "same sign, interval excluding 0, in every source", inputs=inputs, heads=heads,
                    analysis_sha256=sha256(__file__), limits=limits)
        (out / f"components_heads_{robot}.json").write_text(json.dumps(c0.json_safe(hrep), indent=1))
        hl = [f"Part C: head by head on recorded context ({robot}); per-source values in the JSON (heads.<model>.by_source); advantage = |ensemble mean - observed| - |head - observed| "
              "(normalized, mean over the group's dims; > 0: head closer); same direction = share of dims where every head "
              "misses on the side of the ensemble mean"]
        hscopes = heads if robot == "anymal" else {f"{n} {k}": v for n, h in heads.items() for k, v in h.items()}
        each = lambda h: [("pooled", h["pooled"])] + list(h.get("by_source", {}).items())
        for n, h in hscopes.items():          # first: what a `head -60` must show (review 26 Nov)
            if "repeats" in h:
                r = h["repeats"]
                hl.append(f"{n}: head advantages that COUNT (same head and sign in every source {r['sources']}): "
                          + (", ".join(f"{k} {v}" for k, v in r["repeated"].items()) or "none"))
                hl.append(f"{n}: one-source-only advantages (noise by the repeat rule): {len(r['some_sources_only'])}")
            for src, hs in each(h):
                d = [k for k in hs if k.startswith("d")][-1]
                for kind in ("fall", "nonfall"):
                    x = hs[d]["sum"][kind]
                    hl.append(f"  {n} {src} {d} sum {kind} advantage: " + " ".join(
                        f"h{i} {a:+.3f} {_ci(c)}" for i, (a, c) in enumerate(zip(x["advantage"], x["advantage_ci"]))))
        hl.append("LEAVE ONE HEAD OUT (recorded context; e_phys = ensemble-mean physical RMS error, sum, per depth, "
                  "fall | non-fall; groups at the last depth; warning = fall - non-fall H [95 %] and AUROC of u_ctx)")
        for n, h in hscopes.items():
            for src, hs in each(h):
                hl += _loo_lines(n, src, hs["leave_one_out"])
        for n, h in hscopes.items():
            for src, hs in each(h):
                hl.append(f"MODEL {n} {src}:")
                hl += _head_lines(hs)
        (out / f"components_heads_{robot}.txt").write_text("\n".join(hl) + "\n")
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="robot", required=True)
    a = sub.add_parser("anymal")
    a.add_argument("--model", action="append", required=True, help="NAME=WM[:FIT_MANIFEST] (ETH, E1V, E1R2)")
    a.add_argument("--trace", action="append", required=True, help="NAME=PATH (batch A, batch C, round0_1701)")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--per-model-eligible", action="store_true", help="each model its own eligible traces (default: the "
                   "intersection, so every model reads the same windows); a model with none is skipped. Separate tables only")
    a.add_argument("--n-boot", type=int, default=10_000)
    b = sub.add_parser("go2")
    b.add_argument("--model", action="append", required=True, help="NAME=WM[:NORM] (G0, D2, D6b)")
    b.add_argument("--norm", type=Path, required=True, help="default normalizer")
    b.add_argument("--probe", action="append", required=True, help="NAME=PATH (911, 912)")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--n-boot", type=int, default=10_000)
    x = p.parse_args(argv)
    split = lambda s: (s.split("=", 1)[0], *(s.split("=", 1)[1].split(":", 1) + [None])[:2])
    if x.robot == "anymal":
        run_anymal([split(s) for s in x.model], [tuple(s.split("=", 1)) for s in x.trace], x.out,
                   use_intersection=not x.per_model_eligible, n_boot=x.n_boot)
    else:
        run_go2([split(s) for s in x.model], x.norm, [tuple(s.split("=", 1)) for s in x.probe], x.out, n_boot=x.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
