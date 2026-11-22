#!/usr/bin/env python3
"""A4 D4, part 2 (22 Nov 2026, Charbel's decision after D3): test first. Score a held-out recording of the D3 policy
(0.30 friction) with the UNCHANGED qualified model (741b352f...) before anything is fitted on it. The same scorer on
s2's probe 913 (same friction, same model) is the reference. Read-only, CPU; trains nothing.
Everything below was fixed before any D3 recording was read.

Windows (go2_c0_check.windows, K = 0): a fall window ends at the terminal row T; it needs 32 recorded rows before
j = T - 25 in the same episode (falls in the first 57 rows are excluded and counted). Each fall is matched to one
surviving window of equal length from a time-out episode of the same recording (nearest standardized command and
speed at the window start, without replacement, seeded order).

Two arms on the SAME windows, the same 32-row context before j and the same recorded actions (memory rebuilt from the
32 rows, as warm imagination resets); steps m = 0..25 predict rows j..T:
  recorded : the recorded state is fed at every step (teacher forced); all heads read the same input.
  free     : head h drives its own copy with its own predicted state (as imagination); head h's stop output and
             state are read on copy h.
Per window, head and step: stop probability p (stop if p > 0.5, as imagination), base-velocity xy error (m/s) and tilt
error (deg) of the predicted row against the recorded row (raw units), and the exact PPO disagreement u (std over the
five heads, ddof 1, summed over the 45 normalized dims; free arm: on copy h).
Reported per recording, window kind and arm:
  caught / false stop : a head stops at a step that predicts one of the last 11 rows (m = 15..25), as D2's Q2; share
                        over windows, mean over heads; also any stop over m = 0..25;
  errors              : velocity and tilt error at m = 15 (0.2 s before T) and m = 25 (T), mean over heads per window,
                        then median and mean over windows;
  u                   : geometric mean at m = 15 and 25; warning ratio falls / surviving; AUROC falls vs surviving;
  Q2 reference        : D2's Q2 / Q3 (memory from the episode start, recorded) on all falls and surviving episodes.
Readings (fixed now; both can hold):
  R1 missed before rollout : recorded-arm caught share on D3's fall windows < 0.5.
  R2 lost in rollout       : free minus recorded caught share on D3's fall windows <= -0.15 with the bootstrap 95 % CI
                             upper bound < 0 (paired by window; one window per fall episode; 10,000 resamples, seed 0).
  R3 D3-specific (descriptive): recorded-arm caught share on D3's falls minus on s2's falls <= -0.15 with the CI
                             upper bound < 0 (each recording resampled on its own).
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
import go2_c0_check as c0  # noqa: E402
from go2_d2_qualify import GRAV, VXY, load, tilt_deg  # noqa: E402
from pre_reset_transitions import sha256  # noqa: E402

H_CTX, DEPTH, LAST = 32, 25, 11
STEPS = (15, 25)
R_MARGIN = 0.15


def _err(pred_n, true_n, norm):
    """Velocity xy error (m/s) and tilt error (deg) in raw units; [..., 45] normalized inputs."""
    sd_, mu = norm["state_std"], norm["state_mean"]
    p, t = pred_n * sd_ + mu, true_n * sd_ + mu
    v = np.linalg.norm(p[..., VXY] - t[..., VXY], axis=-1)
    a = np.abs(tilt_deg(p[..., GRAV]) - tilt_deg(t[..., GRAV]))
    return v, a


@torch.no_grad()
def paired(sd, eps, rows, norm, device, depth=DEPTH):
    """Both arms on windows rows = [(episode, T)]. Returns dict of arrays [n, E, depth + 1] (recorded u: [n, depth + 1])."""
    E, n = len(sd.state_heads), len(rows)
    starts = [(i, t - depth) for i, t in rows]
    ctx_s = np.stack([eps[i].S[j - H_CTX:j] for i, j in starts])
    ctx_a = np.stack([eps[i].A[j - H_CTX + 1:j + 1] for i, j in starts])
    true = np.stack([eps[i].S[j:j + depth + 1] for i, j in starts])                     # [n, depth + 1, 45]
    act = lambda m: np.stack([eps[i].A[j + m:j + m + 1] for i, j in starts])            # action paired with row j+m
    out = {k: np.empty((n, E, depth + 1), np.float64) for k in ("rec_p", "rec_v", "rec_t", "free_p", "free_v", "free_t", "free_u")}
    out["rec_u"] = np.empty((n, depth + 1), np.float64)
    T = lambda a: torch.from_numpy(a).to(device)
    # recorded arm
    sd.reset()
    xs, xa = T(ctx_s), T(ctx_a)
    for m in range(depth + 1):
        means, logits = rp.all_heads(sd, xs, xa)
        mn = means.cpu().numpy().transpose(1, 0, 2)                                     # [n, E, 45]
        out["rec_p"][:, :, m] = torch.sigmoid(logits).cpu().numpy().T
        out["rec_u"][:, m] = c0.ppo_spread(means).cpu().numpy()
        out["rec_v"][:, :, m], out["rec_t"][:, :, m] = _err(mn, true[:, None, m], norm)
        if m < depth:
            xs, xa = T(true[:, m:m + 1]), T(act(m + 1))
    # free arm: head h drives copy h
    sd.reset()
    rep = lambda a: np.repeat(a, E, axis=0)
    drive = torch.arange(E, device=device).repeat(n)
    cols = torch.arange(n * E, device=device)
    xs, xa = T(rep(ctx_s)), T(rep(ctx_a))
    for m in range(depth + 1):
        means, logits = rp.all_heads(sd, xs, xa)
        own = means[drive, cols]                                                         # [n * E, 45]
        out["free_p"][:, :, m] = torch.sigmoid(logits[drive, cols]).view(n, E).cpu().numpy()
        out["free_u"][:, :, m] = c0.ppo_spread(means).view(n, E).cpu().numpy()
        v, a = _err(own.view(n, E, -1).cpu().numpy(), true[:, None, m], norm)
        out["free_v"][:, :, m], out["free_t"][:, :, m] = v, a
        if m < depth:
            xs, xa = own.unsqueeze(1), T(rep(act(m + 1)))
    sd.reset()
    return out


def stop_any(p, lo, hi):
    """[n, E, steps] -> [n, E] bool: a stop (p > 0.5) at a step in lo..hi."""
    return (np.nan_to_num(p[:, :, lo:hi + 1], nan=0.0) > 0.5).any(-1)


def boot_mean(x, n_boot=10_000, seed=0):
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return [float("nan"), float("nan")]
    g = np.random.default_rng(seed)
    b = x[g.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def boot_diff(x, y, n_boot=10_000, seed=0):
    """CI of mean(x) - mean(y), independent resamples."""
    g = np.random.default_rng(seed)
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    bx = x[g.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    by = y[g.integers(0, len(y), size=(n_boot, len(y)))].mean(1)
    d = bx - by
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def summarize(arr, kind):
    """Per arm summaries for one window kind; arr from paired()."""
    lo = DEPTH - LAST + 1
    s = {}
    for arm in ("rec", "free"):
        caught = stop_any(arr[f"{arm}_p"], lo, DEPTH)                                   # [n, E]
        anyst = stop_any(arr[f"{arm}_p"], 0, DEPTH)
        u = arr["rec_u"] if arm == "rec" else np.exp(np.log(arr["free_u"]).mean(1))      # [n, steps]
        d = dict(per_window_caught=caught.mean(1).tolist(), stop_last11=float(caught.mean()),
                 stop_last11_per_head=caught.mean(0).tolist(), stop_any=float(anyst.mean()))
        for m in STEPS:
            v, t = arr[f"{arm}_v"][:, :, m].mean(1), arr[f"{arm}_t"][:, :, m].mean(1)
            d[f"m{m}"] = dict(vxy_err_median=float(np.median(v)), vxy_err_mean=float(v.mean()),
                              tilt_err_median=float(np.median(t)), tilt_err_mean=float(t.mean()),
                              u_geomean=float(np.exp(np.log(u[:, m]).mean())), u=u[:, m].tolist())
        s[arm] = d
    s["n"] = int(arr["rec_p"].shape[0])
    s["kind"] = kind
    return s


def q2_reference(sd, eps, device):
    det, fs = [], []
    for ep in eps:
        P, _ = rp.episode_context(sd, ep, device)
        if ep.kind == "fall":
            det.append(np.nan_to_num(P[:, max(1, ep.L - LAST):ep.L], nan=0.0).max(1) > 0.5)
        else:
            fs.append(np.nan_to_num(P[:, 1:], nan=0.0).max(1) > 0.5)
    return dict(falls=len(det), surviving=len(fs), caught=float(np.mean(det)) if det else float("nan"),
                false_stop=float(np.mean(fs)) if fs else float("nan"))


def score_probe(sd, trace, norm, name, device, n_boot):
    eps, _ = c0.episodes_of(trace, norm, name)
    fw, sw, counts = c0.windows(eps, 0)
    res = dict(counts=counts, q2_reference=q2_reference(sd, eps, device))
    fa, sa = paired(sd, eps, fw, norm, device), paired(sd, eps, sw, norm, device)
    res["fall"], res["surviving"] = summarize(fa, "fall"), summarize(sa, "surviving")
    d0 = [np.max(np.abs(x["free_u"][:, :, 0] - x["rec_u"][:, None, 0]) / x["rec_u"][:, None, 0]) for x in (fa, sa) if len(x["rec_u"])]
    res["depth0_max_rel_diff"] = float(max(d0)) if d0 else float("nan")     # both arms see the same inputs at m = 0
    for arm in ("rec", "free"):
        res["fall"][arm]["caught_ci"] = boot_mean(res["fall"][arm]["per_window_caught"], n_boot)
        for m in STEPS:
            uf, us = np.array(res["fall"][arm][f"m{m}"]["u"]), np.array(res["surviving"][arm][f"m{m}"]["u"])
            res[f"{arm}_m{m}_warning_ratio"] = float(np.exp(np.log(uf).mean() - np.log(us).mean()))
            res[f"{arm}_m{m}_auroc"] = c0.auroc(uf, us)
    d = np.array(res["fall"]["free"]["per_window_caught"]) - np.array(res["fall"]["rec"]["per_window_caught"])
    res["free_minus_rec_caught"] = dict(mean=float(d.mean()) if len(d) else float("nan"), ci=boot_mean(d, n_boot))
    res["fall_T_steps"] = [int(t) for _, t in fw]
    return res


def readings(d3, ref):
    r1 = d3["fall"]["rec"]["stop_last11"] < 0.5
    fr = d3["free_minus_rec_caught"]
    r2 = bool(fr["mean"] <= -R_MARGIN and fr["ci"][1] < 0)
    if ref is None:
        return dict(R1_missed_before_rollout=bool(r1), R2_lost_in_rollout=r2, R3_d3_specific=None)
    diff = d3["fall"]["rec"]["stop_last11"] - ref["fall"]["rec"]["stop_last11"]
    ci = boot_diff(d3["fall"]["rec"]["per_window_caught"], ref["fall"]["rec"]["per_window_caught"])
    return dict(R1_missed_before_rollout=bool(r1), R2_lost_in_rollout=r2,
                R3_d3_specific=bool(diff <= -R_MARGIN and ci[1] < 0), R3_diff=float(diff), R3_ci=ci)


def text(rep):
    L = ["A4 D4 test first (read-only; unchanged qualified model; rules fixed before the D3 recording was read)",
         f"model {rep['wm_sha256'][:12]}; windows end at the terminal row; recorded vs free arm from the same 32-row context"]
    for name, r in rep["probes"].items():
        c, q = r["counts"], r["q2_reference"]
        L.append(f"[{name}] {c['falls_used']} fall windows ({c['falls_too_short']} too short), matched surviving windows; "
                 f"Q2 reference (memory from start): caught {q['caught']:.3f} of {q['falls']} falls, false stop {q['false_stop']:.3f} "
                 f"of {q['surviving']} surviving")
        for arm, lab in (("rec", "recorded"), ("free", "free")):
            f, s = r["fall"][arm], r["surviving"][arm]
            L.append(f"  {lab:8s}: caught (last 11 rows) {f['stop_last11']:.3f} {np.round(f['caught_ci'], 3).tolist()}, "
                     f"any stop {f['stop_any']:.3f}; surviving windows stopped {s['stop_last11']:.3f} (any {s['stop_any']:.3f})")
            for m in STEPS:
                L.append(f"    m{m}: falls vxy err median {f[f'm{m}']['vxy_err_median']:.3f} m/s, tilt err median "
                         f"{f[f'm{m}']['tilt_err_median']:.2f} deg; surviving {s[f'm{m}']['vxy_err_median']:.3f} m/s, "
                         f"{s[f'm{m}']['tilt_err_median']:.2f} deg; u falls {f[f'm{m}']['u_geomean']:.3f} vs surviving "
                         f"{s[f'm{m}']['u_geomean']:.3f} (ratio {r[f'{arm}_m{m}_warning_ratio']:.2f}, AUROC {r[f'{arm}_m{m}_auroc']:.3f})")
        fr = r["free_minus_rec_caught"]
        L.append(f"  free - recorded caught on fall windows: {fr['mean']:+.3f} {np.round(fr['ci'], 3).tolist()}; "
                 f"depth-0 check (max rel diff) {r['depth0_max_rel_diff']:.1e}")
    rd = rep["readings"]
    L.append(f"R1 missed before rollout: {rd['R1_missed_before_rollout']}; R2 lost in rollout: {rd['R2_lost_in_rollout']}; "
             f"R3 D3-specific: {rd['R3_d3_specific']}" + (f" (diff {rd['R3_diff']:+.3f} {np.round(rd['R3_ci'], 3).tolist()})"
                                                       if rd.get("R3_diff") is not None else ""))
    return "\n".join(L) + "\n"


def run(wm, norm_path, probes, out, n_boot=10_000, device="cpu", d3_name="d3"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    torch.set_num_threads(8)
    sd, _, norm, diff = rp.load_model(wm, norm_path, device)
    res, meta = {}, {}
    for name, path in probes:
        tr, m = load(path)
        res[name] = score_probe(sd, tr, norm, name, device, n_boot)
        meta[name] = dict(path=str(path), sha256=sha256(path), checkpoint_sha256=m.get("checkpoint_sha256"),
                          friction=(m.get("friction_shift") or {}).get("factor"))
    ref = next((res[n] for n, _ in probes if n != d3_name), None)
    rep = dict(protocol="A4 D4 test first (rules fixed before the D3 recording was read)", wm=str(wm), wm_sha256=sha256(wm),
               normalizer_sha256=sha256(norm_path), normalizer_max_diff=diff, probes_meta=meta, probes=res,
               readings=readings(res[d3_name], ref), analysis_sha256=sha256(__file__), wall_s=time.time() - t0)
    (out / "test_first.json").write_text(json.dumps(rep, indent=1, default=float))
    t = text(rep)
    (out / "test_first.txt").write_text(t)
    print(t, end="", flush=True)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", type=Path, required=True)
    p.add_argument("--norm", type=Path, required=True)
    p.add_argument("--probe", action="append", required=True, help="name=path; the first must be the D3 recording")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=10_000)
    a = p.parse_args(argv)
    probes = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in a.probe]
    run(a.wm, a.norm, probes, a.out, a.n_boot, d3_name=probes[0][0])


if __name__ == "__main__":
    main()
