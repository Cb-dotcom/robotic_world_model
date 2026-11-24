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
Limits: the models differ in more than failure data (data volume, recipe, seed; G0 has no friction data); ETH's
training data are unknown; the windows were seen before; associations, not causes.
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


# ---------------------------------------------------------------- windows in one layout for both robots
def anymal_windows(trace, contract, seed=0):
    """Fall and matched non-fall windows of one trace, as C0: Xs/Xa [W, 31 + H, .] normalized (index i is row t-31+i),
    Y [W, H, 45] normalized next states, clusters [W]."""
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
        out[kind] = dict(xs=xs, xa=xa, y=y, origins=o, clusters=np.array([f"{n}/{ep[t, n]}" for t, n in o]))
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
        out[kind] = dict(xs=xs, xa=xa, y=y, origins=np.array(rows), clusters=np.array([eps[i].cluster for i, _ in rows]))
    return out, counts


# ---------------------------------------------------------------- the three arms with every head's mean
@torch.no_grad()
def arm_means(model, xs, xa, horizon, depths):
    """{arm: [P, len(depths), heads, W, 45]} head means at the requested depths. P = 1 for ctx and car, heads for imag."""
    W = xs.shape[0]
    res = {"ctx": [], "car": [], "imag": []}
    for k in depths:                                           # ctx: memory restarted for each target
        model.reset()
        means, _, _, _ = c0.forward_all(model, xs[:, k:k + 32], xa[:, k:k + 32])
        res["ctx"].append(means.numpy())
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


def shares(fall, nonfall, d0_nonfall):
    """Joint-miss and agree-and-wrong shares per group from per-window metrics ([P, W] arrays; mean over paths)."""
    out = {}
    for gname in GROUPS:
        f, n, z = fall[gname], nonfall[gname], d0_nonfall[gname]
        p90_e0 = np.percentile(z["e"], 90)
        med_u, p90_e = np.median(n["u"]), np.percentile(n["e"], 90)
        jm = (f["outside"] >= 0.5) & (f["e"] > p90_e0)
        aw = (f["u"] < med_u) & (f["e"] > p90_e)
        jm_n = (n["outside"] >= 0.5) & (n["e"] > p90_e0)
        aw_n = (n["u"] < med_u) & (n["e"] > p90_e)
        out[gname] = dict(joint_miss_fall=float(jm.mean()), joint_miss_nonfall=float(jm_n.mean()),
                          agree_wrong_fall=float(aw.mean()), agree_wrong_nonfall=float(aw_n.mean()),
                          e_fall=float(f["e"].mean()), e_nonfall=float(n["e"].mean()),
                          e_phys_fall=float(f["e_phys"].mean()), e_phys_nonfall=float(n["e_phys"].mean()),
                          u_geomean_fall=float(np.exp(np.log(np.maximum(f["u"], 1e-300)).mean())),
                          u_geomean_nonfall=float(np.exp(np.log(np.maximum(n["u"], 1e-300)).mean())),
                          coverage_fall=float(f["coverage"].mean()), coverage_nonfall=float(n["coverage"].mean()),
                          thresholds=dict(nonfall_p90_e_d0=float(p90_e0), nonfall_median_u=float(med_u),
                                          nonfall_p90_e=float(p90_e)))
        if "tilt_deg" in f:
            out[gname]["tilt_err_deg_fall"] = float(f["tilt_deg"].mean())
            out[gname]["tilt_err_deg_nonfall"] = float(n["tilt_deg"].mean())
    return out


def h_ratio(u_ctx_f, u_imag_f, u_ctx_n, u_imag_n):
    """C0 point estimate on these windows: H = -mean over windows of mean over paths log(u_imag / u_ctx)."""
    hf = -float((np.log(u_imag_f) - np.log(u_ctx_f)).mean())
    hn = -float((np.log(u_imag_n) - np.log(u_ctx_n)).mean())
    return dict(H_fall=hf, H_nonfall=hn, ratio=hf / hn if hn != 0 else float("nan"))


def model_table(model, stats_np, windows, horizon, depths):
    """All metrics of one model on one window set."""
    m = {}
    for kind in ("fall", "nonfall"):
        w = windows[kind]
        arms = arm_means(model, w["xs"], w["xa"], horizon, depths)
        tgt = w["y"].numpy()[:, list(depths)]
        m[kind] = {arm: arm_metrics(arms[arm], tgt, stats_np) for arm in ARMS}
    table = {}
    for arm in ARMS:
        for di, d in enumerate(depths):
            table[f"{arm}/d{d}"] = shares(m["fall"][arm][di], m["nonfall"][arm][di], m["nonfall"][arm][0])
    last = len(depths) - 1
    u = lambda kind, arm: m[kind][arm][last]["sum"]["u"]
    table["c0_point"] = h_ratio(u("fall", "ctx")[0], u("fall", "imag"), u("nonfall", "ctx")[0], u("nonfall", "imag"))
    table["imag_over_ctx_fall"] = float(np.exp(np.log(u("fall", "imag")).mean() - np.log(u("fall", "ctx")).mean()))
    table["windows"] = {k: int(len(windows[k]["clusters"])) for k in ("fall", "nonfall")}
    return table


def predictions(tables, anymal_pairs=(("E1V", "ETH"), ("E1R2", "ETH")), go2_pairs=(("D6b", "G0"),), endpoint="imag/d25"):
    """P1-P4 held / not held (None when a model is absent)."""
    def jm(name, group="sum"):
        return tables[name][endpoint][group]["joint_miss_fall"] if name in tables else None
    out = {}
    pairs = [p for p in anymal_pairs + go2_pairs if p[0] in tables and p[1] in tables]
    out["P1"] = {f"{a}<{b}": jm(a) < jm(b) for a, b in pairs}
    out["P2"] = {f"{a}>{b}": tables[a]["imag_over_ctx_fall"] > tables[b]["imag_over_ctx_fall"] for a, b in pairs}
    out["P3"] = {f"{a}<{b}": tables[a]["c0_point"]["ratio"] < tables[b]["c0_point"]["ratio"] for a, b in pairs}
    out["P4"] = {}
    for name in ("ETH", "G0"):
        if name in tables:
            s = {g: jm(name, g) for g in SIX}
            out["P4"][name] = dict(largest=max(s, key=s.get), held=max(s, key=s.get) == "gravity", shares=s)
    for k in ("P1", "P2", "P3"):
        out[k] = dict(per_pair=out[k], held=all(out[k].values()) if out[k] else None)
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
def run_anymal(models, traces, out, seed=0, use_intersection=True):
    """models: [(name, wm path, fit manifest or None)]; traces: [(name, path)]."""
    from analyse_transition_fidelity import load_model
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    tsha = {n: sha256(p) for n, p in traces}
    eligible, flags = {}, {}
    for name, wm, man in models:
        used = manifest_hashes(man)
        eligible[name] = [n for n, _ in traces if tsha[n] not in used]
        flags[name] = dict(manifest=str(man) if man else None, excluded=[n for n, _ in traces if tsha[n] in used],
                           no_manifest=man is None)
    common = [n for n, _ in traces if all(n in e for e in eligible.values())]
    tables, norm_ids = {}, {}
    for name, wm, _ in models:
        model, contract = load_model(Path(wm))
        stats_np = {k: v.numpy() for k, v in contract.statistics.items()}
        norm_ids[name] = normalizer_id(stats_np)
        use = common if use_intersection else eligible[name]
        if not use:
            raise ValueError(f"{name}: no eligible trace")
        merged = {"fall": [], "nonfall": []}
        for tn, tp in traces:
            if tn not in use:
                continue
            with np.load(tp, allow_pickle=False) as z:
                validate_trace(z)
                tr = {k: z[k] for k in z.files if k != "metadata"}
            w = anymal_windows(tr, contract, seed)
            for kind in merged:
                merged[kind].append(w[kind])
        win = {k: dict(xs=torch.cat([x["xs"] for x in v]), xa=torch.cat([x["xa"] for x in v]),
                       y=torch.cat([x["y"] for x in v]), clusters=np.concatenate([x["clusters"] for x in v]))
               for k, v in merged.items()}
        tables[name] = model_table(model, stats_np, win, c0.HORIZON, ANYMAL_DEPTHS)
        tables[name]["traces_used"] = use
        tables[name]["wm_sha256"] = sha256(wm)
    return _write(out, "anymal", tables, norm_ids, flags, dict(traces={n: tsha[n] for n, _ in traces}, common=common))


def run_go2(models, norm_path, probes, out, seed=0, ks=(10, 0)):
    """models: [(name, wm path, norm path or None (then norm_path))]; probes: [(name, path)]."""
    import go2_c0_check as g
    import go2_g0_replay as rp
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    tables, norm_ids = {}, {}
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
        tables[name] = {"wm_sha256": sha256(wm), "norm_sha256": sha256(npath)}
        for k in ks:
            win, counts = go2_windows(eps, k, seed=seed)
            t = model_table(sd, stats_np, win, 26, (0, 25))
            tables[name][f"K{k}"] = dict(t, counts=counts)
    flat = {n: tables[n]["K10"] for n in tables}
    return _write(out, "go2", tables, norm_ids, {}, dict(probes={n: sha256(p) for n, p in probes}), flat_for_pred=flat)


def _write(out, robot, tables, norm_ids, flags, inputs, flat_for_pred=None):
    pred = predictions(flat_for_pred or tables)
    rep = dict(protocol=f"disagreement vs error by component ({robot}); draft rules 24 Nov 2026 §2; descriptive",
               inputs=inputs, eligibility=flags, normalizers=norm_ids, common_scale_normalized=common_scale(norm_ids),
               tables=tables, predictions=pred, analysis_sha256=sha256(__file__),
               limits=["models differ in more than failure data (volume, recipe, seed)", "ETH's training data are unknown",
                       "the windows were seen before", "associations, not causes"])
    (out / f"components_{robot}.json").write_text(json.dumps(c0.json_safe(rep), indent=1))
    lines = [f"Disagreement vs error by component ({robot}); descriptive; windows seen before"]
    for k, v in rep["common_scale_normalized"].items():
        lines.append(f"  normalized units common for {k}: {v}")
    for k in ("P1", "P2", "P3"):
        lines.append(f"  {k}: held {pred[k]['held']} {pred[k]['per_pair']}")
    lines.append(f"  P4: held {pred['P4'].get('held')} " + str({n: v['largest'] for n, v in pred['P4'].items() if isinstance(v, dict)}))
    (out / f"components_{robot}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="robot", required=True)
    a = sub.add_parser("anymal")
    a.add_argument("--model", action="append", required=True, help="NAME=WM[:FIT_MANIFEST] (ETH, E1V, E1R2)")
    a.add_argument("--trace", action="append", required=True, help="NAME=PATH (batch A, batch C, round0_1701)")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--per-model-eligible", action="store_true", help="each model its own eligible traces (default: the "
                   "intersection, so every model reads the same windows)")
    b = sub.add_parser("go2")
    b.add_argument("--model", action="append", required=True, help="NAME=WM[:NORM] (G0, D2, D6b)")
    b.add_argument("--norm", type=Path, required=True, help="default normalizer")
    b.add_argument("--probe", action="append", required=True, help="NAME=PATH (911, 912)")
    b.add_argument("--out", type=Path, required=True)
    x = p.parse_args(argv)
    split = lambda s: (s.split("=", 1)[0], *(s.split("=", 1)[1].split(":", 1) + [None])[:2])
    if x.robot == "anymal":
        run_anymal([split(s) for s in x.model], [tuple(s.split("=", 1)) for s in x.trace], x.out,
                   use_intersection=not x.per_model_eligible)
    else:
        run_go2([split(s) for s in x.model], x.norm, [tuple(s.split("=", 1)) for s in x.probe], x.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
