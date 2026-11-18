#!/usr/bin/env python3
"""Plan v6 cause table from c0_battery outputs: how much of the healing each refit removes, against a
control model, on the same windows. CPU only, no model runs (reads c0_windows.npz).

* closure = 1 - H_treat(25) / H_control(25) (plan §2, §11), with H = -HI summed over the 45 dimensions;
  reported only when the control's H(25) has a 95 % CI above 0, otherwise "inconclusive".
* Spreads are re-expressed in the control's normalization before summing (per-dimension scale
  std_model / std_control), so a model with another normalizer (C6) is read on the same coordinates;
  its H in its own coordinates is given too.
* Doubt E18: change of log u_real and of log u_imag (geometric means), so a closure that comes only
  from a lower u_real is visible.
* Errors: real-context one-step base-velocity RMSE over all window steps; imagined base-velocity RMSE at
  d = 1 and d = 25 (driving head's own path). Changes in % of the control.
* Paired cluster bootstrap: the same resampled episodes for control and treatment, stratified by source.
* Noise floor: the same numbers for refits of the control recipe with other seeds (--noise).
The cause rules of plan §2 are applied by reading this table; they are not automated here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from c0_battery import COMPONENTS, ci, cluster_bootstrap, json_safe

D_ONE, D_LAST = 1, 25


def load(folder, sources):
    with np.load(Path(folder) / "c0_windows.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    out = {"std": data["meta/state_std"].astype(float)}
    for s in sources:
        for kind in ("fall", "nonfall"):
            key = f"{s}/{kind}/"
            if key + "u_real" not in data:
                raise KeyError(f"{folder}: no source {s}")
            out[(s, kind)] = {k[len(key):]: v for k, v in data.items() if k.startswith(key)}
    return out


def window_values(run, control_std, sources, kind):
    """Per-window columns: HI(25) per component in control units, HI(25) sum in own units,
    log u_real / log u_imag sums (control units), squared velocity errors."""
    cols, clusters, strata = [], [], []
    scale = run["std"] / control_std                                              # [45]
    for s in sources:
        r = run[(s, kind)]
        ur, ui = r["u_real"][D_LAST] * scale, r["u_imag"][:, D_LAST] * scale       # [W,45], [heads,W,45]
        his = []
        for lo, hi in COMPONENTS.values():
            his.append(np.log(ui[..., lo:hi].sum(-1) / ur[None, :, lo:hi].sum(-1)).mean(0))
        own = np.log(r["u_imag"][:, D_LAST].sum(-1) / r["u_real"][D_LAST].sum(-1)[None]).mean(0)
        cols.append(np.column_stack(his + [own, np.log(ur.sum(-1)), np.log(ui.sum(-1)).mean(0),
                                           r["se_real_vel"].mean(0), r["se_imag_vel"][:, D_ONE].mean(0),
                                           r["se_imag_vel"][:, D_LAST].mean(0)]))
        clusters.append(r["clusters"])
        strata.append([s] * len(r["clusters"]))
    return np.concatenate(cols), np.concatenate(clusters), np.concatenate(strata)


NC = len(COMPONENTS)
COL = {"own": NC, "log_ur": NC + 1, "log_ui": NC + 2, "se_real": NC + 3, "se_d1": NC + 4, "se_d25": NC + 5}


def compare(control, treat, sources, n_boot, seed):
    """Paired comparison of one treatment run with the control run."""
    res = {}
    for kind in ("fall", "nonfall"):
        for s in sources:
            if not np.array_equal(control[(s, kind)]["origins"], treat[(s, kind)]["origins"]):
                raise ValueError(f"windows differ for {s}/{kind}: rerun both with the same traces and seed")
        vc, cl, st = window_values(control, control["std"], sources, kind)
        vt, _, _ = window_values(treat, control["std"], sources, kind)
        point, boot = cluster_bootstrap(np.hstack([vc, vt]), cl, st, n_boot, seed)
        c, t = point[:vc.shape[1]], point[vc.shape[1]:]
        bc, bt = boot[:, :vc.shape[1]], boot[:, vc.shape[1]:]
        sum_i = list(COMPONENTS).index("sum")
        h_c, h_t, hb_c, hb_t = -c[sum_i], -t[sum_i], -bc[:, sum_i], -bt[:, sum_i]
        out = dict(H_control=float(h_c), H_control_ci=ci(hb_c), H_treat=float(h_t), H_treat_ci=ci(hb_t),
                   H_treat_own_units=float(-t[COL["own"]]),
                   dlog_u_real=float(t[COL["log_ur"]] - c[COL["log_ur"]]),
                   dlog_u_real_ci=ci(bt[:, COL["log_ur"]] - bc[:, COL["log_ur"]]),
                   dlog_u_imag=float(t[COL["log_ui"]] - c[COL["log_ui"]]),
                   dlog_u_imag_ci=ci(bt[:, COL["log_ui"]] - bc[:, COL["log_ui"]]))
        for name in ("se_real", "se_d1", "se_d25"):
            j = COL[name]
            out[f"{name}_rmse_control"] = float(np.sqrt(c[j]))
            out[f"{name}_rmse_treat"] = float(np.sqrt(t[j]))
            out[f"{name}_change"] = float(np.sqrt(t[j] / c[j]) - 1)
            out[f"{name}_change_ci"] = ci(np.sqrt(bt[:, j] / bc[:, j]) - 1)
        if kind == "fall":
            valid = ci(hb_c)[0] > 0
            comp = {}
            for i, name in enumerate(COMPONENTS):
                hc, ht = -c[i], -t[i]
                if ci(-bc[:, i])[0] > 0:
                    comp[name] = dict(closure=float(1 - ht / hc), closure_ci=ci(1 - (-bt[:, i]) / (-bc[:, i])))
                else:
                    comp[name] = dict(closure=None, note="control H CI includes 0")
            out["closure"] = comp["sum"].get("closure")
            out["closure_ci"] = comp["sum"].get("closure_ci")
            out["closure_valid"] = bool(valid)
            out["closure_by_component"] = comp
        res[kind] = out
    return res


def table(results):
    first = next(iter(results.values()))
    lines = [f"control: falls H(25) = {first['fall']['H_control']:.3f} [{first['fall']['H_control_ci'][0]:.3f}, "
             f"{first['fall']['H_control_ci'][1]:.3f}]; non-falls H(25) = {first['nonfall']['H_control']:.3f} "
             f"[{first['nonfall']['H_control_ci'][0]:.3f}, {first['nonfall']['H_control_ci'][1]:.3f}]; "
             f"imagined vel RMSE d25 = {first['fall']['se_d25_rmse_control']:.3f} m/s"]
    lines += ["model           | falls H(25) [95% CI]     | closure [95% CI]        | dlog u_real | dlog u_imag | "
             "1-step vel err | imag err d1 | imag err d25 | non-falls H(25) | gravity closure"]
    for name, r in results.items():
        f, n = r["fall"], r["nonfall"]
        clo = (f"{f['closure']:+.2f} [{f['closure_ci'][0]:+.2f}, {f['closure_ci'][1]:+.2f}]"
               if f["closure"] is not None else "inconclusive")
        g = f["closure_by_component"]["gravity"]
        gc = f"{g['closure']:+.2f}" if g["closure"] is not None else "n/a"
        lines.append(f"{name:15s} | {f['H_treat']:.3f} [{f['H_treat_ci'][0]:.3f}, {f['H_treat_ci'][1]:.3f}] | {clo:23s} | "
                     f"{f['dlog_u_real']:+.3f}      | {f['dlog_u_imag']:+.3f}      | {f['se_real_change']:+.1%}         | "
                     f"{f['se_d1_change']:+.1%}      | {f['se_d25_change']:+.1%}       | {n['H_treat']:.3f}           | {gc}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--control", required=True, metavar="NAME=DIR")
    p.add_argument("--treat", action="append", default=[], metavar="NAME=DIR")
    p.add_argument("--noise", action="append", default=[], metavar="NAME=DIR",
                   help="refits of the control recipe with other seeds (noise floor)")
    p.add_argument("--sources", default="", help="comma list, e.g. batchA,batchC (default: all of the control)")
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="fresh output folder")
    a = p.parse_args()
    if a.out.exists():
        p.error("output folder exists; use a fresh one")
    if not a.treat and not a.noise:
        p.error("give at least one --treat or --noise")
    pairs = {}
    for item in [a.control] + a.noise + a.treat:
        name, _, folder = item.partition("=")
        if not name or not folder:
            p.error(f"bad {item!r}; use NAME=DIR")
        if name in pairs:
            p.error(f"duplicate name {name}")
        pairs[name] = Path(folder)
    cname = a.control.partition("=")[0]
    if a.sources:
        sources = a.sources.split(",")
    else:
        with np.load(pairs[cname] / "c0_windows.npz") as z:
            sources = sorted({k.split("/")[0] for k in z.files if not k.startswith("meta/")})
    control = load(pairs[cname], sources)
    results = {}
    for item in a.noise + a.treat:
        name = item.partition("=")[0]
        results[("noise: " if item in a.noise else "") + name] = compare(control, load(pairs[name], sources),
                                                                         sources, a.bootstrap, a.seed)
    a.out.mkdir(parents=True)
    meta = dict(control=cname, folders={k: str(v) for k, v in pairs.items()}, sources=sources,
                bootstrap=a.bootstrap, seed=a.seed, depth=D_LAST)
    (a.out / "compare.json").write_text(json.dumps(json_safe(dict(meta=meta, results=results)), indent=1, allow_nan=False))
    text = f"Cause table vs control '{cname}' on {', '.join(sources)} (d = {D_LAST}, paired cluster bootstrap)\n" + table(results) + "\n"
    (a.out / "compare.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
