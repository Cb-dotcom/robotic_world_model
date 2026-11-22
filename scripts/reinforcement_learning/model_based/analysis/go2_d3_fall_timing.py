#!/usr/bin/env python3
"""A4 D4, part 1 (22 Nov 2026, Charbel's decision after D3): where do D3's extra falls occur? Read-only, CPU, no model.

Input: the saved D3 evaluation folders (`<eval>/{D3,S2}_cmd10_s{42,43}/episodes.csv` and `episodes_summary.json`,
written by eval_episode_stats.py). Descriptive only; every quantity below was fixed before the files were read.

Per policy (D3, S2) and seed, and pooled:
  - falls, completed episodes, exposure (envs x steps x step_dt), falls per 1000 s with the exact Poisson CI;
  - time after reset at the fall (episode length x step_dt), right-closed bins (0, 1], (1, 2.56], (2.56, 5.12],
    (5.12, 10], (10, end] s; the share within 5.12 s (<= 256 steps = the imagined-episode cap in training);
  - time since the last command change at the fall (steps_since_cmd_change x step_dt; at reset the counter restarts),
    right-closed bins (0, 0.5], (0.5, 1], (1, 2], (2, 5], (5, end] s; falls before any command change (counter =
    episode length);
  - commanded speed at the fall (cmd_speed_end), bins [0, 0.5), [0.5, 1.0), [1.0, 1.5), [1.5, inf) m/s (counts only:
    the files hold no exposure per command speed);
  - measured distance = sum over episodes of mean_speed_act x counted_steps x step_dt (base-frame horizontal speed
    integrated over counted steps) and falls per km (measured, no assumption on command exposure);
  - excess falls D3 - S2 per bin (pooled seeds; same exposure).
Tracking and speed in the summaries count only non-terminal steps, so they can improve while falls rise.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from fall_rate import falls_per_1000s  # noqa: E402

TAGS, SEEDS = ("D3", "S2"), (42, 43)
RESET_BINS = (0.0, 1.0, 2.56, 5.12, 10.0, np.inf)
CMD_BINS = (0.0, 0.5, 1.0, 2.0, 5.0, np.inf)
SPEED_BINS = (0.0, 0.5, 1.0, 1.5, np.inf)
CAP_S = 5.12


def labels(edges, right=False):
    end = lambda b: "end" if np.isinf(b) else f"{b:g}"
    if right:
        return [f"({a:g}, {end(b)}]" for a, b in zip(edges[:-1], edges[1:])]
    return [f"[{a:g}, {end(b)})" for a, b in zip(edges[:-1], edges[1:])]


def hist(x, edges, right=False):
    """Counts per bin: [a, b) by default, (a, b] with right=True (the first bin also takes x == edges[0])."""
    x, e = np.asarray(x, np.float64), np.asarray(edges, np.float64)
    if not right:
        return np.histogram(x, bins=e)[0].astype(int).tolist()
    tol = 1e-9
    out = [int(((x > a + tol) & (x <= b + tol)).sum()) for a, b in zip(e[:-1], e[1:])]
    out[0] += int((np.abs(x - e[0]) <= tol).sum())
    return out


def read_cell(folder):
    """episodes_summary.json as eval_episode_stats.py writes it: run settings at the top level (step_dt,
    friction_shift, ...) and EpisodeStats.summary() under "summary" (falls, exposure_s, counted_steps, ...)."""
    folder = Path(folder)
    summ = json.loads((folder / "episodes_summary.json").read_text())
    if "summary" not in summ or "step_dt" not in summ:
        raise ValueError(f"{folder}: episodes_summary.json is not in the eval_episode_stats.py format")
    with open(folder / "episodes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    ep = dict(length=np.array([int(r["length"]) for r in rows]),
              counted=np.array([int(r["counted_steps"]) for r in rows]),
              fall=np.array([r["fall"] == "True" for r in rows]),
              censored=np.array([r["censored"] == "True" for r in rows]),
              since_cmd=np.array([int(r["steps_since_cmd_change"]) for r in rows]),
              cmd_end=np.array([float(r["cmd_speed_end"]) for r in rows]),
              speed_act=np.array([float(r["mean_speed_act"]) for r in rows]))
    return summ, ep


def stats(summ, ep):
    dt = float(summ["step_dt"])
    inner = summ["summary"]
    exposure = float(inner["exposure_s"])
    f = ep["fall"] & ~ep["censored"]
    t_reset = ep["length"][f] * dt
    t_cmd = ep["since_cmd"][f] * dt
    dist_m = float(np.nansum(np.nan_to_num(ep["speed_act"], nan=0.0) * ep["counted"]) * dt)
    ref = float(inner["speed_act_per_step"]) * float(inner["counted_steps"]) * dt     # the evaluator's own aggregate
    if not np.isclose(dist_m, ref, rtol=1e-6, atol=1e-6):
        raise ValueError(f"distance from episodes.csv {dist_m:.3f} m differs from the summary's {ref:.3f} m")
    n = int(f.sum())
    rate, lo, hi = falls_per_1000s(n, exposure)
    return dict(falls=n, completed=int((~ep["censored"]).sum()), exposure_s=exposure, step_dt=dt,
                falls_per_1000s=[rate, lo, hi], falls_in_summary=int(inner["falls"]),
                reset_counts=hist(t_reset, RESET_BINS, right=True), within_cap=int((t_reset <= CAP_S + 1e-9).sum()),
                cmd_counts=hist(t_cmd, CMD_BINS, right=True), before_any_change=int((ep["since_cmd"][f] == ep["length"][f]).sum()),
                speed_counts=hist(ep["cmd_end"][f], SPEED_BINS), distance_m=dist_m,
                falls_per_km=(1000.0 * n / dist_m) if dist_m > 0 else float("nan"),
                median_fall_s=float(np.median(t_reset)) if n else float("nan"),
                friction=(summ.get("friction_shift") or {}).get("factor"))


def pool(cells):
    keys_sum = ("falls", "completed", "exposure_s", "within_cap", "before_any_change", "distance_m")
    out = {k: sum(c[k] for c in cells) for k in keys_sum}
    for k in ("reset_counts", "cmd_counts", "speed_counts"):
        out[k] = np.sum([c[k] for c in cells], axis=0).astype(int).tolist()
    out["falls_per_1000s"] = list(falls_per_1000s(out["falls"], out["exposure_s"]))
    out["falls_per_km"] = 1000.0 * out["falls"] / out["distance_m"] if out["distance_m"] > 0 else float("nan")
    return out


def analyse(eval_dir):
    res = {}
    for tag in TAGS:
        per = {}
        for s in SEEDS:
            summ, ep = read_cell(Path(eval_dir) / f"{tag}_cmd10_s{s}")
            per[str(s)] = stats(summ, ep)
            if per[str(s)]["falls"] != per[str(s)]["falls_in_summary"]:
                raise ValueError(f"{tag} s{s}: falls in episodes.csv differ from the summary")
        res[tag] = dict(per_seed=per, pooled=pool(list(per.values())))
    d, s = res["D3"]["pooled"], res["S2"]["pooled"]
    if abs(d["exposure_s"] - s["exposure_s"]) > 1e-6:
        raise ValueError("D3 and S2 exposures differ")
    res["excess_D3_minus_S2"] = {k: (np.array(d[k]) - np.array(s[k])).tolist()
                                 for k in ("reset_counts", "cmd_counts", "speed_counts")}
    res["excess_D3_minus_S2"]["falls"] = d["falls"] - s["falls"]
    res["excess_D3_minus_S2"]["within_cap"] = d["within_cap"] - s["within_cap"]
    res["bins"] = dict(reset_s=labels(RESET_BINS, True), since_cmd_change_s=labels(CMD_BINS, True),
                       cmd_speed_mps=labels(SPEED_BINS))
    return res


def text(res):
    L = ["A4 D4 part 1: D3 fall timing (read-only; saved evaluations; 0.30 friction, 10 s commands)"]
    b = res["bins"]
    for tag in TAGS:
        p = res[tag]["pooled"]
        r = p["falls_per_1000s"]
        seeds = ", ".join(f"s{s} {c['falls']}" for s, c in res[tag]["per_seed"].items())
        L.append(f"{tag}: falls {p['falls']} ({seeds}); {r[0]:.2f} per 1000 s [{r[1]:.2f}, {r[2]:.2f}]; "
                 f"distance {p['distance_m'] / 1000:.2f} km -> {p['falls_per_km']:.1f} falls per km")
        L.append(f"  after reset {dict(zip(b['reset_s'], p['reset_counts']))}; within {CAP_S} s: {p['within_cap']} "
                 f"({p['within_cap'] / max(p['falls'], 1):.2f})")
        L.append(f"  since command change {dict(zip(b['since_cmd_change_s'], p['cmd_counts']))}; before any change: "
                 f"{p['before_any_change']}")
        L.append(f"  commanded speed at the fall {dict(zip(b['cmd_speed_mps'], p['speed_counts']))}")
    e = res["excess_D3_minus_S2"]
    L.append(f"excess falls D3 - S2: total {e['falls']}; within {CAP_S} s {e['within_cap']}; "
             f"after reset {dict(zip(b['reset_s'], e['reset_counts']))}")
    L.append(f"  since command change {dict(zip(b['since_cmd_change_s'], e['cmd_counts']))}; "
             f"commanded speed {dict(zip(b['cmd_speed_mps'], e['speed_counts']))}")
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=False)
    res = analyse(a.eval)
    (a.out / "d3_fall_timing.json").write_text(json.dumps(res, indent=1, default=float))
    t = text(res)
    (a.out / "d3_fall_timing.txt").write_text(t)
    print(t, end="")


if __name__ == "__main__":
    main()
