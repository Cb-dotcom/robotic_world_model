#!/usr/bin/env python3
"""A4 D4 part 1b and D7 (22 Nov 2026, Charbel's review 20:47): fall risk per reset and per second at risk, by time
after a reset. Read-only, CPU, no model. Fixed before the files were read.

Input: saved evaluation folders `<eval>/<TAG>_cmd10_s{42,43}/episodes.csv` + `episodes_summary.json`
(eval_episode_stats.py layout), for a policy tag (D3 or D7) and the reference S2.

Per policy, pooled over seeds 42 and 43 (every episode counts, censored ones too; a censored episode adds exposure,
never a fall):
  resets            episodes started (completed + censored)
  falls per reset   falls / resets, and the share of resets that end in a fall within 32 and within 40 rows
  per bin of rows after the reset, right-closed: (0, 10], (10, 20], (20, 32], (32, 40], (40, 50], (50, 128],
  (128, 256], (256, 500], (500, end]:
      at risk       episodes still running at the bin start (length > a)
      exposure      steps spent in the bin, summed over episodes (min(length, b) - a), in seconds
      falls         falls whose terminal row lies in the bin
      hazard        falls per second at risk, exact Poisson 95 % CI (fall_rate.falls_per_1000s / 1000)
      conditional   falls / at risk (probability of falling in the bin given the bin was reached)
  hazard ratio policy / S2 per bin (descriptive; no CI on the ratio)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from fall_rate import falls_per_1000s  # noqa: E402
from go2_d3_fall_timing import SEEDS, read_cell  # noqa: E402

EDGES = (0, 10, 20, 32, 40, 50, 128, 256, 500, np.inf)


def label(a, b):
    return f"({a}, {'end' if np.isinf(b) else int(b)}]"


def cell_arrays(eval_dir, tag):
    L, F, dts = [], [], set()
    for s in SEEDS:
        summ, ep = read_cell(Path(eval_dir) / f"{tag}_cmd10_s{s}")
        dts.add(float(summ["step_dt"]))
        L.append(ep["length"])
        F.append(ep["fall"] & ~ep["censored"])
    if len(dts) != 1:
        raise ValueError(f"{tag}: step_dt differs between seeds")
    return np.concatenate(L).astype(np.int64), np.concatenate(F), dts.pop()


def hazards(length, fall, dt):
    out = dict(resets=int(len(length)), falls=int(fall.sum()))
    out["falls_per_reset"] = out["falls"] / max(out["resets"], 1)
    for k in (32, 40):
        out[f"fall_within_{k}_rows_per_reset"] = float((fall & (length <= k)).sum() / max(out["resets"], 1))
    bins = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        at_risk = int((length > a).sum())
        steps = np.clip(np.minimum(length, b) - a, 0, None).sum()
        exp_s = float(steps * dt)
        n = int((fall & (length > a) & (length <= b)).sum())
        r, lo, hi = falls_per_1000s(n, exp_s) if exp_s > 0 else (float("nan"),) * 3
        bins.append(dict(bin=label(a, b), at_risk=at_risk, exposure_s=exp_s, falls=n,
                         hazard_per_s=[r / 1000, lo / 1000, hi / 1000],
                         conditional=n / at_risk if at_risk else float("nan")))
    out["bins"] = bins
    out["exposure_s"] = float(sum(b["exposure_s"] for b in bins))
    return out


def analyse(eval_dir, tag="D3", ref="S2"):
    res = {}
    for t in (tag, ref):
        L, F, dt = cell_arrays(eval_dir, t)
        res[t] = hazards(L, F, dt)
    if abs(res[tag]["exposure_s"] - res[ref]["exposure_s"]) > 1e-6 * max(res[ref]["exposure_s"], 1.0):
        raise ValueError("exposures differ between the policies")
    res["hazard_ratio"] = [
        (b1["hazard_per_s"][0] / b2["hazard_per_s"][0]) if b2["hazard_per_s"][0] > 0 else float("nan")
        for b1, b2 in zip(res[tag]["bins"], res[ref]["bins"])]
    res["tags"] = [tag, ref]
    return res


def text(res):
    tag, ref = res["tags"]
    L = [f"Startup hazard (read-only; saved evaluations): {tag} vs {ref}, per reset and per second at risk"]
    for t in (tag, ref):
        r = res[t]
        L.append(f"{t}: {r['resets']} resets, {r['falls']} falls -> {r['falls_per_reset']:.3f} falls per reset; "
                 f"fall within 32 rows {r['fall_within_32_rows_per_reset']:.3f}, within 40 rows "
                 f"{r['fall_within_40_rows_per_reset']:.3f} per reset; exposure {r['exposure_s']:.0f} s")
    L.append(f"{'rows after reset':>16s} | " + " | ".join(
        f"{t}: at risk, falls, hazard/s [CI], P(fall|reached)" for t in (tag, ref)) + " | ratio")
    for i, b in enumerate(res[tag]["bins"]):
        c = res[ref]["bins"][i]
        f = lambda x: (f"{x['at_risk']:5d} {x['falls']:4d} {x['hazard_per_s'][0]:.4f} "
                       f"[{x['hazard_per_s'][1]:.4f}, {x['hazard_per_s'][2]:.4f}] {x['conditional']:.3f}")
        L.append(f"{b['bin']:>16s} | {f(b)} | {f(c)} | {res['hazard_ratio'][i]:.2f}")
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--tag", default="D3")
    p.add_argument("--ref", default="S2")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=False)
    res = analyse(a.eval, a.tag, a.ref)
    (a.out / "startup_hazard.json").write_text(json.dumps(res, indent=1, default=float))
    t = text(res)
    (a.out / "startup_hazard.txt").write_text(t)
    print(t, end="")


if __name__ == "__main__":
    main()
