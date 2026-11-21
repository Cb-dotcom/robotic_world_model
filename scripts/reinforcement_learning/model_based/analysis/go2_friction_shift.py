#!/usr/bin/env python3
"""A4 D1 (21 Nov 2026): falls and recoveries in Go2 recordings under a ground-friction shift; level selection.

Definitions (fixed before any D1 recording; A4 §1):
  fall      : an episode that ends by physical termination (terminated and not time-out);
  near-fall : a row where body tilt crosses 30 deg upward (|projected gravity xy| > sin 30 deg = 0.5), within an
              episode, at least 10 rows after the previous near-fall of that episode;
  recovery  : a near-fall followed by >= 50 rows of the same episode with no fall (the episode may end later by
              time-out or a fall more than 50 rows on). Near-falls with fewer than 50 rows left are censored;
  recovery window (for model checks): rows onset - 10 .. onset + 50 of the episode.
Level selection: levels in order (0.60, 0.45, 0.30); the chosen level is the FIRST whose selection recording has
>= 15 falls, >= 15 recoveries and a fall fraction (falls / ended episodes) <= 0.40. None meets it -> no level.
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

from pre_reset_transitions import sha256, validate_trace  # noqa: E402

TILT_XY = 0.5          # sin(30 deg)
GAP, RECOVER, PRE = 10, 50, 10
LEVELS = (0.60, 0.45, 0.30)


def episodes(done):
    T, N = done.shape
    out = []
    for n in range(N):
        s = 0
        for e in np.flatnonzero(done[:, n]):
            out.append((n, s, int(e), True))
            s = int(e) + 1
        if s <= T - 1:
            out.append((n, s, T - 1, False))
    return out


def count(trace):
    """Counts and recovery windows of one recording (pre-reset schema)."""
    g = trace["next_state"][..., 6:8]
    tilt = np.sqrt((g ** 2).sum(-1)) > TILT_XY
    term, tout = trace["terminated"], trace["time_out"]
    c = dict(episodes=0, ended=0, falls=0, time_outs=0, near_falls=0, recoveries=0, censored=0, near_falls_before_fall=0)
    windows = []
    for n, s, e, finished in episodes(trace["done"]):
        c["episodes"] += 1
        fall = bool(finished and term[e, n] and not tout[e, n])
        c["ended"] += finished
        c["falls"] += fall
        c["time_outs"] += bool(finished and tout[e, n])
        last = -10 ** 9
        for t in range(s + 1, e + 1):
            if tilt[t, n] and not tilt[t - 1, n] and t - last >= GAP:
                last = t
                c["near_falls"] += 1
                if fall and e - t < RECOVER:
                    c["near_falls_before_fall"] += 1
                elif e - t < RECOVER:
                    c["censored"] += 1
                else:
                    c["recoveries"] += 1
                    windows.append(dict(env=n, episode_first=s, onset=t, lo=max(s, t - PRE), hi=t + RECOVER))
    c["fall_fraction"] = c["falls"] / c["ended"] if c["ended"] else float("nan")
    return c, windows


def choose(results, min_falls=15, min_recoveries=15, max_fall_fraction=0.40):
    """results: [(level, counts)] in ladder order -> the first level meeting the rule, else None."""
    for level, c in results:
        if c["falls"] >= min_falls and c["recoveries"] >= min_recoveries and c["fall_fraction"] <= max_fall_fraction:
            return level
    return None


def load(path):
    with np.load(path, allow_pickle=False) as z:
        meta = validate_trace(z)
        return {k: z[k] for k in z.files if k != "metadata"}, meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("count")
    a1.add_argument("--trace", type=Path, required=True)
    a1.add_argument("--out", type=Path, required=True)
    a1.add_argument("--min-falls", type=int, default=None)
    a1.add_argument("--min-recoveries", type=int, default=None)
    a2 = sub.add_parser("choose")
    a2.add_argument("--counts", type=Path, nargs="+", required=True, help="count JSONs in ladder order")
    a2.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.cmd == "count":
        tr, meta = load(a.trace)
        fs = meta.get("friction_shift") or {}
        if not fs.get("verified"):
            raise SystemExit(f"{a.trace}: no verified friction shift in the metadata")
        c, w = count(tr)
        rec = dict(trace=str(a.trace), trace_sha256=sha256(a.trace), friction_factor=fs["factor"], seed=meta.get("seed"),
                   num_envs=meta.get("num_envs"), num_steps=meta.get("num_steps"), counts=c, recovery_windows=w,
                   analysis_sha256=sha256(__file__))
        ok = True
        if a.min_falls is not None or a.min_recoveries is not None:
            ok = c["falls"] >= (a.min_falls or 0) and c["recoveries"] >= (a.min_recoveries or 0)
            rec["minimum_met"] = ok
        a.out.write_text(json.dumps(rec, indent=1))
        print(f"[d1] {a.trace.name}: friction x{fs['factor']}, {c['ended']} ended episodes, {c['falls']} falls "
              f"(fraction {c['fall_fraction']:.3f}), {c['time_outs']} time-outs, {c['near_falls']} near-falls: "
              f"{c['recoveries']} recoveries, {c['near_falls_before_fall']} before a fall, {c['censored']} censored"
              + ("" if "minimum_met" not in rec else f"; minimum met: {ok}"), flush=True)
        return 0 if ok else 3
    results = []
    for path in a.counts:
        r = json.loads(path.read_text())
        results.append((r["friction_factor"], r["counts"]))
    level = choose(results)
    out = dict(rule="first level in ladder order with >= 15 falls, >= 15 recoveries, fall fraction <= 0.40",
               ladder=[lv for lv, _ in results], counts={str(lv): c for lv, c in results}, chosen=level,
               analysis_sha256=sha256(__file__))
    a.out.write_text(json.dumps(out, indent=1))
    print(f"[d1] chosen friction level: {level}", flush=True)
    return 0 if level is not None else 4


if __name__ == "__main__":
    sys.exit(main())
