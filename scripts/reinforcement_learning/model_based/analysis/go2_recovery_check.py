#!/usr/bin/env python3
"""A4 (22 Nov 2026, Charbel's question): can the baseline policy s2 come back from a near-fall in physics?
Read-only, CPU, descriptive. Data: existing s2 recordings (no world model involved).

Near-fall event: body tilt crosses 30 deg upward (|projected gravity xy| > sin 30 deg), at least 10 rows after the
previous event of the same episode (same detection as go2_friction_shift.py).
Outcome per event (A4 D1b definitions, fixed 21 Nov; 50 rows = 1 s):
  fell        : the episode ends by a fall within 250 rows (5 s) of the onset;
  recovered   : no fall within 250 rows, AND tilt back below 10 deg within 50 rows, AND base-velocity tracking error
                |v_xy - command_xy| < 0.25 m/s on some row within 100 rows;
  not recovered: no fall within 250 rows but upright or tracking not regained in time;
  censored    : the episode ends by time-out (or the recording ends) within 250 rows with no fall.
Reported per recording and by the maximum tilt reached in the 50 rows after the onset (30-40, 40-50, 50-60, > 60 deg).
Reading (fixed before the run): the "point of no return" is the lowest tilt bin whose recovered share among
uncensored events is <= 0.10 (bins with fewer than 5 uncensored events are not read).
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

ONSET_XY, UPRIGHT_DEG, TRACK = 0.5, 10.0, 0.25
GAP, UP_WIN, TRACK_WIN, FALL_WIN, PEAK_WIN = 10, 50, 100, 250, 50
BINS = ((30, 40), (40, 50), (50, 60), (60, 181))


def tilt(g):
    g = np.asarray(g, np.float64)
    return np.degrees(np.arccos(np.clip(-g[..., 2] / np.maximum(np.linalg.norm(g, axis=-1), 1e-9), -1, 1)))


def events(trace):
    ns, done = trace["next_state"], trace["done"]
    term, tout, cmd = trace["terminated"], trace["time_out"], trace["command"]
    T, N = done.shape
    out = []
    for n in range(N):
        s = 0
        ends = list(np.flatnonzero(done[:, n])) + ([T - 1] if not done[T - 1, n] else [])
        for e in ends:
            e = int(e)
            finished = bool(done[e, n])
            fall = bool(finished and term[e, n] and not tout[e, n])
            g = ns[s:e + 1, n, 6:9]
            tl = tilt(g)
            xy = np.linalg.norm(g[:, :2], axis=1) > ONSET_XY
            err = np.linalg.norm(ns[s:e + 1, n, 0:2] - cmd[s:e + 1, n, 0:2], axis=1)
            last = -10 ** 9
            for k in range(1, len(tl)):
                if xy[k] and not xy[k - 1] and k - last >= GAP:
                    last = k
                    rows_left = len(tl) - 1 - k
                    peak = float(tl[k:k + PEAK_WIN + 1].max())
                    if fall and rows_left <= FALL_WIN:
                        outcome = "fell"
                    elif rows_left < FALL_WIN:
                        outcome = "censored"
                    else:
                        up = (tl[k + 1:k + UP_WIN + 1] < UPRIGHT_DEG).any()
                        tr = (err[k + 1:k + TRACK_WIN + 1] < TRACK).any()
                        outcome = "recovered" if (up and tr) else "not recovered"
                    out.append(dict(env=n, episode_first=s, onset=s + k, peak_tilt=peak, outcome=outcome))
            s = e + 1
    return out


def summarize(evs):
    def count(sel):
        c = {k: sum(e["outcome"] == k for e in sel) for k in ("recovered", "not recovered", "fell", "censored")}
        unc = c["recovered"] + c["not recovered"] + c["fell"]
        c["uncensored"] = unc
        c["recovered_share"] = c["recovered"] / unc if unc else float("nan")
        return c
    s = dict(all=count(evs), by_peak_tilt={})
    for lo, hi in BINS:
        s["by_peak_tilt"][f"{lo}-{hi if hi < 181 else 'max'}"] = count([e for e in evs if lo <= e["peak_tilt"] < hi])
    ponr = None
    for k, c in s["by_peak_tilt"].items():
        if c["uncensored"] >= 5 and c["recovered_share"] <= 0.10:
            ponr = k
            break
    s["point_of_no_return_bin"] = ponr
    return s


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", action="append", required=True, help="name=path, repeatable")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    rep = dict(protocol="A4 s2 recovery check (physics recordings, descriptive)", analysis_sha256=sha256(__file__), sets={})
    lines = ["A4 s2 recovery check (physics only; no world model)"]
    for spec in a.trace:
        name, path = spec.split("=", 1)
        with np.load(path, allow_pickle=False) as z:
            meta = validate_trace(z)
            tr = {k: z[k] for k in ("next_state", "command", "done", "terminated", "time_out")}
        evs = events(tr)
        s = summarize(evs)
        rep["sets"][name] = dict(trace=path, trace_sha256=sha256(path),
                                 friction=(meta.get("friction_shift") or {}).get("factor", 1.0), summary=s, events=evs)
        c = s["all"]
        lines.append(f"{name} (friction x{rep['sets'][name]['friction']}): {len(evs)} near-falls: recovered {c['recovered']}, "
                     f"not recovered {c['not recovered']}, fell {c['fell']}, censored {c['censored']}; recovered share "
                     f"{c['recovered_share']:.2f} of {c['uncensored']}")
        lines.append("  by peak tilt: " + "; ".join(f"{k} deg: {v['recovered']}/{v['uncensored']}"
                                                    for k, v in s["by_peak_tilt"].items())
                     + f"; point of no return: {s['point_of_no_return_bin']}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as f:
        json.dump(rep, f, indent=1, default=float)
    print("\n".join(lines), flush=True)
    return rep


if __name__ == "__main__":
    main()
