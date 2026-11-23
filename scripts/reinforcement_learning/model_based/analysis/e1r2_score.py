#!/usr/bin/env python3
"""A4 E1-R2 readout (rules fixed 23 Nov 2026, before any E1-R2 run; A4 §3): E1 on ANYmal with a fresh data
collection (data seed 102) and a fresh model fit (fit seed 2), frozen-model arm B' vs updated-model arm D', both
lambda 1, PPO seeds 1-3.

Inputs: the cells eval_episode_stats.py writes, `<dir>/<TAG>_cmd<10|30>_s<42|43>/episodes_summary.json`, one set per
arm and PPO seed, given as --arm B1=DIR:TAG ... --arm D3=DIR:TAG (arm letter B or D, then the PPO seed).
Per PPO seed and command schedule: falls pooled over evaluation seeds 42 + 43 (12,800 s) with an exact Poisson
interval; track_per_step and speed_ratio as the mean of the two evaluation seeds (go2_g0_gate.cell, task-agnostic).
Across seeds: the mean of the per-seed rates (as E1-R), the D'/B' ratio of the means, and the seeds with D' below B'.
All cells of one schedule and evaluation seed must share the protocol (task, envs, steps, seed, std, command
resampling, pushes); otherwise the readout stops.

Reading (descriptive; 3 pairs allow no significance test), at 10 s commands:
  NOT READABLE           a policy has speed_ratio < 0.80 (it does not move; E1 V3); its falls say nothing about safety
  SAME DIRECTION AS E1   D' falls < B' falls in every seed, and D' track_per_step >= B' track_per_step - 0.030 in each
  OPPOSITE               D' falls > B' falls in at least 2 of 3 seeds
  MIXED                  otherwise
30 s commands are reported with the same numbers, not read.

--expect TAG:CMD=N (repeatable) turns on a known-answer check of the reader on saved cells: every named pooled
fall count must equal N ("CROSSCHECK: PASS"). The E1-R2 launcher runs it on E1-R's saved cells before any GPU work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from go2_g0_gate import SEEDS, ProtocolMismatch, cell  # noqa: E402

CMDS, SPEED_MIN, TRACK_MARGIN = (10, 30), 0.80, 0.030


def parse_arms(items):
    arms = {}
    for item in items:
        tag, sep, loc = item.partition("=")
        folder, sep2, cell_tag = loc.rpartition(":")
        if not sep or not sep2 or len(tag) < 2 or tag[0] not in "BD" or not tag[1:].isdigit():
            raise SystemExit(f"--arm must look like B1=DIR:TAG or D3=DIR:TAG, got {item!r}")
        arms[tag] = (Path(folder), cell_tag)
    return arms


def read_cells(arms):
    out = {tag: {cmd: cell(folder, cell_tag, cmd) for cmd in CMDS} for tag, (folder, cell_tag) in arms.items()}
    for cmd in CMDS:
        for i, s in enumerate(SEEDS):
            protos = {tag: out[tag][cmd]["per_seed"][i]["protocol"] for tag in out}
            first = next(iter(protos.values()))
            for tag, p in protos.items():
                if p != first:
                    raise ProtocolMismatch(f"cmd {cmd} s{s}: {tag} protocol {p} != {first}")
    return out


def summarise(cells):
    seeds = sorted({int(t[1:]) for t in cells if t.startswith("B")} & {int(t[1:]) for t in cells if t.startswith("D")})
    res = {}
    for cmd in CMDS:
        per = []
        for s in seeds:
            b, d = cells[f"B{s}"][cmd], cells[f"D{s}"][cmd]
            per.append(dict(seed=s, b_falls=b["falls"], d_falls=d["falls"], b_rate=b["rate_per_1000s"],
                            d_rate=d["rate_per_1000s"], b_ci=b["ci95"], d_ci=d["ci95"], b_track=b["track_per_step"],
                            d_track=d["track_per_step"], b_speed=b["speed_ratio"], d_speed=d["speed_ratio"],
                            d_below=d["falls"] < b["falls"], d_above=d["falls"] > b["falls"],
                            track_ok=d["track_per_step"] >= b["track_per_step"] - TRACK_MARGIN))
        mb = sum(p["b_rate"] for p in per) / len(per) if per else float("nan")
        md = sum(p["d_rate"] for p in per) / len(per) if per else float("nan")
        res[cmd] = dict(per_seed=per, mean_b_rate=mb, mean_d_rate=md, ratio=(md / mb) if mb else float("nan"),
                        d_below=sum(p["d_below"] for p in per), d_above=sum(p["d_above"] for p in per), seeds=len(per))
    r10 = res[10]["per_seed"]
    if not r10:
        reading = "NOT READABLE (no paired seeds)"
    elif any(min(p["b_speed"], p["d_speed"]) < SPEED_MIN for p in r10):
        reading = "NOT READABLE (a policy does not move at 10 s commands)"
    elif all(p["d_below"] and p["track_ok"] for p in r10):
        reading = "SAME DIRECTION AS E1"
    elif sum(p["d_above"] for p in r10) >= 2:
        reading = "OPPOSITE"
    else:
        reading = "MIXED"
    return dict(seeds=seeds, by_cmd=res, reading=reading)


def check_expect(cells, expect):
    rows, ok = [], True
    for item in expect:
        key, sep, n = item.partition("=")
        tag, sep2, cmd = key.partition(":")
        if not sep or not sep2:
            raise SystemExit(f"--expect must look like B2:10=238, got {item!r}")
        got = cells[tag][int(cmd)]["falls"]
        good = got == int(n)
        ok &= good
        rows.append(f"  {tag} {cmd} s: falls {got}, expected {n} -> {'ok' if good else 'DIFFERENT'}")
    return ok, rows


def text(r, expect_rows=None, expect_ok=None):
    L = ["A4 E1-R2 readout: frozen-model arm B' vs updated-model arm D' (lambda 1; falls pooled over eval seeds 42 + 43)"]
    for cmd in CMDS:
        c = r["by_cmd"][cmd]
        L.append(f"{cmd} s commands:")
        for p in c["per_seed"]:
            L.append(f"  seed {p['seed']}: B' {p['b_falls']} ({p['b_rate']:.2f} per 1000 s, track {p['b_track']:.3f}, speed "
                     f"{p['b_speed']:.2f}) | D' {p['d_falls']} ({p['d_rate']:.2f}, track {p['d_track']:.3f}, speed "
                     f"{p['d_speed']:.2f}) | D' below B': {p['d_below']}")
        L.append(f"  mean rate B' {c['mean_b_rate']:.2f} vs D' {c['mean_d_rate']:.2f} per 1000 s; D'/B' {c['ratio']:.2f}; "
                 f"D' below in {c['d_below']} of {c['seeds']} seeds")
    L.append(f"READING (10 s commands): {r['reading']}")
    if expect_rows is not None:
        L += ["Known-answer check of the reader:"] + expect_rows + [f"CROSSCHECK: {'PASS' if expect_ok else 'FAIL'}"]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, help="B1=DIR:TAG ... D3=DIR:TAG")
    p.add_argument("--expect", action="append", default=None, help="known answer TAG:CMD=FALLS (crosscheck)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    cells = read_cells(parse_arms(a.arm))
    r = summarise(cells)
    exp_ok, exp_rows = (check_expect(cells, a.expect) if a.expect else (None, None))
    r["crosscheck"] = None if a.expect is None else dict(passed=exp_ok, rows=exp_rows)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "e1r2_score.json").write_text(json.dumps(dict(result=r, cells=cells), indent=1, default=float))
    t = text(r, exp_rows, exp_ok)
    (a.out / "e1r2_score.txt").write_text(t)
    print(t, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
