#!/usr/bin/env python3
"""A4 D3 milestone gate (rules fixed 21-22 Nov 2026, A4 §1): does one short update from s2 in the qualified Go2 model
keep s2 walking under the frozen friction shift?

Physics evaluation (s2's G6 protocol: Go2-Start-v0, 32 envs x 10000 steps, std 0.5, 10 s commands, seeds 42 + 43,
12,800 s per policy) of the D3 policy and of UNCHANGED s2, both under the same verified ground-friction factor.
Milestone met only if all hold: speed_ratio(D3) >= 0.80; track_per_step(D3) >= track_per_step(s2) - 0.030;
falls(D3) <= 1.25 x falls(s2) + 3 (counts at matched exposure). Falls at comparable tracking are the primary
adaptation metric; all three numbers are reported.
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

SPEED_MIN, TRACK_MARGIN, FALL_FACTOR, FALL_SLACK = 0.80, 0.030, 1.25, 3


def frictions(folder, tag):
    out = []
    for s in SEEDS:
        d = json.loads((Path(folder) / f"{tag}_cmd10_s{s}" / "episodes_summary.json").read_text())
        fs = d.get("friction_shift") or {}
        if not fs.get("verified"):
            raise ProtocolMismatch(f"{tag} seed {s}: no verified friction shift in the summary")
        out.append(float(fs["factor"]))
    return out


def score(eval_dir, factor):
    d, s = cell(eval_dir, "D3", 10), cell(eval_dir, "S2", 10)
    for a, b in zip(d["per_seed"], s["per_seed"]):
        if a["protocol"] != b["protocol"]:
            raise ProtocolMismatch(f"D3 protocol {a['protocol']} != s2 protocol {b['protocol']}")
    for tag in ("D3", "S2"):
        if any(abs(f - factor) > 1e-6 for f in frictions(eval_dir, tag)):
            raise ProtocolMismatch(f"{tag}: friction factor differs from {factor}")
    fall_bar = FALL_FACTOR * s["falls"] + FALL_SLACK
    checks = dict(speed=d["speed_ratio"] >= SPEED_MIN, tracking=d["track_per_step"] >= s["track_per_step"] - TRACK_MARGIN,
                  falls=d["falls"] <= fall_bar)
    return dict(rule=dict(speed_min=SPEED_MIN, track_margin=TRACK_MARGIN, fall_factor=FALL_FACTOR, fall_slack=FALL_SLACK,
                          friction=factor, endpoint="10 s commands, seeds 42 + 43", tracking_bar=s["track_per_step"] - TRACK_MARGIN,
                          fall_bar=fall_bar),
                d3=d, s2=s, checks=checks, verdict="MILESTONE MET" if all(checks.values()) else "NOT MET")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", type=Path, required=True, help="folder with D3_cmd10_s{42,43} and S2_cmd10_s{42,43}")
    p.add_argument("--friction", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    r = score(a.eval, a.friction)
    with a.out.open("x") as f:
        json.dump(r, f, indent=1)
    d, s = r["d3"], r["s2"]
    print(f"A4 D3 gate (friction x{a.friction}, 10 s commands, 12,800 s per policy)")
    print(f"  falls: D3 {d['falls']} ({d['rate_per_1000s']:.2f} per 1000 s {[round(x, 2) for x in d['ci95']]}) vs s2 {s['falls']} "
          f"({s['rate_per_1000s']:.2f} {[round(x, 2) for x in s['ci95']]}); bar {r['rule']['fall_bar']:.2f} -> {r['checks']['falls']}")
    print(f"  speed_ratio: D3 {d['speed_ratio']:.3f} vs s2 {s['speed_ratio']:.3f}; bar {SPEED_MIN} -> {r['checks']['speed']}")
    print(f"  track_per_step: D3 {d['track_per_step']:.4f} vs s2 {s['track_per_step']:.4f}; bar {r['rule']['tracking_bar']:.4f} "
          f"-> {r['checks']['tracking']}")
    print(f"VERDICT: {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
