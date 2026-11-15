#!/usr/bin/env python3
"""Score a Go2 start-policy candidate against the pass gate fixed on 14 Nov 2026 (decisions-log). CPU.

Inputs: the candidate's gait_metrics.py JSON, and evaluation folders (eval_episode_stats.py logs)
for the candidate and the June walker, named <tag>_cmd<10|30>_s<seed>/eval.log.

Bars (all must pass):
  G1 base height >= 0.25 m in >= 99 % of samples at every speed   (low_body_fraction_lt_0p25m <= 0.01)
  G2 0 falls at every speed (terminations and plan-definition events)
  G3 speed ratio 0.85-1.10 for every speed >= 0.2 m/s
  G4 foot clearance >= 3 cm for every speed >= 0.2 m/s
  G5 pitch std <= 0.6 deg and trot index >= 0.8 for every speed >= 0.2 m/s
  G6 evaluation falls per 1000 s at 10 s and at 30 s commands <= the June walker's (same task)

With --reference-gait (iteration 3, decision 15 Nov 2026): a gait_metrics.py JSON of the reference walker
(Unitree's Go2 recipe, model_18200, same gait test). G4 changes and two posture bars are added:
  G4 foot clearance >= 0.8 x the reference's lowest clearance (its speeds >= 0.2 m/s), every speed >= 0.2 m/s
  G7 base height within +-0.03 m of the reference at every speed both measured
  G8 base height std (bounce) <= 2 x the reference's at every speed both measured
Without --reference-gait the gate is the 14 Nov gate, unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys


def eval_falls(folder: Path, tag: str, cmd: int, num_envs: int, num_steps: int, dt: float):
    """Pooled falls per 1000 commanded s over the seeds found, or None."""
    falls = seconds = 0.0
    found = 0
    for log in sorted(folder.glob(f"{tag}_cmd{cmd}_s*/eval.log")):
        text = log.read_text(errors="replace")
        block = text[text.rfind("EPISODE STATS"):] if "EPISODE STATS" in text else ""
        m = re.search(r"falls:\s*([0-9]+)", block)
        if not m:
            continue
        falls += int(m.group(1))
        seconds += num_envs * num_steps * dt
        found += 1
    return (1000.0 * falls / seconds, found) if found else (None, 0)


HEIGHT_TOL_M = 0.03
BOUNCE_FACTOR = 2.0
CLEARANCE_FACTOR = 0.8


def _num(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def check(gait: dict, cand: dict, june: dict, ref: dict | None = None):
    rows = []
    s = {round(float(k), 3): v for k, v in gait["summary"].items()}
    speeds = sorted(s)
    moving = [v for v in speeds if v >= 0.2 - 1e-9]

    def row(name, ok, detail):
        rows.append((name, bool(ok), detail))

    row("G1 body >= 0.25 m in >= 99 %", all(s[v].get("low_body_fraction_lt_0p25m", 1.0) <= 0.01 for v in speeds),
        {v: s[v].get("low_body_fraction_lt_0p25m") for v in speeds})
    row("G2 no falls", all(s[v]["falls"] == 0 and s[v].get("plan_fall_events", 1) == 0 for v in speeds),
        {v: (s[v]["falls"], s[v].get("plan_fall_events")) for v in speeds})
    row("G3 speed ratio 0.85-1.10", all(0.85 <= s[v]["speed_ratio"] <= 1.10 for v in moving),
        {v: round(s[v]["speed_ratio"], 3) for v in moving})
    r = None if ref is None else {round(float(k), 3): v for k, v in ref["summary"].items()}
    if r is None:
        row("G4 clearance >= 3 cm", all(s[v]["foot_clearance_cm_mean"] >= 3.0 for v in moving),
            {v: round(s[v]["foot_clearance_cm_mean"], 2) for v in moving})
    else:
        ref_clear = [r[v]["foot_clearance_cm_mean"] for v in r if v >= 0.2 - 1e-9 and _num(r[v].get("foot_clearance_cm_mean"))]
        bar = CLEARANCE_FACTOR * min(ref_clear) if ref_clear else float("nan")
        row(f"G4 clearance >= {CLEARANCE_FACTOR} x reference min ({bar:.2f} cm)",
            bool(ref_clear) and all(_num(s[v]["foot_clearance_cm_mean"]) and s[v]["foot_clearance_cm_mean"] >= bar for v in moving),
            {v: round(s[v]["foot_clearance_cm_mean"], 2) for v in moving})
    row("G5 pitch std <= 0.6 and trot >= 0.8", all(s[v]["pitch_std_deg"] <= 0.6 and s[v]["trot_index"] >= 0.8 for v in moving),
        {v: (round(s[v]["pitch_std_deg"], 2), round(s[v]["trot_index"], 3)) for v in moving})
    ok6, det6 = True, {}
    for cmd in (10, 30):
        c, j = cand.get(cmd), june.get(cmd)
        det6[cmd] = (c, j)
        ok6 = ok6 and c is not None and j is not None and c <= j
    row("G6 eval falls/1000 s <= June walker", ok6, det6)
    if r is not None:
        common = [v for v in moving if v in r]
        h = {v: (s[v].get("base_height_mean_m"), r[v].get("base_height_mean_m")) for v in common}
        row(f"G7 base height within +-{HEIGHT_TOL_M} m of reference",
            bool(common) and all(_num(a) and _num(b) and abs(a - b) <= HEIGHT_TOL_M for a, b in h.values()),
            {v: (round(a, 3) if _num(a) else a, round(b, 3) if _num(b) else b) for v, (a, b) in h.items()})
        bo = {v: (s[v].get("base_height_std_cm"), r[v].get("base_height_std_cm")) for v in common}
        row(f"G8 bounce (height std) <= {BOUNCE_FACTOR} x reference",
            bool(common) and all(_num(a) and _num(b) and a <= BOUNCE_FACTOR * b for a, b in bo.values()),
            {v: (round(a, 2) if _num(a) else a, round(b, 2) if _num(b) else b) for v, (a, b) in bo.items()})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gait", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--tag", required=True, help="candidate eval folder prefix, e.g. vel")
    p.add_argument("--june-tag", default="june")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=10000)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--reference-gait", type=Path, default=None, help="iteration-3 gate: reference walker gait JSON")
    a = p.parse_args(argv)
    gait = json.loads(a.gait.read_text())
    ref = json.loads(a.reference_gait.read_text()) if a.reference_gait else None
    cand = {c: eval_falls(a.eval_dir, a.tag, c, a.num_envs, a.num_steps, a.dt)[0] for c in (10, 30)}
    june = {c: eval_falls(a.eval_dir, a.june_tag, c, a.num_envs, a.num_steps, a.dt)[0] for c in (10, 30)}
    rows = check(gait, cand, june, ref)
    which = "decision 14 Nov 2026" if ref is None else f"decision 15 Nov 2026, reference {ref.get('label')}"
    lines = [f"Start-policy gate ({which}): {gait.get('label')} ({gait.get('checkpoint')})"]
    lines += [f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}" for name, ok, detail in rows]
    passed = all(ok for _, ok, _ in rows)
    lines.append(f"GATE PASS: {passed}")
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
