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
"""
from __future__ import annotations

import argparse
import json
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


def check(gait: dict, cand: dict, june: dict):
    rows = []
    s = {float(k): v for k, v in gait["summary"].items()}
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
    row("G4 clearance >= 3 cm", all(s[v]["foot_clearance_cm_mean"] >= 3.0 for v in moving),
        {v: round(s[v]["foot_clearance_cm_mean"], 2) for v in moving})
    row("G5 pitch std <= 0.6 and trot >= 0.8", all(s[v]["pitch_std_deg"] <= 0.6 and s[v]["trot_index"] >= 0.8 for v in moving),
        {v: (round(s[v]["pitch_std_deg"], 2), round(s[v]["trot_index"], 3)) for v in moving})
    ok6, det6 = True, {}
    for cmd in (10, 30):
        c, j = cand.get(cmd), june.get(cmd)
        det6[cmd] = (c, j)
        ok6 = ok6 and c is not None and j is not None and c <= j
    row("G6 eval falls/1000 s <= June walker", ok6, det6)
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
    a = p.parse_args(argv)
    gait = json.loads(a.gait.read_text())
    cand = {c: eval_falls(a.eval_dir, a.tag, c, a.num_envs, a.num_steps, a.dt)[0] for c in (10, 30)}
    june = {c: eval_falls(a.eval_dir, a.june_tag, c, a.num_envs, a.num_steps, a.dt)[0] for c in (10, 30)}
    rows = check(gait, cand, june)
    lines = [f"Start-policy gate (decision 14 Nov 2026): {gait.get('label')} ({gait.get('checkpoint')})"]
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
