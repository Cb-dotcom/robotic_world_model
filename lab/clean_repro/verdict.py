#!/usr/bin/env python3
"""Verdict for the clean-repro test and the lambda sweep. Read-only; prints a report.

Usage: python3 verdict.py <OUT dir>
Layout: <OUT>/<run>/train/{manifest.json,metrics.jsonl}, <OUT>/<run>/eval_<proto>_<seed>/episodes_summary.json
Protocols: std = Pretrain task, 10 s commands, pushes on (decides); cmd30 = same with 30 s commands;
play = Isaac-Velocity-Flat-Anymal-D-Play-v0 defaults (reported only).

LOCKED RULE (approved before any result; never change it afterwards):
  per clean policy, pooled eval seeds 42+43, protocol std:
    WORKS  : falls <= 6 and tracking >= 0.859 and speed_ratio >= 0.8
    BETTER : not WORKS, falls <= 78 (half of run 1601's 156/689), tracking >= 0.859, speed_ratio >= 0.8
    SAME   : otherwise (tag SLOW if speed_ratio < 0.8)
  overall: R1 and R2 in the same class -> that class; else MIXED (train R3, take the majority).
"""
import json
import statistics
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
REF = {"ceiling (ETH online policy)": (3, 641), "our paper-mode run 1601": (156, 689)}


def summary(d):
    p = d / "episodes_summary.json"
    if not p.is_file():
        return None
    r = json.loads(p.read_text())
    return r["summary"]


def pool(run, proto):
    rs = [summary(OUT / run / f"eval_{proto}_{s}") for s in (42, 43)]
    rs = [r for r in rs if r]
    if len(rs) < 2:
        return None, len(rs)
    steps = sum(r["counted_steps"] for r in rs)
    cmd = sum(r["speed_cmd_per_step"] * r["counted_steps"] for r in rs)
    return {
        "falls": sum(r["falls"] for r in rs),
        "episodes": sum(r["episodes_completed"] for r in rs),
        "tracking": sum(r["track_per_step"] * r["counted_steps"] for r in rs) / steps,
        "speed_ratio": sum(r["speed_act_per_step"] * r["counted_steps"] for r in rs) / cmd,
    }, 2


def classify(p):
    ok = p["tracking"] >= 0.859 and p["speed_ratio"] >= 0.8
    if ok and p["falls"] <= 6:
        return "WORKS"
    if ok and p["falls"] <= 78:
        return "BETTER"
    return "SAME" + (" (SLOW)" if p["speed_ratio"] < 0.8 else "")


def imagined(run):
    p = OUT / run / "train" / "metrics.jsonl"
    if not p.is_file():
        return {}
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    last = rows[-50:]
    keys = sorted({k for r in last for k in r if any(w in k.lower() for w in
                   ("uncertainty", "track", "mean_reward", "episode_length", "stand_still"))})
    return {k: statistics.mean(r[k] for r in last if k in r) for k in keys} | {"iterations_logged": len(rows)}


def fmt(p):
    return (f"falls {p['falls']}/{p['episodes']} ({100 * p['falls'] / max(p['episodes'], 1):.1f}%/ep), "
            f"tracking {p['tracking']:.3f}, speed {p['speed_ratio']:.3f}")


runs = sorted(d.name for d in OUT.iterdir() if d.is_dir() and (d / "train").is_dir())
print("References (std protocol):", "; ".join(f"{k} {a}/{b}" for k, (a, b) in REF.items()))
classes = {}
for run in runs:
    man = OUT / run / "train" / "manifest.json"
    lam = json.loads(man.read_text()).get("lambda_used") if man.is_file() else "?"
    print(f"\n== {run}  (lambda={lam})")
    for proto in ("std", "cmd30"):
        p, n = pool(run, proto)
        print(f"  {proto:5s}: " + (fmt(p) if p else f"{n}/2 evaluations finished"))
        if proto == "std" and p:
            classes[run] = classify(p)
    s10, s30 = pool(run, "std")[0], pool(run, "cmd30")[0]
    if s10 and s30:
        r10, r30 = s10["falls"] / s10["episodes"], s30["falls"] / s30["episodes"]
        print(f"  falls from command changes (rate 10 s - rate 30 s): {100 * (r10 - r30):.1f} points per episode")
    play = summary(OUT / run / "eval_play_42")
    if play:
        print(f"  play : falls {play['falls']}/{play['episodes_completed']}, tracking {play['track_per_step']:.3f}, "
              f"speed {play['speed_act_per_step'] / max(play['speed_cmd_per_step'], 1e-9):.3f}  (reported only)")
    im = imagined(run)
    if im:
        print("  imagination (mean of last 50 iterations):", json.dumps({k: round(v, 4) for k, v in im.items()}))
    if run in classes:
        print(f"  CLASS (locked rule, repro runs only): {classes[run]}")

ceil = summary(OUT / "ceiling" / "eval_play_42")
if ceil:
    print(f"\nceiling play: falls {ceil['falls']}/{ceil['episodes_completed']}, tracking {ceil['track_per_step']:.3f}")

rep = [classes[r].split()[0] for r in ("R1", "R2", "R3") if r in classes]
if len(rep) == 3:
    best = max(set(rep), key=rep.count)
    print("\nREPRO VERDICT:", best if rep.count(best) >= 2 else "NO MAJORITY (3 different classes)", f"(R1-R3: {rep})")
elif len(rep) == 2:
    print("\nREPRO VERDICT:", rep[0] if rep[0] == rep[1] else "MIXED (train R3; majority decides)")
else:
    print(f"\nREPRO VERDICT: pending ({len(rep)}/2 clean policies scored)")
