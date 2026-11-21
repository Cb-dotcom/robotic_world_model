#!/usr/bin/env python3
"""Plan v6 amendment A3 §4 (21 Nov 2026): score gate G0, the first Go2 gate.

Inputs: the full-precision summaries eval_episode_stats.py writes beside its CSV,
`<dir>/<TAG>_cmd<10|30>_s<42|43>/episodes_summary.json`, of the G0 policy and of s2's G6 check
(`logs/p2/startunitree_1115_2350/eval/unitree_s2_cmd*_s*/`). The protocol must be the same in every pair of cells
(task, envs, steps, seed, std, command resampling, push setting); otherwise the gate stops.

PASS only if both hold at 10 s commands (A2 movement gate, unchanged by A3):
  speed_ratio(G0) >= 0.80                      mean of seeds 42 and 43
  track_per_step(G0) >= T_s2 - 0.030           T_s2 = the UNROUNDED mean of s2's track_per_step, seeds 42 and 43,
                                               read from the logs' full-precision values
30 s numbers and falls per 1000 s are reported, not gated. A valid miss stops the scored U/R study (A3 §4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fall_rate import falls_per_1000s  # noqa: E402

SEEDS, EXPOSURE_PER_SEED = (42, 43), 32 * 10000 * 0.02
SPEED_MIN, TRACK_MARGIN = 0.80, 0.030
FIELDS = ("falls", "episodes_completed", "track_per_step", "speed_ratio")
PROTOCOL = ("task", "num_envs", "num_steps", "seed", "std", "cmd_resample_s")


class ProtocolMismatch(ValueError):
    pass


def summary(path):
    """Full-precision summary and protocol of one evaluation cell."""
    d = json.loads(Path(path).read_text())
    s = d["summary"]
    out = {name: float(s[name]) for name in FIELDS}
    out["protocol"] = {k: d.get(k) for k in PROTOCOL}
    out["protocol"]["push_enabled"] = bool((d.get("push_robot") or {}).get("enabled", False))
    return out


def cell(folder, tag, cmd):
    rows = [summary(Path(folder) / f"{tag}_cmd{cmd}_s{s}" / "episodes_summary.json") for s in SEEDS]
    falls = int(sum(r["falls"] for r in rows))
    rate, lo, hi = falls_per_1000s(falls, EXPOSURE_PER_SEED * len(rows))
    return dict(falls=falls, rate_per_1000s=rate, ci95=[lo, hi],
                speed_ratio=sum(r["speed_ratio"] for r in rows) / len(rows),
                track_per_step=sum(r["track_per_step"] for r in rows) / len(rows), per_seed=rows)


def score(g0_dir, g0_tag, s2_dir, s2_tag):
    res = {cmd: dict(g0=cell(g0_dir, g0_tag, cmd), s2=cell(s2_dir, s2_tag, cmd)) for cmd in (10, 30)}
    for cmd in (10, 30):
        for a, b in zip(res[cmd]["g0"]["per_seed"], res[cmd]["s2"]["per_seed"]):
            if a["protocol"] != b["protocol"]:
                raise ProtocolMismatch(f"{cmd} s: G0 protocol {a['protocol']} != s2 protocol {b['protocol']}")
    g, s = res[10]["g0"], res[10]["s2"]
    t_ref = s["track_per_step"]
    checks = dict(speed=g["speed_ratio"] >= SPEED_MIN, tracking=g["track_per_step"] >= t_ref - TRACK_MARGIN)
    return dict(rule=dict(speed_min=SPEED_MIN, track_margin=TRACK_MARGIN, endpoint="10 s, seeds 42 + 43",
                          s2_reference_track_unrounded=t_ref, tracking_bar=t_ref - TRACK_MARGIN),
                results=res, checks=checks, verdict="PASS" if all(checks.values()) else "FAIL")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--g0", type=Path, required=True, help="eval folder of the G0 policy")
    p.add_argument("--g0-tag", default="G0")
    p.add_argument("--s2", type=Path, required=True, help="eval folder of s2's G6 check")
    p.add_argument("--s2-tag", default="unitree_s2")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    r = score(a.g0, a.g0_tag, a.s2, a.s2_tag)
    a.out.write_text(json.dumps(r, indent=1))
    lines = ["Plan v6 amendment A3: gate G0 (Go2 policy trained in imagination, scored in physics)"]
    for cmd in (10, 30):
        for name in ("g0", "s2"):
            x = r["results"][cmd][name]
            lines.append(f"  {cmd} s {name}: falls {x['falls']:3d} -> {x['rate_per_1000s']:.2f} per 1000 s "
                         f"[{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]; speed_ratio {x['speed_ratio']:.4f}; "
                         f"track_per_step {x['track_per_step']:.6f}")
    c, rule = r["checks"], r["rule"]
    lines.append(f"checks (10 s): speed_ratio >= 0.80 {c['speed']}; track_per_step >= {rule['tracking_bar']:.6f} "
                 f"(s2 {rule['s2_reference_track_unrounded']:.6f} - 0.030) {c['tracking']}")
    lines.append(f"VERDICT: {r['verdict']} (A3 §4: a valid FAIL stops the scored U/R study; fallback defined)")
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    a.out.with_suffix(".txt").write_text(text)


if __name__ == "__main__":
    main()
