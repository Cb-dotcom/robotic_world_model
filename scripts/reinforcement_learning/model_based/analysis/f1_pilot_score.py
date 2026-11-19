#!/usr/bin/env python3
"""Plan v6 amendment A1 §5: score the F1 development pilot against arm D (run 2004), go / no-go.

Inputs: eval_episode_stats.py logs `<dir>/<TAG>_cmd<10|30>_s<42|43>/eval.log` of the pilot and of D1.
Endpoint: falls per 1000 s at 10 s commands, seeds 42 + 43 pooled (falls added; exposure 32 x 10000 x 0.02 s per
seed). GO only if all hold (fixed 19 Nov, before training):
  1. pilot falls <= 0.7 x D1 falls (10 s, pooled);
  2. pilot speed_ratio >= 0.8 (mean of the two seeds, 10 s);
  3. pilot track_per_step >= D1 track_per_step - 0.03 (means of the two seeds, 10 s).
30 s numbers are reported, not part of the decision. NO-GO: report and stop, no retuning. GO: a proposal of
the full study may follow; it is not started automatically.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fall_rate import falls_per_1000s  # noqa: E402

SEEDS, EXPOSURE_PER_SEED = (42, 43), 32 * 10000 * 0.02
FALL_BAR, SPEED_MIN, TRACK_MARGIN = 0.7, 0.8, 0.03
FIELDS = ("falls", "episodes_completed", "track_per_step", "speed_ratio")


def stats_block(path):
    text = Path(path).read_text(errors="replace")
    if "EPISODE STATS" not in text:
        raise ValueError(f"{path}: no EPISODE STATS block")
    block = text[text.rfind("EPISODE STATS"):]
    out = {}
    for name in FIELDS:
        m = re.search(rf"^\s*{name}:\s*([-+0-9.eE]+|nan)\s*$", block, flags=re.MULTILINE)
        if not m:
            raise ValueError(f"{path}: no '{name}:' line")
        out[name] = float(m.group(1))
    return out


def arm(folder, tag, cmd):
    rows = [stats_block(Path(folder) / f"{tag}_cmd{cmd}_s{s}" / "eval.log") for s in SEEDS]
    falls = int(sum(r["falls"] for r in rows))
    exposure = EXPOSURE_PER_SEED * len(rows)
    rate, lo, hi = falls_per_1000s(falls, exposure)
    return dict(falls=falls, exposure_s=exposure, rate_per_1000s=rate, ci95=[lo, hi],
                speed_ratio=sum(r["speed_ratio"] for r in rows) / len(rows),
                track_per_step=sum(r["track_per_step"] for r in rows) / len(rows), per_seed=rows)


def score(pilot_dir, pilot_tag, d_dir, d_tag):
    res = {cmd: dict(pilot=arm(pilot_dir, pilot_tag, cmd), d1=arm(d_dir, d_tag, cmd)) for cmd in (10, 30)}
    p, d = res[10]["pilot"], res[10]["d1"]
    checks = dict(falls=p["falls"] <= FALL_BAR * d["falls"], speed=p["speed_ratio"] >= SPEED_MIN,
                  tracking=p["track_per_step"] >= d["track_per_step"] - TRACK_MARGIN)
    verdict = "GO" if all(checks.values()) else "NO-GO"
    return dict(rules=dict(fall_bar=FALL_BAR, speed_min=SPEED_MIN, track_margin=TRACK_MARGIN, endpoint="10 s, seeds 42+43"),
                results=res, checks=checks, verdict=verdict,
                rate_ratio_10s=(p["rate_per_1000s"] / d["rate_per_1000s"]) if d["falls"] else float("nan"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot", type=Path, required=True, help="eval folder of the pilot")
    p.add_argument("--pilot-tag", default="F1")
    p.add_argument("--d1", type=Path, required=True, help="eval folder of arm D (E1-V: logs/p2/e1v_1113_0939/eval)")
    p.add_argument("--d1-tag", default="D")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    r = score(a.pilot, a.pilot_tag, a.d1, a.d1_tag)
    a.out.write_text(json.dumps(r, indent=1))
    lines = ["Plan v6 amendment A1: F1 development pilot (run 2601) vs arm D (run 2004), one training seed"]
    for cmd in (10, 30):
        for name in ("pilot", "d1"):
            x = r["results"][cmd][name]
            lines.append(f"  {cmd} s {name:5s}: falls {x['falls']:3d} -> {x['rate_per_1000s']:.2f} per 1000 s "
                         f"[{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]; speed_ratio {x['speed_ratio']:.3f}; "
                         f"track_per_step {x['track_per_step']:.4f}")
    c = r["checks"]
    lines.append(f"checks (10 s): falls <= 0.7 x D1 {c['falls']}; speed_ratio >= 0.8 {c['speed']}; "
                 f"tracking >= D1 - 0.03 {c['tracking']}; rate ratio {r['rate_ratio_10s']:.2f}")
    lines.append(f"VERDICT: {r['verdict']} (development check, one seed; NO-GO = report and stop)")
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    a.out.with_suffix(".txt").write_text(text)


if __name__ == "__main__":
    main()
