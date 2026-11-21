#!/usr/bin/env python3
"""A4 development check D0 (21 Nov 2026): cold vs warm imagination resets through the real PPO runner.

Two short PPO runs in the saved G0 world model with every G0 setting (s2 reward contract, round-0 start data,
lambda 1, PPO seed 1000, paper recipe), changing ONLY the reset history: paper (cold: one state, zero memory)
vs paper_warm (GRU memory rebuilt from the sampled 32-step history after every reset). Development evidence,
not a scored result; G0 stays a valid FAIL.

Reading rule, fixed before the run (C): the cold-start trap is removed by warm resets if, at the last common
logged iteration, the warm run's mean imagined episode length is >= 25 steps AND >= 5 x the cold run's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def series(log_path):
    """{iteration: (mean_reward, mean_episode_length)} from an rsl_rl console log."""
    text = Path(log_path).read_text(errors="replace")
    out = {}
    for b in re.split(r"Learning iteration\s+", text)[1:]:
        m = re.match(r"(\d+)/(\d+)", b)
        rew = re.search(r"Mean reward:\s*([-+0-9.eE]+)", b)
        eln = re.search(r"Mean episode length:\s*([-+0-9.eE]+)", b)
        if m and rew and eln:
            out[int(m.group(1))] = (float(rew.group(1)), float(eln.group(1)))
    return out


def compare(cold_log, warm_log, marks=(50, 100, 200), *, expected_iterations=None):
    c, w = series(cold_log), series(warm_log)
    if expected_iterations is not None:
        expected_last = expected_iterations - 1
        cold_last = max(c, default=None)
        warm_last = max(w, default=None)
        if cold_last != expected_last or warm_last != expected_last:
            raise ValueError(
                f"expected both logs to end at iteration {expected_last}; cold={cold_last}, warm={warm_last}"
            )
    common = sorted(set(c) & set(w))
    if not common:
        raise ValueError("no common logged iteration with mean reward and mean episode length")
    last = common[-1]
    rows = {str(i): dict(cold=dict(zip(("mean_reward", "mean_episode_length"), c[i])),
                         warm=dict(zip(("mean_reward", "mean_episode_length"), w[i])))
            for i in [*[m for m in marks if m in c and m in w], last]}
    lc, lw = c[last][1], w[last][1]
    read = bool(lw >= 25.0 and lw >= 5.0 * lc)
    return dict(last_common_iteration=last, iterations=rows, cold_last_length=lc, warm_last_length=lw,
                trap_removed_by_warm_resets=read,
                rule="warm mean imagined episode length >= 25 steps and >= 5 x cold, at the last common iteration")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cold-log", type=Path, required=True)
    p.add_argument("--warm-log", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-iterations", type=int, help="require the final iteration in both logs")
    a = p.parse_args(argv)
    try:
        r = compare(a.cold_log, a.warm_log, expected_iterations=a.expected_iterations)
    except ValueError as error:
        p.error(str(error))
    a.out.write_text(json.dumps(r, indent=1))
    lines = ["A4 D0: cold vs warm imagination resets, G0 model, G0 settings (development check)",
             "iteration | cold: mean reward, mean episode length | warm: mean reward, mean episode length"]
    for it, v in r["iterations"].items():
        lines.append(f"{it:>9s} | {v['cold']['mean_reward']:+.3f}, {v['cold']['mean_episode_length']:.2f} | "
                     f"{v['warm']['mean_reward']:+.3f}, {v['warm']['mean_episode_length']:.2f}")
    lines.append(f"Reading (rule fixed before the run): trap removed by warm resets: "
                 f"{'YES' if r['trap_removed_by_warm_resets'] else 'NO'} ({r['rule']})")
    print("\n".join(lines), flush=True)
    return r


if __name__ == "__main__":
    main()
