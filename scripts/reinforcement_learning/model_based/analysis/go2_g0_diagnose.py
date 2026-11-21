#!/usr/bin/env python3
"""Plan v6 amendment A3 §4 (21 Nov 2026): read-only validity check of a failed G0 run. Changes nothing.

A3: only an independently verified implementation or measurement defect invalidates a G0 run. This tool
collects the evidence on the saved run; it trains nothing and does not touch the gate verdict.
  1. physics: episode lengths of the G0 policy (eval CSVs, 10 s commands) vs s2's G6 CSVs;
  2. action support: the G0 policy's actions on s2's RECORDED observations vs s2's recorded actions and the
     world model's action normalizer (share outside s2's per-joint range, share beyond mean +- 3 std);
  3. imagination: the G0 policy in the fitted model, built exactly as the trainer built it (go2_study),
     with the same statistics as the s2 probe, plus its action magnitudes in imagination;
  4. training log: imagined mean reward and episode length at iterations 100, 500, 1000, last.
Descriptive output; the reading (defect or valid failure) is written by a person with the evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MB = HERE.parent
for p in (str(HERE), str(MB)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pre_reset_transitions import sha256, validate_trace  # noqa: E402


def episode_lengths(folder, tag, cmd=10, seeds=(42, 43)):
    lens, falls = [], 0
    for s in seeds:
        path = Path(folder) / f"{tag}_cmd{cmd}_s{s}" / "episodes.csv"
        if not path.is_file():
            return None
        with path.open() as f:
            for row in csv.DictReader(f):
                if row["censored"] == "True":
                    continue
                lens.append(int(row["length"]))
                falls += row["fall"] == "True"
    a = np.asarray(lens)
    if not len(a):
        return dict(episodes=0)
    return dict(episodes=int(len(a)), falls=int(falls), median_steps=float(np.median(a)),
                p10_steps=float(np.percentile(a, 10)), p90_steps=float(np.percentile(a, 90)),
                share_under_25_steps=float(np.mean(a < 25)))


def action_support(policy, trace, stats, rows=20000, seed=0):
    obs = trace["policy_observation"].reshape(-1, trace["policy_observation"].shape[-1])
    act = trace["action"].reshape(-1, trace["action"].shape[-1])
    pick = np.random.default_rng(seed).choice(len(obs), size=min(rows, len(obs)), replace=False)
    with torch.no_grad():
        g0 = policy(torch.from_numpy(obs[pick]).float()).numpy()
    lo, hi = act.min(0), act.max(0)
    m, s = stats["action_mean"], stats["action_std"]
    outside_range = (g0 < lo) | (g0 > hi)
    outside_3sd = np.abs((g0 - m) / s) > 3
    return dict(rows=int(len(pick)),
                g0_abs_mean=float(np.abs(g0).mean()), s2_abs_mean=float(np.abs(act[pick]).mean()),
                g0_abs_max=float(np.abs(g0).max()), s2_abs_max=float(np.abs(act).max()),
                g0_std_per_joint=[float(x) for x in g0.std(0)], s2_std_per_joint=[float(x) for x in act.std(0)],
                share_outside_s2_range=float(outside_range.mean()),
                share_outside_s2_range_per_joint=[float(x) for x in outside_range.mean(0)],
                share_beyond_3sd=float(outside_3sd.mean()))


def training_trajectory(log_path, marks=(100, 500, 1000)):
    text = Path(log_path).read_text(errors="replace")
    blocks = re.split(r"Learning iteration\s+", text)[1:]
    out = {}
    last = None
    for b in blocks:
        m = re.match(r"(\d+)/(\d+)", b)
        if not m:
            continue
        it = int(m.group(1))
        rew = re.search(r"Mean reward:\s*([-+0-9.eE]+)", b)
        eln = re.search(r"Mean episode length:\s*([-+0-9.eE]+)", b)
        rec = dict(mean_reward=float(rew.group(1)) if rew else None,
                   mean_episode_length=float(eln.group(1)) if eln else None)
        if it in marks:
            out[str(it)] = rec
        last = (it, rec)
    if last:
        out[f"last ({last[0]})"] = last[1]
    return out


class Recording:
    """Wraps a policy and records the magnitude of the actions it gives in imagination."""

    def __init__(self, policy, lo, hi):
        self.policy, self.lo, self.hi = policy, torch.as_tensor(lo), torch.as_tensor(hi)
        self.abs_sum = self.n = self.outside = 0.0
        self.abs_max = 0.0

    def __call__(self, obs):
        a = self.policy(obs)
        self.abs_sum += float(a.abs().sum())
        self.n += a.numel()
        self.abs_max = max(self.abs_max, float(a.abs().max()))
        self.outside += float(((a < self.lo) | (a > self.hi)).sum())
        return a

    def summary(self):
        return dict(abs_mean=self.abs_sum / max(self.n, 1), abs_max=self.abs_max,
                    share_outside_s2_range=self.outside / max(self.n, 1))


def run(g0_policy, wm, start_data, asset_trace, physics_trace, g0_eval, s2_eval, train_log, out,
        envs=256, steps=1000, seed=0, device="cuda:0"):
    import go2_imagination_probe as probe
    from policy_loop import Policy
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    with np.load(physics_trace, allow_pickle=False) as z:
        validate_trace(z)
        trace = {k: z[k] for k in z.files if k != "metadata"}
    policy = Policy(g0_policy)
    exp, cfg, contract, start = probe.build(wm, start_data, asset_trace, envs, device)
    stats = {k: v.reshape(-1).cpu().numpy() for k, v in
             zip(("state_mean", "state_std", "action_mean", "action_std"),
                 (exp.dataset.state_data_mean, exp.dataset.state_data_std,
                  exp.dataset.action_data_mean, exp.dataset.action_data_std))}
    act = trace["action"].reshape(-1, trace["action"].shape[-1])
    support = action_support(policy, trace, stats)
    rec = Recording(policy, act.min(0), act.max(0))
    imag = probe.imagination_stats(exp, rec, steps, seed)
    imag["actions"] = rec.summary()
    report = dict(protocol="A3 §4 G0 validity evidence (read-only)", g0_policy=str(g0_policy),
                  g0_policy_sha256=sha256(g0_policy), wm=str(wm), wm_sha256=sha256(wm),
                  physics_episodes=dict(g0=episode_lengths(g0_eval, "G0"), s2=episode_lengths(s2_eval, "unitree_s2")),
                  action_support_on_s2_observations=support, imagination_g0=imag,
                  training=training_trajectory(train_log), analysis_sha256=sha256(__file__))
    (out / "g0_diagnose.json").write_text(json.dumps(report, indent=1, default=float))
    pe = report["physics_episodes"]
    lines = ["A3 G0 validity evidence (read-only; the gate verdict is unchanged)"]
    for k in ("g0", "s2"):
        e = pe[k]
        if e and e.get("episodes"):
            lines.append(f"physics {k}: {e['episodes']} episodes, {e['falls']} falls; length median {e['median_steps']:.0f} "
                         f"steps (p10 {e['p10_steps']:.0f}, p90 {e['p90_steps']:.0f}); share < 25 steps {e['share_under_25_steps']:.3f}")
        else:
            lines.append(f"physics {k}: no episode CSV")
    s = support
    lines.append(f"actions on s2's recorded observations ({s['rows']} rows): |a| mean G0 {s['g0_abs_mean']:.3f} vs s2 "
                 f"{s['s2_abs_mean']:.3f}; max G0 {s['g0_abs_max']:.2f} vs s2 {s['s2_abs_max']:.2f}; outside s2's per-joint "
                 f"range {s['share_outside_s2_range']:.3f}; beyond 3 std of the model's action normalizer {s['share_beyond_3sd']:.3f}")
    lines.append(f"G0 in imagination ({imag['envs']} x {imag['steps']}): track {imag['track_per_step']:.4f}, speed_ratio "
                 f"{imag['speed_ratio']:.4f}, terminations {imag['terminations_per_1000s']:.2f} per 1000 s, task reward "
                 f"{imag['task_reward_per_s']:+.3f} per s, penalty {imag['penalty_per_s']:+.3f} per s; actions |a| mean "
                 f"{imag['actions']['abs_mean']:.3f}, max {imag['actions']['abs_max']:.2f}, outside s2's range "
                 f"{imag['actions']['share_outside_s2_range']:.3f}")
    for k, v in report["training"].items():
        lines.append(f"training it {k}: mean reward {v['mean_reward']}, mean episode length {v['mean_episode_length']}")
    text = "\n".join(lines) + "\n"
    (out / "g0_diagnose.txt").write_text(text)
    print(text, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--g0-policy", type=Path, required=True)
    p.add_argument("--wm", type=Path, required=True)
    p.add_argument("--start-data", type=Path, required=True)
    p.add_argument("--asset-trace", type=Path, required=True)
    p.add_argument("--physics", type=Path, required=True, help="s2's round-0 recording")
    p.add_argument("--g0-eval", type=Path, required=True)
    p.add_argument("--s2-eval", type=Path, required=True)
    p.add_argument("--train-log", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    run(a.g0_policy, a.wm, a.start_data, a.asset_trace, a.physics, a.g0_eval, a.s2_eval, a.train_log, a.out,
        a.envs, a.steps, 0, a.device)


if __name__ == "__main__":
    main()
