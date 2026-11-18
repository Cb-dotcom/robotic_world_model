#!/usr/bin/env python3
"""Plan v6 C10 (descriptive only): optimism gap per policy = imagined termination rate per 1000 imagined s
minus real fall rate per 1000 s.

Imagined rate: the policy runs in the imagination it was trained in, rebuilt on the CPU from the run's own
`run_args.json` (resolved config, world-model checkpoint, termination rule, warm-reset mode, reward options);
every config value must match the recorded one except the device, the number of imagined envs and the WM
path when it is overridden. The loop is the training loop of `policy_training.py` (RNN: only the last row
after the first step; the env resets ended episodes and samples new start histories). Terminations are
imagined dones that are not time-outs. Two action rules: deterministic (`act_inference`, as the real
evaluation) and sampled (`act`, as during training).

Real rate: falls / evaluated seconds from the existing real evaluations (seeds 42 + 43 pooled, 12,800 s per
cell, 10 s and 30 s commands), given in a JSON file whose rows cite the run-log.

No rule and no causal claim (plan §2 C10, clarified 15 Nov): the gap contains the real fall rate, so any
correlation with real falls is partly built in. Reported per policy.

Usage (CPU, robotic_world_model root):
  c10_optimism_gap.py --policy 1601=logs/offline/anymal_d_flat/<run>_1601 [...] \
      --real analysis/c10_real_falls.json --out logs/p2/c10_<date> [--num-envs 640 --steps 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MB = HERE.parent
for p in (str(HERE), str(MB)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pre_reset_transitions import sha256  # noqa: E402

ALLOWED = {"experiment_config.device", "environment_config.num_envs", "model_architecture_config.resume_path"}


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def apply_resolved(cfg, resolved):
    """Set every section field of `cfg` from the run's resolved_config (dataclass sections only)."""
    for section, values in resolved.items():
        obj = getattr(cfg, section, None)
        if obj is None or not isinstance(values, dict) or not hasattr(obj, "__dataclass_fields__"):
            if hasattr(cfg, section) and not isinstance(values, dict):
                setattr(cfg, section, values)
            continue
        for key, value in values.items():
            if key in obj.__dataclass_fields__:
                setattr(obj, key, value)
    return cfg


def config_differences(cfg, resolved):
    """Recorded keys that the rebuilt config does not carry with the same value (refused), and keys that exist
    only in today's code (reported: the run predates them, so their defaults are assumed to match)."""
    a, b = flatten(asdict(cfg)), flatten(resolved)
    refused = sorted(k for k in set(b) - ALLOWED if not _same(a.get(k, object()), b[k]))
    newer = sorted(k for k in set(a) - set(b) - ALLOWED)
    return refused, newer


def build_experiment(run_dir, num_envs, wm_override=None):
    """ModelBasedExperiment on the CPU from run_dir/run_args.json; returns (experiment, record, notes)."""
    import train as T
    import termination_rule
    from envs import BaseEnv
    record = json.loads((Path(run_dir) / "run_args.json").read_text())
    resolved = record["resolved_config"]
    args = record.get("args", {})
    task = record.get("task") or args.get("task")
    cfg = apply_resolved(T.resolve_task_config(task), resolved)
    diffs, newer = config_differences(cfg, resolved)
    if diffs:
        raise RuntimeError(f"{run_dir}: config cannot be rebuilt exactly; differing keys {diffs[:10]}")
    wm = wm_override or resolved["model_architecture_config"]["resume_path"]
    if not wm or not Path(wm).is_file():
        raise FileNotFoundError(f"{run_dir}: world model {wm} not found")
    recorded = (record.get("world_model_checkpoint") or {})
    wm_sha = sha256(wm)
    if isinstance(recorded, dict) and recorded.get("sha256") and recorded["sha256"] != wm_sha:
        raise RuntimeError(f"{run_dir}: world model sha256 {wm_sha[:12]} != recorded {recorded['sha256'][:12]}")
    cfg.experiment_config.device = "cpu"
    cfg.environment_config.num_envs = int(num_envs)
    cfg.model_architecture_config.resume_path = str(wm)
    # p15_train patches, as recorded
    mode = str(args.get("mode", record.get("mode", "paper")))
    rule, thr = args.get("termination_rule", "selected"), args.get("termination_threshold", 0.5)
    termination_rule.install(rule, thr, uncertainty_threshold=args.get("termination_uncertainty"))
    BaseEnv.warm_start_resets = mode.endswith("_warm")
    BaseEnv.only_positive_rewards = bool(args.get("only_positive_rewards", False))
    BaseEnv.alive_bonus = float(args.get("alive_bonus", 0.0) or 0.0)
    T.args_cli = types.SimpleNamespace(**{k: args.get(k) for k in ("task", "run_num", "seed", "mode")},
                                       init_policy_checkpoint=None)
    exp = T.ModelBasedExperiment(**cfg.experiment_config.to_dict())
    exp.prepare_environment(**cfg.environment_config.to_dict())
    exp.prepare_model(**cfg.model_architecture_config.to_dict())
    pol = cfg.policy_architecture_config.to_dict()
    pol["resume_path"] = None
    exp.prepare_policy(**pol)
    exp.prepare_data(**cfg.data_config.to_dict())
    want = (record.get("normalizer_contract") or {}).get("norm_stats_sha256")
    got = exp.normalization_contract.norm_stats_sha256
    if want and want != got:
        raise RuntimeError(f"{run_dir}: normalizer sha256 {got[:12]} != recorded {want[:12]}")
    notes = dict(norm_stats_sha256=got, config_keys_newer_than_run=newer, task=task, mode=mode, termination_rule=rule, termination_threshold=thr,
                 warm_start_resets=BaseEnv.warm_start_resets, wm=str(wm), wm_sha256=wm_sha,
                 penalty=cfg.environment_config.uncertainty_penalty_weight,
                 observation_noise=cfg.environment_config.observation_noise,
                 max_episode_length=cfg.environment_config.max_episode_length,
                 step_dt=cfg.environment_config.step_dt,
                 command_resample_interval_range=cfg.environment_config.command_resample_interval_range,
                 event_interval_range=cfg.environment_config.event_interval_range)
    return exp, record, notes


def load_policy(exp, checkpoint):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    exp.actor_critic.load_state_dict(ck["model_state_dict"])
    exp.actor_critic.eval()
    return ck.get("iter")


@torch.no_grad()
def imagined_terminations(exp, steps, seed, deterministic):
    """Training loop of policy_training.learn without learning. Returns counts and imagined seconds."""
    from run_provenance import seed_everything
    seed_everything(seed)
    env = exp.env
    state_history, action_history = env.prepare_imagination()
    rnn = env.system_dynamics.architecture_config["type"] in ("rnn", "rssm")
    terminated = time_outs = 0
    per_step = np.zeros(steps, np.int64)
    for i in range(steps):
        if rnn and env.common_step_counter > 0:
            state_history, action_history = state_history[:, -1:], action_history[:, -1:]
        obs = env.get_imagination_observation(state_history, action_history)
        actions = exp.actor_critic.act_inference(obs) if deterministic else exp.actor_critic.act(obs)
        _, _, dones, extras, state_history, action_history, _ = env.imagination_step(actions, state_history,
                                                                                      action_history)
        to = extras["time_outs"].bool()
        term = dones.bool() & ~to
        per_step[i] = int(term.sum())
        terminated += int(term.sum())
        time_outs += int(to.sum())
    seconds = env.num_envs * steps * env._step_dt
    return dict(terminations=terminated, time_outs=time_outs, imagined_s=seconds,
                rate_per_1000s=1000.0 * terminated / seconds, per_step_first_last=[int(per_step[:50].sum()),
                                                                                  int(per_step[-50:].sum())])


def real_rates(real, name):
    row = real["policies"].get(name)
    if row is None:
        raise KeyError(f"no real evaluation row for {name}")
    out = {}
    for sched in ("10s", "30s"):
        if sched in row:
            falls, seconds = row[sched]["falls"], row[sched].get("seconds", real["seconds_per_cell"])
            out[sched] = dict(falls=falls, seconds=seconds, rate_per_1000s=1000.0 * falls / seconds,
                              ci95_per_1000s=poisson_ci(falls, seconds))
    return out


def poisson_ci(k, seconds, alpha=0.05):
    """Exact (Garwood) Poisson interval of k events in `seconds`, per 1000 s."""
    from scipy.stats import chi2
    lo = 0.0 if k == 0 else chi2.ppf(alpha / 2, 2 * k) / 2
    hi = chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / 2
    return [1000.0 * lo / seconds, 1000.0 * hi / seconds]


def run(policies, real_path, out, num_envs=640, steps=1000, seed=0, checkpoint_name="policy_2499.pt",
        wm_overrides=None):
    torch.set_num_threads(8)
    out.mkdir(parents=True, exist_ok=False)
    real = json.loads(Path(real_path).read_text())
    rows = {}
    for name, run_dir in policies:
        run_dir = Path(run_dir)
        ck = run_dir / checkpoint_name
        if not ck.is_file():
            raise FileNotFoundError(f"{name}: {ck} not found")
        t0 = time.time()
        exp, record, notes = build_experiment(run_dir, num_envs, (wm_overrides or {}).get(name))
        it = load_policy(exp, ck)
        imag = {mode: imagined_terminations(exp, steps, seed, mode == "deterministic")
                for mode in ("deterministic", "sampled")}
        rr = real_rates(real, name)
        gap = {f"{mode}_minus_real_{s}": imag[mode]["rate_per_1000s"] - rr[s]["rate_per_1000s"]
               for mode in imag for s in rr}
        rows[name] = dict(run_dir=str(run_dir), checkpoint=str(ck), checkpoint_sha256=sha256(ck), policy_iter=it,
                          run_num=record.get("run_num"), seed=record.get("seed"), config=notes, imagined=imag,
                          real=rr, gap_per_1000s=gap, wall_s=round(time.time() - t0, 1))
        print(f"[c10] {name}: imagined det {imag['deterministic']['rate_per_1000s']:.2f} / sampled "
              f"{imag['sampled']['rate_per_1000s']:.2f} per 1000 imagined s; real 10 s "
              f"{rr.get('10s', {}).get('rate_per_1000s', float('nan')):.2f} per 1000 s", flush=True)
    meta = dict(num_envs=num_envs, steps=steps, seed=seed, real_source=str(real_path), real_sha256=sha256(real_path),
                analysis_sha256=sha256(__file__), reading="descriptive only (plan v6 §2 C10, clarified 15 Nov)")
    (out / "c10_report.json").write_text(json.dumps(dict(meta=meta, policies=rows), indent=1))
    lines = ["Plan v6 C10 optimism gap (descriptive only; no rule, no causal claim)",
             f"imagination: {num_envs} envs x {steps} steps per policy and action rule, seed {seed}; "
             "real: seeds 42 + 43 pooled",
             "policy | model (sha) | imagined terminations per 1000 imagined s: deterministic / sampled | "
             "real falls per 1000 s: 10 s [95% CI] / 30 s | gap (deterministic - real 10 s)"]
    for name, r in rows.items():
        i, rr = r["imagined"], r["real"]
        lines.append(f"{name:6s} | {Path(r['config']['wm']).name} ({r['config']['wm_sha256'][:8]}) | "
                     f"{i['deterministic']['rate_per_1000s']:.2f} / {i['sampled']['rate_per_1000s']:.2f} | "
                     f"{rr['10s']['rate_per_1000s']:.2f} [{rr['10s']['ci95_per_1000s'][0]:.2f}, "
                     f"{rr['10s']['ci95_per_1000s'][1]:.2f}] / {rr.get('30s', {}).get('rate_per_1000s', float('nan')):.2f} | "
                     f"{r['gap_per_1000s'].get('deterministic_minus_real_10s', float('nan')):+.2f}")
    text = "\n".join(lines) + "\n"
    (out / "c10_report.txt").write_text(text)
    print(text, flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", action="append", required=True, metavar="NAME=RUN_DIR")
    p.add_argument("--wm", action="append", default=[], metavar="NAME=PATH",
                   help="world model when run_args.json's resume_path moved (sha256 still checked)")
    p.add_argument("--real", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=640)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-name", default="policy_2499.pt")
    a = p.parse_args()
    if a.out.exists():
        p.error("output folder exists")
    pols = []
    for item in a.policy:
        name, _, path = item.partition("=")
        if not name or not path:
            p.error(f"bad --policy {item!r}")
        pols.append((name, path))
    wms = dict(item.partition("=")[::2] for item in a.wm)
    run(pols, a.real, a.out, a.num_envs, a.steps, a.seed, a.checkpoint_name, wms)


if __name__ == "__main__":
    main()
