# SPDX-License-Identifier: BSD-3-Clause
"""Per-episode real-env evaluation of an offline policy: falls counted per episode, time to fall,
clean tracking for ALL envs, and a check of the old summary's aggregation (doubt C7).

Same rollout as eval_go2_offline_policy.py (same task, policy loading, normal resets). Differences:
- tracks command vs measured velocity for every env (not only --trace_envs);
- counts falls per episode (done and not time_out) instead of the averaged Episode_Termination logs;
- writes one CSV row per episode (length, fall, censored, tracking, speed);
- prints the env's own averaged logs (computed exactly like eval_go2_offline_policy.py) next to an
  emulation of that aggregation, so the C7 hypothesis is tested, not assumed.

eval_go2_offline_policy.py is not changed (P1.5 numbers stay reproducible).

Run (in container, GPU):
  PYTHONUNBUFFERED=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/eval_episode_stats.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
    --checkpoint <abs path to policy_2499.pt> --num_envs 32 --num_steps 2000 --headless \
    --out_csv <path>/episodes.csv
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Per-episode evaluation of an offline policy in the real env.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_steps", type=int, default=2000)
parser.add_argument("--std", type=float, default=0.5, help="Tracking sigma (Isaac Lab track_lin_vel_xy_exp: sqrt(0.25)).")
parser.add_argument("--no_push", action="store_true", default=False, help="Disable the push_robot interval event (diagnosis).")
parser.add_argument("--cmd_resample_s", type=float, default=None, help="Set the command resampling time (s), e.g. 30 = never within a 20 s episode.")
parser.add_argument("--out_csv", type=str, default=None, help="Per-episode CSV (default: next to checkpoint).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import collections
import csv
import json

import gymnasium as gym
import torch

import rsl_rl.runners as rsl_runners

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import mbrl.tasks  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "model_based", "analysis"))
from episode_stats import EpisodeStats, format_summary  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.no_push:
        env_cfg.events.push_robot = None
        print("[eval_ep] push_robot event DISABLED")
    if args_cli.cmd_resample_s is not None:
        env_cfg.commands.base_velocity.resampling_time_range = (args_cli.cmd_resample_s, args_cli.cmd_resample_s)
        print(f"[eval_ep] command resampling_time_range = {env_cfg.commands.base_velocity.resampling_time_range}")
    try:
        print("[eval_ep] push_robot cfg:", env_cfg.events.push_robot)
        print("[eval_ep] command resampling_time_range:", env_cfg.commands.base_velocity.resampling_time_range)
    except Exception as e:
        print(f"[eval_ep] (could not read event/command cfg: {e})")
    if not args_cli.checkpoint:
        raise ValueError("--checkpoint must point to a policy_*.pt (offline) or model_*.pt (pretrain)")
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[eval_ep] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device
    base = env.unwrapped

    runner_cls = getattr(rsl_runners, agent_cfg.class_name)
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ckpt = torch.load(resume_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    runner.alg.policy.load_state_dict(state)
    runner.alg.policy.eval()
    policy = runner.get_inference_policy(device=device)

    step_dt = float(base.step_dt)
    ep_len_s = float(base.max_episode_length_s)
    command_resampling_s = list(base.command_manager.get_term("base_velocity").cfg.resampling_time_range)
    try:
        push_cfg = base.event_manager.get_term_cfg("push_robot")
    except ValueError:  # disabled event terms are absent from the manager
        push_cfg = None
    push_metadata = {"enabled": push_cfg is not None}
    if push_cfg is not None:
        push_metadata.update({"mode": push_cfg.mode,
                              "interval_range_s": push_cfg.interval_range_s,
                              "velocity_range": push_cfg.params.get("velocity_range")})
    print(f"[eval_ep] step_dt={step_dt} episode_length_s={ep_len_s} max_episode_length={base.max_episode_length}")
    try:
        print("[eval_ep] termination terms:", base.termination_manager.active_terms)
        print("[eval_ep] track_lin_vel_xy_exp cfg:", base.reward_manager.get_term_cfg("track_lin_vel_xy_exp"))
    except Exception as e:  # informational only
        print(f"[eval_ep] (could not introspect managers: {e})")

    stats = EpisodeStats(args_cli.num_envs, step_dt, ep_len_s, std=args_cli.std)
    log_sums = collections.defaultdict(float)
    log_counts = collections.defaultdict(int)
    warned = False

    obs = env.get_observations()
    for t in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, extras = env.step(actions)
        done = dones.bool()
        if isinstance(extras, dict) and "time_outs" in extras:
            time_out = extras["time_outs"].bool()
        else:
            time_out = base.termination_manager.time_outs.bool()
            if not warned:
                print("[eval_ep] extras has no 'time_outs'; using termination_manager.time_outs")
                warned = True
        cmd = base.command_manager.get_command("base_velocity")[:, :2]
        act = base.scene["robot"].data.root_lin_vel_b[:, :2]
        stats.step(cmd.detach().cpu().numpy(), act.detach().cpu().numpy(),
                   done.detach().cpu().numpy(), time_out.detach().cpu().numpy())
        # the old summary's aggregation, unchanged (eval_go2_offline_policy.py lines 159-164)
        log = extras.get("log", {}) if isinstance(extras, dict) else {}
        for k, v in log.items():
            try:
                log_sums[k] += float(v); log_counts[k] += 1
            except (TypeError, ValueError):
                pass
        if (t + 1) % 200 == 0:
            print(f"[eval_ep] {t+1}/{args_cli.num_steps} steps, episodes={len(stats.episodes)}")
    stats.finish()
    s = stats.summary()

    old = {k: log_sums[k] / max(log_counts[k], 1) for k in sorted(log_sums)}
    old_track = old.get("Episode_Reward/track_lin_vel_xy_exp", float("nan"))
    old_bc = old.get("Episode_Termination/base_contact", float("nan"))
    old_to = old.get("Episode_Termination/time_out", float("nan"))
    print("\n" + format_summary(s))
    print("---------------- C7 CHECK (old summary vs emulation) ----------------")
    print(f"  logged track_lin_vel_xy_exp (old method): {old_track:.4f}   emulated: {s['emulated_logged_track']:.4f}")
    print(f"  logged base_contact          (old method): {old_bc:.4f}   emulated: {s['emulated_logged_base_contact']:.4f}")
    print(f"  logged time_out              (old method): {old_to:.4f}   emulated: {s['emulated_logged_time_out']:.4f}")
    print(f"  (variant 'log cleared every step': track {s['emulated_reset_steps_track']:.4f}, "
          f"base_contact {s['emulated_reset_steps_base_contact']:.4f}, time_out {s['emulated_reset_steps_time_out']:.4f})")
    ratio = old_bc / (old_bc + old_to) if (old_bc + old_to) > 0 else float("nan")
    print(f"  (variant 'zero log from step 0': track {s['emulated_logged_track_from_step0']:.4f}, "
          f"base_contact {s['emulated_logged_base_contact_from_step0']:.4f}, time_out {s['emulated_logged_time_out_from_step0']:.4f})")
    print(f"  old falls ratio base_contact/(base_contact+time_out): {ratio:.4f}   per-episode fall rate: {s['fall_rate_completed']:.4f}")
    print("----------------------------------------------------------------------")

    out_csv = args_cli.out_csv or os.path.join(os.path.dirname(resume_path), "episodes.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats.episodes[0].keys()) if stats.episodes else ["env"])
        w.writeheader()
        for e in stats.episodes:
            w.writerow(e)
    with open(os.path.splitext(out_csv)[0] + "_summary.json", "w") as f:
        json.dump({"checkpoint": resume_path, "task": args_cli.task, "num_envs": args_cli.num_envs,
                   "num_steps": args_cli.num_steps, "no_push": args_cli.no_push,
                   "cmd_resample_s": args_cli.cmd_resample_s,
                   "seed": int(env_cfg.seed) if env_cfg.seed is not None else None,
                   "std": args_cli.std, "step_dt": step_dt, "episode_length_s": ep_len_s,
                   "command_resampling_time_range_s": command_resampling_s,
                   "push_robot": push_metadata, "summary": s, "old_logs": old}, f, indent=1)
    print(f"[eval_ep] wrote {out_csv}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
