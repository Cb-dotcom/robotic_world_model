# SPDX-License-Identifier: BSD-3-Clause
"""Record real-env trajectories of a policy for world-model replay (P1.5 follow-up, 15 Oct 2026).

Same rollout as eval_episode_stats.py (same task, policy loading, normal resets, same switches).
Saves one .npz with, per step t and env n (all float32 / bool):
  state  [T, N, 45]  obs_buf["system_state"] after env.step (the world-model state)
  action [T, N, 12]  obs_buf["system_action"] after env.step (= the action applied in step t)
  cmd    [T, N, 3]   base_velocity command after env.step (vx, vy, wz; base frame)
  done   [T, N]      env was reset inside step t (so row t is the post-reset spawn state)
  time_out [T, N]    the reset in step t was a time-out (else a fall)
Row layout matches the dataset collector (state and last action of the same step).

Run (in container, GPU):
  PYTHONUNBUFFERED=1 /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/record_real_trajectories.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 --checkpoint <policy.pt> \
    --num_envs 32 --num_steps 2000 --headless --out <path>.npz
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record real-env trajectories for world-model replay.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_steps", type=int, default=2000)
parser.add_argument("--no_push", action="store_true", default=False)
parser.add_argument("--cmd_resample_s", type=float, default=None)
parser.add_argument("--out", type=str, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json

import gymnasium as gym
import numpy as np
import torch

import rsl_rl.runners as rsl_runners

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import mbrl.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.no_push:
        env_cfg.events.push_robot = None
    if args_cli.cmd_resample_s is not None:
        env_cfg.commands.base_velocity.resampling_time_range = (args_cli.cmd_resample_s, args_cli.cmd_resample_s)
    print("[record] push_robot cfg:", env_cfg.events.push_robot)
    print("[record] command resampling_time_range:", env_cfg.commands.base_velocity.resampling_time_range)
    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required")
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[record] checkpoint: {resume_path}")

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

    T, N = args_cli.num_steps, args_cli.num_envs
    S = np.zeros((T, N, 45), np.float32)
    A = np.zeros((T, N, 12), np.float32)
    C = np.zeros((T, N, 3), np.float32)
    D = np.zeros((T, N), bool)
    TO = np.zeros((T, N), bool)

    obs = env.get_observations()
    for t in range(T):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, extras = env.step(actions)
        full = base.obs_buf
        S[t] = full["system_state"].detach().cpu().numpy()
        A[t] = full["system_action"].detach().cpu().numpy()
        C[t] = base.command_manager.get_command("base_velocity").detach().cpu().numpy()
        D[t] = dones.bool().detach().cpu().numpy()
        to = extras["time_outs"] if isinstance(extras, dict) and "time_outs" in extras else base.termination_manager.time_outs
        TO[t] = (to.bool() & dones.bool()).detach().cpu().numpy()
        if (t + 1) % 200 == 0:
            print(f"[record] {t + 1}/{T} steps, resets so far={int(D[:t + 1].sum())}")

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    meta = {"checkpoint": resume_path, "task": args_cli.task, "num_envs": N, "num_steps": T,
            "no_push": args_cli.no_push, "cmd_resample_s": args_cli.cmd_resample_s,
            "step_dt": float(base.step_dt), "episode_length_s": float(base.max_episode_length_s)}
    np.savez_compressed(args_cli.out, state=S, action=A, cmd=C, done=D, time_out=TO, meta=json.dumps(meta))
    print(f"[record] wrote {args_cli.out}: falls={int((D & ~TO).sum())} time_outs={int(TO.sum())}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
