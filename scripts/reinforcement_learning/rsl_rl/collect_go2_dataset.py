# SPDX-License-Identifier: BSD-3-Clause
"""Collect an offline RWM-U dataset by rolling out a trained Go2 policy.

Writes <out> in the offline loader's column order:
  [ system_state(45) | system_action(12) | system_contact(8) | system_termination(1) ] = 66
No extension columns: system_extension is disabled in ObservationsCfg_PRETRAIN.

Matches the shipped ANYmal reference: ONE continuous single-env trajectory,
~10000 steps, terminations ~0 (clean policy never falls), no time-limit reset.

Run (in container, GPU):
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/collect_go2_dataset.py \
    --task Template-Isaac-Velocity-Flat-Unitree-Go2-Pretrain-v0 \
    --checkpoint <abs path to pretrain_ens5/model_2000.pt> \
    --num_steps 10000 --num_envs 1 --headless
"""
import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect offline RWM-U dataset from a trained Go2 policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (keep 1 for one continuous trajectory).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--num_steps", type=int, default=10000, help="Timesteps to record (rows in the CSV).")
parser.add_argument("--output", type=str, default=None, help="CSV output path (default assets/data/go2/state_action_data_0.csv).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import pandas as pd

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import mbrl.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    # one continuous trajectory: disable the time-limit reset so the env never resets
    # mid-collection (matches the reference dataset's unbroken stream)
    env_cfg.episode_length_s = 1.0e9

    # resolve checkpoint
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[collect] loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    out = args_cli.output or os.path.join("assets", "data", "go2", "state_action_data_0.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    obs = env.get_observations()
    rows = []
    for t in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        full = env.unwrapped.obs_buf  # dict of all groups, post-step
        row = torch.cat(
            [full["system_state"], full["system_action"], full["system_contact"], full["system_termination"]],
            dim=-1,
        )  # [num_envs, 66]
        rows.append(row.detach().to("cpu"))
        if (t + 1) % 1000 == 0:
            print(f"[collect] {t + 1}/{args_cli.num_steps} steps")

    data = torch.cat(rows, dim=0).numpy()  # [num_steps * num_envs, 66]
    pd.DataFrame(data).to_csv(out, header=False, index=False)
    term_count = int(data[:, -1].sum())
    print(f"[collect] wrote {data.shape[0]} rows x {data.shape[1]} cols -> {out} (terminations={term_count})")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
