# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate an offline-trained Go2 policy in the REAL Go2 env.

Loads an offline RWM-U/MOPO policy (policy_*.pt, an rsl_rl ActorCritic state dict)
into the real Pretrain env and rolls it out with NORMAL episode resets, measuring
real episode length, per-term episode rewards (tracking etc.), and the termination
breakdown (base_contact = fall vs time_out). Imagined metrics can be gamed by
world-model exploitation; this is the only test of whether the policy walks.

Run (in container, GPU):
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/eval_go2_offline_policy.py \
    --task Template-Isaac-Velocity-Flat-Unitree-Go2-Pretrain-v0 \
    --checkpoint <abs path to policy_499.pt OR pretrain model_2000.pt> \
    --num_envs 32 --num_steps 2000 --headless
"""
import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate an offline Go2 policy in the real env.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of envs for averaging.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--num_steps", type=int, default=2000, help="Rollout steps.")
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
import collections

import rsl_rl.runners as rsl_runners

from isaaclab.envs import DirectRLEnvCfg, DirectMARLEnvCfg, ManagerBasedRLEnvCfg
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
    # NOTE: do NOT disable the time-limit reset -- we want normal episodes so that
    # falls (base_contact) and time-outs both terminate naturally and are measured.

    if not args_cli.checkpoint:
        raise ValueError("--checkpoint must point to a policy_*.pt (offline) or model_*.pt (pretrain)")
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[eval] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device

    # Build the runner as pretrain/collect did (same actor-critic arch), but load
    # ONLY the actor-critic weights -- offline checkpoints carry no ensemble, so
    # runner.load() would not apply.
    runner_cls = getattr(rsl_runners, agent_cfg.class_name)
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    ckpt = torch.load(resume_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ac = getattr(getattr(runner, "alg", runner), "actor_critic", None) or getattr(runner, "actor_critic", None)
    if ac is None:
        raise RuntimeError("could not locate actor_critic on runner")
    missing, unexpected = ac.load_state_dict(state, strict=False)
    print(f"[eval] loaded actor-critic (missing={list(missing)}, unexpected={list(unexpected)})")
    ac.eval()
    policy = runner.get_inference_policy(device=device)

    # --- obs-layout trap check: print the real policy group's term order ---
    om = env.unwrapped.observation_manager
    try:
        print("[eval] policy obs terms:", om.active_terms["policy"])
        print("[eval] policy obs dims :", om.group_obs_term_dim["policy"])
    except Exception as e:
        print(f"[eval] (could not introspect obs manager: {e})")

    # --- rollout with normal resets ---
    log_sums = collections.defaultdict(float)
    log_counts = collections.defaultdict(int)
    ep_lengths = []
    rew_sum = 0.0
    cur_len = torch.zeros(args_cli.num_envs, device=device)

    obs = env.get_observations()
    for t in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, extras = env.step(actions)
        cur_len += 1
        rew_sum += float(rew.mean().item())
        log = extras.get("log", {}) if isinstance(extras, dict) else {}
        for k, v in log.items():
            try:
                log_sums[k] += float(v); log_counts[k] += 1
            except (TypeError, ValueError):
                pass
        for i in dones.nonzero(as_tuple=False).flatten().tolist():
            ep_lengths.append(int(cur_len[i].item())); cur_len[i] = 0.0
        if (t + 1) % 200 == 0:
            print(f"[eval] {t+1}/{args_cli.num_steps} steps, episodes={len(ep_lengths)}")

    print("\n================ EVAL SUMMARY ================")
    print(f"steps={args_cli.num_steps} envs={args_cli.num_envs} episodes_completed={len(ep_lengths)}")
    if ep_lengths:
        print(f"mean real episode length: {sum(ep_lengths)/len(ep_lengths):.1f} "
              f"(min={min(ep_lengths)}, max={max(ep_lengths)})")
    else:
        print("no episodes completed in window (all still running -> long/stable, good sign)")
    print(f"mean per-step total reward: {rew_sum/args_cli.num_steps:.4f}")
    print("--- averaged episode logs (Episode_Reward/* , Episode_Termination/*) ---")
    for k in sorted(log_sums):
        print(f"  {k}: {log_sums[k]/max(log_counts[k],1):.4f}")
    print("==============================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()