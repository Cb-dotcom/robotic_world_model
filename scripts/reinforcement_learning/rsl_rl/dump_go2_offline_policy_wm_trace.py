# SPDX-License-Identifier: BSD-3-Clause
"""Roll out an offline Go2 policy in the real env and dump WM-input trace.

Writes env-major CSV:
  [system_state(45) | system_action(12) | system_contact(8) | system_termination(1)] = 66

This is for exploit-rollout diagnostics: score the actual real-env trajectory
driven by an offline policy under a fitted ensemble world model.
"""

import argparse
import sys

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Dump real-env WM trace for an offline Go2 policy.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--normal_resets", action="store_true", default=True)
parser.add_argument("--trace_envs", type=int, default=6)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import pandas as pd
import numpy as np
import gymnasium as gym
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

    # Keep normal resets for exploit diagnostics. We want the actual failure/reset behavior.
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[dump] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device

    runner_cls = getattr(rsl_runners, agent_cfg.class_name)
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    ckpt = torch.load(resume_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    runner.alg.policy.load_state_dict(state)
    runner.alg.policy.eval()
    print("[dump] loaded policy weights into runner.alg.policy")
    policy = runner.get_inference_policy(device=device)

    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)

    log_sums = collections.defaultdict(float)
    log_counts = collections.defaultdict(int)
    ep_lengths = []
    cur_len = torch.zeros(args_cli.num_envs, device=device)

    base = env.unwrapped
    n_trace = min(args_cli.trace_envs, args_cli.num_envs)
    tr_cmd, tr_act = [], []

    rows = []
    done_count = 0
    obs = env.get_observations()

    for t in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, extras = env.step(actions)

        cur_len += 1
        done_count += int(dones.sum().item())

        full = env.unwrapped.obs_buf
        row = torch.cat(
            [
                full["system_state"],
                full["system_action"],
                full["system_contact"],
                full["system_termination"],
            ],
            dim=-1,
        )
        rows.append(row.detach().cpu())

        if n_trace > 0:
            try:
                cmd = base.command_manager.get_command("base_velocity")[:n_trace]
                lin = base.scene["robot"].data.root_lin_vel_b[:n_trace, :2]
                ang = base.scene["robot"].data.root_ang_vel_b[:n_trace, 2:3]
                tr_cmd.append(cmd.detach().cpu().clone())
                tr_act.append(torch.cat([lin, ang], dim=1).detach().cpu().clone())
            except Exception as e:
                if t == 0:
                    print(f"[dump] velocity trace disabled: {e}")
                n_trace = 0

        log = extras.get("log", {}) if isinstance(extras, dict) else {}
        for k, v in log.items():
            try:
                log_sums[k] += float(v)
                log_counts[k] += 1
            except (TypeError, ValueError):
                pass

        for i in dones.nonzero(as_tuple=False).flatten().tolist():
            ep_lengths.append(int(cur_len[i].item()))
            cur_len[i] = 0.0

        if (t + 1) % 200 == 0:
            print(f"[dump] {t + 1}/{args_cli.num_steps} steps, dones={done_count}, episodes={len(ep_lengths)}")

    data_tne = torch.stack(rows, dim=0)              # [T, N, 66]
    data_nte = data_tne.transpose(0, 1).contiguous() # [N, T, 66]

    # Mark env-boundary seams so later windowing never crosses env_i -> env_i+1.
    data_nte[:, -1, -1] = 1.0

    data = data_nte.view(-1, data_nte.shape[-1]).numpy()
    pd.DataFrame(data).to_csv(args_cli.output, header=False, index=False)

    term_count = int(data[:, -1].sum())
    seam_count = args_cli.num_envs
    interior_terms = term_count - seam_count

    print("\n================ DUMP SUMMARY ================")
    print(f"rows={data.shape[0]} cols={data.shape[1]}")
    print(f"num_envs={args_cli.num_envs} num_steps={args_cli.num_steps}")
    print(f"terminations_with_env_seams={term_count}")
    print(f"env_boundary_seams={seam_count}")
    print(f"interior_terms_est={interior_terms}")
    print(f"dones_from_env_step={done_count}")
    if ep_lengths:
        print(f"mean real episode length={sum(ep_lengths)/len(ep_lengths):.1f} min={min(ep_lengths)} max={max(ep_lengths)}")
    else:
        print("no episodes completed")
    print(f"saved_csv={args_cli.output}")

    if n_trace > 0 and len(tr_cmd) > 0:
        cmd = torch.stack(tr_cmd).numpy()
        act = torch.stack(tr_act).numpy()
        npz_out = os.path.splitext(args_cli.output)[0] + "_vel_trace.npz"
        np.savez(npz_out, cmd=cmd, act=act)
        c = cmd[..., :2].reshape(-1, 2)
        a = act[..., :2].reshape(-1, 2)
        err_xy = np.linalg.norm(c - a, axis=-1)
        print(f"saved_vel_trace={npz_out}")
        print(f"mean_speed_cmd_actual={np.linalg.norm(c, axis=-1).mean():.3f}/{np.linalg.norm(a, axis=-1).mean():.3f}")
        print(f"mean_xy_error={err_xy.mean():.3f}")
        for d, nm in [(0, "vx"), (1, "vy")]:
            if c[:, d].std() > 1e-6 and a[:, d].std() > 1e-6:
                cc = float(np.corrcoef(c[:, d], a[:, d])[0, 1])
            else:
                cc = float("nan")
            print(f"corr_cmd_{nm}_act_{nm}={cc:+.3f}")
    print("==============================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
