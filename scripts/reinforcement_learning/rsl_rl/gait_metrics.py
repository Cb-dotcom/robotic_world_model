# SPDX-License-Identifier: BSD-3-Clause
"""Measure the gait of a Go2 policy in simulation, with the same conditions for any Isaac Lab task.

Same conditions for every policy: flat plane, no pushes or external forces, no observation noise,
episodes of 40 s, commands held fixed per env (forward speeds --speeds, no lateral or yaw command,
no standing envs). The actor is rebuilt from the checkpoint's `actor.*` weights (ELU or the agent's
activation; empirical observation normalizer applied if the checkpoint has one), so the script does not
depend on the rsl_rl version that trained it.

  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/gait_metrics.py --pkg mbrl \
      --task Template-Isaac-Velocity-Flat-Unitree-Go2-Baseline-v0 --checkpoint <model.pt> --label ours_4c \
      --out <fresh.json> --headless
  ... --pkg unitree --task Unitree-Go2-Velocity --checkpoint <model_18200.pt> --label unitree_18200 ...
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/gait_metrics.py --compare a.json b.json
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gait_metrics_lib as G  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--compare", nargs="+", type=Path, help="print a table from saved JSON files and exit")
parser.add_argument("--pkg", choices=("mbrl", "unitree"))
parser.add_argument("--task")
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--label")
parser.add_argument("--out", type=Path)
parser.add_argument("--speeds", default="0.3,0.45,0.6,0.9")
parser.add_argument("--envs_per_speed", type=int, default=16)
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--warmup_s", type=float, default=2.0)
parser.add_argument("--seed", type=int, default=7)
known, _ = parser.parse_known_args()
if known.compare:
    named = {}
    for p in known.compare:
        d = json.loads(p.read_text())
        named[d["label"]] = {float(k): v for k, v in d["summary"].items()}
    print(G.table(named))
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not (args.pkg and args.task and args.checkpoint and args.label and args.out):
    parser.error("--pkg, --task, --checkpoint, --label and --out are required")
if args.out.exists():
    parser.error("--out must not exist")
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

if args.pkg == "mbrl":
    import mbrl.tasks  # noqa: F401
else:
    import unitree_rl_lab.tasks  # noqa: F401

ACT = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU, "lrelu": nn.LeakyReLU}


def build_actor(state, activation):
    """MLP from actor.<i>.weight/bias keys; activation between linear layers."""
    idx = sorted({int(k.split(".")[1]) for k in state if k.startswith("actor.") and k.endswith(".weight")})
    layers = []
    for j, i in enumerate(idx):
        w, b = state[f"actor.{i}.weight"], state[f"actor.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if j < len(idx) - 1:
            layers.append(ACT[activation]())
    return nn.Sequential(*layers)


def obs_normalizer(state):
    for prefix in ("actor_obs_normalizer.", "obs_normalizer."):
        if f"{prefix}_mean" in state:
            mean, std = state[f"{prefix}_mean"], state[f"{prefix}_std"]
            return lambda x: (x - mean.to(x.device)) / (std.to(x.device) + 1e-2)
    return None


def policy_obs(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    if not torch.is_tensor(obs):
        obs = obs["policy"]
    return obs


def main():
    speeds_list = [float(s) for s in args.speeds.split(",")]
    num_envs = args.envs_per_speed * len(speeds_list)
    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = 40.0
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    if getattr(env_cfg, "curriculum", None) is not None:
        for name in ("terrain_levels", "lin_vel_cmd_levels"):
            if hasattr(env_cfg.curriculum, name):
                setattr(env_cfg.curriculum, name, None)
    for name in ("push_robot", "base_external_force_torque"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    env_cfg.observations.policy.enable_corruption = False
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e6, 1.0e6)
    if hasattr(cmd, "heading_command"):
        cmd.heading_command = False
    if hasattr(cmd, "rel_standing_envs"):
        cmd.rel_standing_envs = 0.0
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg, render_mode=None), clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    try:
        state = torch.load(args.checkpoint, map_location=base.device, weights_only=False)["model_state_dict"]
        activation = getattr(agent_cfg.policy, "activation", "elu")
        actor = build_actor(state, activation).to(base.device).eval()
        norm = obs_normalizer(state)
        robot, sensor = base.scene["robot"], base.scene["contact_forces"]
        foot_ids, foot_names = robot.find_bodies([f"{f}_foot" for f in G.FEET], preserve_order=True)
        sensor_ids, _ = sensor.find_bodies([f"{f}_foot" for f in G.FEET], preserve_order=True)
        term = base.command_manager.get_term("base_velocity")
        v_env = torch.tensor(np.repeat(speeds_list, args.envs_per_speed), device=base.device, dtype=torch.float32)
        dt = base.step_dt
        T = int(round(args.seconds / dt))

        def set_cmd():
            term.vel_command_b[:, 0] = v_env
            term.vel_command_b[:, 1:] = 0.0
            if hasattr(term, "is_standing_env"):
                term.is_standing_env[:] = False

        set_cmd()
        obs = env.get_observations()
        in_dim = actor[0].in_features
        rec = {k: [] for k in ("contact", "foot_z", "foot_vxy", "pitch", "roll", "height", "vx", "vy", "wz",
                                "tau_abs", "power", "done", "time_out")}
        for t in range(T):
            with torch.inference_mode():
                x = policy_obs(obs)
                if x.shape[-1] != in_dim:
                    raise RuntimeError(f"policy obs {x.shape[-1]} != actor input {in_dim}")
                if norm is not None:
                    x = norm(x)
                obs, _, dones, extras = env.step(actor(x))
                set_cmd()
                d = robot.data
                forces = sensor.data.net_forces_w[:, sensor_ids, :].norm(dim=-1)
                roll, pitch, _ = euler_xyz_from_quat(d.root_quat_w)
                wrap = lambda a: torch.atan2(torch.sin(a), torch.cos(a))  # noqa: E731
                timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
                rec["contact"].append((forces > 1.0).cpu().numpy())
                rec["foot_z"].append(d.body_pos_w[:, foot_ids, 2].cpu().numpy())
                rec["foot_vxy"].append(d.body_lin_vel_w[:, foot_ids, :2].norm(dim=-1).cpu().numpy())
                rec["pitch"].append(wrap(pitch).cpu().numpy())
                rec["roll"].append(wrap(roll).cpu().numpy())
                rec["height"].append(d.root_pos_w[:, 2].cpu().numpy())
                vb = d.root_lin_vel_b
                rec["vx"].append(vb[:, 0].cpu().numpy())
                rec["vy"].append(vb[:, 1].cpu().numpy())
                rec["wz"].append(d.root_ang_vel_b[:, 2].cpu().numpy())
                rec["tau_abs"].append(d.applied_torque.abs().mean(dim=-1).cpu().numpy())
                rec["power"].append((d.applied_torque * d.joint_vel).abs().sum(dim=-1).cpu().numpy())
                rec["done"].append(dones.bool().cpu().numpy())
                rec["time_out"].append(timeouts.cpu().numpy())
            if (t + 1) % 250 == 0:
                print(f"[gait] {t + 1}/{T}", flush=True)
        arrays = {k: np.stack(v) for k, v in rec.items()}
        summary = G.summarize(arrays, np.repeat(speeds_list, args.envs_per_speed), dt, int(round(args.warmup_s / dt)))
        report = {"label": args.label, "task": args.task, "pkg": args.pkg, "checkpoint": str(args.checkpoint.resolve()),
                  "activation": activation, "obs_normalizer": norm is not None, "policy_obs_dim": in_dim,
                  "foot_names": list(foot_names), "dt": dt, "seconds": args.seconds, "warmup_s": args.warmup_s,
                  "seed": args.seed, "conditions": "plane, no pushes/external forces, no obs noise, fixed forward commands",
                  "summary": {str(k): v for k, v in summary.items()}}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(G.table({args.label: summary}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
