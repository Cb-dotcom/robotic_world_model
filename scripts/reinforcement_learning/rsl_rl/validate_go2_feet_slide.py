# SPDX-License-Identifier: BSD-3-Clause
"""Check the kinematic Go2 feet_slide (model_based/envs/go2_kinematics.py) against the simulator.

Runs a saved Go2 policy in the Pretrain task (pushes off, so the post-step state equals the
state the reward saw) and compares, on envs that did not reset in that step:
  1. per-foot horizontal speed: kinematics from the 45-dim system_state vs asset body_lin_vel_w
  2. the unweighted feet_slide term: kinematics + observed foot contact vs the reward manager
  3. foot-contact gate: system_contact foot bits vs the sensor gate used by the reward
for three base COM offsets: URDF, zero, and the asset's own value.

Pass rule (fixed before the run): for the best offset, speed RMSE <= 0.02 m/s,
term mean absolute error <= 5% of the mean simulator term, gate agreement >= 0.999.
Only a pass allows the feet_slide weight in configs/go2_flat_cfg.py to change from 0 to -0.25.
"""
import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Isaac-Velocity-Flat-Unitree-Go2-Pretrain-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", type=Path, required=True, help="fresh .json path")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.out.exists() or not args.checkpoint:
    parser.error("Require a fresh --out and --checkpoint")
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import rsl_rl.runners as rsl_runners  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import mbrl.tasks  # noqa: F401,E402

MB = Path(__file__).resolve().parents[1] / "model_based"
sys.path.insert(0, str(MB / "envs"))
import go2_kinematics as K  # noqa: E402

EXPECTED_JOINTS = [f"{leg}_{j}_joint" for j in ("hip", "thigh", "calf") for leg in K.LEGS]
EXPECTED_FEET = [f"{leg}_foot" for leg in K.LEGS]
BARS = {"speed_rmse_m_s": 0.02, "term_rel_mae": 0.05, "gate_agreement": 0.999}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg.seed = args.seed
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        env_cfg.sim.device = args.device
    env_cfg.events.push_robot = None
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg, render_mode=None), clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    try:
        robot, sensor = base.scene["robot"], base.scene["contact_forces"]
        if list(robot.joint_names) != EXPECTED_JOINTS:
            raise RuntimeError(f"Joint order differs from go2_kinematics: {robot.joint_names}")
        foot_ids, foot_names = robot.find_bodies(".*_foot", preserve_order=False)
        sensor_ids, sensor_names = sensor.find_bodies(".*_foot", preserve_order=False)
        if list(foot_names) != EXPECTED_FEET or list(sensor_names) != EXPECTED_FEET:
            raise RuntimeError(f"Foot order differs: robot {foot_names}, sensor {sensor_names}")
        default_q = robot.data.default_joint_pos[0].cpu().numpy()
        if not np.allclose(default_q, K.DEFAULT_Q, atol=1e-6):
            raise RuntimeError(f"Default joint positions differ: {default_q.tolist()}")
        base_id = robot.find_bodies("base")[0][0]
        options = {"urdf": K.URDF_BASE_COM, "zero": (0.0, 0.0, 0.0)}
        com_b = getattr(robot.data, "body_com_pos_b", None)
        asset_com = None if com_b is None else com_b[0, base_id].cpu().numpy().astype(float).tolist()
        if asset_com is not None:
            options["asset"] = tuple(asset_com)

        names = list(base.reward_manager.active_terms)
        k = names.index("feet_slide")
        weight = float(base.reward_manager.get_term_cfg("feet_slide").weight)

        runner = getattr(rsl_runners, agent_cfg.class_name)(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        checkpoint = Path(args.checkpoint).resolve()
        runner.alg.policy.load_state_dict(torch.load(checkpoint, map_location=base.device, weights_only=False)["model_state_dict"])
        runner.alg.policy.eval()
        policy = runner.get_inference_policy(device=base.device)

        true_speed, sim_term, gate_obs, gate_sim = [], [], [], []
        imag_speed = {key: [] for key in options}
        imag_term = {key: [] for key in options}
        obs = env.get_observations()
        for t in range(args.num_steps):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
                keep = ~dones.bool()
                if not keep.any():
                    continue
                st = base.obs_buf["system_state"][keep]
                contact = base.obs_buf["system_contact"][keep][:, 4:8]
                v, w, g, q, dq = st[:, 0:3], st[:, 3:6], st[:, 6:9], st[:, 9:21], st[:, 21:33]
                true_speed.append(robot.data.body_lin_vel_w[keep][:, foot_ids, :2].norm(dim=-1).cpu())
                sim_term.append((base.reward_manager._step_reward[keep, k] / weight).cpu())
                forces = sensor.data.net_forces_w_history[keep][:, :, sensor_ids, :].norm(dim=-1).max(dim=1)[0]
                gate_sim.append((forces > 1.0).float().cpu())
                gate_obs.append(contact.float().cpu())
                for key, c0 in options.items():
                    imag_speed[key].append(K.foot_horizontal_speed(v, w, g, q, dq, c0).cpu())
                    imag_term[key].append(K.feet_slide(v, w, g, q, dq, contact, c0).cpu())
            if (t + 1) % 200 == 0:
                print(f"[feet-slide check] {t + 1}/{args.num_steps}", flush=True)

        ts, st_, gs, go = (torch.cat(x).double() for x in (true_speed, sim_term, gate_sim, gate_obs))
        mean_term = float(st_.abs().mean())
        results = {}
        for key in options:
            s = torch.cat(imag_speed[key]).double()
            m = torch.cat(imag_term[key]).double()
            err = s - ts
            results[key] = {
                "base_com_offset": list(options[key]),
                "speed_rmse_m_s": float(err.pow(2).mean().sqrt()),
                "speed_max_abs_err_m_s": float(err.abs().max()),
                "speed_corr": float(np.corrcoef(s.flatten().numpy(), ts.flatten().numpy())[0, 1]),
                "term_mae": float((m - st_).abs().mean()),
                "term_rel_mae": float((m - st_).abs().mean() / max(mean_term, 1e-9)),
            }
        best = min(results, key=lambda key: results[key]["speed_rmse_m_s"])
        agreement = float((gs == go).double().mean())
        passed = (results[best]["speed_rmse_m_s"] <= BARS["speed_rmse_m_s"]
                  and results[best]["term_rel_mae"] <= BARS["term_rel_mae"]
                  and agreement >= BARS["gate_agreement"])
        report = {
            "pass": bool(passed), "best_offset": best, "bars": BARS, "results": results,
            "gate_agreement": agreement, "samples_env_steps": int(ts.shape[0]),
            "mean_sim_term": mean_term, "mean_true_speed_m_s": float(ts.mean()),
            "feet_slide_weight_in_task": weight, "asset_base_com": asset_com,
            "task": args.task, "seed": args.seed, "num_envs": args.num_envs, "num_steps": args.num_steps,
            "push_enabled": False, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "script_sha256": sha256(__file__), "kinematics_sha256": sha256(MB / "envs/go2_kinematics.py"),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as f:
            json.dump(report, f, indent=2)
        print(json.dumps({k2: report[k2] for k2 in ("pass", "best_offset", "gate_agreement", "results")}, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
