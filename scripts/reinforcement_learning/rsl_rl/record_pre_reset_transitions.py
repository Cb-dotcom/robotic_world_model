# SPDX-License-Identifier: BSD-3-Clause
"""Record true pre-reset ANYmal transitions through IsaacLab recorder hooks.

This is a diagnostic protocol. An active recorder causes IsaacLab to compute
observations once more before reset, including policy-noise RNG draws. Both
policies use this protocol; counts need not reproduce the earlier evaluator.
"""
import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_steps", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", type=Path, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.out.exists() or args.num_envs < 1 or args.num_steps < 1 or not args.checkpoint:
    parser.error("Require a fresh output path, positive steps/envs and --checkpoint")
if args.task != "Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0":
    parser.error("This diagnostic is validated for ANYmal-D Pretrain only")
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import inspect
import json
import subprocess

import gymnasium as gym
import numpy as np
import torch
import rsl_rl.runners as rsl_runners
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import mbrl.tasks  # noqa: F401

MB = Path(__file__).resolve().parents[1] / "model_based"
sys.path.insert(0, str(MB / "analysis"))
from pre_reset_transitions import TransitionBuffer, array, sha256

buffer = TransitionBuffer(args.num_steps, args.num_envs)
policy_observation = None


def timer_fields(env):
    data = env.scene["contact_forces"].data
    # Full sensor order is saved in metadata. Offline reward timers use feet only.
    ids, _ = env.scene["contact_forces"].find_bodies(".*FOOT")
    return {"current_air_time": data.current_air_time[:, ids],
            "last_air_time": data.last_air_time[:, ids],
            "current_contact_time": data.current_contact_time[:, ids],
            "last_contact_time": data.last_contact_time[:, ids]}


class PreResetRecorder(RecorderTerm):
    def record_pre_step(self):
        env = self._env
        buffer.begin(state=env.observation_manager.compute_group("system_state"),
                     action=env.action_manager.action,
                     previous_action=env.action_manager.prev_action,
                     command=env.command_manager.get_command("base_velocity"),
                     policy_observation=policy_observation,
                     **timer_fields(env))
        return None, None

    def record_post_step(self):
        env = self._env
        # Private cached reward vector is intentional: calling reward functions
        # again could mutate their state. Pin/validate its shape and total below.
        terms = env.reward_manager._step_reward
        if terms.shape != (env.num_envs, len(env.reward_manager.active_terms)):
            raise RuntimeError("Unexpected IsaacLab cached reward layout")
        torch.testing.assert_close(terms.sum(-1) * env.step_dt, env.reward_buf, rtol=1e-5, atol=1e-6)
        if not torch.equal(env.obs_buf["system_termination"].squeeze(-1).bool(), env.reset_terminated.bool()):
            raise RuntimeError("World-model termination label differs from physical termination")
        buffer.end(next_state=env.obs_buf["system_state"],
                   observed_action=env.obs_buf["system_action"],
                   contact=env.obs_buf["system_contact"],
                   termination_observation=env.obs_buf["system_termination"],
                   terminated=env.reset_terminated.bool(), time_out=env.reset_time_outs.bool(),
                   done=env.reset_buf.bool(), episode_step=env.episode_length_buf,
                   reward=env.reward_buf, reward_terms=terms,
                   joint_acc=env.scene["robot"].data.joint_acc)
        return None, None

    def record_pre_reset(self, env_ids):
        if env_ids is not None and buffer.index:
            buffer.verify_before_reset(env_ids, self._env.observation_manager.compute_group("system_state"),
                                       self._env.action_manager.action)
        return None, None


def check_hook_order():
    source = inspect.getsource(ManagerBasedRLEnv.step)
    hooks = ("self.action_manager.process_action(", "self.recorder_manager.record_pre_step()",
             "self.termination_manager.compute()", "self.reward_manager.compute(",
             "self.recorder_manager.record_post_step()", "self.recorder_manager.record_pre_reset(",
             "self._reset_idx(", "self.command_manager.compute(")
    positions = [source.index(hook) for hook in hooks]
    if positions != sorted(positions):
        raise RuntimeError("IsaacLab hook order changed; recording is unsafe")
    return sha256(inspect.getsourcefile(ManagerBasedRLEnv))


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    global policy_observation
    env_source_hash = check_hook_order()
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    env_cfg.seed = agent_cfg.seed
    env_cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        env_cfg.sim.device = args.device
    env_cfg.events.push_robot = None
    env_cfg.commands.base_velocity.resampling_time_range = (10.0, 10.0)
    recorder_cfg = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE,
                                          export_in_record_pre_reset=False)
    recorder_cfg.transitions = RecorderTermCfg(class_type=PreResetRecorder)
    env_cfg.recorders = recorder_cfg
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    try:
        for name in ("system_state", "system_action", "system_contact", "system_termination"):
            group_cfg = getattr(env_cfg.observations, name)
            if group_cfg.enable_corruption:
                raise RuntimeError(f"Diagnostic group is noisy: {name}")
        runner = getattr(rsl_runners, agent_cfg.class_name)(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        checkpoint = Path(args.checkpoint).resolve()
        ckpt = torch.load(checkpoint, map_location=base.device, weights_only=False)
        runner.alg.policy.load_state_dict(ckpt["model_state_dict"])
        runner.alg.policy.eval()
        policy = runner.get_inference_policy(device=base.device)
        obs = env.get_observations()
        for t in range(args.num_steps):
            with torch.inference_mode():
                policy_observation = obs["policy"].clone()
                actions = policy(obs)
                obs, _, done, _ = env.step(actions)
            if buffer.index != t + 1 or not np.array_equal(array(done).astype(bool), buffer.data["done"][t]):
                raise RuntimeError("Recorder did not capture exactly one transition per environment step")
            if (t + 1) % 200 == 0:
                print(f"[pre-reset] {t + 1}/{args.num_steps}; resets={int(buffer.data['done'][:t+1].sum())}", flush=True)
        reward_names = list(base.reward_manager.active_terms)
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        meta = dict(task=args.task, seed=agent_cfg.seed, num_envs=args.num_envs,
                    num_steps=args.num_steps, step_dt=float(base.step_dt), push_enabled=False,
                    command_resampling_s=[10.0, 10.0], checkpoint=str(checkpoint),
                    checkpoint_sha256=sha256(checkpoint), git_commit=git,
                    isaaclab_env_source_sha256=env_source_hash,
                    recorder_sha256=sha256(__file__), storage_sha256=sha256(MB / "analysis/pre_reset_transitions.py"),
                    policy_noise=bool(env_cfg.observations.policy.enable_corruption),
                    termination_names=list(base.termination_manager.active_terms),
                    extra_observation_compute=True, clip_actions=agent_cfg.clip_actions,
                    action_space="ActionManager.action: wrapper-clipped, unscaled policy action; joint target scaling remains in simulator",
                    reward_names=reward_names,
                    reward_weights={name: base.reward_manager.get_term_cfg(name).weight for name in reward_names},
                    reward_units="reward_terms are weighted per-second; reward=sum(terms)*step_dt",
                    sensor_body_names=list(base.scene["contact_forces"].body_names),
                    timer_foot_body_names=base.scene["contact_forces"].find_bodies(".*FOOT")[1],
                    terminal_semantics="Physical termination and timeout stored separately; terminal next_state captured before reset")
        buffer.save(args.out, meta)
        sidecar = args.out.with_suffix(".json")
        with sidecar.open("x") as stream:
            json.dump(dict(meta, trace_sha256=sha256(args.out),
                           falls=int((buffer.data["done"] & ~buffer.data["time_out"]).sum())), stream, indent=2)
        print(f"[pre-reset] wrote {args.out}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
