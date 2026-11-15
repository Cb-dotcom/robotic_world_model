from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def unitree_lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    limit_lin_vel_x: tuple[float, float] = (-1.0, 1.0),
    limit_lin_vel_y: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Widen the x/y velocity command ranges by 0.1 m/s on each side once per episode length while the mean
    tracking reward exceeds 80 % of its weight. Logic copied from unitree_rl_lab 4960b84
    (tasks/locomotion/mdp/curriculums.py lin_vel_cmd_levels, Apache-2.0); the limit ranges are passed as
    parameters instead of a command-cfg subclass (start-policy iteration 4, decision 16 Nov 2026)."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_lin_vel_x[0], limit_lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_lin_vel_y[0], limit_lin_vel_y[1],
            ).tolist()
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)
