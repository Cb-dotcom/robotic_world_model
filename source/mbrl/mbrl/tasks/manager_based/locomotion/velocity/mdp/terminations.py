"""Fall definition of plan v6 §5 for the Go2 start-policy tasks (decision 14 Nov 2026).

A fall = base height below 0.15 m or body tilt above 60 degrees (angle between the body z axis and
world up, from the projected gravity). The same rule is used as a termination and, OR-ed with the
base contact, as the world model's system_termination label, so the pre-reset recorder's check
(label == reset_terminated) still holds.
"""
from __future__ import annotations

import math

import torch
from isaaclab.managers import SceneEntityCfg

PLAN_FALL_HEIGHT_M = 0.15
PLAN_FALL_TILT_RAD = math.radians(60.0)


def plan_fall(env, minimum_height: float = PLAN_FALL_HEIGHT_M, limit_angle: float = PLAN_FALL_TILT_RAD,
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """True where the base is below `minimum_height` (world z) or tilted beyond `limit_angle`."""
    asset = env.scene[asset_cfg.name]
    low = asset.data.root_pos_w[:, 2] < minimum_height
    tilt = torch.acos(torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0))
    return low | (tilt > limit_angle)


def base_contact_or_plan_fall(env, threshold: float, sensor_cfg: SceneEntityCfg,
                              minimum_height: float = PLAN_FALL_HEIGHT_M, limit_angle: float = PLAN_FALL_TILT_RAD,
                              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """[N, 1] bool: base contact (as the base_contact termination) OR plan_fall."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    contact = (torch.max(torch.norm(forces, dim=-1), dim=1)[0] > threshold).any(dim=-1)
    return (contact | plan_fall(env, minimum_height, limit_angle, asset_cfg)).unsqueeze(-1)
