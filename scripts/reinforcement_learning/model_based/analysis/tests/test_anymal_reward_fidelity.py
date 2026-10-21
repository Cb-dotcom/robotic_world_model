"""CPU checks of imagined ANYmal rewards against physical quantities."""
from pathlib import Path
import sys

import pytest
import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from envs.anymal_d_flat import AnymalDFlatEnv


@pytest.mark.parametrize("dt", [0.01, 0.02])
@pytest.mark.parametrize("acceleration", [0.0, 3.0, -4.0])
def test_acceleration_reward_uses_previous_velocity(dt, acceleration):
    env = AnymalDFlatEnv.__new__(AnymalDFlatEnv)
    env._num_envs = 2
    env._step_dt = dt
    env.device = "cpu"
    env.base_velocity = torch.zeros(2, 3)
    env.obs_last_action = torch.zeros(2, 12)
    obs = torch.zeros(2, 48)
    # Position must not enter the finite difference of velocity.
    obs[:, 12:24] = torch.tensor([0.5, -0.75])[:, None]
    previous_velocity = torch.linspace(-2.0, 2.0, 24).reshape(2, 12)
    obs[:, 24:36] = previous_velocity
    env.last_obs = TensorDict({"policy": obs}, batch_size=[2])
    state = {
        "base_lin_vel": torch.zeros(2, 3),
        "base_ang_vel": torch.zeros(2, 3),
        "projected_gravity": torch.tensor([[0., 0., -1.]]).repeat(2, 1),
        "joint_pos": torch.ones(2, 12) * 0.25,
        "joint_vel": previous_velocity + acceleration * dt,
        "joint_torque": torch.zeros(2, 12),
    }
    contacts = {"thigh_contact": None, "foot_contact": None}
    action = torch.zeros(2, 12)
    expected = torch.full((2,), 12 * acceleration ** 2)
    env._compute_imagination_reward_terms(state, action, None, contacts)
    torch.testing.assert_close(env.imagination_reward_per_step["dof_acc_l2"], expected)
    # The next call must use the just-produced velocity as its previous value.
    state["joint_vel"] = state["joint_vel"] + acceleration * dt
    env._compute_imagination_reward_terms(state, action, None, contacts)
    torch.testing.assert_close(env.imagination_reward_per_step["dof_acc_l2"], expected)
