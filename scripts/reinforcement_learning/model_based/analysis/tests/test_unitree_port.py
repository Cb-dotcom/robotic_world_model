"""CPU tests of the start-policy iteration 4 port of Unitree's Go2 recipe (stub env; Isaac Lab not needed).

- every reward weight of UnitreeGo2FlatEnvCfg_START_UNITREE equals unitree_rl_lab 4960b84
  (robots/go2/velocity_env_cfg.py RewardsCfg), read from the source with ast
- the copied curriculum, energy and air-time-variance functions behave as Unitree's
"""
import ast
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[5]
VEL = ROOT / "source/mbrl/mbrl/tasks/manager_based/locomotion/velocity"

# unitree_rl_lab 4960b84 RewardsCfg (Unitree name -> our term name, weight)
UNITREE = {
    "track_lin_vel_xy_exp": 1.5, "track_ang_vel_z_exp": 0.75, "lin_vel_z_l2": -2.0, "ang_vel_xy_l2": -0.05,
    "joint_vel": -0.001, "dof_acc_l2": -2.5e-7, "dof_torques_l2": -2e-4, "action_rate_l2": -0.1,
    "dof_pos_limits": -10.0, "energy": -2e-5, "flat_orientation_l2": -2.5, "joint_pos": -0.7,
    "feet_air_time": 0.1, "air_time_variance": -1.0, "feet_slide": -0.1, "undesired_contacts": -1.0,
}


def load(name, stub_managers=True):
    saved = {k: sys.modules.get(k) for k in ("isaaclab", "isaaclab.managers")}
    if stub_managers:
        stub = types.ModuleType("isaaclab.managers")
        stub.SceneEntityCfg = lambda name, **kw: types.SimpleNamespace(name=name, **kw)
        sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
        sys.modules["isaaclab.managers"] = stub
    try:
        spec = importlib.util.spec_from_file_location(f"port_{name}", VEL / "mdp" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


def config_weights():
    tree = ast.parse((VEL / "config/go2/flat_env_cfg.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnitreeGo2FlatEnvCfg_START_UNITREE")
    weights, removed = {}, set()
    for node in ast.walk(cls):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        # r.<term>.weight = value
        if isinstance(t, ast.Attribute) and t.attr == "weight" and isinstance(t.value, ast.Attribute):
            weights[t.value.attr] = ast.literal_eval(node.value)
        # r.<term> = RewardTermCfg(..., weight=value) or None
        elif isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "r":
            if isinstance(node.value, ast.Constant) and node.value.value is None:
                removed.add(t.attr)
            elif isinstance(node.value, ast.Call):
                kw = {k.arg: k.value for k in node.value.keywords}
                weights[t.attr] = ast.literal_eval(kw["weight"])
    return weights, removed


def test_every_reward_weight_is_unitrees():
    weights, removed = config_weights()
    assert weights == UNITREE
    assert removed == {"stand_still"}


class Ranges:
    def __init__(self, x, y):
        self.lin_vel_x, self.lin_vel_y = x, y


def curriculum_env(mean_reward_per_s, step, x=(-0.1, 0.1), y=(-0.1, 0.1)):
    ranges = Ranges(x, y)
    term = types.SimpleNamespace(cfg=types.SimpleNamespace(ranges=ranges))
    sums = {"track_lin_vel_xy_exp": torch.full((4,), mean_reward_per_s * 20.0)}
    rm = types.SimpleNamespace(get_term_cfg=lambda name: types.SimpleNamespace(weight=1.5), _episode_sums=sums)
    env = types.SimpleNamespace(command_manager=types.SimpleNamespace(get_term=lambda name: term), reward_manager=rm,
                                max_episode_length_s=20.0, max_episode_length=1000, common_step_counter=step,
                                device="cpu")
    return env, ranges


def test_curriculum_widens_only_on_good_tracking_at_episode_boundaries():
    c = load("curriculums")
    env, r = curriculum_env(1.3, 2000)                    # > 0.8 x 1.5 = 1.2, boundary step
    c.unitree_lin_vel_cmd_levels(env, [0, 1, 2, 3])
    assert r.lin_vel_x == pytest.approx([-0.2, 0.2]) and r.lin_vel_y == pytest.approx([-0.2, 0.2])
    env, r = curriculum_env(1.3, 2001)                    # not a boundary step
    c.unitree_lin_vel_cmd_levels(env, [0])
    assert tuple(r.lin_vel_x) == (-0.1, 0.1)
    env, r = curriculum_env(1.1, 3000)                    # tracking too poor
    c.unitree_lin_vel_cmd_levels(env, [0])
    assert tuple(r.lin_vel_x) == (-0.1, 0.1)
    env, r = curriculum_env(1.4, 1000, x=(-1.0, 1.0), y=(-0.95, 0.95))
    out = c.unitree_lin_vel_cmd_levels(env, [0], limit_lin_vel_x=(-1.0, 1.0), limit_lin_vel_y=(-1.0, 1.0))
    assert r.lin_vel_x == pytest.approx([-1.0, 1.0]) and r.lin_vel_y == pytest.approx([-1.0, 1.0]) and float(out) == 1.0


def test_energy_and_air_time_variance():
    m = load("rewards")
    robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_vel=torch.tensor([[1.0, -2.0]]),
                                                             applied_torque=torch.tensor([[3.0, 4.0]])))
    env = types.SimpleNamespace(scene={"robot": robot})
    assert m.unitree_energy(env, types.SimpleNamespace(name="robot", joint_ids=slice(None))).item() == pytest.approx(11.0)
    air = torch.tensor([[0.2, 0.2, 0.9, 0.2]])            # 0.9 clipped to 0.5
    contact = torch.tensor([[0.3, 0.3, 0.3, 0.3]])
    sensor = types.SimpleNamespace(cfg=types.SimpleNamespace(track_air_time=True),
                                   data=types.SimpleNamespace(last_air_time=air, last_contact_time=contact))
    env = types.SimpleNamespace(scene=types.SimpleNamespace(sensors={"contact_forces": sensor}))
    got = m.unitree_air_time_variance_penalty(env, types.SimpleNamespace(name="contact_forces", body_ids=slice(None)))
    assert got.item() == pytest.approx(torch.var(torch.tensor([0.2, 0.2, 0.5, 0.2])).item())
    sensor.cfg.track_air_time = False
    with pytest.raises(RuntimeError):
        m.unitree_air_time_variance_penalty(env, types.SimpleNamespace(name="contact_forces", body_ids=slice(None)))
