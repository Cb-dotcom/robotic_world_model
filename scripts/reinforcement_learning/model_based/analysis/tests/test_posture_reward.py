"""CPU test of mdp/rewards.py joint_position_penalty (start-policy iteration 3) with a stub env."""
import importlib.util
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[5]
FILE = ROOT / "source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/mdp/rewards.py"


def load():
    stub = types.ModuleType("isaaclab.managers")
    stub.SceneEntityCfg = lambda name, **kw: types.SimpleNamespace(name=name)
    saved = {k: sys.modules.get(k) for k in ("isaaclab", "isaaclab.managers")}
    sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    sys.modules["isaaclab.managers"] = stub
    try:
        spec = importlib.util.spec_from_file_location("posture_rewards", FILE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


def make_env(joint_dev, cmd, body_vx):
    n = len(cmd)
    default = torch.zeros(n, 12)
    pos = default.clone()
    pos[:, 0] = torch.tensor(joint_dev)
    vel = torch.zeros(n, 3)
    vel[:, 0] = torch.tensor(body_vx)
    robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=pos, default_joint_pos=default, root_lin_vel_b=vel))
    commands = torch.zeros(n, 3)
    commands[:, 0] = torch.tensor(cmd)
    cm = types.SimpleNamespace(get_command=lambda name: commands)
    return types.SimpleNamespace(scene={"robot": robot}, command_manager=cm)


def test_joint_position_penalty_matches_unitree_definition():
    m = load()
    # rows: moving with command; no command but body fast; no command and slow (x5); exactly at default
    env = make_env([0.3, 0.3, 0.3, 0.0], [0.5, 0.0, 0.0, 0.0], [0.5, 0.5, 0.1, 0.0])
    out = m.joint_position_penalty(env, types.SimpleNamespace(name="robot"), stand_still_scale=5.0, velocity_threshold=0.3)
    assert torch.allclose(out, torch.tensor([0.3, 0.3, 1.5, 0.0]))
