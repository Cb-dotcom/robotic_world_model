"""CPU test of mdp/terminations.py (plan v6 fall rule) with a stub env; Isaac Lab not needed."""
import importlib.util
import math
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[5]
FILE = ROOT / "source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/mdp/terminations.py"


def load():
    """Import terminations.py with a stub isaaclab.managers; restore sys.modules afterwards."""
    stub = types.ModuleType("isaaclab.managers")
    stub.SceneEntityCfg = lambda name, **kw: types.SimpleNamespace(name=name, body_ids=kw.get("body_ids", [0]))
    saved = {k: sys.modules.get(k) for k in ("isaaclab", "isaaclab.managers")}
    sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    sys.modules["isaaclab.managers"] = stub
    try:
        spec = importlib.util.spec_from_file_location("plan_terminations", FILE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


class Scene(dict):
    def __init__(self, robot, sensor):
        super().__init__(robot=robot)
        self.sensors = {"contact_forces": sensor}


def make_env(heights, tilts_deg, forces):
    t = torch.tensor(tilts_deg).float() * math.pi / 180
    g = torch.stack([torch.sin(t), torch.zeros_like(t), -torch.cos(t)], dim=-1)
    root = torch.zeros(len(heights), 3)
    root[:, 2] = torch.tensor(heights)
    robot = types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=root, projected_gravity_b=g))
    hist = torch.zeros(len(heights), 3, 1, 3)          # [N, history, bodies, xyz]
    hist[:, 1, 0, 2] = torch.tensor(forces).float()
    sensor = types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w_history=hist))
    return types.SimpleNamespace(scene=Scene(robot, sensor))


def test_plan_fall_and_label():
    m = load()
    e = make_env([0.33, 0.12, 0.30, 0.30, 0.30], [5.0, 5.0, 59.0, 61.0, 5.0], [0.0, 0.0, 0.0, 0.0, 5.0])
    robot = types.SimpleNamespace(name="robot")
    assert m.plan_fall(e, asset_cfg=robot).tolist() == [False, True, False, True, False]
    label = m.base_contact_or_plan_fall(e, 1.0, types.SimpleNamespace(name="contact_forces", body_ids=[0]),
                                        asset_cfg=robot)
    assert label.shape == (5, 1) and label[:, 0].tolist() == [False, True, False, True, True]
