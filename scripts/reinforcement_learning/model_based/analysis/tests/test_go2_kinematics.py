"""CPU tests for envs/go2_kinematics.py (no simulator)."""
from pathlib import Path
import sys

import pytest
import torch

MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MB / "envs"))
import go2_kinematics as K  # noqa: E402

DEFAULT = torch.tensor(K.DEFAULT_Q, dtype=torch.float64)


@pytest.fixture(autouse=True)
def float64_default():
    """float64 inside this module only; the old module-level switch leaked into every later test file."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def test_straight_legs_geometry():
    q_rel = (torch.zeros(12) - DEFAULT)[None]  # absolute joint angles all zero
    r, _ = K.foot_positions_and_velocities(q_rel, torch.zeros(1, 12))
    expected = torch.tensor([[0.1934, 0.0465 + 0.0955, -0.426], [0.1934, -0.1420, -0.426],
                             [-0.1934, 0.1420, -0.426], [-0.1934, -0.1420, -0.426]])
    torch.testing.assert_close(r[0], expected)


def test_hip_rotation_moves_foot_outward():
    q_abs = torch.zeros(12)
    q_abs[0] = 0.3  # FL hip about +x: foot y increases (straight leg below the hip)
    r, _ = K.foot_positions_and_velocities((q_abs - DEFAULT)[None], torch.zeros(1, 12))
    assert r[0, 0, 1] > 0.1420


def test_joint_velocity_matches_finite_difference():
    g = torch.Generator().manual_seed(0)
    q = 0.5 * torch.randn(64, 12, generator=g)
    dq = torch.randn(64, 12, generator=g)
    eps = 1e-6
    r_plus, _ = K.foot_positions_and_velocities(q + eps * dq, dq)
    r_minus, _ = K.foot_positions_and_velocities(q - eps * dq, dq)
    _, r_dot = K.foot_positions_and_velocities(q, dq)
    torch.testing.assert_close(r_dot, (r_plus - r_minus) / (2 * eps), rtol=1e-6, atol=1e-8)


def test_pure_translation_and_gravity_projection():
    v = torch.tensor([[0.3, -0.4, 0.7]])
    g = torch.tensor([[0.0, 0.0, -1.0]])
    speed = K.foot_horizontal_speed(v, torch.zeros(1, 3), g, torch.zeros(1, 12), torch.zeros(1, 12))
    torch.testing.assert_close(speed, torch.full((1, 4), 0.5))


def test_vertical_motion_of_tilted_base_has_no_horizontal_speed():
    tilt = 0.4
    g_b = torch.tensor([[-torch.sin(torch.tensor(tilt)), 0.0, -torch.cos(torch.tensor(tilt))]])
    up_world_in_base = -g_b
    s_vert = K.foot_horizontal_speed(2.0 * up_world_in_base, torch.zeros(1, 3), g_b,
                                     torch.zeros(1, 12), torch.zeros(1, 12))
    torch.testing.assert_close(s_vert, torch.zeros(1, 4), atol=1e-12, rtol=0)


def test_rotation_term_and_com_offset():
    # base spinning about z with no joint motion: foot speed = |omega| * horizontal distance to COM
    w = torch.tensor([[0.0, 0.0, 2.0]])
    grav = torch.tensor([[0.0, 0.0, -1.0]])
    q_rel = (torch.zeros(12) - DEFAULT)[None]
    c0 = (0.02, 0.0, 0.0)
    s = K.foot_horizontal_speed(torch.zeros(1, 3), w, grav, q_rel, torch.zeros(1, 12), base_com_offset=c0)
    r, _ = K.foot_positions_and_velocities(q_rel, torch.zeros(1, 12))
    d = (r[0, :, :2] - torch.tensor(c0[:2])).norm(dim=-1)
    torch.testing.assert_close(s[0], 2.0 * d)


def test_feet_slide_gates_on_contact():
    v = torch.tensor([[1.0, 0.0, 0.0]])
    grav = torch.tensor([[0.0, 0.0, -1.0]])
    contact = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    out = K.feet_slide(v, torch.zeros(1, 3), grav, torch.zeros(1, 12), torch.zeros(1, 12), contact)
    torch.testing.assert_close(out, torch.tensor([2.0]))
