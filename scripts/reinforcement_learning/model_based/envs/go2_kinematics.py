"""Go2 leg kinematics for imagination rewards (no simulator).

feet_slide in Isaac Lab (isaaclab_tasks ... locomotion/velocity/mdp/rewards.py) is
    sum_feet  || horizontal world velocity of the foot body ||  *  contact_gate
This module reconstructs the foot velocity from the 45-dim world-model state:
    u_i = v_C + omega x (r_i(q) - c_0) + J_i(q) dq          (all in the base frame)
    speed_i = sqrt(|u_i|^2 - (u_i . g_hat)^2)                 (remove the gravity-axis component)
v_C is the base velocity in the state, c_0 the base COM offset from the root-link origin
(Isaac Lab reports the root COM velocity), r_i the foot position relative to the root-link
origin. Yaw is not needed: the horizontal plane is the plane normal to gravity.

Geometry: unitree go2_description URDF (hip at (+-0.1934, +-0.0465, 0), thigh offset
+-0.0955 in y, thigh and calf 0.213 m; hip axis x, thigh/calf axis y; foot COM at the foot
origin). Joint order is joint-major (all hips, all thighs, all calves), legs FL, FR, RL, RR.
State joint positions are relative to the default pose. The simulator check
(rsl_rl/validate_go2_feet_slide.py) decides which base COM offset matches the asset.
"""
from __future__ import annotations

import torch

LEGS = ("FL", "FR", "RL", "RR")
HIP_OFFSET = ((0.1934, 0.0465, 0.0), (0.1934, -0.0465, 0.0), (-0.1934, 0.0465, 0.0), (-0.1934, -0.0465, 0.0))
THIGH_Y = (0.0955, -0.0955, 0.0955, -0.0955)
L_THIGH = 0.213
L_CALF = 0.213
# Isaac Lab UNITREE_GO2_CFG init_state.joint_pos, joint-major order
DEFAULT_Q = (0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5)
URDF_BASE_COM = (0.021112, 0.0, -0.005366)


def _consts(ref):
    kw = dict(dtype=ref.dtype, device=ref.device)
    return (torch.tensor(HIP_OFFSET, **kw), torch.tensor(THIGH_Y, **kw), torch.tensor(DEFAULT_Q, **kw))


def foot_positions_and_velocities(joint_pos_rel, joint_vel):
    """Foot positions r [B,4,3] and joint-induced velocities r_dot [B,4,3] in the base frame."""
    hip_offset, thigh_y, default_q = _consts(joint_pos_rel)
    q = joint_pos_rel + default_q
    qh, qt, qc = q[:, 0:4], q[:, 4:8], q[:, 8:12]
    dh, dt, dc = joint_vel[:, 0:4], joint_vel[:, 4:8], joint_vel[:, 8:12]
    s1, c1 = torch.sin(qt), torch.cos(qt)
    s12, c12 = torch.sin(qt + qc), torch.cos(qt + qc)
    # leg vector in the hip frame (after the hip rotation is removed)
    wx = -L_THIGH * s1 - L_CALF * s12
    wy = thigh_y.expand_as(qt)
    wz = -L_THIGH * c1 - L_CALF * c12
    wx_dot = (-L_THIGH * c1 - L_CALF * c12) * dt + (-L_CALF * c12) * dc
    wz_dot = (L_THIGH * s1 + L_CALF * s12) * dt + (L_CALF * s12) * dc
    sh, ch = torch.sin(qh), torch.cos(qh)
    # rotate about x by the hip angle
    rx = wx
    ry = ch * wy - sh * wz
    rz = sh * wy + ch * wz
    r = torch.stack([rx, ry, rz], dim=-1) + hip_offset
    rx_dot = wx_dot
    ry_dot = (-sh * wy - ch * wz) * dh - sh * wz_dot
    rz_dot = (ch * wy - sh * wz) * dh + ch * wz_dot
    r_dot = torch.stack([rx_dot, ry_dot, rz_dot], dim=-1)
    return r, r_dot


def foot_horizontal_speed(base_lin_vel, base_ang_vel, projected_gravity, joint_pos_rel, joint_vel,
                          base_com_offset=URDF_BASE_COM):
    """Horizontal (gravity-normal) speed of each foot [B,4], in m/s."""
    r, r_dot = foot_positions_and_velocities(joint_pos_rel, joint_vel)
    c0 = torch.as_tensor(base_com_offset, dtype=r.dtype, device=r.device)
    omega = base_ang_vel[:, None, :].expand_as(r)
    u = base_lin_vel[:, None, :] + torch.cross(omega, r - c0, dim=-1) + r_dot
    g_hat = projected_gravity / projected_gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    along = (u * g_hat[:, None, :]).sum(-1)
    return torch.sqrt(torch.clamp((u * u).sum(-1) - along * along, min=0.0))


def feet_slide(base_lin_vel, base_ang_vel, projected_gravity, joint_pos_rel, joint_vel, foot_contact,
               base_com_offset=URDF_BASE_COM):
    """Unweighted Isaac Lab feet_slide term [B]: sum over feet of contact * horizontal speed."""
    speed = foot_horizontal_speed(base_lin_vel, base_ang_vel, projected_gravity, joint_pos_rel, joint_vel,
                                  base_com_offset)
    return (speed * foot_contact).sum(-1)
