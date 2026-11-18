import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import e16_reward_check as e16
from configs.go2_flat_cfg import S2_REWARD_CONTRACT, s2_reward_term_weights
from envs.go2_kinematics import feet_slide as go2_feet_slide
from pre_reset_transitions import SCHEMA

DT = 0.02
DEFAULT = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]
LIMITS = [[-1.0, 1.0]] * 4 + [[-1.5, 3.4]] * 4 + [[-2.7, -0.9]] * 4


def sim_terms(tr, extra_hip_contacts):
    """Independent numpy versions of the simulator's terms (Isaac Lab / unitree_rl_lab formulas)."""
    s, nx, c, a, pa, cmd = (tr[k] for k in ("state", "next_state", "contact", "action", "previous_action", "command"))
    v, w, g, jp, jv, tau = nx[..., 0:3], nx[..., 3:6], nx[..., 6:9], nx[..., 9:21], nx[..., 21:33], nx[..., 33:45]
    foot = c[..., 4:] > 0.5
    cur_air, last_air = tr["current_air_time"].copy(), tr["last_air_time"].copy()
    cur_con, last_con = tr["current_contact_time"].copy(), tr["last_contact_time"].copy()
    first_c, first_d = (cur_air > 0) & foot, (cur_con > 0) & ~foot
    last_air = np.where(first_c, cur_air + DT, last_air)
    cur_air = np.where(~foot, cur_air + DT, 0)
    last_con = np.where(first_d, cur_con + DT, last_con)
    cur_con = np.where(foot, cur_con + DT, 0)
    q = jp + np.asarray(DEFAULT)
    lim = np.asarray(LIMITS)
    moving = (np.linalg.norm(cmd, axis=-1) > 0) | (np.linalg.norm(v[..., :2], axis=-1) > 0.3)
    fs = go2_feet_slide(*(torch.from_numpy(x.reshape(-1, x.shape[-1])).float() for x in (v, w, g, jp, jv)),
                        torch.from_numpy(c[..., 4:].reshape(-1, 4)).float(),
                        base_com_offset=(0.021112, 0.0, -0.005366)).numpy().reshape(v.shape[:-1])
    raw = {
        "track_lin_vel_xy_exp": np.exp(-np.sum((cmd[..., :2] - v[..., :2]) ** 2, -1) / 0.25),
        "track_ang_vel_z_exp": np.exp(-(cmd[..., 2] - w[..., 2]) ** 2 / 0.25),
        "lin_vel_z_l2": v[..., 2] ** 2,
        "ang_vel_xy_l2": np.sum(w[..., :2] ** 2, -1),
        "joint_vel": np.sum(jv ** 2, -1),
        "dof_acc_l2": np.sum(((jv - s[..., 21:33]) / DT) ** 2, -1),
        "dof_torques_l2": np.sum(tau ** 2, -1),
        "action_rate_l2": np.sum((a - pa) ** 2, -1),
        "dof_pos_limits": np.sum(-np.clip(q - lim[:, 0], None, 0) + np.clip(q - lim[:, 1], 0, None), -1),
        "energy": np.sum(np.abs(jv) * np.abs(tau), -1),
        "flat_orientation_l2": np.sum(g[..., :2] ** 2, -1),
        "joint_pos": np.linalg.norm(jp, axis=-1) * np.where(moving, 1.0, 5.0),
        "feet_air_time": np.sum((last_air - 0.5) * ((cur_con > 0) & (cur_con < DT + 1e-8)), -1)
        * (np.linalg.norm(cmd[..., :2], axis=-1) > 0.1),
        "air_time_variance": np.var(np.clip(last_air, None, 0.5), -1, ddof=1) + np.var(np.clip(last_con, None, 0.5), -1, ddof=1),
        "feet_slide": fs,
        "undesired_contacts": np.sum(c[..., :4], -1) + extra_hip_contacts,
    }
    names = list(S2_REWARD_CONTRACT)
    return names, np.stack([raw[n] * S2_REWARD_CONTRACT[n]["weight"] for n in names], -1).astype(np.float32)


def synthetic_trace(T=60, N=16, seed=0, extra_hip=True):
    rng = np.random.default_rng(seed)
    st = rng.normal(0, .3, (T, N, 45)).astype(np.float32)
    st[..., 6:9] = np.array([0, 0, -1]) + rng.normal(0, .05, (T, N, 3))
    st[..., 9:21] = rng.normal(0, .6, (T, N, 12))                     # some joints beyond the soft limits
    st[:, :3, 0:2] *= 0.1                                             # slow standing envs (below)
    nx = np.empty_like(st)
    nx[:-1], nx[-1] = st[1:], st[-1]
    contact = (rng.random((T, N, 8)) < .5).astype(np.float32)
    timers = {k: rng.uniform(0, .6, (T, N, 4)).astype(np.float32) for k in
              ("current_air_time", "last_air_time", "current_contact_time", "last_contact_time")}
    # a foot is either in the air or in contact before the step
    in_air = rng.random((T, N, 4)) < .5
    timers["current_air_time"] *= in_air
    timers["current_contact_time"] *= ~in_air
    cmd = rng.uniform(-1, 1, (T, N, 3)).astype(np.float32)
    cmd[:, :3] = 0.0                                                  # standing envs: the x5 branch of joint_pos
    tr = dict(state=st, next_state=nx, contact=contact, action=rng.normal(0, .5, (T, N, 12)).astype(np.float32),
              previous_action=rng.normal(0, .5, (T, N, 12)).astype(np.float32), command=cmd, **timers)
    hip = (rng.random((T, N)) < .3).astype(np.float32) if extra_hip else 0.0
    names, terms = sim_terms(tr, hip)
    done = np.zeros((T, N), bool)
    meta = dict(schema=SCHEMA, verified_terminal_rows=0, step_dt=DT, task="Template-Isaac-Velocity-Flat-Unitree-Go2-Start-Unitree-Check-v0",
                reward_names=names, reward_weights={n: S2_REWARD_CONTRACT[n]["weight"] for n in names},
                reward_params={"feet_air_time": {"threshold": 0.5, "sensor_cfg": "x"},
                               "joint_pos": {"stand_still_scale": 5.0, "velocity_threshold": 0.3}},
                default_joint_pos=DEFAULT, soft_joint_pos_limits=LIMITS)
    tr.update(reward_terms=terms, done=done, terminated=done, time_out=done, metadata=np.array(json.dumps(meta)))
    return tr, meta


def test_contract_weights_cover_every_s2_term():
    w = s2_reward_term_weights()
    assert len(w) == 16 and w["dof_pos_limits"] == -10.0 and w["feet_air_time"] == 0.1 and "stand_still" not in w


def test_imagined_terms_match_independent_simulator_formulas(tmp_path):
    tr, meta = synthetic_trace()
    p = tmp_path / "t.npz"
    np.savez(p, **tr)
    rep = e16.run(p, tmp_path / "out")
    terms = rep["terms"]
    for name, r in terms.items():
        if name == "undesired_contacts":
            assert r["verdict"] == "DIFFERS" and r["mean_bias"] > 0          # hip/calf contacts are missing
        else:
            assert r["verdict"].startswith("matches"), (name, r)
            assert r["rel_mae"] < 1e-4, (name, r)
    assert terms["feet_air_time"]["timer_order"] == "timers_first"
    assert "undesired_contacts" in (tmp_path / "out" / "e16_report.txt").read_text()


def test_contract_problems_stop_the_check(tmp_path):
    tr, meta = synthetic_trace(extra_hip=False)
    bad = json.loads(str(tr["metadata"]))
    bad["reward_weights"]["energy"] = -1e-3
    bad["reward_params"]["feet_air_time"]["threshold"] = 0.4
    bad["reward_names"] = bad["reward_names"] + ["new_term"]
    probs = e16.contract_problems(bad, S2_REWARD_CONTRACT)
    assert any("energy" in x for x in probs) and any("threshold" in x for x in probs) and any("new_term" in x for x in probs)
    old = dict(tr)
    m = json.loads(str(tr["metadata"]))
    del m["soft_joint_pos_limits"]
    old["metadata"] = np.array(json.dumps(m))
    p = tmp_path / "old.npz"
    np.savez(p, **old)
    with pytest.raises(e16.ContractError, match="soft_joint_pos_limits"):
        e16.run(p, tmp_path / "o2")


def test_old_go2_configs_are_unchanged():
    """The new terms are emitted but count only when a weights dict names them; June / Start-v0 defaults stay."""
    from configs.go2_flat_cfg import Go2FlatConfig
    from envs.go2_flat import Go2FlatEnv
    w = Go2FlatConfig().environment_config.reward_term_weights
    assert not {"joint_vel", "energy", "joint_pos", "air_time_variance", "undesired_contacts"} & set(w)
    assert Go2FlatEnv.feet_air_time_threshold == 0.25 and Go2FlatEnv.timers_before_rewards is False
    assert Go2FlatEnv.soft_joint_pos_limits is None
