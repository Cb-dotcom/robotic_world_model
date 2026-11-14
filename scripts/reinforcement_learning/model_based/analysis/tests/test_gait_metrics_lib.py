"""CPU tests for rsl_rl/gait_metrics_lib.py (synthetic trot)."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "rsl_rl"))
import gait_metrics_lib as G  # noqa: E402


def synthetic(T=400, N=4, dt=0.02, clearance=0.08, period=20):
    t = np.arange(T)
    phase = (t % period) < period // 2                  # True: pair A (FL, RR) in stance
    contact = np.zeros((T, N, 4), bool)
    contact[:, :, 0] = contact[:, :, 3] = phase[:, None]
    contact[:, :, 1] = contact[:, :, 2] = ~phase[:, None]
    foot_z = np.full((T, N, 4), 0.02)
    for f in range(4):
        sw = ~contact[:, :, f]
        # half-sine bump during each swing, peak = clearance above stance height
        k = np.zeros((T, N))
        for n in range(N):
            run = 0
            for i in range(T):
                run = run + 1 if sw[i, n] else 0
                k[i, n] = run
        foot_z[:, :, f] += np.where(sw, clearance * np.sin(np.pi * k / (period // 2 + 1)), 0.0)
    a = dict(contact=contact, foot_z=foot_z, foot_vxy=np.where(contact, 0.01, 0.5),
             pitch=np.full((T, N), np.radians(3.0)) + np.radians(1.0) * np.sin(t)[:, None],
             roll=np.zeros((T, N)), height=np.full((T, N), 0.30), vx=np.full((T, N), 0.45),
             vy=np.zeros((T, N)), wz=np.zeros((T, N)), tau_abs=np.ones((T, N)), power=np.ones((T, N)),
             done=np.zeros((T, N), bool), time_out=np.zeros((T, N), bool))
    return a


def test_trot_metrics():
    a = synthetic()
    s = G.summarize(a, np.full(4, 0.45), 0.02, warmup=40)[0.45]
    assert 7.0 < s["foot_clearance_cm_mean"] <= 8.0
    assert abs(s["duty_factor"] - 0.5) < 1e-6
    assert s["trot_index"] == 1.0
    assert abs(s["step_freq_hz_per_foot"] - 1 / (20 * 0.02)) < 0.15
    assert abs(s["foot_slip_mps_in_contact"] - 0.01) < 1e-9
    assert abs(s["pitch_mean_deg"] - 3.0) < 0.2 and 0.5 < s["pitch_std_deg"] < 1.0
    assert s["falls"] == 0 and abs(s["speed_ratio"] - 1.0) < 1e-9


def test_reset_breaks_swing_and_counts_falls():
    a = synthetic()
    a["done"][200, 1] = True
    a["time_out"][300, 2] = True
    a["done"][300, 2] = True
    s = G.summarize(a, np.full(4, 0.45), 0.02, warmup=40)[0.45]
    assert s["falls"] == 1
    mask = G.valid_mask(a["done"], 40)
    assert not mask[201:241, 1].any() and mask[241, 1]


def test_speed_groups_and_table():
    a = synthetic(N=4)
    a["vx"][:, 2:] = 0.9
    out = G.summarize(a, np.array([0.45, 0.45, 0.9, 0.9]), 0.02, warmup=40)
    assert set(out) == {0.45, 0.9} and abs(out[0.9]["forward_speed_mps"] - 0.9) < 1e-9
    txt = G.table({"x": out, "y": out})
    assert "== command 0.90 m/s" in txt and "foot_clearance_cm_mean" in txt


def test_plan_fall_definition_and_reward_terms():
    a = synthetic()
    a["height"][100:150, 0] = 0.12          # body down for 50 steps in env 0
    a["roll"][300:320, 1] = np.radians(70)  # tipped in env 1
    a["rew"] = np.stack([np.ones((400, 4)), -np.ones((400, 4)) * 0.5], axis=-1)
    s = G.summarize(a, np.full(4, 0.45), 0.02, warmup=40, reward_names=["track", "slide"])[0.45]
    assert s["plan_fall_events"] == 2
    assert abs(s["plan_fall_time_fraction"] - 70 / (4 * 360)) < 1e-9
    assert s["reward_terms_mean"] == {"track": 1.0, "slide": -0.5}
    assert "r:slide" in G.table({"x": {0.45: s}})
