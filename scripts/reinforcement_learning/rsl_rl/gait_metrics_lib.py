"""Gait metrics from recorded simulator arrays (pure numpy, CPU-testable). Used by gait_metrics.py.

Arrays (T steps x N envs): contact[T,N,4] bool, foot_z[T,N,4] m, foot_vxy[T,N,4] m/s, pitch/roll[T,N] rad,
height[T,N] m, vx/vy/wz[T,N] (base frame), tau_abs[T,N] Nm (mean over joints), power[T,N] W (sum |tau*qd|),
done[T,N] bool, time_out[T,N] bool. Feet are ordered FL, FR, RL, RR. Each env has one commanded forward
speed (speeds[N]). Rows before `warmup` steps and rows of an episode's first `warmup` steps are not scored.
"""
from __future__ import annotations

import numpy as np

FEET = ("FL", "FR", "RL", "RR")
PLAN_FALL_HEIGHT_M = 0.15   # plan v6 §5: real fall = base height < 0.15 m or tilt > 60 deg
PLAN_FALL_TILT_DEG = 60.0


def plan_fall_state(height, roll, pitch):
    """Boolean [T,N]: base below 0.15 m or body tilt beyond 60 degrees (plan v6 fall definition).
    Tilt = angle between body z and world up = arccos(cos(roll) cos(pitch)), as mdp.plan_fall."""
    tilt = np.degrees(np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1.0, 1.0)))
    return (height < PLAN_FALL_HEIGHT_M) | (tilt > PLAN_FALL_TILT_DEG)


def plan_fall_events(state, mask):
    """Number of entries into the fall state inside valid rows (one per env until it leaves the state)."""
    prev = np.vstack([np.zeros((1, state.shape[1]), bool), state[:-1]])
    return int((state & ~prev & mask).sum())


def valid_mask(done, warmup):
    """True where the env has run >= warmup steps since its last reset (and since t = 0)."""
    T, N = done.shape
    since = np.zeros(N, dtype=int)
    mask = np.zeros((T, N), dtype=bool)
    for t in range(T):
        mask[t] = since >= warmup
        since = np.where(done[t], 0, since + 1)
    return mask


def swing_clearances(contact, foot_z, mask, done):
    """Per swing: max foot height in swing minus the foot height at the next touchdown.

    Returns a list of (env, foot, clearance_m, swing_steps). A swing counts only if it lies wholly
    inside valid rows and no reset happens during it.
    """
    T, N, F = contact.shape
    out = []
    for n in range(N):
        for f in range(F):
            in_swing, peak, length, ok = False, -np.inf, 0, False
            for t in range(T):
                if done[t, n]:
                    in_swing, peak, length, ok = False, -np.inf, 0, False
                    continue
                c = contact[t, n, f]
                if not in_swing and not c and t > 0 and contact[t - 1, n, f]:
                    in_swing, peak, length, ok = True, foot_z[t, n, f], 1, bool(mask[t, n])
                elif in_swing and not c:
                    peak, length = max(peak, foot_z[t, n, f]), length + 1
                    ok = ok and bool(mask[t, n])
                elif in_swing and c:
                    if ok and mask[t, n]:
                        out.append((n, f, float(peak - foot_z[t, n, f]), length))
                    in_swing = False
    return out


def summarize(a, speeds, dt, warmup, reward_names=None):
    """Summary per commanded speed: a dict speed -> metrics. Optional a["rew"] [T,N,K] = weighted
    reward terms per step (names in reward_names) -> mean per term."""
    mask = valid_mask(a["done"], warmup)
    fall_state = plan_fall_state(a["height"], a["roll"], a["pitch"])
    swings = swing_clearances(a["contact"], a["foot_z"], mask, a["done"])
    result = {}
    for v in sorted(set(float(s) for s in speeds)):
        envs = np.flatnonzero(np.isclose(speeds, v))
        m = mask[:, envs]
        if not m.any():
            continue
        sel = lambda x: x[:, envs][m]  # noqa: E731
        c = a["contact"][:, envs][m]                     # [K,4]
        vxy = a["foot_vxy"][:, envs][m]
        clear = [s for s in swings if s[0] in set(envs.tolist())]
        clear_m = np.array([s[2] for s in clear]) if clear else np.array([np.nan])
        swing_s = np.array([s[3] for s in clear]) * dt if clear else np.array([np.nan])
        pitch, roll = np.degrees(sel(a["pitch"])), np.degrees(sel(a["roll"]))
        valid_time = m.sum() * dt                        # env-seconds
        fl, fr, rl, rr = (c[:, i] for i in range(4))
        falls = int((a["done"][:, envs] & ~a["time_out"][:, envs]).sum())
        fs = fall_state[:, envs]
        result[v] = {
            "env_seconds": float(valid_time),
            "plan_fall_events": plan_fall_events(fs, m),
            "plan_fall_time_fraction": float(fs[m].mean()),
            "low_body_fraction_lt_0p25m": float((sel(a["height"]) < 0.25).mean()),
            "falls": falls,
            "falls_per_1000s": float(1000.0 * falls / max(valid_time, 1e-9)),
            "forward_speed_mps": float(sel(a["vx"]).mean()),
            "speed_ratio": float(sel(a["vx"]).mean() / v) if v > 0 else float("nan"),
            "lateral_speed_rms_mps": float(np.sqrt((sel(a["vy"]) ** 2).mean())),
            "yaw_rate_rms_dps": float(np.degrees(np.sqrt((sel(a["wz"]) ** 2).mean()))),
            "foot_clearance_cm_mean": float(100 * np.nanmean(clear_m)),
            "foot_clearance_cm_p10_p90": [float(100 * np.nanpercentile(clear_m, 10)), float(100 * np.nanpercentile(clear_m, 90))],
            "swing_time_s": float(np.nanmean(swing_s)),
            "swings_counted": len(clear),
            "duty_factor": float(c.mean()),
            "step_freq_hz_per_foot": float(len(clear) / 4 / max(valid_time, 1e-9)),
            "foot_slip_mps_in_contact": float(vxy[c].mean()) if c.any() else float("nan"),
            "trot_index": float(((fl == rr) & (fr == rl)).mean()),
            "pitch_mean_deg": float(pitch.mean()),
            "pitch_std_deg": float(pitch.std()),
            "roll_std_deg": float(roll.std()),
            "base_height_mean_m": float(sel(a["height"]).mean()),
            "base_height_std_cm": float(100 * sel(a["height"]).std()),
            "torque_abs_mean_nm": float(sel(a["tau_abs"]).mean()),
            "power_w": float(sel(a["power"]).mean()),
        }
        if reward_names is not None and "rew" in a:
            rew = a["rew"][:, envs][m]                   # [K_valid, n_terms]
            result[v]["reward_terms_mean"] = {name: float(rew[:, i].mean()) for i, name in enumerate(reward_names)}
    return result


ROWS = [
    ("forward_speed_mps", "{:.3f}"), ("speed_ratio", "{:.3f}"), ("foot_clearance_cm_mean", "{:.2f}"),
    ("swing_time_s", "{:.3f}"), ("duty_factor", "{:.3f}"), ("step_freq_hz_per_foot", "{:.2f}"),
    ("foot_slip_mps_in_contact", "{:.4f}"), ("trot_index", "{:.3f}"), ("pitch_mean_deg", "{:.2f}"),
    ("pitch_std_deg", "{:.2f}"), ("roll_std_deg", "{:.2f}"), ("base_height_mean_m", "{:.3f}"),
    ("base_height_std_cm", "{:.2f}"), ("lateral_speed_rms_mps", "{:.3f}"), ("yaw_rate_rms_dps", "{:.2f}"),
    ("torque_abs_mean_nm", "{:.2f}"), ("power_w", "{:.1f}"), ("falls", "{:d}"),
    ("plan_fall_events", "{:d}"), ("plan_fall_time_fraction", "{:.3f}"), ("low_body_fraction_lt_0p25m", "{:.3f}"),
    ("env_seconds", "{:.0f}"),
]


def table(named):
    """named: {label: summarize() output}. Returns a text table, one block per speed."""
    labels = list(named)
    speeds = sorted({v for s in named.values() for v in s})
    lines = []
    for v in speeds:
        lines.append(f"== command {v:.2f} m/s")
        lines.append(f"{'metric':28s}" + "".join(f"{lab:>22s}" for lab in labels))
        for key, fmt in ROWS:
            cells = []
            for lab in labels:
                val = named[lab].get(v, {}).get(key)
                cells.append(f"{(fmt.format(val) if val is not None else '-'):>22s}")
            lines.append(f"{key:28s}" + "".join(cells))
        terms = sorted({t for lab in labels for t in named[lab].get(v, {}).get("reward_terms_mean", {})})
        for t in terms:
            cells = [named[lab].get(v, {}).get("reward_terms_mean", {}).get(t) for lab in labels]
            lines.append(f"{'r:' + t:28s}" + "".join(f"{('{:.4f}'.format(c) if c is not None else '-'):>22s}" for c in cells))
    return "\n".join(lines)
