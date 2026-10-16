# SPDX-License-Identifier: BSD-3-Clause
"""Per-episode statistics for a real-env policy evaluation (numpy only, no Isaac Sim).

Used by scripts/reinforcement_learning/rsl_rl/eval_episode_stats.py (P1.5 fall diagnosis, doubt C7).

Why this exists (doubt C7, 15 Oct 2026). eval_go2_offline_policy.py averages extras["log"] over
every rollout step. Isaac Lab writes extras["log"] only when some env resets, and the dict stays
until the next reset. So each "Episode_Reward/*" value in that summary is a *time-weighted* mean of
the most recent reset batch, where each batch value is mean(episode_sum) / episode_length_s. Short
(fallen) episodes therefore pull "tracking" down. This module computes clean numbers directly from
command and measured velocity, and also *emulates* the old aggregation, so the C7 hypothesis can be
checked against the env's own logged values.

Conventions:
- Tracking per step r = exp(-|cmd_xy - act_xy|^2 / std^2) (Isaac Lab track_lin_vel_xy_exp, weight 1).
- The measurement taken right after a step in which an env was done is the post-reset (spawn)
  state, so it is NOT counted for either episode. The env's reward for that step used the last
  pre-reset state, which the caller never sees; this costs one step per episode.
- fall = done and not time_out.
- Episodes still running at the end are "censored" (kept, flagged, never counted as falls).
"""
from __future__ import annotations

import numpy as np

FALL_BINS = (0, 50, 100, 250, 500, 750, 1001)


class EpisodeStats:
    def __init__(self, n_envs: int, step_dt: float, episode_length_s: float, std: float = 0.5):
        self.n = int(n_envs)
        self.dt = float(step_dt)
        self.T = float(episode_length_s)
        self.std = float(std)
        self.episodes: list[dict] = []
        self._reset_acc(np.arange(self.n))
        self._start = np.zeros(self.n, dtype=np.int64)
        self._len = np.zeros(self.n, dtype=np.int64)
        self._t = 0
        # emulation of the old aggregation (stale extras["log"])
        self._last_batch_track = None   # Episode_Reward/track... of the last reset batch
        self._stale_track_sum = 0.0
        self._last_outcome = np.zeros(self.n, dtype=np.int8)  # 0 none, 1 fall, 2 time_out
        self._stale_fall_sum = 0.0
        self._stale_to_sum = 0.0
        self._stale_steps = 0
        # variant: extras["log"] cleared every step -> only reset steps are averaged
        self._batch_track: list[float] = []
        self._batch_fall_frac: list[float] = []
        self._batch_to_frac: list[float] = []
        self.all_step_r: list[float] = []  # per-step tracking, all counted env-steps
        # command changes (xy part only; it changes only when the command is resampled)
        self._prev_cmd = np.full((self.n, 2), np.nan)
        self._last_change_len = np.zeros(self.n, dtype=np.int64)
        self._last_jump = np.full(self.n, np.nan)

    def _reset_acc(self, ids):
        if not hasattr(self, "_r_sum"):
            self._r_sum = np.zeros(self.n)
            self._r_cnt = np.zeros(self.n, dtype=np.int64)
            self._e_sum = np.zeros(self.n)
            self._spd_cmd = np.zeros(self.n)
            self._spd_act = np.zeros(self.n)
            self._r_tail: list[list[float]] = [[] for _ in range(self.n)]
        self._r_sum[ids] = 0.0
        self._r_cnt[ids] = 0
        self._e_sum[ids] = 0.0
        self._spd_cmd[ids] = 0.0
        self._spd_act[ids] = 0.0
        for i in np.atleast_1d(ids):
            self._r_tail[int(i)] = []

    def step(self, cmd, act, done, time_out):
        """Call once per env.step with arrays for all envs.

        cmd, act: [n, >=2] base-frame velocity command and measurement taken AFTER env.step.
        done, time_out: [n] bool from the same env.step.
        """
        cmd = np.asarray(cmd, dtype=np.float64)[:, :2]
        act = np.asarray(act, dtype=np.float64)[:, :2]
        done = np.asarray(done, dtype=bool).reshape(-1)
        time_out = np.asarray(time_out, dtype=bool).reshape(-1) & done
        self._len += 1

        # a done env was reset inside env.step: its measurement is the spawn state -> not counted
        use = ~done
        err = np.linalg.norm(cmd - act, axis=-1)
        r = np.exp(-(err ** 2) / self.std ** 2)
        self._r_sum[use] += r[use]
        self._r_cnt[use] += 1
        self._e_sum[use] += err[use]
        self._spd_cmd[use] += np.linalg.norm(cmd, axis=-1)[use]
        self._spd_act[use] += np.linalg.norm(act, axis=-1)[use]
        for i in np.nonzero(use)[0]:
            tail = self._r_tail[i]
            tail.append(float(r[i]))
            if len(tail) > 50:
                tail.pop(0)
        self.all_step_r.extend(r[use].tolist())
        jump = np.linalg.norm(cmd - self._prev_cmd, axis=-1)
        changed = use & ~np.isnan(jump) & (jump > 1e-6)
        self._last_change_len[changed] = self._len[changed]
        self._last_jump[changed] = jump[changed]
        self._prev_cmd[use] = cmd[use]

        ids = np.nonzero(done)[0]
        if len(ids):
            batch_vals = []
            for i in ids:
                ep = self._close(i, fall=not time_out[i], censored=False)
                batch_vals.append(ep["isaac_track"])
                self._last_outcome[i] = 2 if time_out[i] else 1
            self._last_batch_track = float(np.mean(batch_vals))
            self._batch_track.append(self._last_batch_track)
            self._batch_fall_frac.append(float(np.mean(self._last_outcome == 1)))
            self._batch_to_frac.append(float(np.mean(self._last_outcome == 2)))
            self._reset_acc(ids)
            self._start[ids] = self._t + 1
            self._len[ids] = 0
            self._prev_cmd[ids] = cmd[ids]  # the new episode's first command (sampled at reset)
            self._last_change_len[ids] = 0
            self._last_jump[ids] = np.nan

        if self._last_batch_track is not None:
            self._stale_track_sum += self._last_batch_track
            self._stale_fall_sum += float(np.mean(self._last_outcome == 1))
            self._stale_to_sum += float(np.mean(self._last_outcome == 2))
            self._stale_steps += 1
        self._t += 1

    def _close(self, i, fall, censored):
        c = max(int(self._r_cnt[i]), 1)
        ep = {
            "env": int(i),
            "start": int(self._start[i]),
            "length": int(self._len[i]),
            "counted_steps": int(self._r_cnt[i]),
            "fall": bool(fall),
            "censored": bool(censored),
            "mean_track": float(self._r_sum[i] / c) if self._r_cnt[i] else float("nan"),
            # what Isaac Lab logs for this episode: sum(r * dt) / episode_length_s (weight 1)
            "isaac_track": float(self._r_sum[i] * self.dt / self.T),
            "mean_err_xy": float(self._e_sum[i] / c) if self._r_cnt[i] else float("nan"),
            "mean_speed_cmd": float(self._spd_cmd[i] / c) if self._r_cnt[i] else float("nan"),
            "mean_speed_act": float(self._spd_act[i] / c) if self._r_cnt[i] else float("nan"),
            "track_last50": float(np.mean(self._r_tail[i])) if self._r_tail[i] else float("nan"),
            # steps from the last command change (or episode start) to the end of the episode
            "steps_since_cmd_change": int(self._len[i] - self._last_change_len[i]),
            "last_cmd_jump": float(self._last_jump[i]),
            "cmd_speed_end": float(np.linalg.norm(self._prev_cmd[i])) if not np.isnan(self._prev_cmd[i]).any() else float("nan"),
        }
        self.episodes.append(ep)
        return ep

    def finish(self):
        for i in range(self.n):
            if self._len[i] > 0:
                self._close(i, fall=False, censored=True)
        self._len[:] = 0

    def summary(self) -> dict:
        eps = self.episodes
        done = [e for e in eps if not e["censored"]]
        falls = [e for e in done if e["fall"]]
        tos = [e for e in done if not e["fall"]]
        n_all = len(eps)
        counted = [e for e in eps if e["counted_steps"] > 0]
        counted_steps = sum(e["counted_steps"] for e in counted)
        speed_cmd_sum = sum(e["mean_speed_cmd"] * e["counted_steps"] for e in counted)
        speed_act_sum = sum(e["mean_speed_act"] * e["counted_steps"] for e in counted)
        out = {
            "episodes_completed": len(done),
            "episodes_censored": n_all - len(done),
            "falls": len(falls),
            "time_outs": len(tos),
            "fall_rate_completed": len(falls) / len(done) if done else float("nan"),
            "fall_rate_all": len(falls) / n_all if n_all else float("nan"),
            "track_per_step": float(np.mean(self.all_step_r)) if self.all_step_r else float("nan"),
            "counted_steps": counted_steps,
            "speed_cmd_per_step": speed_cmd_sum / counted_steps if counted_steps else float("nan"),
            "speed_act_per_step": speed_act_sum / counted_steps if counted_steps else float("nan"),
            "speed_ratio": speed_act_sum / speed_cmd_sum if speed_cmd_sum > 0 else float("nan"),
            "track_per_episode": _nanmean([e["mean_track"] for e in eps]),
            "track_per_episode_no_fall": _nanmean([e["mean_track"] for e in eps if not e["fall"]]),
            "track_per_episode_fall": _nanmean([e["mean_track"] for e in falls]),
            "track_last50_before_fall": _nanmean([e["track_last50"] for e in falls]),
            "isaac_track_per_episode": _nanmean([e["isaac_track"] for e in done]),
            "emulated_logged_track": self._stale_track_sum / self._stale_steps if self._stale_steps else float("nan"),
            "emulated_logged_base_contact": self._stale_fall_sum / self._stale_steps if self._stale_steps else float("nan"),
            "emulated_logged_time_out": self._stale_to_sum / self._stale_steps if self._stale_steps else float("nan"),
            "emulated_reset_steps_track": _nanmean(self._batch_track),
            "emulated_reset_steps_base_contact": _nanmean(self._batch_fall_frac),
            "emulated_reset_steps_time_out": _nanmean(self._batch_to_frac),
            # old method if env.reset() at step 0 already wrote a zero-valued log (C7 follow-up)
            "emulated_logged_track_from_step0": self._stale_track_sum / self._t if self._t else float("nan"),
            "emulated_logged_base_contact_from_step0": self._stale_fall_sum / self._t if self._t else float("nan"),
            "emulated_logged_time_out_from_step0": self._stale_to_sum / self._t if self._t else float("nan"),
            "falls_within_100_steps_of_cmd_change": sum(1 for e in falls if e["steps_since_cmd_change"] <= 100),
            "median_steps_since_cmd_change_at_fall": float(np.median([e["steps_since_cmd_change"] for e in falls])) if falls else float("nan"),
            "speed_cmd_fall": _nanmean([e["mean_speed_cmd"] for e in falls]),
            "speed_cmd_no_fall": _nanmean([e["mean_speed_cmd"] for e in eps if not e["fall"]]),
        }
        lens = np.array([e["length"] for e in falls], dtype=np.int64)
        hist, _ = np.histogram(lens, bins=FALL_BINS)
        out["time_to_fall_bins"] = [f"{a}-{b - 1}" for a, b in zip(FALL_BINS[:-1], FALL_BINS[1:])]
        out["time_to_fall_counts"] = hist.tolist()
        out["time_to_fall_median"] = float(np.median(lens)) if len(lens) else float("nan")
        return out


def _nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else float("nan")


def format_summary(s: dict) -> str:
    lines = ["================ EPISODE STATS ================"]
    for k, v in s.items():
        if k.startswith("time_to_fall_") and isinstance(v, list):
            continue
        lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    lines.append("  time_to_fall (steps): " + ", ".join(
        f"{b}={c}" for b, c in zip(s["time_to_fall_bins"], s["time_to_fall_counts"])))
    lines.append("===============================================")
    return "\n".join(lines)
