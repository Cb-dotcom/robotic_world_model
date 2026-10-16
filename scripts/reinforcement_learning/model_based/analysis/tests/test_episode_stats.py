"""CPU tests for analysis/episode_stats.py (P1.5 fall diagnosis, doubt C7)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # analysis/

from episode_stats import EpisodeStats, format_summary  # noqa: E402

DT, T = 0.02, 20.0  # ANYmal-D: 1000 steps per episode


def _run(n, steps, cmd_fn, act_fn, done_fn, to_fn):
    st = EpisodeStats(n, DT, T, std=0.5)
    for t in range(steps):
        st.step(cmd_fn(t), act_fn(t), done_fn(t), to_fn(t))
    st.finish()
    return st


def test_perfect_tracking_time_out_only():
    n = 3
    st = _run(n, 2000,
              lambda t: np.tile([0.5, 0.0], (n, 1)), lambda t: np.tile([0.5, 0.0], (n, 1)),
              lambda t: np.full(n, (t + 1) % 1000 == 0), lambda t: np.full(n, (t + 1) % 1000 == 0))
    s = st.summary()
    assert s["episodes_completed"] == 6 and s["falls"] == 0 and s["episodes_censored"] == 0
    assert s["track_per_step"] == pytest.approx(1.0)
    assert s["track_per_episode"] == pytest.approx(1.0)
    # Isaac-style per-episode value = 999 counted steps * dt / 20 s (done step not seen)
    assert s["isaac_track_per_episode"] == pytest.approx(999 * DT / T)
    assert all(e["length"] == 1000 for e in st.episodes)


def test_falls_counted_per_episode_and_time_to_fall():
    n = 2
    # env 0 falls at steps 50 and 150 (lengths 50, 100), then runs to the end (censored);
    # env 1 never falls and is censored at 300 steps
    fall_at = {49, 149}
    done = lambda t: np.array([t in fall_at, False])
    st = _run(n, 300, lambda t: np.zeros((n, 2)), lambda t: np.zeros((n, 2)), done, lambda t: np.zeros(n, bool))
    s = st.summary()
    assert s["falls"] == 2 and s["time_outs"] == 0 and s["episodes_censored"] == 2
    assert s["fall_rate_completed"] == pytest.approx(1.0)
    assert s["fall_rate_all"] == pytest.approx(0.5)
    lens = sorted(e["length"] for e in st.episodes if e["fall"])
    assert lens == [50, 100]
    assert s["time_to_fall_counts"][:3] == [0, 1, 1]  # 50 -> bin 50-99, 100 -> bin 100-249
    assert s["time_to_fall_bins"][0] == "0-49"


def test_done_step_measurement_not_counted():
    # the measurement after a done step is the spawn state: a huge error there must not count
    n = 1
    act = lambda t: np.array([[5.0, 0.0]]) if t == 9 else np.array([[0.5, 0.0]])
    st = _run(n, 20, lambda t: np.array([[0.5, 0.0]]), act,
              lambda t: np.array([t == 9]), lambda t: np.array([False]))
    assert st.summary()["track_per_step"] == pytest.approx(1.0)


def test_speed_aggregation_uses_counted_steps_and_excludes_reset_measurements():
    # Different episode lengths and speeds expose episode-mean / length-weighting errors.
    # The first episode closes at step 1 (one counted step); the second closes at
    # step 4 (two counted steps); the last is censored (one counted step).
    commands = [1.0, 1000.0, 2.0, 4.0, 2000.0, 3.0]
    speeds = [0.5, 9000.0, 1.0, 3.0, 8000.0, 1.5]
    st = _run(1, len(commands), lambda t: np.array([[commands[t], 0.0]]),
              lambda t: np.array([[speeds[t], 0.0]]),
              lambda t: np.array([t in (1, 4)]), lambda t: np.array([t == 4]))
    assert [e["length"] for e in st.episodes] == [2, 3, 1]
    assert [e["counted_steps"] for e in st.episodes] == [1, 2, 1]
    s = st.summary()
    assert s["counted_steps"] == 4
    assert s["speed_cmd_per_step"] == pytest.approx(10.0 / 4)
    assert s["speed_act_per_step"] == pytest.approx(6.0 / 4)
    assert s["speed_ratio"] == pytest.approx(6.0 / 10)


def test_speed_aggregation_skips_episodes_without_measurements():
    st = _run(1, 2, lambda t: np.array([[1.0, 0.0]]), lambda t: np.array([[0.5, 0.0]]),
              lambda t: np.array([t == 0]), lambda t: np.array([False]))
    assert [e["counted_steps"] for e in st.episodes] == [0, 1]
    assert np.isnan(st.episodes[0]["mean_speed_cmd"])
    assert st.summary()["speed_ratio"] == pytest.approx(0.5)


@pytest.mark.parametrize("case", ["empty", "all_done", "zero_command"])
def test_speed_aggregation_without_valid_denominator(case):
    st = _run(1, 0 if case == "empty" else 2,
              lambda t: np.array([[0.0 if case == "zero_command" else 1.0, 0.0]]),
              lambda t: np.array([[0.5, 0.0]]),
              lambda t: np.array([case == "all_done"]), lambda t: np.array([False]))
    s = st.summary()
    assert np.isnan(s["speed_ratio"])
    if case == "zero_command":
        assert s["speed_cmd_per_step"] == 0.0
        assert s["speed_act_per_step"] == 0.5
    else:
        assert s["counted_steps"] == 0
        assert np.isnan(s["speed_cmd_per_step"])
        assert np.isnan(s["speed_act_per_step"])


def test_tracking_formula():
    n = 1
    st = _run(n, 10, lambda t: np.array([[1.0, 0.0]]), lambda t: np.array([[0.5, 0.0]]),
              lambda t: np.array([False]), lambda t: np.array([False]))
    assert st.summary()["track_per_step"] == pytest.approx(np.exp(-0.25 / 0.25))
    assert st.episodes[0]["mean_err_xy"] == pytest.approx(0.5)


def test_emulation_stale_log_pulls_tracking_down():
    """C7 mechanism: an early short fall batch stays in extras['log'] for a long time.

    4 envs, perfect tracking. Env 0 falls at step 100. All others time out at 1000.
    Batch values: fall batch = 99*dt/20 = 0.099, time-out batch = 999*dt/20 = 0.999.
    Stale average from step 100 to 1000 (901 values of 0.099, then 1 value of the 0.999 batch
    at step 999... and env 0's own time-out batch etc.). The emulated value must be far below the
    per-step tracking (1.0), and the reset-steps-only variant must be the plain mean of batches.
    """
    n = 4
    def done(t):
        d = np.zeros(n, bool)
        if t == 99:
            d[0] = True
        if t == 999:
            d[1:] = True
        return d
    def to(t):
        d = np.zeros(n, bool)
        if t == 999:
            d[1:] = True
        return d
    st = _run(n, 1000, lambda t: np.zeros((n, 2)), lambda t: np.zeros((n, 2)), done, to)
    s = st.summary()
    assert s["track_per_step"] == pytest.approx(1.0)
    b1, b2 = 99 * DT / T, 999 * DT / T
    expect_stale = (b1 * 900 + b2 * 1) / 901  # steps 99..998 hold b1 (900 steps), step 999 holds b2
    assert s["emulated_logged_track"] == pytest.approx(expect_stale)
    assert s["emulated_reset_steps_track"] == pytest.approx((b1 + b2) / 2)
    # termination emulation: fraction of ALL envs whose last episode ended by that term
    expect_bc = (0.25 * 900 + 0.25 * 1) / 901
    expect_to = (0.0 * 900 + 0.75 * 1) / 901
    assert s["emulated_logged_base_contact"] == pytest.approx(expect_bc)
    assert s["emulated_logged_time_out"] == pytest.approx(expect_to)


def test_censored_never_a_fall_and_format():
    n = 2
    st = _run(n, 30, lambda t: np.zeros((n, 2)), lambda t: np.zeros((n, 2)),
              lambda t: np.zeros(n, bool), lambda t: np.zeros(n, bool))
    s = st.summary()
    assert s["falls"] == 0 and s["episodes_censored"] == 2
    assert np.isnan(s["fall_rate_completed"])
    txt = format_summary(s)
    assert "EPISODE STATS" in txt and "time_to_fall" in txt


def test_command_change_tracking():
    n = 1
    # command 0.5 until step 99, then 1.0; fall at step 149 -> 49 steps after the change step (t=100)
    cmd = lambda t: np.array([[0.5 if t < 100 else 1.0, 0.0]])
    st = _run(n, 300, cmd, cmd, lambda t: np.array([t == 149]), lambda t: np.array([False]))
    ep = [e for e in st.episodes if e["fall"]][0]
    assert ep["length"] == 150
    assert ep["steps_since_cmd_change"] == 49
    assert ep["last_cmd_jump"] == pytest.approx(0.5)
    assert ep["cmd_speed_end"] == pytest.approx(1.0)
    s = st.summary()
    assert s["falls_within_100_steps_of_cmd_change"] == 1
    # the next episode starts with the post-reset command: no false "change" at its step 1
    nxt = [e for e in st.episodes if e["censored"]][0]
    assert nxt["steps_since_cmd_change"] == nxt["length"]


def test_zero_log_from_step0_variant():
    n = 2
    st = _run(n, 100, lambda t: np.zeros((n, 2)), lambda t: np.zeros((n, 2)),
              lambda t: np.array([t == 19, False]), lambda t: np.zeros(n, bool))
    s = st.summary()
    # stale values start at step 19 (81 steps); the step-0 variant divides by all 100 steps
    assert s["emulated_logged_base_contact"] == pytest.approx(0.5)
    assert s["emulated_logged_base_contact_from_step0"] == pytest.approx(0.5 * 81 / 100)
