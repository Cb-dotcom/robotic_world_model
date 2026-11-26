# SPDX-License-Identifier: BSD-3-Clause
"""T11 off-robot tools: synthetic data with known answers (no robot, no ROS install).

Rewritten 26 Nov: the 25 Nov suite (commit 3389bfb message) was never committed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.serde import SerdeError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import go2_bag  # noqa: E402
import t11_h2_actuator as h2  # noqa: E402
import t11_log_check as lc  # noqa: E402
import t11_v_check as vc  # noqa: E402

TS = go2_bag.TYPESTORE
SPORT_T = "unitree_go/msg/SportModeState"
LOW_T = "unitree_go/msg/LowState"
CMD_T = "unitree_go/msg/LowCmd"


def write_bag(path: Path, topic: str, msgtype: str, msgs: list[tuple[int, object]]) -> str:
    with Writer(path, version=8) as w:
        c = w.add_connection(topic, msgtype, typestore=TS)
        for t_ns, m in msgs:
            w.write(c, t_ns, bytes(TS.serialize_cdr(m, msgtype)))
    return str(path)


def write_bag_multi(path: Path, streams: dict[str, tuple[str, list[tuple[int, object]]]]) -> str:
    """streams {topic: (msgtype, [(t_ns, msg)])}, written in receive-time order."""
    with Writer(path, version=8) as w:
        conns = {tp: w.add_connection(tp, mt, typestore=TS) for tp, (mt, _) in streams.items()}
        rows = sorted(((t, tp, m) for tp, (_, ms) in streams.items() for t, m in ms), key=lambda r: (r[0], r[1]))
        for t_ns, tp, m in rows:
            w.write(conns[tp], t_ns, bytes(TS.serialize_cdr(m, streams[tp][0])))
    return str(path)


def h2_bag(path: Path, kp: float, kd: float, lag_s: float = 0.010, gain: float = 0.9, offset: float = 0.3) -> str:
    """50 Hz /lowcmd (cmd k at 20 k ms) and 500 Hz /lowstate at odd ms (so each 2 ms lag step is distinct).
    tau_est = gain * DCMotor(kp (q_tgt held lag earlier - q) - kd dq) + offset; torques stay below the clip."""
    rng = np.random.default_rng(0)
    n_cmd, f32 = 200, lambda x: np.asarray(x, np.float32).astype(np.float64)
    t_cmd = 0.02 * np.arange(n_cmd)
    q_tgt = f32(rng.uniform(-0.3, 0.3, (n_cmd, 12)))
    t_st = 0.001 + 0.002 * np.arange(int(0.02 * n_cmd / 0.002) - 1)
    q, dq = f32(rng.uniform(-0.1, 0.1, (len(t_st), 12))), f32(rng.uniform(-1.0, 1.0, (len(t_st), 12)))
    held = h2.held_index(t_cmd, t_st, lag_s)
    keep = held >= 0
    t_st, q, dq, held = t_st[keep], q[keep], dq[keep], held[keep]
    tau = gain * h2.dc_motor(kp * (q_tgt[held] - q) - kd * dq, dq) + offset
    cmds, states = [], []
    for k in range(n_cmd):
        m = go2_bag.default_message(CMD_T)
        for j in range(12):
            m.motor_cmd[j].q, m.motor_cmd[j].kp, m.motor_cmd[j].kd, m.motor_cmd[j].mode = q_tgt[k, j], kp, kd, 1
        cmds.append((int(round(1e9 + t_cmd[k] * 1e9)), m))
    for i in range(len(t_st)):
        m = go2_bag.default_message(LOW_T)
        m.tick = 2 * i
        for j in range(12):
            m.motor_state[j].q, m.motor_state[j].dq, m.motor_state[j].tau_est = q[i, j], dq[i, j], tau[i, j]
        states.append((int(round(1e9 + t_st[i] * 1e9)), m))
    return write_bag_multi(path, {"/lowcmd": (CMD_T, cmds), "/lowstate": (LOW_T, states)})


def sport_bag(path: Path, walk_s: float, v: float = 0.5, still_s: float = 1.0, hz: float = 50.0) -> str:
    """Still, straight walk along x at v m/s (body = world, yaw 0), still."""
    t = np.arange(0.0, still_s + walk_s + still_s, 1.0 / hz)
    vel = np.where((t >= still_s) & (t < still_s + walk_s), v, 0.0)
    x = np.r_[0.0, np.cumsum(vel[:-1]) / hz]
    msgs = []
    for ti, vi, xi in zip(t, vel, x):
        m = go2_bag.default_message(SPORT_T)
        m.velocity[:] = (vi, 0.0, 0.0)
        m.position[:] = (xi, 0.0, 0.0)
        msgs.append((int(1e9 + ti * 1e9), m))
    return write_bag(path, vc.SPORT, SPORT_T, msgs)


# --- layout and maps --------------------------------------------------------------
def test_sportmodestate_payload_is_512_bytes_like_h0():
    raw = TS.serialize_cdr(go2_bag.default_message(SPORT_T), SPORT_T)
    assert len(raw) - 4 == 512  # 4-byte CDR header + payload


def test_joint_maps_inverse_and_named():
    assert (go2_bag.SIM_FROM_SDK[go2_bag.SDK_FROM_SIM] == np.arange(12)).all()
    assert (go2_bag.SDK_FROM_SIM[go2_bag.SIM_FROM_SDK] == np.arange(12)).all()
    assert [go2_bag.SDK_JOINT_NAMES[i] for i in go2_bag.SIM_FROM_SDK] == go2_bag.SIM_JOINT_NAMES


def test_wrong_layout_is_refused():
    raw = bytes(TS.serialize_cdr(go2_bag.default_message(SPORT_T), SPORT_T))
    go2_bag.check_exact_decode(raw, SPORT_T)
    with pytest.raises((ValueError, SerdeError)):  # rosbags or our size check stops it
        go2_bag.check_exact_decode(raw + bytes(8), SPORT_T)


def test_dc_motor_matches_isaaclab_formula():
    tau = np.array([50.0, -50.0, 50.0, -50.0, 5.0])
    dq = np.array([0.0, 0.0, 15.0, 15.0, 15.0])
    # v = 0: +-23.5; v = 15 (half of 30 rad/s): top 23.5 * 0.5 = 11.75, bottom -23.5
    assert np.allclose(h2.dc_motor(tau, dq), [23.5, -23.5, 11.75, -23.5, 5.0])


# --- logger: 50 Hz emulation -----------------------------------------------------
def test_policy_emulation_clean_stream():
    p = lc.emulate_policy_steps(np.arange(0, 10.0, 0.002), np.arange(5000) * 2)
    assert p["flagged_steps_gt_30ms"] == 0 and p["stale_reused_steps"] == 0
    assert p["usable_fraction_of_steps"] == 1.0


def test_stale_reused_step_is_excluded_from_windows():
    # 50 Hz stream 0.5 ms before each grid point: step k reads message k, 20 ms apart, no flag.
    t = np.r_[0.0, 0.02 * np.arange(1, 301) - 0.0005]
    tick = 2 * np.arange(301)
    tick[100] = tick[99]  # message 100 repeats sample 99 (same tick): step 100 is stale
    p = lc.emulate_policy_steps(t, tick)
    assert p["steps"] == 299 and p["flagged_steps_gt_30ms"] == 0
    assert p["stale_reused_steps"] == 1 and p["repeated_state_steps"] == 0
    # runs 99 + 199 steps (the stale step breaks it); without the fix 1 run of 299 = 5 windows
    assert p["usable_windows_nonoverlap"] == 1 + 3
    assert p["usable_fraction_of_steps"] == pytest.approx(298 / 299)
    assert lc.emulate_policy_steps(t)["usable_windows_nonoverlap"] == 5  # no tick: the repeat is unseen
    # no new message before a grid point: the step reuses the same sample (same timestamp)
    t2 = np.r_[0.0, 0.02 * np.arange(1, 301) - 0.0005]
    t2[150] = 0.02 * 150 + 0.005
    p2 = lc.emulate_policy_steps(t2, 2 * np.arange(301))
    assert p2["stale_reused_steps"] == 1 and p2["repeated_state_steps"] == 1


def test_log_check_reads_tick_and_counts_stale(tmp_path):
    msgs = []
    for i in range(1000):  # 500 Hz, 2 s; 20 messages (40 ms) repeat one tick
        m = go2_bag.default_message(LOW_T)
        m.tick = 2 * i if not 500 <= i < 520 else 1000
        msgs.append((int(1e9 + i * 2e6), m))
    res = lc.analyse(write_bag(tmp_path / "low", "/lowstate", LOW_T, msgs), 10)
    p = res["policy_50hz"]
    assert p["flagged_steps_gt_30ms"] == 0
    assert p["stale_reused_steps"] == 1  # two grid points read tick 1000; the second is stale
    assert "stale steps" in lc.render(res)


def test_log_check_counts_lost_ticks_and_late_burst(tmp_path):
    msgs, t_ms = [], 0.0
    for i in range(1000):  # 500 Hz, tick step 2
        if i == 600:
            t_ms += 30.0  # message 600 arrives 30 ms late, 601..614 follow in a burst; ticks continuous
        t_ms += 0.01 if 601 <= i < 615 else 2.0
        if 300 <= i < 319:
            continue  # 19 messages lost: tick jumps 40, receive gap 40 ms
        m = go2_bag.default_message(LOW_T)
        m.tick = 2 * i
        msgs.append((int(1e9 + t_ms * 1e6), m))
    res = lc.analyse(write_bag(tmp_path / "gaps", "/lowstate", LOW_T, msgs), 10)
    k = res["lowstate_tick"]
    assert k["tick_step_mode"] == 2 and k["messages_lost_by_tick"] == 19
    assert k["rx_gaps_gt_20ms_with_continuous_tick"] == 1  # late, not lost
    assert res["topics"]["/lowstate"]["gaps_gt_20ms"] == 2 and res["topics"]["/lowstate"]["gaps_gt_30ms"] == 2
    assert res["policy_50hz"]["flagged_steps_gt_30ms"] >= 1


# --- H2 actuator replay ---------------------------------------------------------------
def test_h2_recovers_lag_gain_offset(tmp_path):
    bag = h2_bag(tmp_path / "h2", kp=25.0, kd=0.5)
    o = tmp_path / "o"
    assert h2.main([bag, "--out", str(o), "--policy-sha256", "abc123"]) == 0
    out = json.loads((o / "h2_actuator.json").read_text())
    assert out["counts"] is True and out["gain_share"] == 1.0
    for name, r in out["joints"].items():
        assert r["lag_ms"] == pytest.approx(10.0), name
        assert r["gain"] == pytest.approx(0.9, abs=1e-3) and r["offset_nm"] == pytest.approx(0.3, abs=1e-3), name
        assert r["rmse_fit_nm"] < 1e-3 and r["dc_clip_rows"] == 0, name


def test_h2_refuses_unitree_gains(tmp_path):
    bag = h2_bag(tmp_path / "h2", kp=40.0, kd=1.0)
    o = tmp_path / "o"
    assert h2.main([bag, "--out", str(o), "--policy-sha256", "abc123"]) == 2  # stop: not action data
    assert "NOT ACTION DATA" in (o / "h2_actuator.txt").read_text()
    assert not (o / "h2_actuator.json").exists()
    o2 = tmp_path / "o2"
    assert h2.main([bag, "--out", str(o2), "--policy-sha256", "abc123", "--allow-other-gains"]) == 0
    assert json.loads((o2 / "h2_actuator.json").read_text())["counts"] is False
    assert "DOES NOT COUNT" in (o2 / "h2_actuator.txt").read_text()
    # our gains but no policy hash: runs, does not count
    o3 = tmp_path / "o3"
    assert h2.main([h2_bag(tmp_path / "h2b", kp=25.0, kd=0.5), "--out", str(o3)]) == 0
    assert json.loads((o3 / "h2_actuator.json").read_text())["counts"] is False


# --- velocity check ----------------------------------------------------------------
def test_v_check_recovers_tape_and_passes_with_bar(tmp_path):
    bag = sport_bag(tmp_path / "w", walk_s=4.0)  # 2.0 m
    rc = vc.main(["--run", bag, "0.50", "2.52", "--out", str(tmp_path / "o"), "--max-rel-err", "0.05"])
    out = json.loads((tmp_path / "o" / "v_check.json").read_text())
    assert out["runs"][0]["sources"]["sport_pos"]["ratio"] == pytest.approx(2.0 / 2.02, abs=0.01)
    assert out["summary"]["sport_pos"]["verdict"] == "PASS" and rc == 0


def test_v_check_missing_short_or_nan_run_is_never_pass(tmp_path):
    good = sport_bag(tmp_path / "w", walk_s=4.0)
    short = sport_bag(tmp_path / "s", walk_s=1.0)  # < --min-walk-s 1.5
    o = tmp_path / "o"
    rc = vc.main(["--run", good, "0.5", "2.5", "--run", short, "0.5", "1.0", "--run", str(tmp_path / "gone"),
                  "0.5", "2.5", "--out", str(o), "--max-rel-err", "0.05"])
    out = json.loads((o / "v_check.json").read_text())
    assert rc == 1
    assert "shorter than --min-walk-s" in out["runs"][1]["error"] and "error" in out["runs"][2]
    assert out["summary"]["sport_pos"] == {**out["summary"]["sport_pos"], "n": 1, "n_expected": 3,
                                           "verdict": "INVALID"}
    assert "2 of 3 runs not analysed" in (o / "v_check.txt").read_text()
    # NaN ratio or a source absent in one run: INVALID even though the finite ratios are perfect
    ok = {"sources": {"a": {"ratio": 1.0}, "b": {"ratio": 1.0}}}
    nan = {"sources": {"a": {"ratio": float("nan")}}}
    summ, _ = vc.summarize([ok, nan], 0.05)
    assert summ["a"]["verdict"] == "INVALID" and summ["b"]["verdict"] == "INVALID"
    assert vc.summarize([ok, ok], 0.05)[0]["a"]["verdict"] == "PASS"
    assert vc.summarize([ok, nan], None)[0]["a"].get("verdict") is None  # no bar: still no verdict
