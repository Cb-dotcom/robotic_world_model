"""A4 D6 stale-torque sensitivity (23 Nov 2026)."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

MB = Path(__file__).resolve().parents[2]
for p in (str(MB), str(MB / "analysis"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("rsl_rl.modules")

import go2_d6_startup_check as sc  # noqa: E402
import go2_d6_torque_sensitivity as ts  # noqa: E402
from test_go2_c0_check import ID, model  # noqa: E402
from test_go2_g0 import fake_go2_wm, restore_go2_class, save  # noqa: E402,F401
from test_go2_startup_prepare import startup_trace  # noqa: E402

NORM = {k: v for k, v in ID.items()}


def test_zero_torque_changes_only_the_reset_torque():
    eps = sc.start_episodes(startup_trace(), NORM)
    z = ts.zero_torque(eps, NORM)
    for a, b in zip(eps, z):
        assert np.all(b.R[33:45] == 0.0) and np.array_equal(a.R[:33], b.R[:33])
        assert np.array_equal(a.S, b.S) and np.array_equal(a.A, b.A)
        assert not np.all(a.R[33:45] == 0.0)                       # the original is untouched


def test_identical_inputs_give_zero_difference(tmp_path, monkeypatch):
    sd = model(tmp_path)
    eps = [e for e in sc.start_episodes(startup_trace(), NORM) if e.L >= 30][:3]
    monkeypatch.setattr(ts, "zero_torque", lambda e, n: e)       # B = A
    r = ts.compare(sd, eps, NORM)
    assert r["max_stop_prob_diff_free"] == 0.0
    assert r["prediction_diff_free"]["step25"]["vxy"] == 0.0 and r["prediction_diff_free"]["step1"]["tilt"] == 0.0


def test_reading():
    def side(v, t, fs=0.0):
        return dict(fidelity_step25=dict(n=30, vxy=v, tilt=t), false_stop_free=dict(n=30, share=fs))
    res = {"s": dict(A_recorded=side(0.20, 4.0), B_zero_torque=side(0.21, 4.2))}
    assert not ts.reading(res, ["s"])["matters"]
    res = {"s": dict(A_recorded=side(0.20, 4.0), B_zero_torque=side(0.20, 4.7))}      # 0.7 deg > 0.5
    assert ts.reading(res, ["s"])["matters"]
    res = {"s": dict(A_recorded=side(0.245, 4.0), B_zero_torque=side(0.255, 4.0))}    # the vxy outcome flips
    assert ts.reading(res, ["s"])["hits"][0]["outcome_changed"]


def test_end_to_end(tmp_path):
    p913 = save(tmp_path / "p913.npz", startup_trace(seed=3))
    np.savez(tmp_path / "n.npz", **ID)
    wm = fake_go2_wm(tmp_path / "wm.pt", ID)
    rep = ts.run(wm, tmp_path / "n.npz", {"s2_913": (p913, None)}, {}, tmp_path / "out")
    assert "matters" in rep["reading"]
    txt = (tmp_path / "out" / "torque_sensitivity.txt").read_text()
    assert txt.rstrip().split("\n")[-1] in ("STALE TORQUE: MATTERS", "STALE TORQUE: NEGLIGIBLE")
    assert json.loads((tmp_path / "out" / "torque_sensitivity.json").read_text())["limits"] == ts.LIMITS
