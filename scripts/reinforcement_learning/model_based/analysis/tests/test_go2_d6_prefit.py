"""A4 D6 pre-fit verifications (22 Nov 2026) on synthetic prepared data."""
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

import go2_d6_prefit as pf  # noqa: E402
import go2_startup_prepare as sp  # noqa: E402
from test_go2_c0_check import ID  # noqa: E402
from test_go2_g0 import fake_go2_wm, restore_go2_class, save  # noqa: E402,F401
from test_go2_startup_prepare import startup_trace  # noqa: E402


@pytest.fixture
def prepared(tmp_path):
    r0 = save(tmp_path / "r0.npz", startup_trace())
    sh = save(tmp_path / "sh.npz", startup_trace(seed=7))
    sp.prepare_startup(r0, sh, tmp_path / "d6", frac=0.1, seed=101)
    np.savez(tmp_path / "n.npz", **ID)
    wm = fake_go2_wm(tmp_path / "wm.pt", ID)
    return tmp_path, r0, sh, wm


def test_all_checks_pass_on_prepared_data(prepared):
    tmp, r0, sh, wm = prepared
    res = pf.run(tmp / "d6", r0, sh, wm, tmp / "n.npz", tmp / "pre")
    assert res["pass"], json.dumps({k: v for k, v in res.items() if isinstance(v, dict)}, default=str)[:2000]
    a, b = res["a_first_transition"], res["b_terminal_targets"]
    assert a["history1_windows"] == a["startup_segments"] > 0
    assert b["in_startup_windows"] >= 1 and b["without_target_count"] == 0
    assert (tmp / "pre" / "prefit.txt").read_text().rstrip().endswith("PRE-FIT: PASS")


def test_a_corrupted_reset_row_fails(prepared):
    tmp, r0, sh, wm = prepared
    name = json.loads((tmp / "d6" / "startup.json").read_text())["segments"][0]
    f = tmp / "d6" / "segments" / f"{name}.csv"
    x = np.loadtxt(f, delimiter=",", ndmin=2)
    x[0, 3] += 1.0
    np.savetxt(f, x, delimiter=",", fmt="%.9g")
    res = pf.run(tmp / "d6", r0, sh, wm, tmp / "n.npz", tmp / "pre")
    assert not res["pass"] and res["a_first_transition"]["mismatched_count"] == 1


def test_b_short_fall_without_startup_windows_fails(prepared):
    tmp, r0, sh, wm = prepared
    j = tmp / "d6" / "startup.json"
    d = json.loads(j.read_text())
    short = [n for n in d["segments"] if n.endswith("env005_r00000")]
    assert short
    d["segments"] = [n for n in d["segments"] if n not in short]
    j.write_text(json.dumps(d))
    res = pf.run(tmp / "d6", r0, sh, wm, tmp / "n.npz", tmp / "pre")
    assert not res["b_terminal_targets"]["ok"] and res["b_terminal_targets"]["without_target_count"] >= 1


def test_d_stale_reset_row_fails(tmp_path):
    tr = startup_trace()
    # make the reset row after env 1's first time-out (row 99) equal to its terminal state: a stale observation
    tr["state"][100, 1] = tr["next_state"][99, 1]
    res = pf.check_reset_rows({"r0": tr})
    assert not res["ok"] and res["r0"]["equal_to_previous_terminal"] == 1


def test_wrong_recording_is_refused(prepared):
    tmp, r0, sh, wm = prepared
    with pytest.raises(ValueError, match="not the recording"):
        pf.run(tmp / "d6", sh, r0, wm, tmp / "n.npz", tmp / "pre")
