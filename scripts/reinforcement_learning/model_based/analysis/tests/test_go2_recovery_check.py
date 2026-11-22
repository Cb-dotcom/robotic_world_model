"""A4 s2 recovery check (22 Nov 2026): event outcomes and the point-of-no-return reading."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import go2_recovery_check as rc  # noqa: E402


def g_of(deg):
    r = np.radians(deg)
    return np.array([np.sin(r), 0.0, -np.cos(r)])


def trace(T=700, N=4):
    ns = np.zeros((T, N, 45)); ns[..., 6:9] = g_of(0)
    cmd = np.zeros((T, N, 3)); done = np.zeros((T, N), bool); term = done.copy(); tout = done.copy()
    ns[100:110, 0, 6:9] = g_of(45); ns[580:583, 0, 6:9] = g_of(70)    # env 0: recovers at 100; second event falls at 600
    done[600, 0] = term[600, 0] = True
    ns[100:200, 1, 6:9] = g_of(35)                                      # env 1: stays tilted 100 rows -> not recovered
    ns[100:105, 2, 6:9] = g_of(40); ns[90:300, 2, 0] = 1.0               # env 2: upright, but never tracks -> not recovered
    ns[650:655, 3, 6:9] = g_of(50); done[690, 3] = tout[690, 3] = True  # env 3: time-out 40 rows later -> censored
    return dict(next_state=ns, command=cmd, done=done, terminated=term, time_out=tout)


def test_outcomes():
    ev = rc.events(trace())
    got = sorted((e["env"], e["onset"], e["outcome"]) for e in ev)
    assert got == [(0, 100, "recovered"), (0, 580, "fell"), (1, 100, "not recovered"), (2, 100, "not recovered"),
                   (3, 650, "censored")]
    peak = {(e["env"], e["onset"]): e["peak_tilt"] for e in ev}
    assert abs(peak[(0, 580)] - 70) < 1e-6 and abs(peak[(0, 100)] - 45) < 1e-6


def test_point_of_no_return():
    mk = lambda tilt, out: dict(peak_tilt=tilt, outcome=out)
    evs = [mk(35, "recovered")] * 6 + [mk(45, "fell")] * 5 + [mk(45, "recovered")] * 1 + [mk(55, "fell")] * 9
    s = rc.summarize(evs)
    assert s["by_peak_tilt"]["30-40"]["recovered_share"] == 1.0 and s["point_of_no_return_bin"] == "50-60"
    assert rc.summarize([mk(55, "fell")] * 3)["point_of_no_return_bin"] is None      # fewer than 5 events: not read
