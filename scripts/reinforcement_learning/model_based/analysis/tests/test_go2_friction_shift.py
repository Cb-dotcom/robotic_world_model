"""A4 D1 (21 Nov 2026): friction shift read-back rule, fall and recovery counts, level choice."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

MB = Path(__file__).resolve().parents[2]
for p in (str(MB / "analysis"), str(MB.parent / "rsl_rl"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import friction_shift as fs  # noqa: E402
import go2_friction_shift as d1  # noqa: E402
from test_c0_battery import random_trace  # noqa: E402


def cfg(static=1.0, dynamic=1.0, mode="multiply"):
    mat = SimpleNamespace(static_friction=static, dynamic_friction=dynamic, friction_combine_mode=mode)
    return SimpleNamespace(scene=SimpleNamespace(terrain=SimpleNamespace(physics_material=mat, prim_path="/World/ground")))


def test_combine_follows_physx_priority():
    assert fs.combine(0.8, 0.6, "average", "multiply") == (pytest.approx(0.48), "multiply")
    assert fs.combine(0.8, 0.6, "max", "multiply") == (0.8, "max")
    assert fs.combine(0.8, 0.6, "average", "average") == (pytest.approx(0.7), "average")


def test_scale_and_check_the_read_back():
    c = cfg()
    rec = fs.scale_ground(c, 0.6)
    assert c.scene.terrain.physics_material.static_friction == pytest.approx(0.6) and rec["nominal"]["static"] == 1.0
    mats = [("/World/ground/physicsMaterial", 0.6, 0.6, "multiply"), ("/World/envs/env_0/Robot/foot_mat", 0.8, 0.6, "average")]
    out = fs.check(rec, mats)
    assert out["verified"] and out["combine_modes_used"] == ["multiply"]
    with pytest.raises(RuntimeError, match="expected the scaled config"):
        fs.check(rec, [("/World/ground/m", 1.0, 1.0, "multiply")])
    with pytest.raises(RuntimeError, match="does not scale"):
        fs.check(rec, [("/World/ground/m", 0.6, 0.6, "multiply"), ("/World/envs/env_0/Robot/m", 0.8, 0.6, "max")])
    with pytest.raises(RuntimeError, match="does not scale"):     # an average-mode ground would not scale by 0.6
        fs.check(fs.scale_ground(cfg(mode="average"), 0.6), [("/World/ground/m", 0.6, 0.6, "average"),
                                                              ("/World/envs/env_0/Robot/m", 0.8, 0.6, "average")])
    with pytest.raises(RuntimeError, match="no physics material"):
        fs.check(rec, [("/World/envs/env_0/Robot/m", 0.8, 0.6, "average")])
    with pytest.raises(ValueError):
        fs.scale_ground(cfg(), 1.5)


def tilted(T=400, N=3):
    tr = random_trace(T=T, N=N, falls=((300, 0),), timeouts=((199, 1),))
    tr["next_state"][..., 6:9] = np.array([0, 0, -1], np.float32)
    g = tr["next_state"]
    g[100:110, 0, 6] = 0.6          # env 0: near-fall at 100, recovers (fall at 300 is > 50 rows later)
    g[280:300, 0, 6] = 0.7          # env 0: near-fall at 280, falls at 300 -> before a fall
    g[150:155, 1, 6] = 0.6          # env 1: near-fall at 150, episode ends 199 (49 rows left) -> censored
    g[250:252, 2, 6] = 0.6          # env 2: near-fall at 250, recovers
    g[255:257, 2, 6] = 0.6          # env 2: 5 rows later -> inside the 10-row gap, not a new near-fall
    tr["state"][1:] = tr["next_state"][:-1]
    return tr


def test_counts_falls_recoveries_and_censoring():
    c, w = d1.count(tilted())
    assert c["falls"] == 1 and c["time_outs"] == 1 and c["ended"] == 2
    assert c["near_falls"] == 4 and c["recoveries"] == 2 and c["near_falls_before_fall"] == 1 and c["censored"] == 1
    assert c["fall_episodes_with_near_fall"] == 1
    assert [(x["env"], x["onset"], x["lo"], x["hi"]) for x in w] == [(0, 100, 90, 150), (2, 250, 240, 300)]
    assert c["fall_fraction"] == 0.5


def test_choose_takes_the_first_level_that_meets_the_rule():
    mk = lambda f, r, ff: dict(falls=f, recoveries=r, fall_fraction=ff)
    assert d1.choose([(0.6, mk(3, 40, .01)), (0.45, mk(20, 30, .1)), (0.3, mk(60, 50, .3))]) == 0.45
    assert d1.choose([(0.6, mk(3, 40, .01)), (0.45, mk(20, 10, .1)), (0.3, mk(90, 50, .5))]) is None


def test_verify_keeps_the_factor_and_count_reads_the_stored_record(tmp_path, monkeypatch):
    """Regression (21 Nov): verify() returned only the read-back, so the stored record lost its factor."""
    import json
    rec = fs.scale_ground(cfg(), 0.6)
    monkeypatch.setattr(fs, "read_materials", lambda: [("/World/ground/terrain/physicsMaterial", 0.6, 0.6, "multiply"),
                                                       ("/World/envs/env_0/Robot/m", 0.8, 0.6, "average")])
    stored = fs.verify(rec)
    assert stored["factor"] == 0.6 and stored["nominal"]["static"] == 1.0 and stored["verified"]
    tr = tilted()
    meta = json.loads(str(tr["metadata"]))
    meta.update(friction_shift=json.loads(json.dumps(stored)), seed=900, num_envs=3, num_steps=400)
    tr["metadata"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "f0.60.npz", **tr)
    (tmp_path / "f0.60.json").write_text("{}")                      # the recorder's sidecar
    assert d1.main(["count", "--trace", str(tmp_path / "f0.60.npz"), "--out", str(tmp_path / "f0.60.counts.json")]) == 0
    assert json.loads((tmp_path / "f0.60.counts.json").read_text())["friction_factor"] == 0.6
    assert (tmp_path / "f0.60.json").read_text() == "{}"
    with pytest.raises(SystemExit, match="sidecar"):
        d1.main(["count", "--trace", str(tmp_path / "f0.60.npz"), "--out", str(tmp_path / "f0.60.json")])
    assert d1.main(["count", "--trace", str(tmp_path / "f0.60.npz"), "--out", str(tmp_path / "p.json"),
                    "--min-falls", "30"]) == 3
    assert d1.main(["count", "--trace", str(tmp_path / "f0.60.npz"), "--out", str(tmp_path / "q.json"),
                    "--min-falls", "1", "--min-time-outs", "1"]) == 0
    assert d1.main(["count", "--trace", str(tmp_path / "f0.60.npz"), "--out", str(tmp_path / "r.json"),
                    "--min-falls", "1", "--min-time-outs", "2"]) == 3
