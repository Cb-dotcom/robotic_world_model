import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_calibrate as cal
import f1_pilot_score as score
from test_c0_battery import identity_contract, random_trace, real_model


def test_origins_stay_inside_one_episode():
    done = np.zeros((400, 3), bool)
    done[150, 0] = done[300, 1] = True
    origins, n = cal.calibration_origins(done, horizon=40, n_windows=10, seed=0)
    for t, e in origins:
        assert not done[t - 31:t + 40, e].any()
    again, _ = cal.calibration_origins(done, horizon=40, n_windows=10, seed=0)
    np.testing.assert_array_equal(origins, again)
    with pytest.raises(ValueError, match="clean windows"):
        cal.calibration_origins(done, horizon=40, n_windows=10_000)


def test_gain_matches_the_mean_penalties():
    tr = random_trace(T=400, N=4, falls=(), seed=3)
    r = cal.calibrate(real_model(), identity_contract(), tr, horizon=30, n_windows=12, seed=0)
    assert r["scale"].shape == (30,) and np.all(r["scale"] > 0)
    assert r["check"]["f1_mean"] == pytest.approx(r["check"]["standard_mean"], rel=1e-9)
    # s(d) maps the member spread to the realized error on average, depth by depth
    np.testing.assert_allclose(r["scale"] * r["m"].mean(1), r["e"].mean(1), rtol=1e-9)
    # at d = 0 every member has the same input and zeroed memory: member spread = head disagreement
    np.testing.assert_allclose(r["m"][0], r["u"][0], rtol=1e-4)


def test_calibration_belongs_to_its_world_model(tmp_path):
    import e1_train
    wm, other = tmp_path / "wm.pt", tmp_path / "other.pt"
    wm.write_bytes(b"model")
    other.write_bytes(b"other")
    import hashlib
    c = dict(members=5, scale=[1.0] * 256, gain=0.5, wm_sha256=[hashlib.sha256(b"model").hexdigest()])
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(c))

    class Env:
        pass
    e1_train.apply_f1_calibration(p, wm, Env)
    assert Env.f1_members == 5 and Env.f1_gain == 0.5 and len(Env.f1_scale) == 256
    with pytest.raises(SystemExit, match="another world model"):
        e1_train.apply_f1_calibration(p, other, Env)
    p.write_text(json.dumps(dict(c, gain=-1.0)))
    with pytest.raises(SystemExit, match="non-positive"):
        e1_train.apply_f1_calibration(p, wm, Env)


def _eval(folder, tag, cmd, seed, falls, speed, track):
    d = folder / f"{tag}_cmd{cmd}_s{seed}"
    d.mkdir(parents=True)
    (d / "eval.log").write_text("noise\n================ EPISODE STATS ================\n"
                                f"  falls: {falls}\n  episodes_completed: 640\n  track_per_step: {track:.4f}\n"
                                f"  speed_ratio: {speed:.4f}\n===============================================\n")


@pytest.mark.parametrize("pilot_falls, speed, track, verdict", [
    ((9, 10), 0.85, 0.150, "GO"),          # 19 <= 0.7 x 28
    ((10, 10), 0.85, 0.150, "NO-GO"),      # 20 > 19.6
    ((5, 5), 0.79, 0.150, "NO-GO"),        # too slow
    ((5, 5), 0.85, 0.115, "NO-GO"),        # tracking < D1 - 0.03
])
def test_go_no_go(tmp_path, pilot_falls, speed, track, verdict):
    for s, f in zip((42, 43), pilot_falls):
        for cmd in (10, 30):
            _eval(tmp_path / "f", "F1", cmd, s, f, speed, track)
    for s, f in zip((42, 43), (14, 14)):
        for cmd in (10, 30):
            _eval(tmp_path / "d", "D", cmd, s, f, 0.9, 0.150)
    r = score.score(tmp_path / "f", "F1", tmp_path / "d", "D")
    assert r["verdict"] == verdict
    assert r["results"][10]["d1"]["falls"] == 28 and r["results"][10]["d1"]["exposure_s"] == 12800
