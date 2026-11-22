"""A4 D2 breakdown (22 Nov 2026): summary flags and an end-to-end run on synthetic data."""
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

import go2_d2_breakdown as bd  # noqa: E402
import go2_round0_prepare as prep  # noqa: E402
from test_go2_g0 import fake_go2_wm, restore_go2_class, round0_trace, save  # noqa: E402,F401


def fall(caught, length=100, above=False, tilt=70.0):
    return dict(heads_stopping=sum(caught), per_head_caught=caught, length=length, max_tilt_last11_deg=tilt,
                u_above_p99=above)


def test_summary_flags():
    F, T = [False] * 5, [True] * 5
    s = bd.summary([fall(F, 20), fall(F, 30), fall(F, 25), fall(T, 200, above=True), fall(T, 300)])
    assert s["missed_by_all"] == 3 and s["share_of_misses_missed_by_all"] == 1.0 and s["detection_short"] == 0.0
    assert s["detection_long"] == 1.0 and s["flags"] == dict(shared_blind_spot=True, post_reset_gap=True, uncertainty_warns=False)
    s = bd.summary([fall([True, False, False, False, False], 200), fall(F, 200, above=True), fall(T, 200)])
    assert s["partial"] == 1 and s["share_of_misses_missed_by_all"] == 0.5 and not s["flags"]["shared_blind_spot"]
    assert s["flags"]["uncertainty_warns"]


def test_breakdown_end_to_end(tmp_path):
    sh = save(tmp_path / "sh.npz", round0_trace(T=300, N=8, seed=4))
    prep.prepare(sh, tmp_path / "m", frac=0.1, seed=101)
    probe_tr = round0_trace(T=300, N=6, seed=3)
    meta = json.loads(str(probe_tr["metadata"])); meta["friction_shift"] = {"factor": 0.3, "verified": True}
    probe_tr["metadata"] = np.array(json.dumps(meta))
    probe = save(tmp_path / "probe.npz", probe_tr)
    with np.load(tmp_path / "m" / "go2_norm.npz") as z:
        stats = {k: z[k] for k in z.files}
    wm = fake_go2_wm(tmp_path / "wm.pt", stats)
    rep = bd.run(wm, tmp_path / "m" / "go2_norm.npz", probe, sh, tmp_path / "m" / "withheld.json", tmp_path / "out")
    assert rep["probe"]["summary"]["falls"] == 2 and set(rep["probe"]["summary"]["flags"]) == {"shared_blind_spot", "post_reset_gap", "uncertainty_warns"}
    f = rep["probe"]["falls"][0]
    assert 0 <= f["heads_stopping"] <= 5 and len(f["per_head_caught"]) == 5 and 0 <= f["u_rank_vs_surviving"] <= 1


def test_episode_pass_logits_equal_the_replay(tmp_path):
    import go2_g0_replay as rp
    from types import SimpleNamespace
    stats = dict(state_mean=np.zeros(45, np.float32), state_std=np.ones(45, np.float32),
                 action_mean=np.zeros(12, np.float32), action_std=np.ones(12, np.float32))
    np.savez(tmp_path / "n.npz", **stats)
    sd, H, norm, _ = rp.load_model(fake_go2_wm(tmp_path / "wm.pt", stats), tmp_path / "n.npz", "cpu")
    g = np.random.default_rng(0)
    ep = SimpleNamespace(S=g.normal(0, .5, (30, 45)).astype(np.float32), A=g.normal(0, .5, (30, 12)).astype(np.float32),
                         L=30, y=np.zeros(30, bool), obs=None)
    LG, U = bd.episode_pass(sd, ep, "cpu")
    _, _, LG2 = rp.episode_context(sd, ep, "cpu", return_logits=True)
    np.testing.assert_allclose(LG[:, 1:], LG2[:, 1:], rtol=1e-5, atol=1e-6)
    assert np.isnan(U[0]) and (U[1:] >= 0).all()
