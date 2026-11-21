"""A4 D2 qualification (22 Nov 2026): rules, fidelity arithmetic, end-to-end run on synthetic data."""
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

import go2_d2_qualify as q  # noqa: E402
import go2_round0_prepare as prep  # noqa: E402
from test_go2_g0 import fake_go2_wm, restore_go2_class, round0_trace, save  # noqa: E402,F401


def measured(k1=0.0, k32=0.0, surv=1.0, v25=0.1, v50=0.2, t50=2.0, det=0.6, fs=0.05):
    return dict(walking=dict(k1=dict(stop_share=[k1] * 5), k32=dict(stop_share=[k32] * 5),
                             warm_rollouts=dict(survival={"100": [surv] * 5}),
                             fidelity=dict(vxy_rmse_25=v25, vxy_rmse_50=v50, tilt_rmse_50=t50)),
                probe=dict(falls=40, surviving=99, falls_detected_share=[det] * 5, surviving_false_stop_share=[fs] * 5))


def test_rules():
    r = q.decide(measured())
    assert r["D2_pass"] and r["Q1"]["reset_type"] == "cold"
    assert q.decide(measured(k1=0.05))["Q1"]["reset_type"] == "warm" and q.decide(measured(k1=0.05))["D2_pass"]
    assert not q.decide(measured(k32=0.02))["Q1"]["pass"]
    assert not q.decide(measured(surv=0.9))["Q1"]["pass"]
    assert not q.decide(measured(v50=0.5))["Q1b"]["pass"] and not q.decide(measured(t50=6))["D2_pass"]
    assert not q.decide(measured(det=0.4))["Q2"]["pass"] and not q.decide(measured(fs=0.2))["Q3"]["pass"]


def test_fidelity_arithmetic():
    norm = dict(state_mean=np.zeros(45), state_std=np.ones(45))
    true = np.zeros((1, 60, 45)); true[..., 8] = -1.0
    pred = np.repeat(true[:, None, 29], 2, 1).copy()           # [n=1, E=2, 45] = the recorded row 29 (step 25 from j=5)
    v, t = q.fidelity(pred, [(0, 5)], true, 25, norm)
    assert v == 0.0 and t == 0.0
    pred[..., 0] += 0.3
    pred[..., 6], pred[..., 8] = np.sin(np.radians(10)), -np.cos(np.radians(10))
    v, t = q.fidelity(pred, [(0, 5)], true, 25, norm)
    assert v == pytest.approx(0.3) and t == pytest.approx(10.0)


def test_qualify_end_to_end(tmp_path):
    from policy_loop import ObservationBuilder, Policy, consistent_policy_trace, save_actor
    ident = dict(state_mean=np.zeros(45, np.float32), state_std=np.ones(45, np.float32),
                 action_mean=np.zeros(12, np.float32), action_std=np.ones(12, np.float32))
    tr = consistent_policy_trace(round0_trace(T=300, N=6), Policy(save_actor(tmp_path / "s2.pt")),
                                 ObservationBuilder("go2_flat", ident), ident)
    r0 = save(tmp_path / "r0.npz", tr)
    prep.prepare(r0, tmp_path / "m", frac=0.4, seed=101)
    probe_tr = round0_trace(T=300, N=6, seed=3)
    meta = json.loads(str(probe_tr["metadata"])); meta["friction_shift"] = {"factor": 0.3, "verified": True}
    probe_tr["metadata"] = np.array(json.dumps(meta))
    probe = save(tmp_path / "probe.npz", probe_tr)
    with np.load(tmp_path / "m" / "go2_norm.npz") as z:
        stats = {k: z[k] for k in z.files}
    wm = fake_go2_wm(tmp_path / "wm.pt", stats)
    import go2_g0_replay as rp
    monkey_frac = rp.heldout
    rp_heldout = lambda trace, path, frac, seed: monkey_frac(trace, path, 0.4, seed)
    q.rp.heldout = rp_heldout
    try:
        rep = q.run(wm, tmp_path / "m" / "go2_norm.npz", r0, tmp_path / "m" / "withheld.json", probe, tmp_path / "out",
                    K=20, start_every=10)
    finally:
        q.rp.heldout = monkey_frac
    res = rep["results"]
    assert res["probe"]["falls"] == 2 and res["probe"]["surviving"] == 3 and res["probe"]["friction"] == 0.3
    assert set(res["rules"]) == {"Q1", "Q1b", "Q2", "Q3", "D2_pass"}
    assert (tmp_path / "out" / "d2_qualify.txt").read_text().startswith("A4 D2 qualification")
