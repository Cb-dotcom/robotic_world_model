"""Disagreement vs error by component: planted answers, arm parity with C0 / C0-M, manifest exclusion, end to end."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip("rsl_rl.modules")

import c0_battery as c0  # noqa: E402
import c0_components as cc  # noqa: E402
import c0m_memory as cm  # noqa: E402
from test_c0_battery import identity_contract, random_trace, real_model  # noqa: E402

STATS = dict(state_mean=np.zeros(45, np.float32), state_std=np.ones(45, np.float32) * 2,
             action_mean=np.zeros(12, np.float32), action_std=np.ones(12, np.float32))


def test_planted_joint_miss_and_agree_wrong():
    rng = np.random.default_rng(0)
    heads, W = 5, 40
    target = rng.normal(0, 1, (W, 45))
    means = target[None] + rng.normal(0, 0.5, (heads, W, 45))          # target inside the heads' range mostly
    # window 0: every head agrees on a value far from the target in gravity (dims 6-8): a planted joint miss
    means[:, 0, 6:9] = target[0, 6:9] + 5.0 + rng.normal(0, 1e-3, (heads, 3))
    # window 1: the target sits outside the range in only 1 of 3 gravity dims: not a joint miss
    means[:, 1, 6:9] = target[1, 6:9] + np.array([5.0, 0.0, 0.0]) + np.array([[0, -1, -1], [0, 1, 1], [1e-3, 0, 0],
                                                                                 [2e-3, .5, .5], [3e-3, -.5, -.5]])
    m = cc.window_metrics(means, target, STATS)
    assert m["gravity"]["outside"][0] == 1.0 and m["gravity"]["outside"][1] == pytest.approx(1 / 3)
    assert m["gravity"]["e_phys"][0] == pytest.approx(2 * m["gravity"]["e"][0])        # state_std 2
    fall = {g: {k: v[None, :2] for k, v in m[g].items()} for g in m}
    non = {g: {k: v[None, 2:] for k, v in m[g].items()} for g in m}
    s = cc.shares(fall, non, non)
    assert s["gravity"]["joint_miss_fall"] == 0.5                       # window 0 only
    assert s["gravity"]["agree_wrong_fall"] == 0.5                      # tiny spread, large error: window 0
    assert m["gravity"]["coverage"][0] == 0.0


def test_arms_match_c0_and_c0m():
    model, contract = real_model(), identity_contract()
    tr = random_trace()
    w = cc.anymal_windows(tr, contract)
    depths = (0, 8, 16, 25)
    arms = cc.arm_means(model, w["fall"]["xs"], w["fall"]["xa"], c0.HORIZON, depths)
    data = {k: tr[k] for k in ("state", "action", "next_state")}
    o = w["fall"]["origins"]
    ctx = c0.real_context_pass(model, contract, data, o, c0.HORIZON)["spread"]
    car = cm.carried_pass(model, contract, data, o)["spread"]
    imag = c0.imagined_pass(model, contract, data, o, c0.HORIZON)["spread"]
    for di, d in enumerate(depths):
        np.testing.assert_allclose(arms["ctx"][0, di].std(0, ddof=1), ctx[d], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(arms["car"][0, di].std(0, ddof=1), car[d], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(arms["imag"][:, di].std(1, ddof=1), imag[:, d], rtol=1e-5, atol=1e-7)


def test_manifest_exclusion_and_common_scale(tmp_path, monkeypatch):
    import analyse_transition_fidelity as fidelity
    model = real_model()
    monkeypatch.setattr(fidelity, "load_model", lambda path: (model, identity_contract()))
    traces = []
    for i, falls in enumerate((((69, 0), (120, 2)), ((80, 1), (130, 0)))):
        p = tmp_path / f"t{i}.npz"
        np.savez(p, **random_trace(falls=falls, seed=i))
        traces.append((f"batch{i}", p))
    from pre_reset_transitions import sha256
    man = tmp_path / "fit_manifest.json"
    man.write_text(json.dumps({"sources": [{"path": "x", "sha256": sha256(traces[1][1])}]}))
    (tmp_path / "a.pt").write_bytes(b"a"); (tmp_path / "b.pt").write_bytes(b"b")
    rep = cc.run_anymal([("ETH", tmp_path / "a.pt", None), ("E1V", tmp_path / "b.pt", man)], traces, tmp_path / "o")
    assert rep["eligibility"]["E1V"]["excluded"] == ["batch1"] and rep["eligibility"]["ETH"]["no_manifest"]
    assert rep["inputs"]["common"] == ["batch0"]
    assert rep["tables"]["ETH"]["traces_used"] == ["batch0"] == rep["tables"]["E1V"]["traces_used"]
    assert rep["common_scale_normalized"] == {"E1V~ETH": True}
    assert set(rep["predictions"]) == {"P1", "P2", "P3", "P4"}
    t = rep["tables"]["ETH"]["imag/d25"]["sum"]
    assert 0 <= t["joint_miss_fall"] <= 1 and 0 <= t["coverage_fall"] <= 1
    assert (tmp_path / "o" / "components_anymal.txt").exists()


def test_predictions_read_the_table():
    def tab(jm, ratio, imag_over_ctx, gravity_top=True):
        g = {k: dict(joint_miss_fall=(0.5 if (k == "gravity") == gravity_top else 0.1)) for k in cc.SIX}
        g["sum"] = dict(joint_miss_fall=jm)
        return {"imag/d25": g, "c0_point": dict(ratio=ratio), "imag_over_ctx_fall": imag_over_ctx}
    t = {"ETH": tab(.4, 2.6, .3), "E1V": tab(.2, 1.5, .5), "E1R2": tab(.3, 3.0, .6), "G0": tab(.5, 2, .2, False),
         "D6b": tab(.6, 1, .3)}
    p = cc.predictions(t)
    assert p["P1"]["held"] is False and p["P1"]["per_pair"] == {"E1V<ETH": True, "E1R2<ETH": True, "D6b<G0": False}
    assert p["P2"]["held"] is True
    assert p["P3"]["per_pair"]["E1R2<ETH"] is False
    assert p["P4"]["ETH"]["held"] and not p["P4"]["G0"]["held"] and p["P4"]["held"] is False


def test_go2_end_to_end(tmp_path):
    from test_go2_g0 import fake_go2_wm, save
    from test_go2_c0_check import ID
    probes = []
    for name, seed in (("p911", 3), ("p912", 5)):
        tr = random_trace(T=400, N=8, falls=((200, 0), (300, 2), (150, 4)),
                          timeouts=((299, 1), (299, 3), (299, 5), (299, 6), (299, 7)), seed=seed)
        tr["contact"] = np.zeros((400, 8, 8), np.float32)
        tr["termination_observation"] = tr["terminated"][..., None].astype(np.float32)
        tr["previous_action"] = np.zeros((400, 8, 12), np.float32)
        probes.append((name, save(tmp_path / f"{name}.npz", tr)))
    np.savez(tmp_path / "n.npz", **ID)
    wm = fake_go2_wm(tmp_path / "wm.pt", ID)
    rep = cc.run_go2([("G0", wm, None), ("D6b", wm, None)], tmp_path / "n.npz", probes, tmp_path / "o")
    k10 = rep["tables"]["G0"]["K10"]
    assert k10["windows"]["fall"] == k10["counts"]["falls_used"] > 0
    assert set(k10) >= {"ctx/d0", "ctx/d25", "car/d25", "imag/d25", "c0_point"}
    import go2_c0_check as g
    old = g.run(wm, tmp_path / "n.npz", probes, tmp_path / "old", n_boot=50)
    assert k10["c0_point"]["H_fall"] == pytest.approx(old["results"]["10"]["fall"]["H"], rel=1e-5)
