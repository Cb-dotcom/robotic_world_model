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
    rep = cc.run_go2([("G0", wm, None), ("D6b", wm, None)], tmp_path / "n.npz", probes, tmp_path / "o", n_boot=50)
    k10 = rep["tables"]["G0"]["K10"]
    assert k10["windows"]["fall"] == k10["counts"]["falls_used"] > 0
    assert set(k10) >= {"ctx/d0", "ctx/d25", "car/d25", "imag/d25", "c0_point", "c0_interval", "by_source", "repeats"}
    import go2_c0_check as g
    old = g.run(wm, tmp_path / "n.npz", probes, tmp_path / "old", n_boot=50)
    assert k10["c0_point"]["H_fall"] == pytest.approx(old["results"]["10"]["fall"]["H"], rel=1e-5)
    # Part B on Go2: the same pair bootstrap as go2_c0_check (stratified by probe, same draws)
    b = k10["c0_interval"]
    assert b["H_fall"] == pytest.approx(k10["c0_point"]["H_fall"], rel=1e-6)          # float32 spreads
    assert b["H_fall_ci"] == pytest.approx(old["results"]["10"]["H_fall_ci"], rel=1e-4, abs=1e-6)
    assert b["H_nonfall_ci"] == pytest.approx(old["results"]["10"]["H_surv_ci"], rel=1e-4, abs=1e-6)
    h = json.loads((tmp_path / "o" / "components_heads_go2.json").read_text())
    assert set(h["heads"]["G0"]) == {"K10", "K0"} and "heads" not in rep["tables"]["G0"]["K10"]


# ---------------------------------------------------------------- T04: intervals, Part B, Part C (known answers)
def _pairs(W=24, sources=("A", "C")):
    strata = np.array([sources[i % len(sources)] for i in range(W)])
    cf = np.array([f"{s}/f{i}" for i, s in enumerate(strata)])
    cn = np.array([f"{s}/n{i}" for i, s in enumerate(strata)])
    return strata, cf, cn


def test_c0_interval_known_answers_and_the_registered_draws():
    strata, cf, cn = _pairs()
    # constant per-window deficits: every draw gives the point value
    r = cc.c0_interval(np.full(24, 2.0), np.full(24, 0.5), strata, cf, cn, n_boot=200)
    assert r["H_fall"] == 2.0 and r["H_nonfall"] == 0.5 and r["ratio"] == 4.0 and r["difference"] == 1.5
    assert r["H_fall_ci"] == pytest.approx([2, 2]) and r["ratio_ci"] == pytest.approx([4, 4])
    assert r["difference_ci"] == pytest.approx([1.5, 1.5]) and not r["nonfall_ci_includes_0"]
    assert r["pairs"] == 24 and r["pair_components"] == 24 and r["nonfinite_ratio_draws"] == 0
    # random deficits: the intervals are the draws of c0m_memory.decompose (one bootstrap, not a new one)
    rng = np.random.default_rng(1)
    logs = {k: (rng.normal(1, .3, (26, 24)), rng.normal(.8, .2, (26, 24)), rng.normal(.1 if k == "fall" else .6, .3, (26, 24)))
            for k in ("fall", "nonfall")}
    dec = cm.decompose(logs, strata, {"fall": cf, "nonfall": cn}, n_boot=300, seed=0, depths=(25,))["d25"]
    h = {k: logs[k][0][25] - logs[k][2][25] for k in logs}
    r = cc.c0_interval(h["fall"], h["nonfall"], strata, cf, cn, n_boot=300, seed=0)
    assert r["H_fall"] == pytest.approx(dec["fall"]["H"]) and r["H_fall_ci"] == pytest.approx(dec["fall"]["H_ci"])
    assert r["H_nonfall_ci"] == pytest.approx(dec["nonfall"]["H_ci"])
    bf, bn = c0.pair_bootstrap(h["fall"], h["nonfall"], strata, cf, cn, 300, 0)
    assert r["difference_ci"] == pytest.approx(c0.ci(bf - bn)) and r["ratio_ci"] == pytest.approx(c0.ci(bf / bn))
    # a zero-crossing non-fall interval is flagged (no verdict is made)
    r = cc.c0_interval(h["fall"], rng.normal(0, 1, 24), strata, cf, cn, n_boot=300)
    assert r["nonfall_ci_includes_0"] and "verdict" not in r


def test_shares_intervals_known_answers():
    strata, cf, cn = _pairs()
    W = 24
    # every fall window identical, every non-fall window identical: intervals collapse to the point values
    rng = np.random.default_rng(0)
    target = rng.normal(0, 1, (45,))
    fall_means = np.tile(target, (5, W, 1)) + np.linspace(-1, 1, 5)[:, None, None] * 0.1
    fall_means[:, :, 6:9] += 3.0                                        # gravity: every head far from the target
    non_means = np.tile(target, (5, W, 1)) + np.linspace(-1, 1, 5)[:, None, None] * 0.1
    tgt = np.tile(target, (W, 1))
    f = {g: {k: v[None] for k, v in d.items()} for g, d in cc.window_metrics(fall_means, tgt, STATS).items()}
    n = {g: {k: v[None] for k, v in d.items()} for g, d in cc.window_metrics(non_means, tgt, STATS).items()}
    s = cc.shares(f, n, n, (strata, cf, cn, 200, 0))
    g = s["gravity"]
    assert g["e_fall"] == pytest.approx(3.0) and g["e_phys_fall"] == pytest.approx(6.0)        # state_std 2
    assert g["ci"]["e_phys_fall"] == pytest.approx([6.0, 6.0]) and g["ci"]["e_phys_diff"] == pytest.approx([6.0, 6.0])
    assert g["coverage_fall"] == 0.0 and g["ci"]["coverage_fall"] == [0.0, 0.0]
    assert g["ci"]["u_geomean_fall"] == pytest.approx([g["u_geomean_fall"]] * 2)
    assert set(g["ci"]) >= {f"{m}_{k}" for m in cc.METRICS for k in ("fall", "nonfall", "diff")} | {"tilt_deg_fall"}
    # a varying metric: its interval is the pair bootstrap of the per-window values
    f["sum"]["e_phys"] = rng.normal(1, .5, (1, W))
    s = cc.shares(f, n, n, (strata, cf, cn, 300, 0))
    bf, _ = c0.pair_bootstrap(f["sum"]["e_phys"][0], n["sum"]["e_phys"][0], strata, cf, cn, 300, 0)
    assert s["sum"]["ci"]["e_phys_fall"] == pytest.approx(c0.ci(bf))
    assert s["sum"]["e_phys_fall"] == pytest.approx(f["sum"]["e_phys"].mean())


def test_head_readout_known_answers():
    strata, cf, cn = _pairs(W=10)
    W, heads, D = 10, 5, 2
    stats = {k: v.astype(np.float64) for k, v in STATS.items()}
    y = np.random.default_rng(2).normal(0, .2, (W, D, 45))
    y[:, :, 6:9] = [0, 0, -0.5]                                            # upright (normalized; std 2 -> g_z -1)
    off = np.arange(1, heads + 1, dtype=float)                             # head h misses by h + 1 in every dim
    means = {k: y.transpose(1, 0, 2)[:, None] + off[None, :, None, None] for k in ("fall", "nonfall")}   # [D, heads, W, 45]
    stop = {"fall": np.zeros((D, heads, W)), "nonfall": np.zeros((D, heads, W))}
    stop["fall"][1, 0] = 0.9                                               # head 0 predicts the stop at the last depth
    tilted = means["fall"].copy()
    tilted[1, 1, :, 6:9] = [0.5, 0, -0.1]                                  # head 1 predicts a large tilt, no stop
    tilted[1, 2, :, 6:9] = [0, 0, -0.5]                                    # head 2 predicts upright, no stop
    means["fall"] = tilted
    r = cc.head_readout(means, stop, {k: y for k in means}, stats, (0, 25), (strata, cf, cn, 200, 0))
    x = r["d0"]["lin_vel"]["fall"]
    assert x["same_direction"] == 1.0 and x["same_direction_ci"] == [1.0, 1.0]       # every head misses upward
    assert x["advantage"] == pytest.approx([2, 1, 0, -1, -2])                       # mean misses by 3
    assert x["advantage_ci"][0] == pytest.approx([2, 2])
    assert np.allclose(x["signed_error"], off[:, None] * 2)                          # physical: std 2
    assert np.allclose(x["ensemble_signed_error"], 6.0)
    assert x["spread"] == pytest.approx(3 * np.std(off, ddof=1))
    assert r["d25"]["stop"]["fall"]["share_above_half"] == [1.0, 0, 0, 0, 0]
    last = r["fall_last_depth"]
    assert last["stop_predicted"][0] == 1.0 and last["tilt_predicted_stop_missed"][1] == 1.0
    assert last["tilt_missed"][2] == 1.0 and last["observed_tilt_share"] == 0.0


def test_repeat_rule():
    per = {"A": {"x": [0.1, 0.3], "y": [0.1, 0.2], "z": [-0.3, -0.1]},
           "C": {"x": [0.2, 0.4], "y": [-0.1, 0.2], "z": [0.1, 0.2]}}
    r = cc.repeats(per, {"x": ("x",), "y": ("y",), "z": ("z",)})
    assert r["repeated"] == {"x": "+"} and sorted(r["some_sources_only"]) == ["y", "z"]
    assert cc.repeats({"A": per["A"]}, {"x": ("x",)})["repeated"] == {}              # one source: nothing repeats


def test_anymal_two_sources_part_b_part_c_and_separate_table(tmp_path, monkeypatch):
    import analyse_transition_fidelity as fidelity
    model = real_model()
    monkeypatch.setattr(fidelity, "load_model", lambda path: (model, identity_contract()))
    traces = []
    for i, falls in enumerate((((69, 0), (120, 2), (100, 1)), ((80, 1), (130, 0), (90, 2)), ((75, 0), (125, 2)))):
        p = tmp_path / f"t{i}.npz"
        np.savez(p, **random_trace(falls=falls, timeouts=(), seed=i))
        traces.append((("batchA", "batchC", "round0_1701")[i], p))
    from pre_reset_transitions import sha256
    man = tmp_path / "summary.json"
    man.write_text(json.dumps({"inputs": {"baseline1701": {"sha256": sha256(traces[2][1])}}}))
    (tmp_path / "a.pt").write_bytes(b"a"); (tmp_path / "b.pt").write_bytes(b"b")
    models = [("ETH", tmp_path / "a.pt", None), ("E1V", tmp_path / "b.pt", man)]
    rep = cc.run_anymal(models, traces, tmp_path / "o", n_boot=100)
    assert rep["inputs"]["common"] == ["batchA", "batchC"] and rep["inputs"]["pairs_excluded_from_common"] == {"round0_1701": 2}
    assert rep["eligibility"]["E1V"]["excluded_pairs"] == 2 and rep["tables"]["E1V"]["pairs_excluded"] == {"round0_1701": 2}
    t = rep["tables"]["ETH"]
    assert t["c0_interval"]["pairs"] == 6 and set(t["by_source"]) == {"batchA", "batchC"}
    assert t["c0_interval"]["H_fall"] == pytest.approx(t["c0_point"]["H_fall"], rel=1e-6)
    assert "ci" in t["ctx/d0"]["gravity"] and "repeats" in t and "heads" not in t
    txt = (tmp_path / "o" / "components_anymal.txt").read_text()
    assert "PART B" in txt and "1.2415" in txt and "SAME FOR EVERY MODEL: True" in txt
    h = json.loads((tmp_path / "o" / "components_heads_anymal.json").read_text())["heads"]["ETH"]
    assert set(h) == {"pooled", "by_source", "repeats"} and len(h["pooled"]["d25"]["gravity"]["fall"]["advantage"]) == 5
    assert "COUNT" in (tmp_path / "o" / "components_heads_anymal.txt").read_text()
    # separate table: round0_1701 alone, each model its own eligible traces; E1V has none and is skipped
    rep = cc.run_anymal(models, traces[2:], tmp_path / "x", n_boot=100, use_intersection=False)
    assert set(rep["tables"]) == {"ETH"} and rep["eligibility"]["E1V"]["skipped"]
    with pytest.raises(ValueError, match="no eligible trace"):
        cc.run_anymal(models, traces[2:], tmp_path / "y", n_boot=100)


def test_part_b_reproduces_the_registered_c0_arithmetic(tmp_path, monkeypatch):
    """ETH-style run on the windows of a real c0_battery.run: same windows, same pair draws (same trace names), so the
    Part B rows equal the registered report's d25 rows, pooled and per source."""
    from test_c0_supplement import c0_run
    out = c0_run(tmp_path, monkeypatch, n_boot=300)
    reg = json.loads((out / "c0_report.json").read_text())["report"]
    traces = [(n, info["path"]) for n, info in json.loads((out / "c0_report.json").read_text())["meta"]["inputs"].items()]
    (tmp_path / "e.pt").write_bytes(b"e")
    rep = cc.run_anymal([("ETH", tmp_path / "e.pt", None)], traces, tmp_path / "b", n_boot=300)
    t = rep["tables"]["ETH"]
    for scope, mine in [("pooled", t["c0_interval"])] + [(s, v["c0_interval"]) for s, v in t["by_source"].items()]:
        r, rule = reg[scope], reg[scope]["c0_rule_d25_sum"]
        assert mine["H_fall"] == pytest.approx(r["fall"]["rows"]["d25/sum"]["H"], rel=1e-5)
        assert mine["H_nonfall"] == pytest.approx(r["nonfall"]["rows"]["d25/sum"]["H"], rel=1e-5)
        if "ratio" in rule:
            assert mine["ratio_ci"] == pytest.approx(rule["ratio_ci"], rel=1e-4)
        else:
            assert mine["H_fall_ci"] == pytest.approx(rule["h_fall_ci"], rel=1e-4, abs=1e-6)
        assert mine["pair_components"] == rule["pair_components"]
