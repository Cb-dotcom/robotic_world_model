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
    assert set(rep["predictions"]) == {"P1", "P2", "P3", "P3_difference", "P4"}
    t = rep["tables"]["ETH"]["imag/d25"]["sum"]
    assert 0 <= t["joint_miss_fall"] <= 1 and 0 <= t["coverage_fall"] <= 1
    assert (tmp_path / "o" / "components_anymal.txt").exists()


def test_predictions_read_the_table():
    def tab(jm, ratio, imag_over_ctx, gravity_top=True, unstable=False, diff=1.0):
        g = {k: dict(joint_miss_fall=(0.5 if (k == "gravity") == gravity_top else 0.1)) for k in cc.SIX}
        g["sum"] = dict(joint_miss_fall=jm)
        return {"imag/d25": g, "c0_point": dict(ratio=ratio), "imag_over_ctx_fall": imag_over_ctx,
                "c0_interval": dict(unstable_denominator=unstable, difference=diff)}
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
    # review 26 Nov: per-recording head readout at the top (head -60), leave one head out, which side, read gate
    top = (tmp_path / "o" / "components_heads_anymal.txt").read_text().splitlines()[:60]
    assert any(x.startswith("ETH: head advantages that COUNT") for x in top)
    for src in ("batchA", "batchC"):
        assert any(x.startswith(f"  ETH {src} d25 sum fall advantage: h0 ") for x in top)
        assert set(h["by_source"][src]["leave_one_out"]) == {"all"} | {f"without_h{i}" for i in range(5)}
    loo = h["pooled"]["leave_one_out"]["all"]["warning_d25"]
    assert loo["difference"] == pytest.approx(t["c0_interval"]["difference"], rel=1e-5)
    assert loo["difference_ci"] == pytest.approx(t["c0_interval"]["difference_ci"], rel=1e-4)
    ws = t["which_side"]["d25"]
    assert ws["H_fall"] == pytest.approx(t["c0_interval"]["H_fall"], rel=1e-5)
    assert ws["H_nonfall"] == pytest.approx(t["c0_interval"]["H_nonfall"], rel=1e-5)
    assert set(t["which_side"]) == {"d16", "d25"} and set(t["by_source"]["batchC"]["which_side"]) == {"d16", "d25"}
    assert "READ GATE ETH batchA" in txt and "READ GATE ETH batchC" in txt and "  SIDE ETH batchA d25 log u" in txt
    # separate table: round0_1701 alone, each model its own eligible traces; E1V has none and is skipped
    rep = cc.run_anymal(models, traces[2:], tmp_path / "x", n_boot=100, use_intersection=False)
    assert set(rep["tables"]) == {"ETH"} and rep["eligibility"]["E1V"]["skipped"]
    assert "READ GATE ETH round0_1701: ratio" in (tmp_path / "x" / "components_anymal.txt").read_text()
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


# ---------------------------------------------------------------- review of the 26 Nov run: known answers
def _lift(m):
    return {g: {k: v[None] for k, v in d.items()} for g, d in m.items()}


def test_zero_spread_window_is_left_out_and_counted():
    strata, cf, cn = _pairs()
    rng = np.random.default_rng(3)
    tgt = rng.normal(0, 1, (24, 45))
    fall, non = tgt[None] + rng.normal(0, .3, (5, 24, 45)), tgt[None] + rng.normal(0, .2, (5, 24, 45))
    fall[:, 0] = fall[0, 0]                                    # window 0: five identical heads, zero spread everywhere
    fm, nm = cc.window_metrics(fall, tgt, STATS), cc.window_metrics(non, tgt, STATS)
    assert fm["sum"]["u"][0] == 0 and fm["sum"]["coverage"][0] == 0.0          # covered only by an exact hit
    s = cc.shares(_lift(fm), _lift(nm), _lift(nm), (strata, cf, cn, 200, 0))["sum"]
    assert s["zero_spread_windows"] == {"fall": 1, "nonfall": 0} and s["zero_spread_pairs_excluded"] == 1
    lf, ln = np.log(fm["sum"]["u"][1:]), np.log(nm["sum"]["u"][1:])
    assert s["u_geomean_fall"] == pytest.approx(np.exp(lf.mean())) and s["u_geomean_nonfall"] == pytest.approx(np.exp(ln.mean()))
    bf, bn = c0.pair_bootstrap(lf, ln, strata[1:], cf[1:], cn[1:], 200, 0)
    assert s["ci"]["log_u_fall"] == pytest.approx(c0.ci(bf)) and s["ci"]["log_u_diff"] == pytest.approx(c0.ci(bf - bn))
    assert all(np.isfinite(v).all() for v in s["ci"].values())                 # no inf / NaN in any interval
    # Part B and the point H: the pair is left out and counted
    h_f = cc.window_h(fm["sum"]["u"], np.stack([fm["sum"]["u"] * .5] * 5))
    h_n = cc.window_h(nm["sum"]["u"], np.stack([nm["sum"]["u"] * .8] * 5))
    assert np.isnan(h_f[0]) and h_f[1:] == pytest.approx(np.log(2))
    r = cc.c0_interval(h_f, h_n, strata, cf, cn, n_boot=200)
    ref = cc.c0_interval(h_f[1:], h_n[1:], strata[1:], cf[1:], cn[1:], n_boot=200)
    assert r["zero_spread_pairs_excluded"] == 1 and r["pairs"] == 23 and r["H_fall"] == pytest.approx(np.log(2))
    assert r["difference_ci"] == pytest.approx(ref["difference_ci"]) and np.isfinite(r["H_fall_ci"]).all()
    p = cc.h_ratio(fm["sum"]["u"], np.stack([fm["sum"]["u"] * .5] * 5), nm["sum"]["u"], np.stack([nm["sum"]["u"] * .8] * 5))
    assert p["zero_spread_pairs_excluded"] == 1 and p["H_fall"] == pytest.approx(np.log(2))
    # same direction: identical heads that hit the target exactly are no miss
    s10, f10, n10 = _pairs(W=10)
    zeros = {k: np.zeros((1, 5, 10, 45)) for k in ("fall", "nonfall")}
    hr = cc.head_readout(zeros, {k: np.zeros((1, 5, 10)) for k in zeros}, {k: np.zeros((10, 1, 45)) for k in zeros},
                         {k: v.astype(np.float64) for k, v in STATS.items()}, (25,), (s10, f10, n10, 50, 0))
    assert hr["d25"]["sum"]["fall"]["same_direction"] == 0.0 and hr["d25"]["sum"]["fall"]["advantage"] == [0.0] * 5


def test_unstable_denominator_suppresses_the_ratio_and_p3():
    W = 200
    strata = np.array(["A"] * W)
    cf, cn = np.array([f"A/f{i}" for i in range(W)]), np.array([f"A/n{i}" for i in range(W)])
    h_f, h_n = np.full(W, 2.0), np.ones(W)
    h_n[0] = -55.0          # CI above 0 (3 copies: 0.16), but ~2 % of draws hold it 4+ times: denominator <= 0
    r = cc.c0_interval(h_f, h_n, strata, cf, cn, n_boot=2000)
    assert r["H_nonfall_ci"][0] > 0 and not r["nonfall_ci_includes_0"]
    assert r["nonpositive_denominator_draws"] > 0 and r["unstable_denominator"]
    txt = cc._c0_text(r)
    assert txt.startswith("fall - non-fall 1.280") and "UNSTABLE DENOMINATOR" in txt and f"{r['ratio']:.2f}" not in txt
    r0 = cc.c0_interval(h_f, np.random.default_rng(0).normal(0, 1, W), strata, cf, cn, n_boot=300)
    assert r0["nonfall_ci_includes_0"] and r0["unstable_denominator"] and "UNSTABLE DENOMINATOR" in cc._c0_text(r0)
    rs = cc.c0_interval(h_f, np.full(W, .5), strata, cf, cn, n_boot=300)
    assert not rs["unstable_denominator"] and "ratio 4.00 [4.000, 4.000]" in cc._c0_text(rs)
    g = {k: dict(joint_miss_fall=.1) for k in cc.SIX + ("sum",)}
    tab = lambda b: {"imag/d25": g, "c0_point": dict(ratio=b["ratio"]), "imag_over_ctx_fall": .5, "c0_interval": b}
    p = cc.predictions({"ETH": tab(rs), "E1V": tab(r), "E1R2": tab(rs)})
    assert p["P3"]["per_pair"] == {"E1V<ETH": None, "E1R2<ETH": False} and p["P3"]["held"] is None
    assert p["P3_difference"]["per_pair"] == {"E1V<ETH": True, "E1R2<ETH": False}


def test_which_side_reproduces_the_points():
    strata, cf, cn = _pairs()
    one = lambda v, P=1: np.full((P, 24), float(v))
    met = lambda u, e, P=1: {g: dict(e=one(e / 2, P), e_phys=one(e, P), u=one(u, P), coverage=one(.5, P),
                                     outside=one(0, P)) for g in cc.GROUPS}
    boot = (strata, cf, cn, 200, 0)
    # ETH, 26 Nov, sum group, d25: geomean u ctx 12.546 / 5.35, imag 3.250 / 2.95
    table = {"ctx/d25": cc.shares(met(12.546, .8), met(5.35, .3), met(5.35, .3), boot),
             "imag/d25": cc.shares(met(3.25, 1.2, 5), met(2.95, .5, 5), met(2.95, .5, 5), boot)}
    ws = cc.which_side(table, (0, 25))
    assert list(ws) == ["d25"]
    r = ws["d25"]
    assert r["H_fall"] == pytest.approx(1.351, abs=5e-4) and r["H_nonfall"] == pytest.approx(np.log(5.35 / 2.95))
    assert r["ctx"]["log_u_diff"] == pytest.approx(np.log(12.546 / 5.35)) and r["ctx"]["log_u_diff_ci"] == pytest.approx([np.log(12.546 / 5.35)] * 2)
    assert r["imag"]["e_phys_fall"] == pytest.approx(1.2) and r["imag"]["e_phys_diff_ci"] == pytest.approx([.7, .7])
    line = "\n".join(cc._side_lines("ETH", "pooled", ws))
    assert "geomean u ctx 12.55/5.35 imag 3.25/2.95 | H fall 1.351" in line


def test_leave_one_head_out_known_answers():
    strata, cf, cn = _pairs()
    W, D = 24, 2
    stats = {k: v.astype(np.float64) for k, v in STATS.items()}
    c = np.array([.1, -.1, .2, 5.0, -.2])                   # head 3 far off: the 5-head mean misses by 1 (normalized)
    ctx = {k: np.broadcast_to(s * c[None, None, :, None, None], (1, D, 5, W, 45)).copy() for k, s in (("fall", 2.), ("nonfall", 1.))}
    imag = {k: np.broadcast_to(ctx[k] * (.5 if k == "fall" else 1.), (5, D, 5, W, 45)).copy() for k in ctx}
    r = cc.leave_one_out({k: dict(ctx=ctx[k], imag=imag[k]) for k in ctx}, {k: np.zeros((W, D, 45)) for k in ctx},
                         stats, (16, 25), (strata, cf, cn, 200, 0))
    assert set(r) == {"all"} | {f"without_h{i}" for i in range(5)}
    e = lambda s, k: r[s]["e_phys"]["d25"]["sum"][k]
    assert e("all", "fall") == pytest.approx(4.0) and e("all", "nonfall") == pytest.approx(2.0)     # state_std 2
    assert e("without_h3", "fall") == pytest.approx(0, abs=1e-12) and e("without_h0", "nonfall") == pytest.approx(2 * 4.9 / 4)
    assert r["without_h3"]["e_phys"]["d16"]["gravity"]["nonfall"] == pytest.approx(0, abs=1e-12)
    for s in r:            # spread halves in imagination on falls only: fall - non-fall H = log 2 for every 4-head set
        for d in (16, 25):
            w = r[s][f"warning_d{d}"]
            assert w["difference"] == pytest.approx(np.log(2)) and w["difference_ci"] == pytest.approx([np.log(2)] * 2)
            assert w["auroc_u_ctx"] == 1.0 and w["pairs"] == 24       # u_ctx(fall) = 2 u_ctx(non-fall) everywhere
    assert any(x.startswith("  LOO ETH batchA without_h3: e_phys sum (d16/d25) fall 0.000/0.000")
               for x in cc._loo_lines("ETH", "batchA", r))
