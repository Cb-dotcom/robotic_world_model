import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import c0_battery as c0
import replay_fall_paths as replay
from pre_reset_transitions import SCHEMA

rsl = pytest.importorskip("rsl_rl.modules")


def real_model(heads=5, seed=0):
    from configs.anymal_d_flat_cfg import AnymalDFlatConfig
    torch.manual_seed(seed)
    mac = AnymalDFlatConfig().model_architecture_config
    return rsl.SystemDynamicsEnsemble(45, 12, 0, 8, 1, "cpu", ensemble_size=heads,
                                     history_horizon=mac.history_horizon,
                                     architecture_config=mac.architecture_config).eval()


class Contract(SimpleNamespace):
    def metadata(self):
        return {"kind": "unit test identity"}


def identity_contract():
    return Contract(statistics=dict(state_mean=torch.zeros(45), state_std=torch.ones(45),
                                    action_mean=torch.zeros(12), action_std=torch.ones(12)))


def random_trace(T=160, N=3, falls=((69, 0), (120, 2)), timeouts=((99, 1),), seed=0):
    rng = np.random.default_rng(seed)
    state = rng.normal(0, .3, (T, N, 45)).astype(np.float32)
    state[:, :, 6:9] += np.array([0, 0, -1], np.float32)
    nxt = np.empty_like(state)
    nxt[:-1] = state[1:]
    nxt[-1] = state[-1]
    done = np.zeros((T, N), bool)
    term = np.zeros((T, N), bool)
    for t, n in falls:
        done[t, n] = term[t, n] = True
    for t, n in timeouts:
        done[t, n] = True
    for t, n in zip(*np.nonzero(done)):
        nxt[t, n] = rng.normal(0, .3, 45)          # terminal next state differs from the reset row
    meta = {"schema": SCHEMA, "verified_terminal_rows": int(done.sum()), "step_dt": .02}
    return dict(state=state, next_state=nxt, action=rng.normal(0, .5, (T, N, 12)).astype(np.float32),
                command=rng.uniform(-1, 1, (T, N, 3)).astype(np.float32), done=done, terminated=term,
                time_out=done & ~term, metadata=np.array(json.dumps(meta)))


def test_forward_all_matches_library_forward_over_recurrent_steps():
    m1 = real_model()
    m2 = copy.deepcopy(m1)
    torch.manual_seed(1)
    xs, xa = torch.randn(3, 32, 45), torch.randn(3, 32, 12)
    ids = torch.full((1, 3, 1), 2, dtype=torch.long)
    m1.reset(); m2.reset()
    for step in range(3):
        mean, _, epi, _, contact, term = m1(xs, xa, ids)
        means, contacts, terms, _ = c0.forward_all(m2, xs, xa)
        torch.testing.assert_close(means[2], mean)
        torch.testing.assert_close(c0.head_spread(means).sum(-1), epi)
        torch.testing.assert_close(contacts[2], contact)
        torch.testing.assert_close(terms[2], term)
        xs, xa = mean[:, None], torch.randn(3, 1, 12)


def test_spreads_equal_the_27_oct_replay_uncertainty():
    """Same algorithm as replay_fall_paths (horizon 25): summed spread = its 'uncertainty'."""
    model, contract, trace = real_model(), identity_contract(), random_trace()
    origins, _ = replay.fall_origins(trace, 25)
    assert len(origins) == 2
    teacher_rows = np.asarray([(t + k, n) for k in range(25) for t, n in origins])
    teacher = replay.predict(model, contract, trace, teacher_rows, 1)[3]          # [heads, rows]
    mine = c0.real_context_spread(model, contract, trace, origins, 25, batch=7).sum(-1)
    for h in range(model.ensemble_size):
        np.testing.assert_allclose(teacher[h].reshape(25, len(origins)), mine, rtol=1e-4, atol=1e-6)
    imagined = replay.imagined_path(model, contract, trace, origins, 25)[3]       # [heads, step, event]
    spread, stop = c0.imagined_spread(model, contract, trace, origins, 25)
    np.testing.assert_allclose(imagined, spread.sum(-1), rtol=1e-4, atol=1e-6)
    assert stop.shape == (model.ensemble_size, 25, len(origins)) and ((stop >= 0) & (stop <= 1)).all()


def test_depth_zero_is_identical_in_both_arms():
    model, contract, trace = real_model(), identity_contract(), random_trace()
    origins, _ = c0.fall_windows(trace)
    u_real = c0.real_context_spread(model, contract, trace, origins, c0.HORIZON)
    u_imag, _ = c0.imagined_spread(model, contract, trace, origins, c0.HORIZON)
    np.testing.assert_allclose(u_imag[:, 0], np.broadcast_to(u_real[0], u_imag[:, 0].shape), rtol=1e-4, atol=1e-6)
    hi0 = c0.log_ratio(u_imag, u_real, "sum")[0]
    assert np.abs(hi0).max() < 1e-4
    assert u_imag.shape == (5, 26, len(origins), 45) and u_real.shape == (26, len(origins), 45)


def test_fall_windows_end_on_the_terminal_row_and_match_replay_rule():
    trace = random_trace()
    origins, counts = c0.fall_windows(trace)
    assert origins.tolist() == [[44, 0], [95, 2]]
    assert counts == dict(physical_falls=2, included=2, short_history_excluded=0)
    np.testing.assert_array_equal(origins, replay.fall_origins(trace, 26)[0])
    trace["done"][70, 2] = True                        # reset inside the second fall's history (rows 64-119)
    assert c0.fall_windows(trace)[0].tolist() == [[44, 0]]


def test_nonfall_candidates_hold_no_reset_from_history_to_survival_margin():
    done = np.zeros((200, 2), bool)
    done[100, 0] = True
    cands = c0.nonfall_candidates(done, stride=1)
    for t, n in cands:
        assert not done[t - 31:t + 26 + 25, n].any()
        assert t + 26 + 25 <= 200
    starts0 = set(cands[cands[:, 1] == 0, 0])
    assert 49 in starts0 and 50 not in starts0          # 49+26+25-1 = 99 < 100; 50 touches row 100
    assert 132 in starts0 and 131 not in starts0         # history must start after row 100
    assert max(cands[cands[:, 1] == 1, 0]) == 200 - 26 - 25


def test_matching_is_nearest_without_replacement():
    trace = random_trace()
    trace["command"][:] = 0
    trace["state"][:, :, 0:2] = 0
    falls = np.array([[44, 0], [95, 2]])
    cands = np.array([[40, 1], [50, 1], [60, 1]])
    trace["command"][44, 0] = (1, 0, 0)
    trace["command"][95, 2] = (1, 0, 0)
    trace["command"][50, 1] = (1, 0, 0)                  # exact match for both; only one may take it
    trace["command"][60, 1] = (.9, 0, 0)
    chosen, dist = c0.match_nonfalls(trace, falls, cands, seed=0)
    assert sorted(map(tuple, chosen)) == [(50, 1), (60, 1)]
    assert sorted(dist)[0] == 0
    with pytest.raises(ValueError):
        c0.match_nonfalls(trace, falls, cands[:1])


def test_cluster_bootstrap_stratified_point_and_ci():
    values = np.array([1., 1., 3., 3., 10.])
    clusters = np.array(["a", "a", "b", "b", "c"])
    strata = np.array(["s1", "s1", "s1", "s1", "s2"])
    point, boot = c0.cluster_bootstrap(values, clusters, strata, n_boot=4000, seed=1)
    assert point == pytest.approx(values.mean())
    assert boot.shape == (4000,)
    # stratum s2 has one cluster (sum 10, always drawn); s1 draws two of a (sum 2) / b (sum 6), 2 rows each:
    # (4 + 10) / 5, (8 + 10) / 5, (12 + 10) / 5
    assert set(np.round(np.unique(boot), 6)) == {2.8, 3.6, 4.4}
    same, boot2 = c0.cluster_bootstrap(np.full(5, 2.0), clusters, strata, n_boot=100)
    assert same == 2.0 and np.allclose(boot2, 2.0)
    vec, bootv = c0.cluster_bootstrap(np.stack([values, -values], 1), clusters, strata, n_boot=10)
    assert vec.shape == (2,) and bootv.shape == (10, 2) and np.allclose(bootv[:, 0], -bootv[:, 1])


def test_c0_rule_readings():
    tight = lambda x: np.full(1000, x) + np.linspace(-.01, .01, 1000)
    assert c0.c0_verdict(3.0, tight(3.0), 1.0, tight(1.0))["verdict"] == "supported"
    assert c0.c0_verdict(1.0, tight(1.0), 1.0, tight(1.0))["verdict"] == "not supported"
    wide_f = np.linspace(.5, 4, 1000)
    wide_n = np.full(1000, 1.0)
    assert c0.c0_verdict(2.0, wide_f, 1.0, wide_n)["verdict"] == "inconclusive"
    r = c0.c0_verdict(.5, tight(.5), -.1, tight(-.1))
    assert r["verdict"] == "supported" and r["reading"].startswith("H_nonfall <= 0")
    assert c0.c0_verdict(-.5, tight(-.5), -.1, tight(-.1))["verdict"] == "not supported"


def test_reproduction_check_tolerance_and_hashes():
    def report(r25, i1, i25, t="x", w="y"):
        steps = lambda first, last: [dict(uncertainty_mean=first)] + [dict(uncertainty_mean=0)] * 23 + [dict(uncertainty_mean=last)]
        return dict(real_context=dict(steps=steps(0, r25)), imagined_context=dict(steps=steps(i1, i25)),
                    trace_sha256=t, wm_sha256=w)
    old = report(9.78, 5.11, 2.66)
    ok = c0.reproduction_check(report(9.78 * 1.04, 5.11, 2.66 * .97), old)
    assert ok["passed"] and ok["same_input_hashes"]
    bad = c0.reproduction_check(report(9.78, 5.11 * 1.06, 2.66, w="z"), old)
    assert not bad["passed"] and not bad["same_input_hashes"]
    # numbers within tolerance but another model or trace: the gate fails (review 18 Nov, second round)
    other = c0.reproduction_check(report(9.78, 5.11, 2.66, w="z"), old)
    assert other["numbers_within_tolerance"] and not other["same_input_hashes"] and not other["passed"]
    no_hash = report(9.78, 5.11, 2.66)
    no_hash.pop("trace_sha256")
    assert not c0.reproduction_check(no_hash, old)["passed"]
    assert bad["rows"]["imagined_step1"]["rel_diff"] == pytest.approx(.06)


def test_end_to_end_writes_reports_and_refuses_overwrite(tmp_path, monkeypatch):
    import analyse_transition_fidelity as fidelity
    model, contract = real_model(), identity_contract()
    monkeypatch.setattr(fidelity, "load_model", lambda path: (model, contract))
    paths = []
    for i, falls in enumerate((((69, 0), (120, 2)), ((80, 1),))):
        p = tmp_path / f"t{i}.npz"
        np.savez(p, **random_trace(falls=falls, seed=i))
        paths.append((f"src{i}", p))
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"not a real checkpoint")
    out = tmp_path / "c0"
    report, repro = c0.run(wm, paths, out, n_boot=200)
    assert repro is None
    for name in ("c0_report.json", "c0_report.txt", "c0_windows.npz"):
        assert (out / name).is_file()
    saved = json.loads((out / "c0_report.json").read_text())
    assert set(saved["report"]) >= {"pooled", "src0", "src1"}
    pooled = saved["report"]["pooled"]
    assert pooled["fall"]["windows"] == 3 and pooled["nonfall"]["windows"] == 3
    assert pooled["hr0_check"]["passed"]
    assert pooled["c0_rule_d25_sum"]["verdict"] in ("supported", "not supported", "inconclusive")
    row = pooled["fall"]["rows"]["d25/gravity"]
    assert row["H"] == pytest.approx(-row["HI"]) and row["HR"] == pytest.approx(np.exp(row["HI"]))
    with np.load(out / "c0_windows.npz") as arrays:
        assert arrays["src0/fall/u_imag"].shape == (5, 26, 2, 45)
    with pytest.raises(FileExistsError):
        c0.run(wm, paths, out, n_boot=10)
    monkeypatch.setattr(sys, "argv", ["c0_battery.py", "--wm", str(wm), "--trace", f"a={paths[0][1]}", "--out", str(out)])
    with pytest.raises(SystemExit):
        c0.main()


def test_errors_and_member_spread_match_replay_paths():
    model, contract, trace = real_model(), identity_contract(), random_trace()
    origins, _ = c0.fall_windows(trace)
    H = c0.HORIZON
    real = c0.real_context_pass(model, contract, trace, origins, H, batch=5)
    rows = np.asarray([(t + k, n) for k in range(H) for t, n in origins])
    pred = replay.predict(model, contract, trace, rows, 1)[0]                          # [heads, rows, 45]
    target = trace["next_state"][rows[:, 0], rows[:, 1]]
    np.testing.assert_allclose(((pred - target[None]) ** 2).mean(-1).mean(0).reshape(H, -1), real["se_norm"], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(((pred - target[None])[..., :3] ** 2).mean(-1).mean(0).reshape(H, -1), real["se_vel"], rtol=1e-4, atol=1e-6)
    imag = c0.imagined_pass(model, contract, trace, origins, H)
    states = replay.imagined_path(model, contract, trace, origins, H)[0]               # [heads, H, W, 45]
    np.testing.assert_allclose(imag["own"], states, rtol=1e-4, atol=1e-5)
    tgt = c0.gather(trace["next_state"], origins, 0, H).transpose(1, 0, 2)
    np.testing.assert_allclose(imag["se_norm"], ((states - tgt[None]) ** 2).mean(-1), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(imag["own"].std(axis=0, ddof=1), states.std(axis=0, ddof=1), rtol=1e-4, atol=1e-5)


def test_model_set_reads_single_head_models_as_one_ensemble(monkeypatch):
    ms = [real_model(heads=1, seed=s) for s in range(3)]
    copies = [copy.deepcopy(m) for m in ms]
    group = c0.ModelSet(ms)
    assert group.ensemble_size == 3
    torch.manual_seed(2)
    xs, xa = torch.randn(4, 32, 45), torch.randn(4, 32, 12)
    group.reset()
    for m in copies:
        m.reset()
    means, contacts, terms, stds = c0.forward_all(group, xs, xa)
    assert means.shape == (3, 4, 45) and contacts.shape == (3, 4, 8) and terms.shape == (3, 4, 1)
    for i, m in enumerate(copies):
        torch.testing.assert_close(means[i], m(xs, xa, torch.zeros((1, 4, 1), dtype=torch.long))[0])
    contract, trace = identity_contract(), random_trace()
    origins, _ = c0.fall_windows(trace)
    out = c0.imagined_pass(group, contract, trace, origins, 6)
    assert out["spread"].shape == (3, 6, len(origins), 45) and np.isfinite(out["spread"]).all()
    import analyse_transition_fidelity as fidelity
    stats = {0: identity_contract(), 1: identity_contract(), 2: identity_contract()}
    stats[2].statistics["state_std"] = torch.full((45,), 2.0)
    monkeypatch.setattr(fidelity, "load_model", lambda p: (real_model(heads=1), stats[int(Path(p).stem[-1])]))
    model, contract, paths = c0.load_models("m0.pt,m1.pt")
    assert isinstance(model, c0.ModelSet) and len(paths) == 2
    with pytest.raises(ValueError, match="different normalization"):
        c0.load_models("m0.pt,m2.pt")


def test_cold_context_changes_the_imagined_arm_only():
    model, contract, trace = real_model(), identity_contract(), random_trace()
    origins, _ = c0.fall_windows(trace)
    warm = c0.imagined_pass(model, contract, trace, origins, 4)
    same = c0.imagined_pass(model, contract, trace, origins, 4, context=32)
    cold = c0.imagined_pass(model, contract, trace, origins, 4, context=1)
    np.testing.assert_array_equal(warm["spread"], same["spread"])
    assert np.abs(cold["spread"] - warm["spread"]).max() > 1e-4
    for bad in (0, 33):
        with pytest.raises(ValueError):
            c0.imagined_pass(model, contract, trace, origins, 4, context=bad)


def test_head_subsets_spread_and_membership():
    subsets = c0.make_subsets(5, (2, 5, 10), n=7, seed=0)
    assert set(subsets) == {2} and subsets[2].shape == (7, 2)
    means = torch.randn(5, 3, 45)
    got = c0.subset_spread(means, subsets)[2]
    for j, idx in enumerate(subsets[2]):
        np.testing.assert_allclose(got[j], means[torch.from_numpy(idx)].std(0).sum(-1).numpy(), rtol=1e-6)
    model, contract, trace = real_model(), identity_contract(), random_trace()
    origins, _ = c0.fall_windows(trace)
    imag = c0.imagined_pass(model, contract, trace, origins, c0.HORIZON, subsets=subsets)
    sub = imag["subsets"][2]                                                            # [n, heads, D, W]
    for j, idx in enumerate(subsets[2]):
        for h in range(5):
            assert np.isfinite(sub[j, h]).all() == (h in idx)


def test_summary_with_cold_context_and_subsets(tmp_path, monkeypatch):
    import analyse_transition_fidelity as fidelity
    model, contract = real_model(), identity_contract()
    monkeypatch.setattr(fidelity, "load_model", lambda path: (model, contract))
    p = tmp_path / "t.npz"
    np.savez(p, **random_trace())
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"x")
    report, _ = c0.run(wm, [("src", p)], tmp_path / "cold", n_boot=100, context=1, head_subsets=(2,))
    pooled = report["pooled"]
    assert pooled["hr0_check"]["passed"] is None and "not applicable" in pooled["hr0_check"]["note"]
    assert set(pooled["fall"]["c8"]) == {"K2", "K_all"} and pooled["fall"]["c8"]["K_all"]["heads"] == 5
    assert pooled["fall"]["c1a"]["ratio"] > 0 and "d25" in pooled["fall"]["errors"]
    with np.load(tmp_path / "cold" / "c0_windows.npz") as arrays:
        assert {"meta/state_std", "src/fall/u_member", "src/fall/se_imag_vel", "src/fall/sub_imag_K2"} <= set(arrays.files)
    monkeypatch.setattr(sys, "argv", ["c0", "--wm", str(wm), "--trace", f"a={p}", "--out", str(tmp_path / "x"),
                                      "--head-subsets", "two"])
    with pytest.raises(SystemExit):
        c0.main()


def test_feedback_modes():
    model, contract, trace = real_model(), identity_contract(), random_trace()
    origins, _ = c0.fall_windows(trace)
    head = c0.imagined_pass(model, contract, trace, origins, 3)
    mean = c0.imagined_pass(model, contract, trace, origins, 3, feedback="mean")
    tilted = c0.imagined_pass(model, contract, trace, origins, 3, feedback="most_tilted")
    rand = c0.imagined_pass(model, contract, trace, origins, 3, feedback="random_step")
    s1 = c0.imagined_pass(model, contract, trace, origins, 3, feedback="sampled", seed=1)
    s1b = c0.imagined_pass(model, contract, trace, origins, 3, feedback="sampled", seed=1)
    assert mean["own"].shape[0] == tilted["own"].shape[0] == 1 and rand["own"].shape[0] == s1["own"].shape[0] == 5
    # step 0 sees the same real inputs in every mode
    np.testing.assert_allclose(mean["own"][0, 0], head["own"][:, 0].mean(0), rtol=1e-5, atol=1e-6)
    tilt = c0.tilt_deg(torch.from_numpy(head["own"][:, 0][..., 6:9])).numpy()           # identity normalization
    pick = tilt.argmax(0)
    np.testing.assert_allclose(tilted["own"][0, 0], head["own"][pick, 0, np.arange(len(origins))], rtol=1e-5, atol=1e-6)
    for p in range(5):
        for w in range(len(origins)):
            assert any(np.allclose(rand["own"][p, 0, w], head["own"][h, 0, w], atol=1e-6) for h in range(5))
    np.testing.assert_array_equal(s1["own"], s1b["own"])
    assert np.abs(s1["own"][:, 0] - head["own"][:, 0]).max() > 1e-4
    np.testing.assert_allclose(head["ratio"][:, 0] * head["spread"][:, 0], np.abs(head["own"][:, 0] - (
        c0.gather(trace["next_state"], origins, 0, 3).transpose(1, 0, 2)[None, 0])), rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError):
        c0.imagined_pass(model, contract, trace, origins, 3, feedback="best")


def test_single_path_mode_summary_is_json_safe(tmp_path, monkeypatch):
    import analyse_transition_fidelity as fidelity
    model, contract = real_model(), identity_contract()
    monkeypatch.setattr(fidelity, "load_model", lambda path: (model, contract))
    p = tmp_path / "t.npz"
    np.savez(p, **random_trace())
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"x")
    report, _ = c0.run(wm, [("src", p)], tmp_path / "mean", n_boot=50, feedback="mean")
    pooled = report["pooled"]
    assert set(pooled["c5_scale"]["s"]) == {f"d{d}" for d in c0.DEPTHS}
    assert 0 <= pooled["fall"]["c5"]["reach_termination"] <= 1
    saved = json.loads((tmp_path / "mean" / "c0_report.json").read_text())
    assert saved["meta"]["feedback"] == "mean" and saved["report"]["pooled"]["fall"]["c1a"]["ratio"] is None
    with np.load(tmp_path / "mean" / "c0_windows.npz") as z:
        assert z["src/fall/ratio"].shape == (1, 26, 2, 45) and z["src/fall/target"].shape == (26, 2, 45)


def test_held_actions_free_run():
    model, contract, trace = real_model(), identity_contract(), random_trace()
    origins, _ = c0.fall_windows(trace)
    rec = c0.imagined_pass(model, contract, trace, origins, 5)
    held = c0.imagined_pass(model, contract, trace, origins, 5, actions_mode="held")
    np.testing.assert_array_equal(rec["own"][:, 0], held["own"][:, 0])                  # step 0 uses the context only
    assert np.abs(rec["own"][:, 1:] - held["own"][:, 1:]).max() > 1e-5
    const = random_trace()
    const["action"][:] = const["action"][0:1]                                           # every row the same action
    a = c0.imagined_pass(model, contract, const, origins, 5)
    b = c0.imagined_pass(model, contract, const, origins, 5, actions_mode="held")
    np.testing.assert_array_equal(a["own"], b["own"])
    with pytest.raises(ValueError):
        c0.imagined_pass(model, contract, trace, origins, 5, actions_mode="zero")


def test_tilt_angle_any_vector_length():
    g = torch.tensor([[0, 0, -1.0], [1.0, 0, 0], [0, 0, 1.0], [0.5, 0, -3.0], [0, 0, -0.2]])
    np.testing.assert_allclose(c0.tilt_deg(g).numpy(), [0, 90, 180, np.degrees(np.arctan2(0.5, 3)), 0], atol=1e-4)


def test_spearman_ties_and_constant_input():
    np.testing.assert_array_equal(c0.average_ranks([3, 1, 3, 2]), [3.5, 1, 3.5, 2])
    assert np.isnan(c0.spearman(np.ones(10), np.ones(10)))
    assert np.isnan(c0.spearman(np.arange(10), np.ones(10)))
    assert np.isnan(c0.spearman([1, 2], [2, 3]))
    assert c0.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)
    assert c0.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    a = np.array([1, 1, 2, 3, 3, 3]); b = np.array([2, 1, 3, 3, 4, 5])
    ra, rb = c0.average_ranks(a), c0.average_ranks(b)
    assert c0.spearman(a, b) == pytest.approx(np.corrcoef(ra, rb)[0, 1])


def test_pair_bootstrap_resamples_pairs_jointly():
    rng = np.random.default_rng(0)
    f = rng.normal(1.0, 0.3, 40)
    n = 0.5 * f + rng.normal(0, 0.05, 40)                  # matched partner, not an exact copy
    strata = np.array(["a"] * 25 + ["b"] * 15)
    fc = np.array([f"{s}/f{i}" for i, s in enumerate(strata)])
    nc = np.array([f"{s}/n{i}" for i, s in enumerate(strata)])  # no shared episode: components = pairs
    bf, bn = c0.pair_bootstrap(f, n, strata, fc, nc, 3000, 0)
    lo, hi = c0.ci(bf / bn)
    assert hi - lo > 1e-3                                  # not a degenerate zero-width interval
    assert lo < f.mean() / n.mean() < hi
    # one draw: both groups use the same pairs
    one = np.array(["s"] * 4)
    bf1, bn1 = c0.pair_bootstrap(np.arange(4.0), 10 * np.arange(4.0), one, [f"s/f{i}" for i in range(4)],
                                 [f"s/n{i}" for i in range(4)], 1, 3)
    assert bn1[0] == pytest.approx(10 * bf1[0])
    with pytest.raises(ValueError):
        c0.pair_bootstrap(f, n[:-1], strata, fc, nc[:-1], 10, 0)


def test_pair_components_join_shared_episodes_transitively():
    # pairs 0 and 1 share non-fall episode x; pair 2's non-fall episode is pair 1's fall episode; pair 3 alone
    lab = c0.pair_components(["s/a", "s/b", "s/c", "s/d"], ["s/x", "s/x", "s/b", "s/y"])
    assert lab[0] == lab[1] == lab[2] and lab[3] != lab[0]
    # one component only: every draw is the full sample, so the interval has zero width
    bf, bn = c0.pair_bootstrap(np.array([1.0, 2, 3]), np.array([1.0, 1, 1]), np.array(["s"] * 3),
                               ["s/a", "s/b", "s/c"], ["s/x", "s/x", "s/b"], 200, 0)
    assert np.ptp(bf) == 0 and bf[0] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="crosses sources"):
        c0.pair_bootstrap(np.ones(2), np.ones(2), np.array(["a", "b"]), ["e1", "e2"], ["e3", "e3"], 10, 0)


def test_pair_bootstrap_widens_when_nonfall_episodes_are_shared():
    # 40 pairs; non-fall windows come from 4 episodes with strong episode effects (10 windows each)
    rng = np.random.default_rng(1)
    f = rng.normal(2.0, 0.1, 40)
    ep = np.repeat(np.arange(4), 10)
    n = np.array([0.4, 0.8, 1.2, 1.6])[ep] + rng.normal(0, 0.02, 40)
    st = np.array(["s"] * 40)
    fc = [f"s/f{i}" for i in range(40)]
    shared = [f"s/n{e}" for e in ep]
    indep = [f"s/n{i}" for i in range(40)]
    w_shared = np.ptp(c0.ci(np.divide(*c0.pair_bootstrap(f, n, st, fc, shared, 4000, 0))))
    w_indep = np.ptp(c0.ci(np.divide(*c0.pair_bootstrap(f, n, st, fc, indep, 4000, 0))))
    assert w_shared > 2 * w_indep


def test_zero_disagreement_stops():
    u_real = np.ones((26, 3, 45))
    u_imag = np.ones((2, 26, 3, 45))
    u_imag[0, 5, 1, :] = 0
    with pytest.raises(c0.ZeroDisagreement, match="windows \\[1\\]"):
        c0.log_ratio(u_imag, u_real, "sum")


def test_gates_stop_the_run(tmp_path, monkeypatch):
    import analyse_transition_fidelity as fidelity
    import replay_fall_paths as replay
    model, contract = real_model(), identity_contract()
    monkeypatch.setattr(fidelity, "load_model", lambda path: (model, contract))
    p = tmp_path / "t.npz"
    np.savez(p, **random_trace())
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"x")
    def report(r25):
        steps = lambda first, last: [dict(uncertainty_mean=first)] + [dict(uncertainty_mean=1.0)] * 23 + [dict(uncertainty_mean=last)]
        return dict(real_context=dict(steps=steps(1.0, r25)), imagined_context=dict(steps=steps(5.0, 2.0)),
                    trace_sha256="x", wm_sha256="y")
    def fake_run(trace, wm_path, out):
        out.mkdir(parents=True)
        (out / "paired_fall_replay.json").write_text(json.dumps(report(11.0)))
    monkeypatch.setattr(replay, "run", fake_run)
    old = tmp_path / "old.json"
    old.write_text(json.dumps(report(9.78)))
    with pytest.raises(c0.GateFailed, match="reproduction"):
        c0.run(wm, [("src", p)], tmp_path / "g1", repro_trace=p, repro_json=old, n_boot=20)
    assert (tmp_path / "g1" / "c0_report.txt").is_file()          # evidence kept
    real_summarize = c0.summarize
    def bad_summary(*args, **kwargs):
        rep = real_summarize(*args, **kwargs)
        rep["pooled"]["hr0_check"]["passed"] = False
        return rep
    monkeypatch.setattr(c0, "summarize", bad_summary)
    with pytest.raises(c0.GateFailed, match="HR\\(0\\)"):
        c0.run(wm, [("src", p)], tmp_path / "g2", n_boot=20)


def _policy_setup(tmp_path, seed=0, normalizer=False):
    import policy_loop as pl
    stats = {k: v for k, v in identity_contract().statistics.items()}
    ck = pl.save_actor(tmp_path / f"policy_{seed}.pt", seed=seed, normalizer=normalizer)
    policy = pl.Policy(ck)
    builder = pl.ObservationBuilder("anymal_d_flat", stats)
    tr = pl.consistent_policy_trace(random_trace(), policy, builder, stats)
    meta = json.loads(str(tr["metadata"]))
    meta.update(checkpoint=str(ck), checkpoint_sha256=pl.sha256(ck), task="Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0",
                policy_noise=False, clip_actions=None)
    tr["metadata"] = np.array(json.dumps(meta))
    return pl, stats, policy, builder, tr, meta


def test_policy_loader_and_checks(tmp_path):
    pl, stats, policy, builder, tr, meta = _policy_setup(tmp_path, normalizer=True)
    assert policy.norm is not None and policy.obs_dim == 48
    c = pl.policy_checks("s", tr, meta, policy, builder, stats, rows=200)
    assert c["passed"] and c["max_abs_action_error"] < 1e-5 and max(c["obs_mean_abs_error"].values()) < 1e-6
    # the observation layout is the imagination env's: command in 9:12, previous action in 36:48
    np.testing.assert_allclose(tr["policy_observation"][5, 1, 9:12], tr["command"][5, 1], atol=1e-6)
    np.testing.assert_allclose(tr["policy_observation"][5, 1, 36:48], tr["action"][4, 1], atol=1e-6)
    other = pl.Policy(pl.save_actor(tmp_path / "other.pt", seed=9))
    with pytest.raises(pl.PolicyMismatch, match="reproduce the recorded actions"):
        pl.policy_checks("s", tr, meta, other, builder, stats, rows=200)
    bad = dict(tr, command=tr["command"] + 0.5)                   # commands that the policy never saw
    with pytest.raises(pl.PolicyMismatch, match="rebuilt observation"):
        pl.policy_checks("s", bad, meta, policy, builder, stats, rows=200)
    with pytest.raises(pl.PolicyMismatch, match="sha256"):
        pl.resolve_policy("s", dict(meta, checkpoint_sha256="0" * 64), {})


def test_closed_loop_pass_feeds_the_policy_its_own_path(tmp_path):
    pl, stats, policy, builder, tr, meta = _policy_setup(tmp_path)
    model, contract = real_model(), identity_contract()
    origins = c0.fall_windows(tr)[0]
    calls = []

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, state, last_a, command):
            calls.append((state.clone(), last_a.clone(), command.clone()))
            return self.inner(state, last_a, command)

    rec = c0.imagined_pass(model, contract, tr, origins, 6)
    clo = c0.imagined_pass(model, contract, tr, origins, 6, actions_mode="policy", policy=policy, builder=Spy(builder))
    np.testing.assert_allclose(clo["spread"][:, 0], rec["spread"][:, 0], atol=1e-6)    # first step: real action
    assert not np.allclose(clo["spread"][:, 3], rec["spread"][:, 3])                    # then the policy acts
    t0 = origins[:, 0]
    first_state, first_last, first_cmd = calls[0]                                       # path 0, step 0
    np.testing.assert_allclose(first_state.numpy(), rec["own"][0, 0], atol=1e-6)       # its own imagined state
    np.testing.assert_allclose(first_last.numpy(), tr["action"][t0, origins[:, 1]], atol=1e-6)
    np.testing.assert_allclose(first_cmd.numpy(), tr["command"][t0 + 1, origins[:, 1]], atol=1e-6)
    _, second_last, _ = calls[1]
    np.testing.assert_allclose(second_last.numpy(), policy(builder(*calls[0])).numpy(), atol=1e-6)
    assert np.isfinite(clo["act_dev"][:, :5]).all() and np.isnan(clo["act_dev"][:, 5]).all()
    with pytest.raises(ValueError, match="needs a policy"):
        c0.imagined_pass(model, contract, tr, origins, 6, actions_mode="policy")


def test_closed_loop_run_writes_policy_checks(tmp_path, monkeypatch):
    import analyse_transition_fidelity as fidelity
    pl, stats, policy, builder, tr, meta = _policy_setup(tmp_path)
    model, contract = real_model(), identity_contract()
    monkeypatch.setattr(fidelity, "load_model", lambda path: (model, contract))
    p = tmp_path / "t.npz"
    np.savez(p, **tr)
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"x")
    c0.run(wm, [("src", p)], tmp_path / "out", n_boot=20, actions_mode="policy")
    rep = json.loads((tmp_path / "out" / "c0_report.json").read_text())
    assert rep["meta"]["actions"] == "policy" and rep["meta"]["policy"]["src"]["checks"]["passed"]
    assert "Closed loop" in (tmp_path / "out" / "c0_report.txt").read_text()
    z = np.load(tmp_path / "out" / "c0_windows.npz")
    assert "src/fall/act_dev" in z.files



def test_noise_free_observation_blocks_are_checked_even_with_noise(tmp_path):
    pl, stats, policy, builder, tr, meta = _policy_setup(tmp_path)
    noisy_meta = dict(meta, policy_noise=True)
    rng = np.random.default_rng(0)
    obs = tr["policy_observation"].copy()
    obs[..., 24:36] += rng.uniform(-1.5, 1.5, obs[..., 24:36].shape)        # joint-velocity noise only
    noisy = dict(tr, policy_observation=obs)
    noisy["action"] = policy(torch.from_numpy(obs.reshape(-1, 48))).numpy().reshape(tr["action"].shape)
    c = pl.policy_checks("s", noisy, noisy_meta, policy, builder, stats, rows=300)
    assert c["passed"] and c["obs_blocks_checked"] == ["command", "last_action"]
    assert c["obs_mean_abs_error"]["joint_vel"] == pytest.approx(0.75, abs=0.1)   # E|U(-1.5, 1.5)|
    bad = dict(noisy, command=noisy["command"] + 0.3)                       # commands carry no noise
    with pytest.raises(pl.PolicyMismatch, match="command"):
        pl.policy_checks("s", bad, noisy_meta, policy, builder, stats, rows=300)


def test_policy_recheck_cli(tmp_path, monkeypatch, capsys):
    import analyse_transition_fidelity as fidelity
    pl, stats, policy, builder, tr, meta = _policy_setup(tmp_path)
    monkeypatch.setattr(fidelity, "load_model", lambda path: (real_model(), identity_contract()))
    p = tmp_path / "t.npz"
    np.savez(p, **tr)
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", ["policy_loop.py", "--wm", str(wm), "--trace", f"src={p}", "--out", str(tmp_path / "c.json")])
    pl.main()
    assert "all checks passed" in capsys.readouterr().out
    assert json.loads((tmp_path / "c.json").read_text())["src"]["checks"]["passed"]
