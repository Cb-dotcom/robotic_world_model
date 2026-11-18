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
        means, contacts, terms = c0.forward_all(m2, xs, xa)
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
