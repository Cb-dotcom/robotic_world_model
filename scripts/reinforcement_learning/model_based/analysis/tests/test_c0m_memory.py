"""C0-M (E29): the carried recorded reading, its gates, and the exact decomposition H = H_mem + H_src."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip("rsl_rl.modules")

import c0_battery as c0  # noqa: E402
import c0m_memory as cm  # noqa: E402
from test_c0_battery import identity_contract, random_trace, real_model  # noqa: E402
from test_c0_supplement import c0_run  # noqa: E402


def test_known_answers_isolate_each_part():
    r = cm.known_answer_checks()
    assert r["passed"], r
    assert r["last_input"]["max_abs_H_mem"] < 1e-6 and r["last_input"]["max_abs_H_src"] > 1e-2
    assert r["memory_length"]["max_abs_H_src"] < 1e-6 and r["memory_length"]["max_abs_H_mem"] > 1e-2
    for kind in ("last_input", "memory_length"):
        assert r[kind]["depth0_parity"] < 1e-6


def test_carried_reading_feeds_the_recorded_rows_of_the_imagined_schedule():
    """A real ensemble: at d = 0 all three readings agree; at d >= 1 u_car differs from u_ctx (memory) and from
    u_imag (states). Feeding the imagined arm the recorded states reproduces u_car exactly."""
    model, contract = real_model(), identity_contract()
    tr = random_trace()
    data = {k: tr[k] for k in ("state", "action", "next_state")}
    origins = np.array([(40, 0), (60, 1), (80, 2)])
    car = cm.carried_pass(model, contract, data, origins)
    ctx = c0.real_context_pass(model, contract, data, origins, c0.HORIZON)
    imag = c0.imagined_pass(model, contract, data, origins, c0.HORIZON)
    assert cm.depth0_parity(ctx["spread"], car["spread"], imag["spread"]) < 1e-5
    assert not np.array_equal(car["spread"][5], ctx["spread"][5])      # memory 37 rows vs 32 (small for an untrained GRU)
    assert not np.allclose(car["spread"][5], imag["spread"][0, 5])
    # an independent step-by-step loop over the library forward gives the same carried spreads
    import torch
    st = c0._normalized(contract, data, origins, -31, c0.HORIZON, "state")
    ac = c0._normalized(contract, data, origins, -31, c0.HORIZON, "action")
    model.reset()
    xs, xa, ref = st[:, :32], ac[:, :32], []
    for k in range(c0.HORIZON):
        means, _, _, _ = c0.forward_all(model, xs, xa)
        ref.append(means.std(0).numpy())
        xs, xa = st[:, 32 + k:33 + k], ac[:, 32 + k:33 + k]
    np.testing.assert_allclose(car["spread"], np.stack(ref), rtol=1e-6)
    # the recorded counterpart of the imagined state: next_state[t + k] == state[t + k + 1] inside an episode
    assert np.array_equal(c0.gather(data["state"], origins, 1, c0.HORIZON),
                          c0.gather(data["next_state"], origins, 0, c0.HORIZON - 1))
    assert torch.is_tensor(st)


def test_decomposition_is_exact_and_uses_the_registered_draws():
    rng = np.random.default_rng(0)
    W, H = 30, 26
    logs = {k: (rng.normal(1, .2, (H, W)), rng.normal(.8, .2, (H, W)), rng.normal(.2 if k == "fall" else .5, .2, (H, W)))
            for k in ("fall", "nonfall")}
    strata = np.array(["a"] * 15 + ["b"] * 15)
    clusters = {"fall": np.array([f"{s}/f{i}" for i, s in enumerate(strata)]),
                "nonfall": np.array([f"{s}/n{i}" for i, s in enumerate(strata)])}
    res = cm.decompose(logs, strata, clusters, n_boot=500, seed=0)
    for d in ("d25", "d16"):
        for kind in ("fall", "nonfall"):
            x = res[d][kind]
            assert abs(x["identity_error"]) < 1e-12
            dd = int(d[1:])
            assert x["H"] == pytest.approx(float((logs[kind][0][dd] - logs[kind][2][dd]).mean()))
    # the H draws equal c0's pair bootstrap of the registered statistic (same seed, same structure)
    d = 25
    reg = {k: logs[k][0][d] - logs[k][2][d] for k in logs}
    rf, rn = c0.pair_bootstrap(reg["fall"], reg["nonfall"], strata, clusters["fall"], clusters["nonfall"], 500, 0)
    assert res["d25"]["fall"]["H_ci"] == pytest.approx(c0.ci(rf))
    assert res["d25"]["nonfall"]["H_ci"] == pytest.approx(c0.ci(rn))


def _row(ratio, rci, fci, nci):
    return dict(src_ratio=dict(ratio=ratio, ratio_ci=rci), fall=dict(H_src_ci=fci), nonfall=dict(H_src_ci=nci),
                unstable_denominator=nci[0] <= 0 <= nci[1])


def test_src_rule_readings():
    assert cm.src_verdict(_row(2.0, [1.2, 3.0], [.2, .5], [.05, .2])) == "SURVIVES"
    assert cm.src_verdict(_row(1.2, [0.9, 1.4], [.2, .5], [.1, .3])) == "NOT SUPPORTED"
    assert cm.src_verdict(_row(2.0, [1.2, 9.0], [-.1, 0.0], [.05, .2])) == "NOT SUPPORTED"
    assert cm.src_verdict(_row(2.0, [0.8, 3.0], [.2, .5], [.05, .2])) == "INCONCLUSIVE"
    assert cm.src_verdict(_row(3.0, [1.3, 40.0], [.2, .5], [-.01, .2])) == "INCONCLUSIVE"


def test_zero_crossing_denominator_is_inconclusive_before_any_ratio_rule():
    """Regression (decision 5, 25 Nov 2026): s53 returned NOT SUPPORTED here because it read the ratio bar first.
    A non-fall H_src interval that includes 0 makes the ratio unusable, whatever the ratio interval says."""
    assert cm.src_verdict(_row(1.0, [-5.0, 1.2], [.2, .5], [-.1, .3])) == "INCONCLUSIVE"     # ratio CI < 1.5
    assert cm.src_verdict(_row(-2.0, [-9.0, 9.0], [-.3, -.1], [-.2, .1])) == "INCONCLUSIVE"   # fall CI <= 0 too


def test_end_to_end_on_registered_style_windows(tmp_path, monkeypatch):
    out = c0_run(tmp_path, monkeypatch, n_boot=300)
    payload = cm.run(out / "c0_windows.npz", out / "c0_report.json", tmp_path / "m", n_boot=300)
    g = payload["gates"]
    assert g["reproduction"]["reproduced"] and g["depth0_parity"]["ok"] and g["known_answers"]["passed"]
    sup = __import__("c0_supplement").analyse(__import__("c0_supplement").load_sources(out / "c0_windows.npz"), 300, 0)
    for d in ("d25", "d16"):
        r = payload["pooled"][d]
        for kind in ("fall", "nonfall"):
            assert r[kind]["H_mem"] + r[kind]["H_src"] == pytest.approx(r[kind]["H"], abs=1e-12)
            assert r[kind]["H"] == pytest.approx(sup[d][kind]["H"], rel=1e-12)       # same H as the supplement
    assert payload["reading"] is None and "READING: not made" in (tmp_path / "m" / "c0m.txt").read_text()
    with np.load(tmp_path / "m" / "c0m_windows.npz") as z:
        assert z["src0/fall/u_car"].shape == (26, 2, 45)
    with pytest.raises(FileExistsError):
        cm.run(out / "c0_windows.npz", out / "c0_report.json", tmp_path / "m", n_boot=300)


def test_gates_stop_before_any_reading(tmp_path, monkeypatch):
    out = c0_run(tmp_path, monkeypatch, n_boot=300)
    # a changed trace: gate 1 stops, nothing is read
    rep = json.loads((out / "c0_report.json").read_text())
    src = rep["meta"]["inputs"]["src0"]["path"]
    t = dict(np.load(src))
    t["state"] = t["state"] + 1e-3
    moved = tmp_path / "moved.npz"
    np.savez(moved, **t)
    with pytest.raises(cm.GateFailed, match="gate 1"):
        cm.run(out / "c0_windows.npz", out / "c0_report.json", tmp_path / "g1", trace_overrides={"src0": moved}, n_boot=300)
    assert not (tmp_path / "g1" / "c0m.json").exists() and (tmp_path / "g1" / "c0m_gates.json").exists()
    # a different model with the same hash check bypassed: depth-0 parity fails (gate 2)
    import analyse_transition_fidelity as fidelity
    other = real_model(seed=7)
    monkeypatch.setattr(fidelity, "load_model", lambda path: (other, identity_contract()))
    with pytest.raises(cm.GateFailed, match="gate 2"):
        cm.run(out / "c0_windows.npz", out / "c0_report.json", tmp_path / "g2", n_boot=300)
    assert not (tmp_path / "g2" / "c0m.json").exists()
    # a broken known-answer check: gate 3
    monkeypatch.setattr(cm, "known_answer_checks", lambda: dict(passed=False))
    with pytest.raises(cm.GateFailed, match="gate 3"):
        cm.run(out / "c0_windows.npz", out / "c0_report.json", tmp_path / "g3", n_boot=300)


def test_cli_requires_a_dated_record_for_a_reading(tmp_path, monkeypatch):
    out = c0_run(tmp_path, monkeypatch, n_boot=300)
    with pytest.raises(SystemExit):
        cm.main(["--windows", str(out / "c0_windows.npz"), "--report", str(out / "c0_report.json"),
                 "--out", str(tmp_path / "x"), "--rule-record", "frozen somewhere"])
    assert cm.main(["--windows", str(out / "c0_windows.npz"), "--report", str(out / "c0_report.json"),
                    "--out", str(tmp_path / "y"), "--n-boot", "300", "--rule-record", "2026-11-25 01:02 CPU rules"]) == 0
    p = json.loads((tmp_path / "y" / "c0m.json").read_text())
    assert p["reading"]["verdict"] in ("SURVIVES", "NOT SUPPORTED", "INCONCLUSIVE")
    assert len(p["reading"]["paired_dH_src_ci"]) == 2
