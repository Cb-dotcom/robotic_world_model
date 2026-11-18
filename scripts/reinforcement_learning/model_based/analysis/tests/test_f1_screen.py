import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_screen as f1
import c0_battery as c0
from test_c0_battery import identity_contract, random_trace, real_model


def test_auroc_known_values():
    assert f1.auroc([3, 4, 5], [0, 1, 2]) == 1.0
    assert f1.auroc([0, 1, 2], [3, 4, 5]) == 0.0
    assert f1.auroc([1, 1], [1, 1]) == 0.5
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 300), rng.normal(0, 1, 300)
    brute = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
    assert f1.auroc(a, b) == pytest.approx(brute)


def _c0_run(tmp_path, monkeypatch, **kwargs):
    import analyse_transition_fidelity as fidelity
    model, contract = real_model(), identity_contract()
    monkeypatch.setattr(fidelity, "load_model", lambda path: (model, contract))
    traces = []
    for name, seed in (("batchA", 0), ("batchC", 1)):
        p = tmp_path / f"{name}.npz"
        np.savez(p, **random_trace(T=220, N=4, falls=((69, 0), (120, 2), (180, 3), (200, 1)), seed=seed))
        traces.append((name, p))
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"x")
    out = tmp_path / "run"
    c0.run(wm, traces, out, n_boot=50, **kwargs)
    return out


def test_screen_reads_c0_windows_and_applies_the_rules(tmp_path, monkeypatch):
    out = _c0_run(tmp_path, monkeypatch)
    r = f1.screen(out, n_boot=200)
    assert r["windows"]["fall"] == r["windows"]["nonfall"] > 0
    w, _ = f1.load_run(out)
    # criterion values follow the definitions
    hr = np.exp(np.log(w["fall"]["m"][25] / w["fall"]["r"][25]).mean())
    assert r["healing"]["HR25"] == pytest.approx(hr)
    s = f1.scale(w["nonfall"])
    assert r["size"]["ratio"] == pytest.approx((s[:, None] * w["nonfall"]["m"]).mean() / w["nonfall"]["u"].mean())
    assert r["criteria"]["no_healing"] == (hr >= 1.0)
    assert set(r["auroc_by_lead"]) == {f"lead{x}" for x in range(10, 26)}
    # scale: the realized error equals s(d) x spread on average over non-falls
    np.testing.assert_allclose(s * w["nonfall"]["m"].mean(1), w["nonfall"]["e_norm"].mean(1), rtol=1e-6)


def test_screen_refuses_other_feedback_or_actions(tmp_path, monkeypatch):
    out = _c0_run(tmp_path, monkeypatch, feedback="mean")
    with pytest.raises(ValueError, match="head feedback"):
        f1.screen(out, n_boot=20)
