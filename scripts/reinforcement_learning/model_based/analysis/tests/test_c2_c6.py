import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import c2_attractor as c2
import c6_saturation as c6
from test_c0_battery import identity_contract, random_trace, real_model


def test_knn_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    q, r = rng.normal(size=(37, 5)), rng.normal(size=(200, 5))
    brute = np.sort(np.linalg.norm(q[:, None] - r[None], axis=-1), axis=1)[:, :10].mean(1)
    np.testing.assert_allclose(c2.knn_distance(q, r, 10, chunk=8), brute, rtol=1e-5)


def synthetic_run(folder, attracted=True, P=2, W=6):
    """Reference = walking at the origin. Real paths move away (distance and tilt rise); imagined paths move
    back toward the origin when `attracted`, else away like the real ones."""
    H = 26
    target = np.zeros((H, W, 45), np.float32)
    own = np.zeros((P, H, W, 45), np.float32)
    ramp = np.linspace(0, 1, H)[:, None]
    target[..., 6] = 0.2 + ramp                                # gravity x grows: tilt and distance rise
    own[..., 6] = (1.2 - ramp) if attracted else (0.2 + ramp)
    target[..., 8] = -1.0                                      # gravity z = -1: tilt angle = atan(x)
    own[..., 8] = -1.0
    arrays = {"meta/state_mean": np.zeros(45, np.float32), "meta/state_std": np.ones(45, np.float32)}
    for kind in ("fall", "nonfall"):
        arrays.update({f"s/{kind}/own": own, f"s/{kind}/target": target,
                       f"s/{kind}/clusters": np.array([f"s/{kind}/{i}" for i in range(W)])})
    folder.mkdir(parents=True)
    np.savez(folder / "c0_windows.npz", **arrays)
    return folder


def test_c2_attraction_rule(tmp_path):
    ref = np.zeros((50, 45), np.float32)
    ref[:, 8] = -1.0                                           # upright walking
    yes = c2.analyse_run(synthetic_run(tmp_path / "a"), ref, ["s"], 10, 200, 0)["fall"]
    assert yes["share_attracted"] == 1.0 and yes["verdict"] == "supported"
    assert yes["share_real_dist_rises"] == 1.0 and yes["share_imag_tilt_falls"] == 1.0
    no = c2.analyse_run(synthetic_run(tmp_path / "b", attracted=False), ref, ["s"], 10, 200, 0)["fall"]
    assert no["share_attracted"] == 0.0 and no["verdict"] == "not supported"
    assert set(yes["profile"]["imag_dist"]) == {"d0", "d1", "d8", "d16", "d25"}


def test_c2_reference_loading(tmp_path):
    csv = tmp_path / "w.csv"
    np.savetxt(csv, np.ones((30, 66)), delimiter=",")
    tr = tmp_path / "t.npz"
    np.savez(tr, **random_trace(T=40, N=2, falls=((30, 0),), timeouts=()))
    ref = c2.load_reference([csv], [tr], 200, 0)
    assert ref.shape == (30 + 80, 45)
    assert c2.load_reference([csv], [tr], 50, 0).shape == (50, 45)


def test_manual_gru_matches_torch_and_saturates():
    torch.manual_seed(0)
    rnn = torch.nn.GRU(7, 16, num_layers=2, batch_first=True)
    x = torch.randn(3, 9, 7)
    out, shares = c6.gru_gate_saturation(rnn, x)
    torch.testing.assert_close(out, rnn(x)[0], rtol=1e-5, atol=1e-6)
    assert shares.shape == (2, 3, 3, 9) and ((shares >= 0) & (shares <= 1)).all()
    big = torch.nn.GRU(7, 16, num_layers=1, batch_first=True)
    with torch.no_grad():
        for p in big.parameters():
            p.mul_(100)
    assert c6.gru_gate_saturation(big, x)[1].mean() > 0.8
    with pytest.raises(TypeError):
        c6.gru_gate_saturation(torch.nn.LSTM(7, 4, batch_first=True), x)


def test_c6_end_to_end(tmp_path, monkeypatch):
    import analyse_transition_fidelity as fidelity
    import c0_battery as c0
    model, contract = real_model(), identity_contract()
    monkeypatch.setattr(fidelity, "load_model", lambda path: (model, contract))
    p = tmp_path / "t.npz"
    np.savez(p, **random_trace())
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"x")
    c0.run(wm, [("src", p)], tmp_path / "run", n_boot=50)
    assert set(c6.trunks(model)) == {"state_base", "auxiliary_base"}
    res = c6.analyse(model, contract, tmp_path / "run", [("src", p)], n_boot=50)
    assert res["verdict"] in ("supported", "not supported", "inconclusive")
    assert set(res["fall"]["by_depth"]) == {"d0", "d8", "d16", "d25"}
    assert 0 <= res["fall"]["pre_fall_saturation"]["state_trunk"] <= 1
    assert "gravity" in res["h_components_d25"] and isinstance(c6.text("m", res), str)
    json.dumps(c6.json_safe(res), allow_nan=False)


def test_c2_tilt_is_an_angle():
    z = np.zeros((2, 45), np.float32)
    z[0, 6:9] = (0, 0, -1)
    z[1, 6:9] = (0.5, 0, -3)
    np.testing.assert_allclose(c2.tilt(z, np.zeros(45), np.ones(45)), [0, np.degrees(np.arctan2(0.5, 3))], atol=1e-4)
    np.testing.assert_allclose(c2.gravity_norm(z, np.zeros(45), np.ones(45)), [1, np.hypot(0.5, 3)], rtol=1e-6)
