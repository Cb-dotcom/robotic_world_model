from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import c0_compare as cmp


def write_run(folder, u_real=2.0, u_imag=1.0, se_real=0.04, se_imag=0.09, std=None, origins_shift=0,
              W=4, H=26, heads=2, sources=("s",)):
    folder.mkdir(parents=True)
    arrays = {"meta/state_mean": np.zeros(45), "meta/state_std": np.ones(45) if std is None else std}
    for s in sources:
        for kind in ("fall", "nonfall"):
            k = f"{s}/{kind}/"
            ur = np.full((H, W, 45), u_real) if np.isscalar(u_real) else u_real
            ui = np.full((heads, H, W, 45), u_imag) if np.isscalar(u_imag) else u_imag
            arrays.update({k + "origins": np.column_stack([np.arange(W) * 100 + 40 + origins_shift, np.zeros(W, int)]),
                           k + "u_real": ur, k + "u_imag": ui, k + "u_member": np.ones((H, W, 45)),
                           k + "stop": np.zeros((heads, H, W)),
                           k + "se_real_norm": np.ones((H, W)), k + "se_real_vel": np.full((H, W), se_real),
                           k + "se_imag_norm": np.ones((heads, H, W)), k + "se_imag_vel": np.full((heads, H, W), se_imag),
                           k + "clusters": np.array([f"{s}/{kind}/{i}" for i in range(W)])})
    np.savez(folder / "c0_windows.npz", **arrays)
    return folder


def test_closure_errors_and_e18_split(tmp_path):
    c = cmp.load(write_run(tmp_path / "c"), ["s"])
    t = cmp.load(write_run(tmp_path / "t", u_imag=1.5, se_real=0.0484), ["s"])
    r = cmp.compare(c, t, ["s"], 200, 0)["fall"]
    assert r["H_control"] == pytest.approx(np.log(2))
    assert r["H_treat"] == pytest.approx(np.log(2 / 1.5))
    assert r["closure"] == pytest.approx(1 - np.log(2 / 1.5) / np.log(2)) and r["closure_valid"]
    assert r["dlog_u_real"] == pytest.approx(0) and r["dlog_u_imag"] == pytest.approx(np.log(1.5))
    assert r["se_real_change"] == pytest.approx(0.1) and r["se_d25_change"] == pytest.approx(0)
    assert r["closure_by_component"]["gravity"]["closure"] == pytest.approx(r["closure"])


def test_other_normalizer_is_read_in_control_units(tmp_path):
    std = np.ones(45)
    std[6:9] = 2.0                                       # e.g. C6 widened gravity std
    u_real = np.full((26, 4, 45), 2.0)
    u_imag = np.full((2, 26, 4, 45), 1.5)
    u_real[..., 6:9] /= 2                                # same physical spread, stored in its own units
    u_imag[..., 6:9] /= 2
    c = cmp.load(write_run(tmp_path / "c"), ["s"])
    plain = cmp.compare(c, cmp.load(write_run(tmp_path / "p", u_imag=1.5), ["s"]), ["s"], 50, 0)["fall"]
    wide = cmp.compare(c, cmp.load(write_run(tmp_path / "w", u_real=u_real, u_imag=u_imag, std=std), ["s"]), ["s"], 50, 0)["fall"]
    assert wide["closure"] == pytest.approx(plain["closure"])
    assert wide["H_treat_own_units"] == pytest.approx(plain["H_treat"])   # uniform ratio: same H in any units


def test_closure_needs_control_healing_and_same_windows(tmp_path):
    c0 = cmp.load(write_run(tmp_path / "c", u_imag=2.0), ["s"])        # control does not heal: H = 0
    r = cmp.compare(c0, cmp.load(write_run(tmp_path / "t"), ["s"]), ["s"], 50, 0)["fall"]
    assert r["closure"] is None and not r["closure_valid"]
    shifted = cmp.load(write_run(tmp_path / "x", origins_shift=1), ["s"])
    with pytest.raises(ValueError, match="windows differ"):
        cmp.compare(c0, shifted, ["s"], 10, 0)
    with pytest.raises(KeyError):
        cmp.load(tmp_path / "c", ["other"])


def test_cli_writes_table_with_noise_rows(tmp_path, monkeypatch):
    c = write_run(tmp_path / "c")
    n = write_run(tmp_path / "n", u_imag=1.05)
    t = write_run(tmp_path / "t", u_imag=1.5)
    out = tmp_path / "out"
    argv = ["c0_compare.py", "--control", f"ctrl={c}", "--noise", f"seed2={n}", "--treat", f"c4={t}",
            "--bootstrap", "100", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    cmp.main()
    text = (out / "compare.txt").read_text()
    assert "noise: seed2" in text and "c4" in text
    with pytest.raises(SystemExit):
        cmp.main()                                                    # refuses an existing output


def test_same_model_twice_gives_zero_closure(tmp_path, monkeypatch):
    rsl = pytest.importorskip("rsl_rl.modules")
    import analyse_transition_fidelity as fidelity
    import c0_battery as c0
    from test_c0_battery import identity_contract, random_trace, real_model
    model, contract = real_model(), identity_contract()
    monkeypatch.setattr(fidelity, "load_model", lambda path: (model, contract))
    p = tmp_path / "t.npz"
    np.savez(p, **random_trace())
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"x")
    c0.run(wm, [("src", p)], tmp_path / "a", n_boot=50)
    c0.run(wm, [("src", p)], tmp_path / "b", n_boot=50)
    r = cmp.compare(cmp.load(tmp_path / "a", ["src"]), cmp.load(tmp_path / "b", ["src"]), ["src"], 50, 0)["fall"]
    assert r["H_treat"] == pytest.approx(r["H_control"])
    assert r["dlog_u_real"] == pytest.approx(0, abs=1e-9) and r["se_real_change"] == pytest.approx(0, abs=1e-9)
    if r["closure"] is not None:
        assert r["closure"] == pytest.approx(0, abs=1e-9)


def test_reach_ratio(tmp_path):
    c = write_run(tmp_path / "c")
    t = write_run(tmp_path / "t")
    z = dict(np.load(t / "c0_windows.npz"))
    z["s/fall/stop"][:, 5, :2] = 0.9                       # treatment: 2 of 4 windows reach the fall on every path
    np.savez(t / "c0_windows.npz", **z)
    zc = dict(np.load(c / "c0_windows.npz"))
    zc["s/fall/stop"][0, 5, 0] = 0.9                       # control: 1 window, one of two paths
    np.savez(c / "c0_windows.npz", **zc)
    r = cmp.compare(cmp.load(c, ["s"]), cmp.load(t, ["s"]), ["s"], 100, 0)["fall"]
    assert r["reach_control"] == pytest.approx(0.125) and r["reach_treat"] == pytest.approx(0.5)
    assert r["reach_ratio"] == pytest.approx(4.0)


def test_reach_with_ensemble_mean_probability_and_zero_spread(tmp_path):
    c = write_run(tmp_path / "c")
    t = write_run(tmp_path / "t")
    z = dict(np.load(t / "c0_windows.npz"))
    z["s/fall/stop_mean"] = np.zeros_like(z["s/fall/stop"])
    z["s/fall/stop_mean"][:, 3, :] = 0.8                   # ensemble-mean rule: every path reaches
    z["s/nonfall/stop_mean"] = np.zeros_like(z["s/nonfall/stop"])
    np.savez(t / "c0_windows.npz", **z)
    r = cmp.compare(cmp.load(c, ["s"]), cmp.load(t, ["s"]), ["s"], 50, 0)["fall"]
    assert r["reach_mean_prob_control"] == 0 and r["reach_mean_prob_treat"] == pytest.approx(1.0)
    zero = write_run(tmp_path / "z", u_imag=0.0)
    with pytest.raises(cmp.ZeroDisagreement):
        cmp.compare(cmp.load(c, ["s"]), cmp.load(zero, ["s"]), ["s"], 10, 0)
