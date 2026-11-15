"""CPU tests for the plan v6 W1 refit inputs (C3 dose folders, C6 normalizer, C5 NLL loss, fit check).

The NLL tests and the tiny fit need rsl_rl (rsl_rl_rwm) importable; the folder and normalizer tests do not.
"""
from pathlib import Path
import csv
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MB))
sys.path.insert(0, str(MB / "refit"))
import make_fall_dose as D  # noqa: E402
import make_widened_norm as W  # noqa: E402
import check_fit as C  # noqa: E402

COLS = 45 + 12 + 8 + 1


def write_segments(root, spec, steps=45):
    """spec: list of (name, is_fall). One env per file, the fitter's layout; returns the manifest rows."""
    root.mkdir(parents=True)
    rng = np.random.default_rng(1)
    with open(root / "manifest.csv", "w") as f:
        f.write("name,steps,rows,num_envs\n")
        for name, fall in spec:
            rows = rng.normal(size=(steps, COLS)).astype(np.float32)
            rows[:, -1] = 0.0
            rows[-1, -1] = 1.0 if fall else 0.0
            np.savetxt(root / f"{name}.csv", rows, delimiter=",", fmt="%.6g")
            f.write(f"{name},{steps},{steps},1\n")


SPEC = ([(f"baseline1701_env{i:03d}_ep000", True) for i in range(8)]
        + [(f"baseline1701_env{i:03d}_ep001", False) for i in range(4)]
        + [("reference_env000_ep000", False), ("reference_env001_ep000", True)])


def test_dose_folders_are_nested_and_keep_everything_else(tmp_path):
    src, out = tmp_path / "segments", tmp_path / "dose"
    write_segments(src, SPEC)
    assert D.main(["--src", str(src), "--out", str(out), "--doses", "0,25,50,100", "--seed", "0"]) == 0
    kept = {}
    for d, n_falls in ((0, 0), (25, 2), (50, 4), (100, 8)):
        folder = out / f"dose{d:03d}"
        with open(folder / "manifest.csv") as f:
            names = [r["name"] for r in csv.DictReader(f)]
        card = json.loads((folder / "cards.json").read_text())
        assert card["falls_kept"] == n_falls and card["real_terms_expected"] == n_falls + 1
        assert "reference_env001_ep000" in names and "reference_env000_ep000" in names
        assert all(f"baseline1701_env{i:03d}_ep001" in names for i in range(4))
        assert len(names) == 6 + n_falls
        for name in names:
            link = folder / f"{name}.csv"
            assert link.is_symlink() and os.path.realpath(link) == str((src / f"{name}.csv").resolve())
        kept[d] = set(card["kept_fall_names"])
    assert kept[0] <= kept[25] <= kept[50] <= kept[100]
    with pytest.raises(SystemExit):
        D.main(["--src", str(src), "--out", str(out)])  # --out must not exist


def test_dose_order_depends_only_on_seed(tmp_path):
    src = tmp_path / "segments"
    write_segments(src, SPEC)
    rows = D.read_manifest(src)
    a = D.plan_doses(rows, src, "baseline1701", [25], 0)[2]
    b = D.plan_doses(list(reversed(rows)), src, "baseline1701", [25], 0)[2]
    assert a == b


def test_widened_norm_changes_only_gravity(tmp_path):
    sys.path.insert(0, str(MB))
    from e1_prepare_data import eth_norm_arrays
    src, out = tmp_path / "eth_norm.npz", tmp_path / "c6" / "wide.npz"
    eth = eth_norm_arrays()
    np.savez(src, **eth)
    assert W.main(["--src", str(src), "--out", str(out)]) == 0
    with np.load(out) as z:
        new = {k: z[k] for k in W.KEYS}
    expect = [math.sin(math.pi / 3) / 3, math.sin(math.pi / 3) / 3, 0.5 / 3]
    assert np.allclose(new["state_std"][[6, 7, 8]], expect, atol=1e-6)
    mask = np.ones(45, bool)
    mask[[6, 7, 8]] = False
    assert np.array_equal(new["state_std"][mask], eth["state_std"][mask])
    for k in ("state_mean", "action_mean", "action_std"):
        assert np.array_equal(new[k], eth[k])
    # 60 deg tilt -> |z| = 3 on x and z components
    tilt = math.pi / 3
    g = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])
    zx = (g - eth["state_mean"][[6, 7, 8]]) / new["state_std"][[6, 7, 8]]
    assert abs(zx[0] - 3) < 1e-4 and abs(zx[2] - 3) < 1e-4
    card = json.loads(Path(str(out) + ".json").read_text())
    assert card["gravity_std_old"] == pytest.approx([0.02, 0.02, 0.04])


def test_widened_norm_rejects_other_layout():
    bad = {"state_mean": np.zeros(45), "state_std": np.ones(45), "action_mean": np.zeros(12), "action_std": np.ones(12)}
    with pytest.raises(ValueError):
        W.widen(bad)


class _Args:
    iters, state_loss, norm_sha256, data_dir, real_terms, heads = 500, None, None, None, None, None


def test_check_fit_flags_each_problem(tmp_path):
    ck = {"iter": 500, "norm_stats_sha256": "abc", "total_real_terms": 7,
          "provenance": {"b3_reset_per_head": True, "state_loss": "gaussian_nll", "data_dir": str(tmp_path), "ensemble_size": 5}}
    a = _Args()
    a.state_loss, a.norm_sha256, a.data_dir, a.real_terms, a.heads = "gaussian_nll", "abc", str(tmp_path), 7, 5
    assert C.problems_for(ck, a) == []
    a.state_loss, a.real_terms = "mse", 8
    assert len(C.problems_for(ck, a)) == 2
    old = {"iter": 500, "provenance": {"b3_reset_per_head": True}}  # pre-W1 checkpoint: no state_loss key = mse
    b = _Args()
    b.state_loss = "mse"
    assert C.problems_for(old, b) == []
    old["provenance"]["b3_reset_per_head"] = False
    assert C.problems_for(old, b) == ["provenance b3_reset_per_head is not True"]


rsl = pytest.importorskip("rsl_rl.modules")
import fit_world_model_go2_pilot_segments as F  # noqa: E402
from configs.anymal_d_flat_cfg import AnymalDFlatConfig  # noqa: E402

ARCH = AnymalDFlatConfig().model_architecture_config.architecture_config


def _model(seed=0):
    torch.manual_seed(seed)
    return rsl.SystemDynamicsEnsemble(45, 12, 0, 8, 1, "cpu", ensemble_size=2, history_horizon=4, architecture_config=ARCH)


def test_state_loss_nll_uses_eth_branch_and_mse_is_untouched():
    model = _model()
    assert F.set_state_loss(model, "mse") is model and "compute_regression_loss" not in vars(model)
    F.set_state_loss(model, "gaussian_nll")
    mean, std, target = torch.randn(6, 45), torch.rand(6, 45) + 0.1, torch.randn(6, 45)
    got, seq = model.compute_regression_loss(mean, std, target)
    assert torch.allclose(got, torch.nn.GaussianNLLLoss()(mean, target, std ** 2)) and float(seq) == 0.0
    with pytest.raises(ValueError):
        F.set_state_loss(_model(), "l1")


def test_state_loss_nll_reaches_compute_loss_and_backprops():
    model = F.set_state_loss(_model(), "gaussian_nll")
    calls = []
    original = model.compute_regression_loss

    def spy(*a, **k):
        calls.append(1)
        return original(*a, **k)
    model.compute_regression_loss = spy
    s, a = torch.randn(3, 6, 45), torch.randn(3, 6, 12)
    c, t = torch.zeros(3, 6, 8), torch.zeros(3, 6, 1)
    model.reset()
    losses = model.compute_loss(s, a, None, c, t, bootstrap=True)
    assert calls and torch.isfinite(losses[0])
    losses[0].backward()
    assert any(p.grad is not None for p in model.state_heads.parameters())


def test_tiny_cpu_fit_records_state_loss_and_runtime(tmp_path):
    seg = tmp_path / "segments"
    write_segments(seg, [("baseline1701_env000_ep000", True), ("reference_env000_ep000", False)], steps=45)
    norm = tmp_path / "norm.npz"
    from e1_prepare_data import eth_norm_arrays
    np.savez(norm, **eth_norm_arrays())
    out = tmp_path / "wm" / "wm.pt"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, str(MB / "refit" / "fit_reset_per_head.py"), "--config", "anymal_d_flat", "--data", str(seg),
           "--output", str(out), "--norm_stats", str(norm), "--iterations", "2", "--num_mini_batches", "1",
           "--mini_batch_size", "4", "--lr", "1e-4", "--termination_pos_weight", "1", "--lazy_windows",
           "--device", "cpu", "--seed", "1", "--state_loss", "gaussian_nll", "--log_interval", "1"]
    run = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert "[fit] state loss = gaussian_nll" in run.stdout
    ck = torch.load(out, map_location="cpu", weights_only=False)
    prov = ck["provenance"]
    assert prov["state_loss"] == "gaussian_nll" and prov["b3_reset_per_head"] is True
    assert set(prov["runtime"]) >= {"torch", "torch_cuda", "isaac_sim", "rsl_rl"}
    a = _Args()
    a.iters, a.state_loss, a.real_terms, a.heads, a.data_dir = 2, "gaussian_nll", 1, 5, str(seg)
    assert C.problems_for(ck, a) == []
