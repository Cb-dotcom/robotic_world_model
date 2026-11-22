"""A4 Go2 C0 check (22 Nov 2026): exact PPO signal, imagined arm, windows, AUROC, end-to-end."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

MB = Path(__file__).resolve().parents[2]
for p in (str(MB), str(MB / "analysis"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("rsl_rl.modules")

import go2_c0_check as c0  # noqa: E402
import go2_g0_replay as rp  # noqa: E402
from test_go2_g0 import fake_go2_wm, restore_go2_class, round0_trace, save  # noqa: E402,F401

ID = dict(state_mean=np.zeros(45, np.float32), state_std=np.ones(45, np.float32),
          action_mean=np.zeros(12, np.float32), action_std=np.ones(12, np.float32))


def model(tmp_path):
    np.savez(tmp_path / "n.npz", **ID)
    sd, H, _, _ = rp.load_model(fake_go2_wm(tmp_path / "wm.pt", ID), tmp_path / "n.npz", "cpu")
    return sd


def ep(L, seed, kind="fall", probe="p"):
    g = np.random.default_rng(seed)
    e = SimpleNamespace(S=g.normal(0, .5, (L, 45)).astype(np.float32), A=g.normal(0, .5, (L, 12)).astype(np.float32),
                        L=L, kind=kind, probe=probe, cluster=f"{probe}/{seed}")
    e.feat = g.normal(size=(L, 4))
    return e


def test_signal_and_imagined_arm_equal_the_library(tmp_path):
    sd = model(tmp_path)
    eps = [ep(100, 1)]
    T = 80
    uc = c0.recorded_u(sd, eps, [(0, T)], "cpu")
    sd.reset()
    _, _, epi, _, _, _ = sd(torch.from_numpy(eps[0].S[T - 32:T][None]), torch.from_numpy(eps[0].A[T - 31:T + 1][None]), None)
    assert uc[0] == pytest.approx(float(epi[0].detach()), rel=1e-5)
    U = c0.imagined_u(sd, eps, [(0, T)], "cpu", depth=3)
    d0 = c0.recorded_u(sd, eps, [(0, T - 3)], "cpu")
    np.testing.assert_allclose(U[0, :, 0], d0[0], rtol=1e-5)
    e = eps[0]; j0 = T - 3
    for h in range(U.shape[1]):                    # head h drives its own copy, as model_ids in imagination
        sd.reset(); ids = torch.full((1, 1, 1), h, dtype=torch.long)
        mean, _, epi, _, _, _ = sd(torch.from_numpy(e.S[j0 - 32:j0][None]), torch.from_numpy(e.A[j0 - 31:j0 + 1][None]), ids)
        for m in range(1, 4):
            mean, _, epi, _, _, _ = sd(mean[:, None], torch.from_numpy(e.A[j0 + m:j0 + m + 1][None]), ids)
            assert U[0, h, m] == pytest.approx(float(epi[0].detach()), rel=1e-4)
    sd.reset()


def test_windows_endpoint_and_matching():
    eps = [ep(120, 1), ep(60, 2), ep(300, 3, "time_out"), ep(300, 4, "time_out"), ep(150, 5, "fall", "q"), ep(300, 6, "time_out", "q")]
    falls, surv, counts = c0.windows(eps, 10)
    assert (0, 109) in falls and counts["falls_too_short"] == 1 and (4, 139) in falls
    for (fi, _), (si, T) in zip(falls, surv):
        assert eps[si].kind == "time_out" and eps[si].probe == eps[fi].probe and T + 25 <= eps[si].L - 1
    assert len(set(surv)) == len(surv)
    with pytest.raises(ValueError, match="surviving windows"):
        c0.windows([ep(120, 1), ep(60, 9, "time_out")], 10)


def test_auroc():
    assert c0.auroc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert c0.auroc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert c0.auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5


def test_bootstrap_keeps_the_matched_windows_together():
    log_ratios = np.array([-1.0, -2.0, -6.0])
    fall_clusters = ["p/f1", "p/f2", "p/f3"]
    survivor_clusters = ["p/s1", "p/s1", "p/s2"]
    strata = ["p"] * len(log_ratios)
    fall, surviving = c0.bootstrap(log_ratios, log_ratios, fall_clusters, survivor_clusters, strata, strata, n_boot=100)
    np.testing.assert_array_equal(fall, surviving)
    assert len(np.unique(fall)) > 1


def test_c0_end_to_end(tmp_path):
    probes = []
    from test_c0_battery import random_trace
    for name, seed in (("p911", 3), ("p912", 5)):
        tr = random_trace(T=400, N=8, falls=((200, 0), (300, 2), (150, 4)),
                          timeouts=((299, 1), (299, 3), (299, 5), (299, 6), (299, 7)), seed=seed)
        tr["contact"] = np.zeros((400, 8, 8), np.float32)
        tr["termination_observation"] = tr["terminated"][..., None].astype(np.float32)
        tr["previous_action"] = np.zeros((400, 8, 12), np.float32)
        probes.append((name, save(tmp_path / f"{name}.npz", tr)))
    np.savez(tmp_path / "n.npz", **ID)
    wm = fake_go2_wm(tmp_path / "wm.pt", ID)
    rep = c0.run(wm, tmp_path / "n.npz", probes, tmp_path / "out", n_boot=200)
    r = rep["results"]["10"]
    assert r["fall"]["n"] == r["surviving"]["n"] > 0 and r["verdict"]["verdict"] in ("supported", "not supported", "inconclusive")
    assert r["fall"]["depth0_max_rel_diff"] < 1e-4 and 0 <= r["auroc"]["recorded"] <= 1
    assert (tmp_path / "out" / "go2_c0.txt").read_text().startswith("A4 Go2 C0 check")
