"""A4 D4 test first (22 Nov 2026): both arms against the library, readings, end-to-end."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

MB = Path(__file__).resolve().parents[2]
for p in (str(MB), str(MB / "analysis"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("rsl_rl.modules")

import go2_c0_check as c0  # noqa: E402
import go2_test_first as tf  # noqa: E402
from test_go2_c0_check import ID, ep, model  # noqa: E402
from test_go2_g0 import fake_go2_wm, restore_go2_class, save  # noqa: E402,F401

NORM = dict(state_mean=np.zeros(45, np.float32), state_std=np.ones(45, np.float32))


def test_both_arms_equal_the_library(tmp_path):
    sd = model(tmp_path)
    e = ep(120, 1)
    T, D = 100, 4
    out = tf.paired(sd, [e], [(0, T)], NORM, "cpu", depth=D)
    j = T - D
    # recorded arm: stateful teacher forcing, the mean termination logit is not used: every head's own logit
    sd.reset()
    s, a = torch.from_numpy(e.S[j - 32:j][None]), torch.from_numpy(e.A[j - 31:j + 1][None])
    for m in range(D + 1):
        means, logits = tf.rp.all_heads(sd, s, a)
        np.testing.assert_allclose(out["rec_p"][0, :, m], torch.sigmoid(logits[:, 0]).numpy(), rtol=1e-5)
        v = np.linalg.norm(means[:, 0, 0:2].numpy() - e.S[j + m, 0:2], axis=-1)
        np.testing.assert_allclose(out["rec_v"][0, :, m], v, rtol=1e-5)
        s, a = torch.from_numpy(e.S[j + m:j + m + 1][None]), torch.from_numpy(e.A[j + m + 1:j + m + 2][None])
    # free arm: head h drives copy h (library call with model_ids = h); u equals the C0 imagined arm
    U = c0.imagined_u(sd, [e], [(0, T)], "cpu", depth=D)
    np.testing.assert_allclose(out["free_u"][0], U[0], rtol=1e-5)
    for h in range(U.shape[1]):
        sd.reset(); ids = torch.full((1, 1, 1), h, dtype=torch.long)
        mean, _, _, _, _, term = sd(torch.from_numpy(e.S[j - 32:j][None]), torch.from_numpy(e.A[j - 31:j + 1][None]), ids)
        assert out["free_p"][0, h, 0] == pytest.approx(float(torch.sigmoid(term).detach().squeeze()), rel=1e-5)
        for m in range(1, D + 1):
            mean, _, _, _, _, term = sd(mean[:, None], torch.from_numpy(e.A[j + m:j + m + 1][None]), ids)
            assert out["free_p"][0, h, m] == pytest.approx(float(torch.sigmoid(term).detach().squeeze()), rel=1e-4)
    # depth 0: both arms see the same inputs
    np.testing.assert_allclose(out["free_u"][0, :, 0], out["rec_u"][0, 0], rtol=1e-5)
    np.testing.assert_allclose(out["free_p"][0, :, 0], out["rec_p"][0, :, 0], rtol=1e-5)
    sd.reset()


def test_stop_window_and_readings():
    p = np.zeros((2, 5, 26))
    p[0, :, 14] = 0.9                       # m = 14 is outside the last 11 rows
    p[1, 2, 25] = 0.9
    assert tf.stop_any(p, 15, 25).tolist() == [[False] * 5, [False, False, True, False, False]]
    mk = lambda rec, free, pw: dict(fall=dict(rec=dict(stop_last11=rec, per_window_caught=pw)),
                                    free_minus_rec_caught=dict(mean=free - rec, ci=[free - rec - .05, free - rec + .05]))
    d3, ref = mk(0.4, 0.1, [0.4] * 50), mk(0.8, 0.8, [0.8] * 50)
    r = tf.readings(d3, ref)
    assert r["R1_missed_before_rollout"] and r["R2_lost_in_rollout"] and r["R3_d3_specific"]
    r = tf.readings(mk(0.7, 0.65, [0.7] * 50), ref)
    assert not r["R1_missed_before_rollout"] and not r["R2_lost_in_rollout"] and not r["R3_d3_specific"]
    assert tf.readings(d3, None)["R3_d3_specific"] is None


def test_end_to_end(tmp_path):
    from test_c0_battery import random_trace
    probes = []
    for name, seed in (("d3", 3), ("s2_913", 5)):
        tr = random_trace(T=400, N=8, falls=((200, 0), (300, 2), (150, 4)),
                          timeouts=((299, 1), (299, 3), (299, 5), (299, 6), (299, 7)), seed=seed)
        tr["contact"] = np.zeros((400, 8, 8), np.float32)
        tr["termination_observation"] = tr["terminated"][..., None].astype(np.float32)
        tr["previous_action"] = np.zeros((400, 8, 12), np.float32)
        probes.append((name, save(tmp_path / f"{name}.npz", tr)))
    np.savez(tmp_path / "n.npz", **ID)
    wm = fake_go2_wm(tmp_path / "wm.pt", ID)
    rep = tf.run(wm, tmp_path / "n.npz", probes, tmp_path / "out", n_boot=200)
    r = rep["probes"]["d3"]
    assert r["fall"]["n"] == r["surviving"]["n"] == r["counts"]["falls_used"] > 0
    assert r["depth0_max_rel_diff"] < 1e-4
    assert set(rep["readings"]) >= {"R1_missed_before_rollout", "R2_lost_in_rollout", "R3_d3_specific"}
    assert (tmp_path / "out" / "test_first.txt").read_text().startswith("A4 D4 test first")
    with pytest.raises(FileExistsError):
        tf.run(wm, tmp_path / "n.npz", probes, tmp_path / "out", n_boot=200)
