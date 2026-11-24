"""C0-M on Go2 (descriptive): the carried recorded reading on go2_c0_check's windows."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip("rsl_rl.modules")

import go2_c0_check as g  # noqa: E402
import go2_c0m as gm  # noqa: E402
from test_go2_c0_check import ID, ep, model  # noqa: E402
from test_go2_g0 import fake_go2_wm, restore_go2_class, save  # noqa: E402,F401


def test_carried_starts_like_the_other_arms_and_follows_the_recorded_rows(tmp_path):
    sd = model(tmp_path)
    eps = [ep(120, 1)]
    T = 90
    U = gm.carried_u(sd, eps, [(0, T)], "cpu", depth=4)
    d0 = g.recorded_u(sd, eps, [(0, T - 4)], "cpu")
    assert U[0, 0] == pytest.approx(d0[0], rel=1e-5)
    I = g.imagined_u(sd, eps, [(0, T)], "cpu", depth=4)
    assert not np.allclose(U[0, 1:], I[0, :, 1:].mean(0))          # recorded states differ from imagined ones
    # a carried path over 4 recorded rows equals one recurrent pass over 36 recorded rows
    import torch
    e = eps[0]; j = T - 4
    sd.reset()
    xs = torch.from_numpy(e.S[j - 32:j + 4][None]); xa = torch.from_numpy(e.A[j - 31:j + 5][None])
    means, _ = g.rp.all_heads(sd, xs, xa)
    assert U[0, -1] == pytest.approx(float(g.ppo_spread(means)[0]), rel=1e-4)
    sd.reset()


def test_go2_c0m_end_to_end(tmp_path):
    from test_c0_battery import random_trace
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
    rep = gm.run(wm, tmp_path / "n.npz", probes, tmp_path / "out", n_boot=200)
    old = g.run(wm, tmp_path / "n.npz", probes, tmp_path / "old", n_boot=200)
    for k in ("10", "0"):
        r = rep["results"][k]
        for name in ("fall", "surviving"):
            assert r[name]["H_mem"] + r[name]["H_src"] == pytest.approx(r[name]["H"], abs=1e-12)
            assert r[name]["H"] == pytest.approx(old["results"][k][name]["H"], rel=1e-9)   # same H as go2_c0_check
        assert max(r["depth0_parity_max_rel"].values()) <= gm.PARITY_RTOL
    assert "no verdict" in (tmp_path / "out" / "go2_c0m.txt").read_text()
