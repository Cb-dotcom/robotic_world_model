"""A4 D6 Q4 startup check (22 Nov 2026): both arms from the physical reset, scoring, rule, end to end."""
import json
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

import go2_d6_startup_check as sc  # noqa: E402
from test_go2_c0_check import ID, model  # noqa: E402
from test_go2_g0 import fake_go2_wm, restore_go2_class, save  # noqa: E402,F401
from test_go2_startup_prepare import startup_trace  # noqa: E402

NORM = {k: v for k, v in ID.items()}


def test_episodes_start_at_a_physical_reset_only():
    tr = startup_trace(T=300, N=6, mid_start_env=2)
    eps = sc.start_episodes(tr, NORM)
    assert all(not (e.env == 2 and e.first == 0) for e in eps)       # the mid-episode start is excluded
    e = next(x for x in eps if x.env == 5 and x.first == 0)           # the short fall
    assert e.kind == "fall" and e.L == 10
    np.testing.assert_allclose(e.R, tr["state"][0, 5])
    only = {(e.env, e.first)}
    assert [x.env for x in sc.start_episodes(tr, NORM, only)] == [5]


@torch.no_grad()
def test_both_arms_equal_the_library(tmp_path):
    sd = model(tmp_path)
    tr = startup_trace(T=300, N=6)
    eps = [e for e in sc.start_episodes(tr, NORM) if e.L >= 12][:2]
    pf, pr, pred = sc.startup_rollouts(sd, eps, steps=6)
    for i, e in enumerate(eps):
        # recorded arm: memory from zero at the reset, the recorded state fed at every step
        sd.reset()
        s, a = torch.from_numpy(e.R[None, None]), torch.from_numpy(e.A[None, 0:1])
        for m in range(6):
            _, logits = sc.rp.all_heads(sd, s, a)
            np.testing.assert_allclose(pr[i, :, m], torch.sigmoid(logits[:, 0]).numpy(), rtol=1e-5)
            s, a = torch.from_numpy(e.S[None, m:m + 1]), torch.from_numpy(e.A[None, m + 1:m + 2])
        # free arm: head h drives its copy (library call with model_ids = h)
        for h in range(pf.shape[1]):
            sd.reset(); ids = torch.full((1, 1, 1), h, dtype=torch.long)
            mean, _, _, _, _, term = sd(torch.from_numpy(e.R[None, None]), torch.from_numpy(e.A[None, 0:1]), ids)
            assert pf[i, h, 0] == pytest.approx(float(torch.sigmoid(term).detach().squeeze()), rel=1e-5)
            np.testing.assert_allclose(pred[i, h, 0], mean[0].detach().numpy(), rtol=1e-5, atol=1e-6)
            for m in range(1, 6):
                mean, _, _, _, _, term = sd(mean[:, None], torch.from_numpy(e.A[None, m:m + 1]), ids)
                assert pf[i, h, m] == pytest.approx(float(torch.sigmoid(term).detach().squeeze()), rel=1e-4)
    # step 1 of both arms sees the same input
    np.testing.assert_allclose(pf[:, :, 0], pr[:, :, 0], rtol=1e-5)
    sd.reset()


def test_fall_scores_and_false_stops():
    class E:
        def __init__(self, L, kind):
            self.L, self.kind = L, kind
    eps = [E(15, "fall"), E(20, "fall"), E(300, "time_out"), E(30, "fall")]
    p = np.zeros((4, 2, 40))
    p[0, 0, 2] = .9          # head 0 stops 12 steps before the terminal transition (row 14): early, not caught
    p[0, 1, 14] = .9         # head 1 stops at the terminal transition: caught
    p[1, :, 12] = .9         # both heads stop at row 12 of a 20-row fall: caught (last 11 = rows 9..19)
    p[2, 1, 39] = .9         # a false stop on the time-out, head 1
    f = sc.fall_scores(p, eps, 0, 25)
    assert f["n"] == 2 and f["caught"] == 0.75 and f["stopped_before_last_11"] == 0.25
    assert sorted(f["leads"]) == [0, 7, 7, 12]
    s = sc.false_stops(p, eps)
    assert s["n"] == 1 and s["share"] == 0.5
    assert sc.fall_scores(p, eps, 25, 40)["n"] == 1


def test_rule():
    ok = dict(fidelity_step25=dict(n=30, vxy=0.2, tilt=4.0), false_stop_free=dict(n=25, share=0.05))
    assert sc.verdict(ok)["passed"]
    few = dict(fidelity_step25=dict(n=10, vxy=0.2, tilt=4.0), false_stop_free=dict(n=25, share=0.05))
    assert not sc.verdict(few)["passed"] and not sc.verdict(few)["enough_episodes"]
    bad = dict(fidelity_step25=dict(n=30, vxy=0.3, tilt=4.0), false_stop_free=dict(n=25, share=0.05))
    assert not sc.verdict(bad)["passed"]


def test_end_to_end(tmp_path):
    p913 = save(tmp_path / "p913.npz", startup_trace(seed=3))
    r0 = save(tmp_path / "r0.npz", startup_trace(seed=4))
    held = tmp_path / "withheld.json"
    held.write_text(json.dumps([{"env": 1, "first": 0, "last": 99, "kind": "time_out"}]))
    np.savez(tmp_path / "n.npz", **ID)
    wm = fake_go2_wm(tmp_path / "wm.pt", ID)
    rep = sc.run(wm, tmp_path / "n.npz", {"s2_913": (p913, None), "r0_withheld": (r0, {(1, 0)})},
                 {"d3_915": (p913, None)}, tmp_path / "out")
    assert set(rep["results"]) == {"s2_913", "r0_withheld", "d3_915"} and "verdict" not in rep["results"]["d3_915"]
    assert rep["results"]["r0_withheld"]["episodes"] == 1
    assert rep["q4_pass"] is False                                    # too few episodes on these fixtures
    assert (tmp_path / "out" / "startup_check.txt").read_text().rstrip().endswith("Q4: FAIL")


def test_no_episode_gives_empty_results(tmp_path):
    sd = model(tmp_path)
    pf, pr, pred = sc.startup_rollouts(sd, [], steps=5)
    assert pf.shape[0] == pr.shape[0] == pred.shape[0] == 0
    r = sc.score(sd, [], NORM, "cpu")
    assert r["episodes"] == 0 and r["fidelity_step25"]["n"] == 0 and not sc.verdict(r)["passed"]
