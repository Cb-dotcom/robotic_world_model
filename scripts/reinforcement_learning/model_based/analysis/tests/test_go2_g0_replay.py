"""A3 follow-up (21 Nov 2026): read-only G0 world-model replay on held-out s2 data."""
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

import go2_g0_replay as rp  # noqa: E402
import go2_round0_prepare as prep  # noqa: E402
from test_go2_g0 import fake_go2_wm, restore_go2_class, round0_trace, save  # noqa: E402,F401 (autouse fixture)

IDENT = dict(state_mean=np.zeros(45, np.float32), state_std=np.ones(45, np.float32),
             action_mean=np.zeros(12, np.float32), action_std=np.ones(12, np.float32))


def model(tmp_path, seed=0):
    np.savez(tmp_path / "norm.npz", **IDENT)
    wm = fake_go2_wm(tmp_path / "wm.pt", IDENT, seed=seed)
    sd, H, norm, diff = rp.load_model(wm, tmp_path / "norm.npz", "cpu")
    return sd, H


def episode(L=60, seed=0):
    g = np.random.default_rng(seed)
    return SimpleNamespace(S=g.normal(0, .5, (L, 45)).astype(np.float32), A=g.normal(0, .5, (L, 12)).astype(np.float32),
                           y=np.zeros(L, bool), L=L, obs=None)


def test_episode_memory_matches_the_library_forward_step_by_step(tmp_path):
    sd, H = model(tmp_path)
    ep = episode(L=12)
    P, ERR = rp.episode_context(sd, ep, "cpu")
    assert np.isnan(P[:, 0]).all()
    for h in range(len(sd.state_heads)):
        sd.reset()
        ids = torch.full((1, 1, 1), h, dtype=torch.long)
        for j in range(1, ep.L):
            mean, _, _, _, _, term = sd(torch.from_numpy(ep.S[j - 1:j][None]), torch.from_numpy(ep.A[j:j + 1][None]), ids)
            assert torch.sigmoid(term).item() == pytest.approx(float(P[h, j]), rel=1e-4, abs=1e-6)
            assert ((mean[0] - torch.from_numpy(ep.S[j])) ** 2).mean().item() == pytest.approx(float(ERR[h, j]), rel=1e-4)
    sd.reset()


def test_rollouts_start_like_the_windows_and_feed_the_driving_head(tmp_path):
    sd, H = model(tmp_path)
    eps = [episode(L=H + 10, seed=1)]
    j0 = H + 1
    Pw, _ = rp.rollouts(sd, eps, [(0, j0)], 4, H, "warm", "cpu")
    Pc, _ = rp.rollouts(sd, eps, [(0, j0)], 4, H, "cold", "cpu")
    P32, _ = rp.windowed(sd, eps, [(0, j0)], H, "cpu")
    P1, _ = rp.windowed(sd, eps, [(0, j0)], 1, "cpu")
    np.testing.assert_allclose(Pw[0, :, 0], P32[:, 0], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(Pc[0, :, 0], P1[:, 0], rtol=1e-4, atol=1e-6)
    ep = eps[0]
    for h in range(len(sd.state_heads)):          # step 2 and 3 by hand with model_ids = h, as in imagination
        sd.reset()
        ids = torch.full((1, 1, 1), h, dtype=torch.long)
        mean, *_ = sd(torch.from_numpy(ep.S[j0 - H:j0][None]), torch.from_numpy(ep.A[j0 - H + 1:j0 + 1][None]), ids)
        for m in (1, 2):
            mean, _, _, _, _, term = sd(mean[:, None], torch.from_numpy(ep.A[j0 + m:j0 + m + 1][None]), ids)
            assert torch.sigmoid(term).item() == pytest.approx(float(Pw[0, h, m]), rel=1e-4, abs=1e-6)
    sd.reset()


def test_survival_and_reading_rule():
    stop = np.zeros((4, 2, 30), bool)
    stop[0, :, 0] = stop[1, :, 9] = True
    s = rp.survival(stop, (1, 10, 25))
    assert s["1"] == [0.75, 0.75] and s["10"] == [0.5, 0.5] and s["25"] == [0.5, 0.5]
    hs = lambda x: dict(stop_share=[x, x])
    ctx = {"1": hs(0.2), "32": hs(0.01)}
    roll = {"warm": dict(starts=3, survival={"25": [0.3, 0.4]}, teacher_forced_survival={"25": [0.99, 1.0]})}
    act = {"1": dict(s2=hs(0.2), g0_sampled=hs(0.3)), "32": dict(s2=hs(0.01), g0_sampled=hs(0.06))}
    r = rp.reading(ctx, roll, act)
    assert r["reset"]["read"] and r["rollout"]["read"] and r["action"]["read"]
    assert not r["action"]["1"]["read"] and r["action"]["32"]["read"]
    r = rp.reading({"1": hs(0.04), "32": hs(0.0)}, {"warm": dict(starts=3, survival={"25": [0.9]},
                                                                  teacher_forced_survival={"25": [1.0]})}, {})
    assert not r["reset"]["read"] and not r["rollout"]["read"] and r["action"]["read"] is None
    empty = {"warm": dict(starts=0, survival={"25": [float("nan")]}, teacher_forced_survival={"25": [float("nan")]})}
    assert rp.reading(ctx, empty, act)["rollout"]["read"] is None          # no start: n/a, not "not read"


def test_replay_end_to_end_checks_its_inputs(tmp_path):
    from policy_loop import ObservationBuilder, Policy, consistent_policy_trace, save_actor
    tr = consistent_policy_trace(round0_trace(T=300, N=6), Policy(save_actor(tmp_path / "s2.pt")),
                                 ObservationBuilder("go2_flat", IDENT), IDENT)
    phys = save(tmp_path / "r0.npz", tr)
    prep.prepare(phys, tmp_path / "m", frac=0.2, seed=101)
    norm = tmp_path / "m" / "go2_norm.npz"
    with np.load(norm) as z:
        stats = {k: z[k] for k in z.files}
    wm = fake_go2_wm(tmp_path / "wm.pt", stats)
    torch.manual_seed(5)
    g0 = save_actor(tmp_path / "g0.pt", seed=5)
    held = tmp_path / "m" / "withheld.json"
    r = rp.run(wm, norm, phys, held, tmp_path / "out", g0_policy=g0, K=20, start_every=10, frac=0.2)
    assert r["rows"] > 0 and set(r["contexts"]) == {"1", "2", "4", "8", "16", "32", "episode"}
    assert r["heldout_episodes"] == len(json.loads(held.read_text()))
    assert set(r["action_swap"]) == {"1", "32"} and set(r["action_swap"]["32"]) == {"s2", "g0_mean", "g0_sampled"}
    assert r["g0_std"] == [1.0] * 12 and r["reading"]["reset"]["read"] in (True, False)
    assert (tmp_path / "out" / "g0_replay.txt").read_text().startswith("A3 follow-up")
    for v in r["contexts"].values():
        assert all(0 <= x <= 1 for x in v["stop_share"])
    # the withheld list must follow the E22 rule on this recording
    bad = json.loads(held.read_text())[1:]
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="E22 rule"):
        rp.run(wm, norm, phys, tmp_path / "bad.json", tmp_path / "out2", K=20, frac=0.2)
    # the normalizer must be the checkpoint's
    np.savez(tmp_path / "other.npz", **{k: v + 1 for k, v in stats.items()})
    with pytest.raises(ValueError, match="differs"):
        rp.run(wm, tmp_path / "other.npz", phys, held, tmp_path / "out3", K=20, frac=0.2)


def test_warm_check_reading_rule(tmp_path):
    import go2_warm_check as wc
    log = lambda lens: "".join(f"  Learning iteration {i}/300  \n  Mean reward: {-0.01 * i:.2f}\n  Mean episode length: {L}\n"
                               for i, L in lens)
    (tmp_path / "c.log").write_text(log([(50, 1.1), (100, 1.2), (299, 1.3)]))
    (tmp_path / "w.log").write_text(log([(50, 20.0), (100, 40.0), (299, 80.0)]))
    r = wc.compare(tmp_path / "c.log", tmp_path / "w.log")
    assert r["last_common_iteration"] == 299 and r["trap_removed_by_warm_resets"] is True
    assert set(r["iterations"]) == {"50", "100", "299"}
    (tmp_path / "w2.log").write_text(log([(50, 2.0), (100, 3.0), (299, 30.0)]))
    assert wc.compare(tmp_path / "c.log", tmp_path / "w2.log")["trap_removed_by_warm_resets"] is True
    (tmp_path / "w3.log").write_text(log([(50, 2.0), (100, 3.0), (299, 6.0)]))
    assert wc.compare(tmp_path / "c.log", tmp_path / "w3.log")["trap_removed_by_warm_resets"] is False


def test_trainer_development_iteration_cap(tmp_path, monkeypatch):
    import e1_train
    import train as T
    from test_go2_g0 import e16_npz
    tr = round0_trace()
    prep.prepare(save(tmp_path / "r0.npz", tr), tmp_path / "m", frac=0.2, seed=101)
    asset, _, _ = e16_npz(tmp_path)
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"x")
    seen = {}
    monkeypatch.setattr(T, "run", lambda cfg: seen.setdefault("cfg", cfg))
    from envs import BaseEnv
    e1_train.main(["--wm-checkpoint", str(wm), "--reward-contract", "s2", "--asset-trace", str(asset),
                   "--start-data", str(tmp_path / "m" / "start_data"), "--max-iterations", "300",
                   "paper_warm", "3102", "go2_flat", "--seed", "1000"])
    assert seen["cfg"].policy_training_config.max_iterations == 300 and BaseEnv.warm_start_resets is True
    assert seen["cfg"].environment_config.num_envs == 4096 and seen["cfg"].environment_config.uncertainty_penalty_weight == -1.0
    BaseEnv.warm_start_resets = False
