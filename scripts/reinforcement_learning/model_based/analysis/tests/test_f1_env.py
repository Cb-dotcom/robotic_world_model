"""F1 in the imagination env (plan v6 amendment A1): runtime checks on the CPU with a complete fake run."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c10_optimism_gap as c10
from test_c10 import fake_run

N, STEPS = 16, 30


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return fake_run(tmp_path_factory.mktemp("f1"), name="f1", num_envs=N, max_len=12)


def make_env(run, f1=False, scale=None, gain=1.0, events=True):
    exp, _, _ = c10.build_experiment(run, num_envs=N)
    env = exp.env
    if not events:
        env.event_interval_range = None
    if f1:
        env.f1_members = 5
        env.f1_scale = scale if scale is not None else np.linspace(1.0, 2.0, 256)
        env.f1_gain = gain
    return exp, env


def rollout(exp, env, steps=STEPS, seed=0):
    from run_provenance import seed_everything
    seed_everything(seed)
    sh, ah = env.prepare_imagination()
    g = torch.Generator().manual_seed(123)
    out = []
    for i in range(steps):
        if env.common_step_counter > 0:
            sh, ah = sh[:, -1:], ah[:, -1:]
        obs = env.get_imagination_observation(sh, ah)
        a = torch.randn(N, 12, generator=g) * 0.5
        _, rew, dones, extras, sh, ah, epi = env.imagination_step(a, sh, ah)
        out.append(dict(state=sh[:, -1].clone(), rew=rew.clone(), dones=dones.clone(), epi=epi.clone(),
                        to=extras["time_outs"].clone(), obs=obs["policy"].clone(),
                        spread=getattr(env, "f1_spread", torch.zeros(N)).clone(),
                        pen=getattr(env, "f1_penalty", torch.zeros(N)).clone(),
                        depth=getattr(env, "f1_depth", torch.zeros(N)).clone()))
    return out


@torch.no_grad()
def test_driving_path_equals_the_standard_path(run):
    """Same seed, same actions: F1 changes only the penalty term (states, dones, epistemic identical)."""
    std_exp, std_env = make_env(run)
    f1_exp, f1_env = make_env(run, f1=True)
    a, b = rollout(std_exp, std_env), rollout(f1_exp, f1_env)
    assert any(x["dones"].any() for x in a)                                  # resets happen inside the rollout
    for x, y in zip(a, b):
        torch.testing.assert_close(y["state"], x["state"], atol=1e-5, rtol=1e-5)
        assert torch.equal(x["dones"], y["dones"]) and torch.equal(x["to"], y["to"])
        torch.testing.assert_close(y["epi"], x["epi"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(y["obs"], x["obs"], atol=1e-5, rtol=1e-5)
    # rewards differ only by the uncertainty term: lambda * (F1 penalty - epistemic) * dt
    w, dt = std_env.uncertainty_penalty_weight, std_env._step_dt
    for x, y in zip(a, b):
        torch.testing.assert_close(y["rew"] - x["rew"], w * (y["pen"] - y["epi"]) * dt, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_spread_at_the_first_prediction_is_the_head_disagreement(run):
    """At d = 0 every member has the same input and zeroed memory, so the member spread equals the epistemic."""
    exp, env = make_env(run, f1=True)
    out = rollout(exp, env, steps=1)
    torch.testing.assert_close(out[0]["spread"], out[0]["epi"], atol=1e-5, rtol=1e-5)
    done = out[0]["dones"].bool()
    assert torch.all(out[0]["depth"][~done] == 1) and torch.all(out[0]["depth"][done] == 0)


@torch.no_grad()
def test_members_follow_their_own_heads(run):
    """Without resets or pushes, member k's path = head k run alone on its own predictions."""
    exp, env = make_env(run, f1=True, events=False)
    env._max_episode_length = 200                                             # no time-outs in 6 steps
    env._parse_terminations = lambda t: torch.zeros(N, dtype=torch.bool)      # no stops
    from run_provenance import seed_everything
    seed_everything(0)
    sh, ah = env.prepare_imagination()
    sh0, ah0 = sh.clone(), ah.clone()
    g = torch.Generator().manual_seed(5)
    acts = [torch.randn(N, 12, generator=g) * 0.5 for _ in range(6)]
    for i, a in enumerate(acts):
        if i > 0:
            sh, ah = sh[:, -1:], ah[:, -1:]
        env.get_imagination_observation(sh, ah)
        env.imagination_step(a, sh, ah)
    spread_env = env.f1_spread.clone()
    model = env.system_dynamics
    paths = []
    for k in range(5):
        model.reset()
        xs, xa = sh0.clone(), ah0.clone()
        ids = torch.full((1, N, 1), k, dtype=torch.long)
        for i, a in enumerate(acts):
            an = env.dataset.normalize(None, a)[1]
            xa = torch.cat([xa[:, 1:], an.unsqueeze(1)], dim=1)
            inp_s, inp_a = (xs, xa) if i == 0 else (xs[:, -1:], xa[:, -1:])
            m, *_ = model.forward(inp_s, inp_a, ids)
            xs = torch.cat([xs[:, 1:], m.unsqueeze(1)], dim=1)
        paths.append(m)
    manual = torch.stack(paths).std(dim=0).sum(dim=1)
    torch.testing.assert_close(spread_env, manual, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_penalty_uses_scale_at_depth_and_resets(run):
    scale = np.arange(256, dtype=np.float32) + 1.0
    exp, env = make_env(run, f1=True, scale=scale, gain=0.5)
    out = rollout(exp, env, steps=STEPS)
    for i, x in enumerate(out):
        d_before = (out[i - 1]["depth"] if i > 0 else torch.zeros(N, dtype=torch.long)).clone()
        if i > 0:
            d_before[out[i - 1]["dones"].bool()] = 0
        torch.testing.assert_close(x["pen"], 0.5 * torch.as_tensor(scale)[d_before] * x["spread"], atol=1e-5, rtol=1e-5)
    done_any = [i for i, x in enumerate(out) if x["dones"].any()]
    assert done_any and torch.all(out[done_any[0]]["depth"][out[done_any[0]]["dones"].bool()] == 0)


def test_f1_refuses_incomplete_settings(run):
    exp, env = make_env(run, f1=True, scale=np.ones(10))
    with pytest.raises(ValueError, match="covers 10 depths"):
        env.prepare_imagination()
    exp, env = make_env(run, f1=True)
    env.f1_gain = None
    with pytest.raises(ValueError, match="frozen calibration"):
        env.prepare_imagination()
