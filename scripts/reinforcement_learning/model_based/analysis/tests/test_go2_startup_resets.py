"""A4 D6 (22 Nov 2026): imagination startup resets start from a recorded physical reset state with zero memory."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                  # analysis/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # model_based/

torch = pytest.importorskip("torch")
try:
    from envs.base import BaseEnv
except Exception:  # pragma: no cover
    pytest.skip("envs not importable", allow_module_level=True)

from test_eval_signals import make_wm_ckpt  # noqa: E402
from eval_signals import load_wm  # noqa: E402

H, N = 32, 6


class _DS:
    def __init__(self, S, A):
        self.S, self.A, self.i = S, A, 0

    def sample_batch(self, n, normalized=True):
        out = self.S[self.i:self.i + n], self.A[self.i:self.i + n]
        self.i += n
        return out

    def normalize(self, s, a):                       # affine, so the test sees that normalization is applied
        return (None if s is None else s * 2.0 + 1.0), (None if a is None else a * 3.0)


def _env(sd, S, A, share, raw):
    env = BaseEnv.__new__(BaseEnv)
    env.system_dynamics = sd
    env._num_envs = N
    env.dataset = _DS(S, A)
    env.init_dataset = _DS(S[:0], A[:0])
    env.init_data_ratio = 0.0
    env.common_step_counter = 5
    env.device = "cpu"
    env.warm_start_resets = True
    env.extras = {"log": {}}
    env.startup_reset_share = share
    env.startup_reset_raw = raw
    return env


@pytest.fixture
def setup(tmp_path):
    ck = tmp_path / "wm.pt"
    make_wm_ckpt(str(ck))
    sd, _, Hc = load_wm(str(ck), "cpu")
    g = torch.Generator().manual_seed(3)
    S_run = torch.randn(N, H, 45, generator=g) * 0.1
    A_run = torch.randn(N, H, 12, generator=g)
    S_new = torch.randn(N, H, 45, generator=g) * 0.1
    A_new = torch.randn(N, H, 12, generator=g)
    raw = (np.random.default_rng(0).normal(0, .1, (7, 45)).astype(np.float32),
           np.random.default_rng(1).normal(0, .5, (7, 12)).astype(np.float32))
    return sd, S_run, A_run, S_new, A_new, raw


def _run_then_reset(sd, S_run, A_run, ids):
    sd.reset()
    sd.forward(S_run, A_run, None)                   # non-zero memory everywhere
    sd.reset_partial(ids)                            # what _reset_idx does before the history reset


@torch.no_grad()
def test_startup_envs_start_at_a_recorded_reset_with_zero_memory(setup):
    sd, S_run, A_run, S_new, A_new, raw = setup
    ids = torch.tensor([4, 1, 2])
    _run_then_reset(sd, S_run, A_run, ids)
    before = {n: getattr(sd, n).memory.hidden_states.clone() for n in ("state_base", "auxiliary_base")}
    env = _env(sd, S_run, A_run, 1.0, raw)
    env.dataset = _DS(S_new[ids], A_new[ids])
    env.dataset.normalize = _DS.normalize.__get__(env.dataset)
    sh, ah = S_run[:, -1:].clone(), A_run[:, -1:].clone()
    torch.manual_seed(0)
    sh, ah = env._reset_imagination_history(ids, sh, ah)
    ns, na = 2.0 * torch.as_tensor(raw[0]) + 1.0, 3.0 * torch.as_tensor(raw[1])
    for k, e in enumerate(ids.tolist()):
        # the last history row is one of the recorded reset states (normalized), its action the recorded previous action
        hit = (ns - sh[e, -1]).abs().amax(-1) < 1e-6
        assert hit.any()
        assert torch.allclose(ah[e, -1], na[int(np.flatnonzero(hit.numpy())[0])], atol=1e-6)
    for n in ("state_base", "auxiliary_base"):
        h = getattr(sd, n).memory.hidden_states
        assert torch.all(h[:, ids] == 0)             # zero memory: no warm-up from the sampled window
        keep = [0, 3, 5]
        assert torch.equal(h[:, keep], before[n][:, keep])
    assert env.extras["log"]["startup_reset_fraction"] == 1.0 and env.startup_reset_count == 3
    # the next imagination step from a startup env equals a fresh start from the reset state (zero memory)
    a = torch.randn(N, 1, 12)
    out = sd.forward(sh, a, None)
    sd2 = sd
    hidden = {n: getattr(sd2, n).memory.hidden_states.clone() for n in ("state_base", "auxiliary_base")}
    sd2.reset()
    ref = sd2.forward(sh[ids], a[ids], None)
    torch.testing.assert_close(out[0][ids], ref[0])
    for n in hidden:
        getattr(sd2, n).memory.hidden_states = hidden[n]


@torch.no_grad()
def test_share_zero_keeps_the_warm_reset(setup):
    sd, S_run, A_run, S_new, A_new, raw = setup
    ids = torch.tensor([4, 1])
    _run_then_reset(sd, S_run, A_run, ids)
    env = _env(sd, S_run, A_run, 0.0, None)
    env.dataset = _DS(S_new[ids], A_new[ids])
    sh, ah = env._reset_imagination_history(ids, S_run[:, -1:].clone(), A_run[:, -1:].clone())
    assert torch.equal(sh[ids, -1], S_new[ids, -1])
    for n in ("state_base", "auxiliary_base"):
        assert torch.any(getattr(sd, n).memory.hidden_states[:, ids] != 0)   # warm memory rebuilt
    assert "startup_reset_fraction" not in env.extras["log"]


def test_share_without_reset_states_is_refused(setup):
    sd, S_run, A_run, _, _, _ = setup
    env = _env(sd, S_run, A_run, 0.5, None)
    with pytest.raises(ValueError, match="startup_reset_raw"):
        env._startup_mask(4)


def test_trainer_wiring_sets_the_startup_share(tmp_path, monkeypatch):
    """e1_train.py --startup-resets/--startup-share: BaseEnv gets the recorded reset states and the share in a _warm
    mode; D6's start data (reset rows included) load through apply_start_data; nothing is trained."""
    from pathlib import Path
    import e1_train
    import train as T
    import go2_startup_prepare as sp
    sys.path.insert(0, HERE)
    from test_go2_g0 import e16_npz, save
    from test_go2_startup_prepare import startup_trace
    sp.prepare_startup(save(tmp_path / "r0.npz", startup_trace()), save(tmp_path / "sh.npz", startup_trace(seed=7)),
                       tmp_path / "d6", frac=0.1, seed=101)
    asset, _, _ = e16_npz(tmp_path)
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"x")
    seen = {}
    monkeypatch.setattr(T, "run", lambda cfg: seen.setdefault("cfg", cfg))
    monkeypatch.setattr(BaseEnv, "startup_reset_share", 0.0)
    monkeypatch.setattr(BaseEnv, "startup_reset_raw", None)
    args = ["--wm-checkpoint", str(wm), "--reward-contract", "s2", "--asset-trace", str(asset),
            "--start-data", str(tmp_path / "d6" / "start_data")]
    e1_train.main(args + ["--startup-resets", str(tmp_path / "d6" / "reset_states.npz"), "--startup-share", "0.25",
                          "paper_warm", "3002", "go2_flat", "--seed", "1000"])
    assert BaseEnv.startup_reset_share == 0.25 and BaseEnv.startup_reset_raw[0].shape[1] == 45
    assert Path(seen["cfg"].data_config.dataset_root) == (tmp_path / "d6").resolve()
    with pytest.raises(SystemExit, match="_warm mode"):
        e1_train.main(args + ["--startup-resets", str(tmp_path / "d6" / "reset_states.npz"), "--startup-share", "0.25",
                              "paper", "3003", "go2_flat", "--seed", "1000"])
    with pytest.raises(SystemExit, match="go together"):
        e1_train.main(args + ["--startup-share", "0.25", "paper_warm", "3004", "go2_flat", "--seed", "1000"])
