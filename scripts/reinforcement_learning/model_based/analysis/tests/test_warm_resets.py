"""CPU tests for the opt-in B7 fix in envs/base.py (warm GRU memory after imagination resets)."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                  # analysis/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # model_based/ (configs, envs)

torch = pytest.importorskip("torch")
try:
    from rsl_rl.modules import SystemDynamicsEnsemble  # noqa: F401
    from envs.base import BaseEnv
except Exception:  # pragma: no cover
    pytest.skip("rsl_rl / envs not importable", allow_module_level=True)

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


def _env(sd, S, A, warm):
    env = BaseEnv.__new__(BaseEnv)
    env.system_dynamics = sd
    env._num_envs = N
    env.dataset = _DS(S, A)
    env.init_dataset = _DS(S[:0], A[:0])
    env.init_data_ratio = 0.0
    env.common_step_counter = 5
    env.device = "cpu"
    if warm:
        env.warm_start_resets = True
    return env


@pytest.fixture
def setup(tmp_path):
    ck = tmp_path / "wm.pt"
    make_wm_ckpt(str(ck))
    sd, _, Hc = load_wm(str(ck), "cpu")
    assert Hc == H
    g = torch.Generator().manual_seed(1)
    S_run = torch.cumsum(torch.randn(N, H, 45, generator=g) * 0.1, 1)
    A_run = torch.randn(N, H, 12, generator=g)
    S_new = torch.cumsum(torch.randn(N, H, 45, generator=g) * 0.1, 1)
    A_new = torch.randn(N, H, 12, generator=g)
    a_next = torch.randn(N, 12, generator=g)
    return sd, S_run, A_run, S_new, A_new, a_next


@torch.no_grad()
@pytest.mark.parametrize("selected_member", [False, True], ids=["ensemble-mean", "selected-member"])
def test_warm_reset_equals_full_window_start(setup, selected_member):
    sd, S_run, A_run, S_new, A_new, a_next = setup
    # Unsorted, nonconsecutive ids catch warming the wrong batch rows or reordering them.
    ids = torch.tensor([4, 1])
    keep = [0, 2, 3, 5]
    model_ids = (torch.arange(N) % sd.ensemble_size).reshape(1, N, 1) if selected_member else None
    ref_model_ids = model_ids[:, ids] if selected_member else None
    # reference: what the very first imagination step does with the full sampled window
    sd.reset()
    reference = sd.forward(
        S_new[ids], torch.cat([A_new[ids, 1:], a_next[ids, None]], 1), ref_model_ids
    )
    ref_hidden = {
        name: getattr(sd, name).memory.hidden_states.clone()
        for name in ("state_base", "auxiliary_base")
    }
    assert reference[4] is not None and reference[4].shape == (len(ids), 8)
    assert reference[5] is not None and reference[5].shape == (len(ids), 1)
    # running envs: memory built from other trajectories
    sd.reset()
    sd.forward(S_run, A_run)
    h_keep = {
        name: getattr(sd, name).memory.hidden_states[:, keep].clone()
        for name in ("state_base", "auxiliary_base")
    }
    sd.reset_partial(ids)                                   # as _reset_idx does
    env = _env(sd, S_new[ids], A_new[ids], warm=True)
    sh = torch.zeros(N, 1, 45); ah = torch.zeros(N, 1, 12)
    sh, ah = env._reset_imagination_history(ids, sh, ah)
    assert torch.equal(sh[ids, 0], S_new[ids, -1]) and torch.equal(ah[ids, 0], A_new[ids, -1])
    for name, before in h_keep.items():
        assert torch.equal(getattr(sd, name).memory.hidden_states[:, keep], before), name
    ah_step = torch.cat([ah[:, 1:], a_next[:, None]], 1)                      # as imagination_step
    actual = sd.forward(sh, ah_step, model_ids)
    # Exercise real auxiliary heads as well as state prediction: termination/contact
    # logits need exactly the same history even when a single ensemble member is used.
    for name, observed, expected in zip(
        ("state", "aleatoric", "epistemic", "extensions", "contacts", "terminations"), actual, reference
    ):
        if expected is None:
            assert observed is None, name
        else:
            assert torch.allclose(observed[ids], expected, atol=1e-5), name
    for name, expected in ref_hidden.items():
        assert torch.allclose(getattr(sd, name).memory.hidden_states[:, ids], expected, atol=1e-5), name


@torch.no_grad()
def test_default_keeps_eth_cold_start(setup):
    sd, S_run, A_run, S_new, A_new, a_next = setup
    ids = torch.tensor([4, 1])
    sd.reset()
    sd.forward(S_run, A_run)
    sd.reset_partial(ids)
    h_before = {
        name: getattr(sd, name).memory.hidden_states.clone()
        for name in ("state_base", "auxiliary_base")
    }
    env = _env(sd, S_new[ids], A_new[ids], warm=False)
    sh = torch.zeros(N, 1, 45); ah = torch.zeros(N, 1, 12)
    env._reset_imagination_history(ids, sh, ah)
    for name, before in h_before.items():
        assert torch.equal(getattr(sd, name).memory.hidden_states, before), name  # ETH behaviour
        assert torch.equal(before[:, ids], torch.zeros_like(before[:, ids])), name  # cold memory


@torch.no_grad()
def test_b3_wrapper_resets_memory_before_every_head(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "refit"))
    import fit_reset_per_head as b3
    from test_memory_checks import _record_entry_memory
    ck = tmp_path / "wm.pt"
    make_wm_ckpt(str(ck))
    sd, _, _ = load_wm(str(ck), "cpu")
    g = torch.Generator().manual_seed(0)
    S = torch.cumsum(torch.randn(8, H + 8, 45, generator=g) * 0.1, 1)
    A = torch.randn(8, H + 8, 12, generator=g)
    c = torch.zeros(8, H + 8, 8); t = torch.zeros(8, H + 8, 1)
    seen, orig = _record_entry_memory(sd)                 # unpatched: heads 1.. see leaked memory
    sd.reset(); sd.compute_loss(S, A, None, c, t, bootstrap=True)
    sd.state_base.memory.forward = orig
    assert seen[0] is None and all(s is not None for s in seen[1:])
    b3.apply_patch()
    try:
        seen, orig = _record_entry_memory(sd)
        sd.reset(); sd.compute_loss(S, A, None, c, t, bootstrap=True)
        sd.state_base.memory.forward = orig
        assert len(seen) == len(sd.state_heads) and all(s is None for s in seen)
    finally:
        b3.remove_patch()
    seen, orig = _record_entry_memory(sd)                 # restored: ETH behaviour again
    sd.reset(); sd.compute_loss(S, A, None, c, t, bootstrap=True)
    sd.state_base.memory.forward = orig
    assert seen[0] is None and all(s is not None for s in seen[1:])
