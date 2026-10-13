"""CPU tests for analysis/memory_checks.py (B3 emulation and B7 rollout)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # analysis/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # model_based/ (configs)

torch = pytest.importorskip("torch")
try:
    from rsl_rl.modules import SystemDynamicsEnsemble  # noqa: F401
except Exception:  # pragma: no cover
    pytest.skip("rsl_rl not importable", allow_module_level=True)

import memory_checks as mc  # noqa: E402
from test_eval_signals import make_trace, make_wm_ckpt  # noqa: E402
from eval_signals import load_wm  # noqa: E402

H = 32


def _batch(B=16, F=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = torch.cumsum(torch.randn(B, H + F, 45, generator=g) * 0.1, dim=1)
    A = torch.randn(B, H + F, 12, generator=g)
    return S, A


def _record_entry_memory(sd):
    """Hook: store the GRU memory at entry of every full-window (length H) call."""
    seen = []
    mem = sd.state_base.memory
    orig = mem.forward

    def fwd(x):
        if x.shape[1] == H:
            seen.append(None if mem.hidden_states is None else mem.hidden_states.clone())
        return orig(x)
    mem.forward = fwd
    return seen, orig


def test_b3_emulation_matches_compute_loss(tmp_path, monkeypatch):
    ck = tmp_path / "wm.pt"
    make_wm_ckpt(str(ck))
    sd, _, Hc = load_wm(str(ck), "cpu")
    assert Hc == H
    S, A = _batch()
    contact = torch.zeros(S.shape[0], S.shape[1], 8)
    term = torch.zeros(S.shape[0], S.shape[1], 1)
    # remove sampling noise so both paths are deterministic
    monkeypatch.setattr(torch, "randn_like", lambda t, **kw: torch.zeros_like(t))

    seen_real, orig = _record_entry_memory(sd)
    sd.reset()
    sd.compute_loss(S, A, None, contact, term, bootstrap=False)
    sd.state_base.memory.forward = orig

    seen_emu, orig = _record_entry_memory(sd)
    sd.reset()
    for hd in sd.state_heads:
        mc.head_training_like(sd, hd, S, A, H, gen=None)
    sd.state_base.memory.forward = orig

    assert len(seen_real) == len(seen_emu) == len(sd.state_heads)
    assert seen_real[0] is None and seen_emu[0] is None      # head 0 starts from zero memory
    for k in range(1, len(sd.state_heads)):                   # heads 1.. start from leaked memory
        assert seen_real[k] is not None
        assert torch.allclose(seen_real[k], seen_emu[k], atol=1e-6)
        assert seen_real[k].abs().max() > 0


def test_b3_report_head0_zero_shift(tmp_path):
    ck = tmp_path / "wm.pt"
    make_wm_ckpt(str(ck))
    sd, _, _ = load_wm(str(ck), "cpu")
    S, A = _batch(B=32)
    lines = []
    res = mc.check_b3(sd, S, A, H, repeats=2, seed=0, say=lines.append)
    assert res["shift"][0] < 1e-5
    assert (res["shift"][1:] > 0).all()
    assert not any("WARNING" in l for l in lines)


def test_b7_warm_step1_equals_scoring_and_cold_differs(tmp_path):
    ck = tmp_path / "wm.pt"
    make_wm_ckpt(str(ck))
    sd, _, _ = load_wm(str(ck), "cpu")
    K = 6
    S, A = _batch(B=8, F=K)
    ids = torch.zeros(1, 8, 1, dtype=torch.long)
    p_w, e_w = mc.rollout(sd, S[:, :H], A[:, 1:H + 1], A[:, H + 1:H + K], ids, warm=True)
    # scoring path: zero memory, full window, head 0
    sd.reset()
    h = sd.state_base(S[:, :H], A[:, 1:H + 1])
    ref = sd.state_heads[0](h, S[:, :H])[0]
    assert torch.allclose(p_w[0], ref, atol=1e-6)
    p_c, _ = mc.rollout(sd, S[:, :H], A[:, 1:H + 1], A[:, H + 1:H + K], ids, warm=False)
    assert p_c.shape == p_w.shape == (K, 8, 45)
    assert not torch.allclose(p_c[0], p_w[0])
    # cold step 1 = one-step input (last state, action row f0) with zero memory
    sd.reset()
    h1 = sd.state_base(S[:, H - 1:H], A[:, H:H + 1])
    assert torch.allclose(p_c[0], sd.state_heads[0](h1, S[:, H - 1:H])[0], atol=1e-6)


def test_clean_window_starts_rules():
    steps, n_env = 100, 3
    data = np.ones((steps * n_env, 66), np.float32)
    data[:, 65] = 0
    data[50, 65] = 1                 # fall in env 0
    data[150, 45:57] = 0             # unmarked reset row in env 1
    s = mc.clean_window_starts(data, steps, H=10, L=5)
    t = s % steps
    assert ((t >= 10) & (t + 5 <= steps)).all()
    for f in s:
        rows = np.arange(f - 10, f + 5)
        assert 50 not in rows and 150 not in rows
        assert len(np.unique(rows // steps)) == 1
    assert (s // steps == 2).sum() == steps - 10 - 5 + 1


def test_main_end_to_end(tmp_path):
    ck = tmp_path / "wm.pt"
    make_wm_ckpt(str(ck))
    tr = tmp_path / "trace.csv"
    make_trace(str(tr), num_envs=4, steps=150)
    out = tmp_path / "rep.txt"
    sys_argv = sys.argv
    sys.argv = ["memory_checks.py", "--trace", str(tr), "--steps_per_env", "150", "--out", str(out),
                "--n_windows", "64", "--rollout", "12", "--repeats", "1", "--device", "cpu", f"a={ck}"]
    try:
        mc.main()
    finally:
        sys.argv = sys_argv
    txt = out.read_text()
    assert "[B3]" in txt and "[B7]" in txt and "WARNING" not in txt
