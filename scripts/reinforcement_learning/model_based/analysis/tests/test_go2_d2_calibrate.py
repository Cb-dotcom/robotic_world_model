"""A4 D2 option B (22 Nov 2026): termination-logit offset rule, checkpoint edit, end-to-end run."""
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

import go2_d2_calibrate as cal  # noqa: E402
import go2_round0_prepare as prep  # noqa: E402
from test_go2_g0 import fake_go2_wm, restore_go2_class, round0_trace, save  # noqa: E402,F401


def test_choose_rule():
    falls = np.array([[1.0, 2.0], [3.0, 4.0]])          # c needed per (fall, head)
    surv = np.array([[5.0, 6.0], [-1.0, 7.0]])          # one pair already fires at 0
    r = cal.choose(falls, surv)
    assert r["offset"] == pytest.approx(2.001) and r["detection"] == 0.5 and r["false_stop"] == 0.25
    assert r["detection_at_0"] == 0.0 and r["false_stop_at_0"] == 0.25 and not r["valid"]      # 0.25 > 0.10
    r = cal.choose(falls, np.full((10, 2), 9.0))
    assert r["valid"] and r["false_stop"] == 0.0
    assert not cal.choose(np.zeros((0, 2)), surv)["valid"]
    r = cal.choose(np.array([[-1.0, -2.0], [3.0, -4.0]]), np.full((10, 2), 9.0))
    assert r["offset"] == 0.0 and r["detection"] == 0.75                                          # no shift needed


def test_offset_moves_only_the_termination_logits(tmp_path):
    stats = dict(state_mean=np.zeros(45, np.float32), state_std=np.ones(45, np.float32),
                 action_mean=np.zeros(12, np.float32), action_std=np.ones(12, np.float32))
    np.savez(tmp_path / "n.npz", **stats)
    wm = fake_go2_wm(tmp_path / "wm.pt", stats)
    heads = cal.apply_offset(wm, tmp_path / "c.pt", 1.5)
    import go2_g0_replay as rp
    a, H, _, _ = rp.load_model(wm, tmp_path / "n.npz", "cpu")
    b, _, _, _ = rp.load_model(tmp_path / "c.pt", tmp_path / "n.npz", "cpu")
    g = torch.Generator().manual_seed(0)
    xs, xa = torch.randn(4, H, 45, generator=g), torch.randn(4, H, 12, generator=g)
    a.reset(); b.reset()
    ma, la = rp.all_heads(a, xs, xa)
    mb, lb = rp.all_heads(b, xs, xa)
    assert len(heads) == la.shape[0]
    torch.testing.assert_close(lb - la, torch.full_like(la, 1.5), atol=1e-5, rtol=0)
    torch.testing.assert_close(ma, mb)
    ck = torch.load(tmp_path / "c.pt", weights_only=False)
    assert ck["provenance"]["termination_logit_offset"] == 1.5
    with pytest.raises(FileExistsError):
        cal.apply_offset(wm, tmp_path / "c.pt", 1.0)


def test_calibrate_end_to_end(tmp_path):
    sh = save(tmp_path / "sh.npz", round0_trace(T=300, N=8, seed=4))
    prep.prepare(sh, tmp_path / "m", frac=0.1, seed=101)
    with np.load(tmp_path / "m" / "go2_norm.npz") as z:
        stats = {k: z[k] for k in z.files}
    wm = fake_go2_wm(tmp_path / "wm.pt", stats)
    rec = cal.run(wm, tmp_path / "m" / "go2_norm.npz", sh, tmp_path / "m" / "withheld.json", tmp_path / "out")
    assert rec["validation_falls"] + rec["validation_surviving"] <= sum(rec["validation_kinds"].values())
    assert (tmp_path / "out" / "calibration.json").is_file()
    assert rec["valid"] == (tmp_path / "out" / "d2c_wm_seed2.pt").is_file()
