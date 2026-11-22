"""A4 D6 (22 Nov 2026): startup windows in the fitter (real history from the reset, one shape per minibatch)."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

MB = Path(__file__).resolve().parents[2]
for p in (str(MB), str(MB / "analysis"), str(MB / "refit"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("rsl_rl.modules")

import fit_world_model_go2_pilot_segments as F  # noqa: E402

COLS = 45 + 12 + 8 + 1


def seg(n, fall, seed):
    g = np.random.default_rng(seed)
    x = g.normal(0, 1, (n, COLS)).astype(np.float32)
    x[:, 57:65] = (g.random((n, 8)) < .5)
    x[:, -1] = 0
    x[-1, -1] = float(fall)
    x[:, 0] = np.arange(n)                      # row index in state dim 0 (raw): lets the test read positions
    return x


def write(tmp, segs, startup):
    d = tmp / "segments"
    d.mkdir()
    lines = ["name,steps,rows,num_envs"]
    for name, x in segs.items():
        np.savetxt(d / f"{name}.csv", x, delimiter=",", fmt="%.9g")
        lines.append(f"{name},{len(x)},{len(x)},1")
    (d / "manifest.csv").write_text("\n".join(lines) + "\n")
    (tmp / "startup.json").write_text(json.dumps({"segments": startup}))
    return d


def test_groups_shapes_rows_and_short_fall(tmp_path):
    segs = {"a": seg(60, False, 0), "b": seg(12, True, 1), "c": seg(50, True, 2)}
    loaded = [(k, len(v), v, 1) for k, v in segs.items()]
    z = torch.zeros(45), torch.ones(45), torch.zeros(12), torch.ones(12)
    g = F.build_startup_groups(loaded, ["a", "b"], 45, 12, 32, 8, "cpu", *z, normalize=True)
    # a: h = 1..31 with f = 8; b (12 rows): h = 1..11 with f = min(8, 12 - h)
    assert sum(v.shape[0] for v in g.values()) == 31 + 11
    assert (1, 8) in g and (11, 1) in g and (31, 8) in g and (32, 8) not in g
    w = g[(1, 8)][torch.arange(g[(1, 8)].shape[0])]
    assert w.shape == (2, 9, COLS) and set(w[:, :, 0].flatten().tolist()) == set(range(9))  # rows 0..8 of the reset
    # the short fall's terminal row is the last target of its windows (h, 12 - h), h = 4..11 ('b' comes after 'a')
    for h in range(4, 12):
        win = g[(h, 12 - h)][torch.arange(g[(h, 12 - h)].shape[0])]
        assert win.shape[1] == 12 and float(win[-1, -1, -1]) == 1.0 and float(win[-1, :-1, -1].sum()) == 0.0
        assert float(win[-1, -1, 0]) == 11.0                   # raw row index (identity normalizer): the 12th row
    with pytest.raises(RuntimeError, match="not in --data"):
        F.build_startup_groups(loaded, ["zz"], 45, 12, 32, 8, "cpu", *z)


def test_first_transition_is_reset_state_and_first_action():
    """h = 1: the library pairs the reset state (row 0) with the action of row 1 and targets row 1."""
    from rsl_rl.modules import SystemDynamicsEnsemble
    from configs.go2_flat_cfg import Go2FlatConfig
    mac = Go2FlatConfig().model_architecture_config
    sd = SystemDynamicsEnsemble(45, 12, 0, 8, 1, "cpu", ensemble_size=1, history_horizon=1,
                                architecture_config=mac.architecture_config)
    s, a = torch.randn(4, 9, 45), torch.randn(4, 9, 12)
    seen = {}
    orig = sd.state_base.forward

    def spy(xs, xa):
        seen.setdefault("first", (xs.clone(), xa.clone()))
        return orig(xs, xa)
    sd.state_base.forward = spy
    sd.reset()
    sd.compute_state_loss(sd.state_heads[0], s, a)
    xs, xa = seen["first"]
    assert torch.equal(xs, s[:, :1]) and torch.equal(xa, a[:, 1:2])


def test_fit_runs_with_startup_minibatches(tmp_path, capsys):
    segs = {"a": seg(60, False, 0), "b": seg(12, True, 1), "c": seg(70, True, 2), "d": seg(45, False, 3)}
    d = write(tmp_path, segs, ["a", "b", "c"])
    import fit_reset_per_head as R
    argv = ["--config", "go2_flat", "--data", str(d), "--output", str(tmp_path / "wm" / "wm.pt"), "--iterations", "3",
            "--num_mini_batches", "4", "--mini_batch_size", "8", "--device", "cpu", "--skip_short_segments",
            "--startup_segments", str(tmp_path / "startup.json"), "--seed", "0", "--log_interval", "1",
            "--save_interval", "3", "--termination_pos_weight", "1"]
    old = sys.argv
    sys.argv = ["fit_reset_per_head.py"] + argv
    try:
        R.main()
    finally:
        sys.argv = old
        R.remove_patch()
    out = capsys.readouterr().out
    assert "startup minibatch probability" in out
    n = int(out.strip().split("startup_minibatches=")[-1].split()[0])
    assert n > 0
    ck = torch.load(tmp_path / "wm" / "wm.pt", map_location="cpu", weights_only=False)
    assert ck["provenance"]["startup"]["windows"] == 31 + 11 + 31 and ck["iter"] == 3
    assert ck["provenance"]["b3_reset_per_head"] is True
