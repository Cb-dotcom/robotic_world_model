"""CPU tests: opt-in imagination termination rule, its calibration, and the probe rules."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MB))
sys.path.insert(0, str(MB / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from rsl_rl.modules import SystemDynamicsEnsemble
except Exception:  # pragma: no cover
    pytest.skip("rsl_rl not importable", allow_module_level=True)

import termination_rule as tr  # noqa: E402
import calibrate_termination as cal  # noqa: E402
from test_pre_reset_transitions import sample_trace  # noqa: E402


def small_anymal(seed=0):
    from configs.anymal_d_flat_cfg import AnymalDFlatConfig
    torch.manual_seed(seed)
    mac = AnymalDFlatConfig().model_architecture_config
    return SystemDynamicsEnsemble(45, 12, 0, 8, 1, "cpu", ensemble_size=5, history_horizon=32,
                                  architecture_config=mac.architecture_config).eval()


@torch.no_grad()
def test_patched_forward_is_identical_and_stashes_all_heads():
    sd = small_anymal()
    xs, xa = torch.randn(7, 32, 45), torch.randn(7, 32, 12)
    ids = torch.randint(0, 5, (1, 7, 1))
    for model_ids in (None, ids):
        sd.reset()
        ref = SystemDynamicsEnsemble.forward(sd, xs, xa, model_ids)
        sd.reset()
        new = tr.ensemble_forward_with_head_terminations(sd, xs, xa, model_ids)
        for a, b in zip(ref, new):
            if a is None:
                assert b is None
            else:
                assert torch.equal(a, b)
        assert sd.last_head_termination_logits.shape == (5, 7, 1)
        if model_ids is not None:
            picked = torch.gather(sd.last_head_termination_logits, 0, ids).squeeze(0)
            assert torch.equal(picked, new[5])
    # recurrent step after the full window also matches
    sd.reset(); SystemDynamicsEnsemble.forward(sd, xs, xa, ids)
    r2 = SystemDynamicsEnsemble.forward(sd, xs[:, -1:], xa[:, -1:], ids)
    sd.reset(); tr.ensemble_forward_with_head_terminations(sd, xs, xa, ids)
    n2 = tr.ensemble_forward_with_head_terminations(sd, xs[:, -1:], xa[:, -1:], ids)
    assert torch.equal(r2[0], n2[0]) and torch.equal(r2[5], n2[5])


def test_decide_rules():
    logit = lambda p: torch.logit(torch.tensor(p))
    heads = torch.stack([logit([0.01, 0.6, 0.04]), logit([0.02, 0.1, 0.04]), logit([0.3, 0.1, 0.04])])[..., None]
    sel = logit([0.3, 0.6, 0.04])[:, None]
    assert tr.decide(sel, heads, "selected", .5).tolist() == [False, True, False]
    assert tr.decide(sel, heads, "max", .2).tolist() == [True, True, False]
    assert tr.decide(sel, heads, "mean", .2).tolist() == [False, True, False]
    # ETH default equals round(): p > 0.5
    p = torch.tensor([0.49, 0.51, 0.9, 0.1])
    assert tr.decide(torch.logit(p)[:, None], None, "selected", .5).tolist() == torch.sigmoid(torch.logit(p)).round().bool().tolist()


def test_install_default_is_noop_and_custom_patches_given_classes():
    class Ens(SystemDynamicsEnsemble):
        pass

    class Env:
        def _parse_terminations(self, terminations):
            return "eth"
    assert tr.install("selected", 0.5, env_cls=Env, ensemble_cls=Ens) is False
    assert Env()._parse_terminations(None) == "eth" and Ens.forward is SystemDynamicsEnsemble.forward
    assert tr.install("max", 0.05, env_cls=Env, ensemble_cls=Ens) is True
    assert Ens.forward is tr.ensemble_forward_with_head_terminations
    assert SystemDynamicsEnsemble.forward is not tr.ensemble_forward_with_head_terminations
    env = Env()
    env.system_dynamics = type("SD", (), {})()
    env.system_dynamics.last_head_termination_logits = torch.logit(torch.tensor([[0.01, 0.2], [0.06, 0.01]]))[..., None]
    out = env._parse_terminations(torch.logit(torch.tensor([[0.01], [0.01]])))
    assert out.tolist() == [True, True]
    env.system_dynamics.last_head_termination_logits = None
    with pytest.raises(RuntimeError):
        env._parse_terminations(torch.zeros(2, 1))
    for bad in (("bogus", .1), ("max", 0.0), ("max", 1.0)):
        with pytest.raises(ValueError):
            tr.install(*bad, env_cls=Env, ensemble_cls=Ens)


def test_calibration_table_and_selection():
    # 5 heads; baseline: 4 surviving (low p), 2 positives; reference: 4 surviving
    pb = np.array([[.001] * 4 + [.04, .3]] * 5)
    pb[0, 4] = .2
    lb = np.array(["surviving_25"] * 4 + ["fall_now", "pre_fall_1_25"])
    pr = np.full((5, 4), .001)
    lr = np.array(["surviving_25"] * 4)
    rows, eth = cal.table({"baseline": pb, "reference": pr}, {"baseline": lb, "reference": lr})
    by = {(r["rule"], r["threshold"]): r for r in rows}
    assert by[("max", .1)]["recall_pooled"] == 1.0 and by[("mean", .1)]["recall_pooled"] == .5
    assert eth["recall_pooled"] == 0.0 and eth["false_rate_baseline"] == 0.0
    chosen = cal.select(rows)
    # max reaches recall 1.0 first at 0.1 (0.2 is not > 0.2); larger thresholds have lower recall
    assert chosen["rule"] == "max" and chosen["threshold"] == .1
    pb2 = pb.copy(); pb2[:, :4] = .9   # every surviving window would terminate
    assert cal.select(cal.table({"baseline": pb2, "reference": pr}, {"baseline": lb, "reference": lr})[0]) is None


def test_calibration_end_to_end_on_synthetic_trace(tmp_path):
    sd = small_anymal(1)
    wm = tmp_path / "wm.pt"
    torch.save({"system_dynamics_state_dict": sd.state_dict()}, wm)
    trace = sample_trace(T=130, N=2)
    for name in ("baseline", "reference"):
        np.savez_compressed(tmp_path / f"{name}.npz", **trace)
    out = tmp_path / "cal.json"
    rc = cal.main(["--baseline", str(tmp_path / "baseline.npz"), "--reference", str(tmp_path / "reference.npz"),
                   "--wm", str(wm), "--out", str(out), "--device", "cpu"])
    rep = json.loads(out.read_text())
    assert rc in (0, 3) and set(rep["horizons"]) == {"1", "8"}
    assert len(rep["horizons"]["8"]["rows"]) == len(cal.RULES) * len(cal.GRID)
    assert (rc == 0) == (rep["chosen"] is not None)
    with pytest.raises(FileExistsError):
        cal.main(["--baseline", str(tmp_path / "baseline.npz"), "--reference", str(tmp_path / "reference.npz"),
                  "--wm", str(wm), "--out", str(out), "--device", "cpu"])


def test_probe_rules_and_cli_flags():
    import run_termination_probe as probe
    import p15_train
    rec = lambda s: {"summary": {"speed_ratio": s}}
    ok = probe.verdicts({"falls": 80, "track_per_step": .9}, [rec(.9), rec(.85)])
    assert ok == {"screen_pass": True, "walker_pass": False, "both_speeds_ok": True}
    assert probe.verdicts({"falls": 5, "track_per_step": .9}, [rec(.9), rec(.9)])["walker_pass"] is True
    assert probe.verdicts({"falls": 90, "track_per_step": .9}, [rec(.9), rec(.9)])["screen_pass"] is False
    assert probe.verdicts({"falls": 5, "track_per_step": .9}, [rec(.9), rec(.79)])["walker_pass"] is False
    a = p15_train.parse_args(["paper", "1801", "anymal_d_flat", "--seed", "1",
                              "--termination-rule", "max", "--termination-threshold", "0.05"])
    assert (a.termination_rule, a.termination_threshold, a.seed) == ("max", .05, 1)
    d = p15_train.parse_args(["paper", "1601", "--seed", "1"])
    assert (d.termination_rule, d.termination_threshold) == ("selected", .5)


def test_uncertainty_stop_rule_and_install():
    logit = lambda p: torch.logit(torch.tensor(p))
    sel = logit([0.01, 0.01, 0.9])[:, None]
    epi = torch.tensor([0.5, 3.0, 0.1])
    assert tr.decide(sel, None, "selected", None, epi, 2.0).tolist() == [False, True, False]
    assert tr.decide(sel, None, "selected", .5, epi, 2.0).tolist() == [False, True, True]
    with pytest.raises(RuntimeError):
        tr.decide(sel, None, "selected", .5, None, 2.0)

    class Ens(SystemDynamicsEnsemble):
        pass

    class Env:
        def _parse_terminations(self, terminations):
            return "eth"
    with pytest.raises(ValueError):
        tr.install("max", None, env_cls=Env, ensemble_cls=Ens)
    with pytest.raises(ValueError):
        tr.install("selected", .5, env_cls=Env, ensemble_cls=Ens, uncertainty_threshold=0.0)
    assert tr.install("selected", None, env_cls=Env, ensemble_cls=Ens, uncertainty_threshold=2.0) is True
    env = Env()
    env.system_dynamics = type("SD", (), {"last_head_termination_logits": None})()
    env.epistemic_uncertainty = torch.tensor([2.5, 1.0])
    assert env._parse_terminations(logit([0.9, 0.9])[:, None]).tolist() == [True, False]


def test_imagined_stop_calibration_rules():
    import calibrate_imagined_stops as cis
    E, K = 5, cis.HORIZON
    # fall paths: termination low everywhere but disagreement high at step 3
    tf = np.full((E, K, 4, E), .001); ef = np.full((E, K, 4), .5); ef[:, 2] = 5.0
    # surviving paths: low everything, one path with p=.04 on one head
    ts = np.full((E, K, 50, E), .0005); es = np.full((E, K, 50), .6); ts[0, 5, 0, 0] = .04
    paths = {"fall": (tf, ef), "surv_baseline": (ts, es), "surv_reference": (ts.copy(), es.copy())}
    rows, eth = cis.evaluate(paths)
    assert eth["recall_fall"] == 0.0
    chosen = cis.select(rows)
    # head rules cannot catch these falls; the least-intervening full-recall rule is u = 4.0
    assert chosen["recall_fall"] == 1.0 and chosen["u"] == 4.0 and chosen["tau"] is None
    # selected-head mapping: term[h, :, :, h]
    t = np.zeros((E, K, 1, E)); t[2, 0, 0, 2] = .9; t[1, 0, 0, 3] = .9
    s = cis.stops(t, np.zeros((E, K, 1)), "selected", .5, None)
    assert s[:, 0].tolist() == [False, False, True, False, False]
    assert cis.stops(t, np.zeros((E, K, 1)), "max", .5, None)[:, 0].tolist() == [False, True, True, False, False]


def test_imagined_stop_calibration_end_to_end(tmp_path):
    import calibrate_imagined_stops as cis
    sd = small_anymal(2)
    wm = tmp_path / "wm.pt"
    torch.save({"system_dynamics_state_dict": sd.state_dict()}, wm)
    trace = sample_trace(T=160, N=2)
    # make env 0's terminal a fall with full 32+25 context (fall at row 69 in sample_trace)
    for name in ("baseline", "reference"):
        np.savez_compressed(tmp_path / f"{name}.npz", **trace)
    out = tmp_path / "stops.json"
    rc = cis.main(["--baseline", str(tmp_path / "baseline.npz"), "--reference", str(tmp_path / "reference.npz"),
                   "--wm", str(wm), "--out", str(out), "--device", "cpu", "--n_surviving", "20"])
    rep = json.loads(out.read_text())
    assert rep["counts"]["fall"] == 1 and rc in (0, 3)
    assert (rc == 0) == (rep["chosen"] is not None)
    first = rep["rows"][0]
    assert set(first) == {"agg", "tau", "u", "recall_fall", "false_path_baseline", "false_path_reference"}


def test_p15_flags_for_uncertainty_stop():
    import p15_train
    a = p15_train.parse_args(["paper", "1801", "--seed", "1", "--termination-threshold", "none",
                              "--termination-uncertainty", "2.5"])
    assert a.termination_threshold is None and a.termination_uncertainty == 2.5
