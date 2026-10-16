"""CPU checks for explicit legacy WM normalization in both analysis tools."""
import hashlib
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

import eval_signals as es  # noqa: E402
import memory_checks as mc  # noqa: E402
from normalization_contract import KEYS  # noqa: E402
from test_eval_signals import make_wm_ckpt  # noqa: E402


@pytest.fixture
def files(tmp_path):
    saved = tmp_path / "saved.pt"
    checkpoint = make_wm_ckpt(str(saved))
    legacy = tmp_path / "legacy.pt"
    torch.save({key: value for key, value in checkpoint.items() if key not in (*KEYS, "normalized")}, legacy)
    stats = tmp_path / "normalizer.npz"
    np.savez(stats, **{key: checkpoint[key].numpy() for key in KEYS})
    rng = np.random.default_rng(9)
    trace_data = rng.normal(size=(70, 66)).astype(np.float32)
    trace_data[:, 65] = 0
    trace_data[50, 65] = 1
    trace = tmp_path / "trace.csv"
    np.savetxt(trace, trace_data, delimiter=",")
    return saved, legacy, stats, checkpoint, trace, trace_data


def test_explicit_file_matches_saved_normalizer_and_preserves_legacy_default(files):
    saved, legacy, stats, original, _trace, data = files
    sd_saved, ck_saved, horizon = es.load_wm(saved, "cpu")
    sd_fixed, ck_fixed, _ = es.load_wm(legacy, "cpu", norm_stats=stats)
    _sd_raw, ck_raw, _ = es.load_wm(legacy, "cpu")
    state, action = data[:, :45].copy(), data[:, 45:57].copy()
    s_saved, a_saved, _ = es.wm_normalize(state, action, ck_saved)
    s_fixed, a_fixed, _ = es.wm_normalize(state, action, ck_fixed)
    s_raw, a_raw, message = es.wm_normalize(state, action, ck_raw)
    np.testing.assert_array_equal(s_fixed, s_saved)
    np.testing.assert_array_equal(a_fixed, a_saved)
    np.testing.assert_array_equal(s_raw, state)
    np.testing.assert_array_equal(a_raw, action)
    assert "using raw inputs" in message
    assert not np.array_equal(s_fixed, s_raw)
    assert not np.array_equal(a_fixed, a_raw)
    indices = [horizon, horizon + 1]
    expected = es.wm_score_indices(sd_saved, s_saved, a_saved, indices, horizon, "cpu", with_logit=True)
    actual = es.wm_score_indices(sd_fixed, s_fixed, a_fixed, indices, horizon, "cpu", with_logit=True)
    np.testing.assert_array_equal(actual, expected)
    assert not torch.load(legacy, weights_only=False).get("normalized", False)
    for key in KEYS:
        assert torch.equal(ck_fixed[key], original[key])
    meta = ck_fixed["_analysis_normalization"]
    assert meta["source"] == "norm_stats_file"
    assert meta["file"]["sha256"] == hashlib.sha256(stats.read_bytes()).hexdigest()


def test_file_cannot_override_saved_checkpoint_stats(files):
    saved, _legacy, stats, original, *_ = files
    _sd, ck, _ = es.load_wm(saved, "cpu", norm_stats=stats)
    assert ck["_analysis_normalization"]["source"] == "checkpoint_verified_against_norm_stats_file"
    changed = {key: original[key].numpy().copy() for key in KEYS}
    changed["state_mean"][0] += 0.1
    np.savez(stats, **changed)
    with pytest.raises(ValueError, match="differs from saved checkpoint"):
        es.load_wm(saved, "cpu", norm_stats=stats)


@pytest.mark.parametrize("defect", ["missing", "shape", "nonfinite", "zero_std", "negative_std", "object"])
def test_invalid_file_is_rejected_before_use(tmp_path, defect):
    values = {"state_mean": np.zeros(45), "state_std": np.ones(45),
              "action_mean": np.zeros(12), "action_std": np.ones(12)}
    if defect == "missing":
        del values["action_std"]
    elif defect == "shape":
        values["state_mean"] = np.zeros(44)
    elif defect == "nonfinite":
        values["action_mean"][0] = np.nan
    elif defect == "zero_std":
        values["state_std"][0] = 0
    elif defect == "negative_std":
        values["action_std"][0] = -1
    else:
        values["state_mean"] = np.array([object()] * 45, dtype=object)
    path = tmp_path / "invalid.npz"
    np.savez(path, **values)
    with pytest.raises(ValueError):
        es.checkpoint_with_norm_stats({}, path)


def test_non_archive_and_ambiguous_checkpoint_are_rejected(tmp_path):
    path = tmp_path / "not_archive.npy"
    np.save(path, np.ones(45))
    with pytest.raises(ValueError, match="NPZ archive"):
        es.checkpoint_with_norm_stats({}, path)
    path = tmp_path / "valid.npz"
    np.savez(path, state_mean=np.zeros(45), state_std=np.ones(45),
             action_mean=np.zeros(12), action_std=np.ones(12))
    ambiguous = {"normalized": False, "state_mean": torch.zeros(45)}
    with pytest.raises(ValueError, match="without normalized=True"):
        es.checkpoint_with_norm_stats(ambiguous, path)
    assert not ambiguous["normalized"]  # rejected input was not mutated


def test_evaluator_flag_records_normalizer_provenance(files, tmp_path):
    _saved, legacy, stats, _ck, trace, _data = files
    args = es.build_parser().parse_args([
        "--trace", str(trace), "--wm", str(legacy), "--norm_stats", str(stats),
        "--num_envs", "1", "--steps_per_env", "70", "--num_neg", "4",
        "--bootstrap", "0", "--device", "cpu", "--out", str(tmp_path / "eval"),
    ])
    result = es.run(args)
    assert result["inputs"]["norm_stats"]["sha256"] == hashlib.sha256(stats.read_bytes()).hexdigest()
    assert result["wm_normalization"]["source"] == "norm_stats_file"
    assert len(result["wm_normalization"]["norm_stats_sha256"]) == 64
    args.wm = None
    with pytest.raises(SystemExit, match="requires --wm"):
        es.run(args)


def test_memory_checks_flag_uses_and_records_explicit_stats(files, tmp_path, monkeypatch):
    _saved, legacy, stats, _ck, trace, _data = files
    report = tmp_path / "memory.txt"
    monkeypatch.setattr(sys, "argv", [
        "memory_checks.py", "--trace", str(trace), "--steps_per_env", "70", "--out", str(report),
        "--norm_stats", str(stats), "--n_windows", "2", "--forecast", "2", "--rollout", "2",
        "--repeats", "1", "--device", "cpu", f"legacy={legacy}",
    ])
    real_load = mc.load_wm
    observed = []

    def capture(*args, **kwargs):
        result = real_load(*args, **kwargs)
        observed.append(result[1])
        return result

    monkeypatch.setattr(mc, "load_wm", capture)
    mc.main()
    assert observed[0]["normalized"]
    text = report.read_text()
    assert "normalized inputs to WM training space" in text
    assert str(stats.resolve()) in text
    assert hashlib.sha256(stats.read_bytes()).hexdigest() in text
