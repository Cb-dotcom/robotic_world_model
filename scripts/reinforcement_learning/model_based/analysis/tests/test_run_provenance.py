"""CPU checks for explicit seeds and identities of the files used by a run."""

import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


MODEL_BASED = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_BASED))
import run_provenance as provenance  # noqa: E402
import p15_train  # noqa: E402


def _draw_initialization_and_samples(seed):
    provenance.seed_everything(seed)
    policy = torch.nn.Sequential(torch.nn.Linear(7, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3))
    return (
        random.random(),
        np.random.normal(size=8),
        torch.cat([parameter.detach().flatten() for parameter in policy.parameters()]),
        torch.randint(0, 1234, (16,)),
        torch.randn(8),
    )


def test_same_seed_reproduces_policy_initialization_and_data_draws():
    first = _draw_initialization_and_samples(42)
    second = _draw_initialization_and_samples(42)
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    for left, right in zip(first[2:], second[2:]):
        assert torch.equal(left, right)


def test_distinct_seed_changes_policy_initialization_and_data_draws():
    first = _draw_initialization_and_samples(42)
    second = _draw_initialization_and_samples(43)
    assert first[0] != second[0]
    assert not np.array_equal(first[1], second[1])
    assert all(not torch.equal(left, right) for left, right in zip(first[2:], second[2:]))


def test_rng_stream_continues_between_draws_without_an_implicit_reset():
    provenance.seed_everything(42)
    first = torch.rand(16)
    second = torch.rand(16)
    assert not torch.equal(first, second)
    provenance.seed_everything(42)
    assert torch.equal(torch.rand(32), torch.cat((first, second)))


@pytest.mark.parametrize("value", [None, True, False, -1, 2**32, 1.5, "1.5", "wrong"])
def test_invalid_or_missing_seed_is_rejected(value):
    with pytest.raises(ValueError, match="seed must be an integer"):
        provenance.seed_everything(value)


@pytest.mark.parametrize("value", [0, 42, np.int64(7), str(2**32 - 1)])
def test_supported_seed_range(value):
    assert provenance.seed_everything(value) == int(value)


def test_cuda_rngs_are_seeded_when_available(monkeypatch):
    seeds = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seeds.append)
    provenance.seed_everything(42)
    assert seeds and all(seed == 42 for seed in seeds)


def test_run_provenance_records_exact_files_and_effective_configuration(tmp_path):
    wm = tmp_path / "world_model.pt"
    policy = tmp_path / "policy.pt"
    wm.write_bytes(b"fixed world-model bytes")
    policy.write_bytes(b"fixed policy bytes")
    args = SimpleNamespace(seed=42, task="anymal_d_flat", run_num=1701, mode="paper_warm", init_policy_checkpoint=str(policy))
    config = {
        "model_architecture_config": {"resume_path": str(wm), "history_horizon": 32},
        "data_config": {"state_data_mean": [2.0], "state_data_std": [3.0]},
    }
    contract = {"source": "checkpoint", "state_data_mean": torch.tensor([2.0]), "state_data_std": np.array([3.0])}
    path = provenance.write_run_args(tmp_path / "run", args, config, normalizer_contract=contract)
    record = json.loads(path.read_text())
    assert path.name == "run_args.json"
    assert record["seed"] == 42 and record["task"] == "anymal_d_flat"
    assert record["mode"] == "paper_warm" and record["run_num"] == 1701
    assert record["args"] == vars(args)
    assert record["resolved_config"] == config
    assert record["world_model_checkpoint"] == {"path": str(wm), "sha256": hashlib.sha256(wm.read_bytes()).hexdigest()}
    assert record["initial_policy_checkpoint"]["sha256"] == hashlib.sha256(policy.read_bytes()).hexdigest()
    assert record["normalizer_contract"] == {"source": "checkpoint", "state_data_mean": [2.0], "state_data_std": [3.0]}
    repo_root = MODEL_BASED.parents[2]
    expected_commit = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    assert record["git"]["commit"] == expected_commit
    assert isinstance(record["git"]["dirty"], bool)
    assert not (path.parent / "run_args.json.tmp").exists()
    first_hash = record["world_model_checkpoint"]["sha256"]
    wm.write_bytes(b"different bytes at the same path")
    provenance.write_run_args(tmp_path / "run", args, config)
    assert json.loads(path.read_text())["world_model_checkpoint"]["sha256"] != first_hash


def test_provenance_rejects_unseeded_run_and_missing_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="explicit training seed"):
        provenance.write_run_args(tmp_path, SimpleNamespace(task="go2_flat"), {})
    with pytest.raises(FileNotFoundError):
        provenance.write_run_args(tmp_path, {"seed": 42}, {}, world_model_checkpoint=tmp_path / "missing.pt")
    assert not (tmp_path / "run_args.json").exists()


def test_p15_accepts_explicit_positional_or_flag_seed():
    positional = p15_train.parse_args(["paper_warm", "1701", "anymal_d_flat", "42"])
    flagged = p15_train.parse_args(["paper_warm", "1701", "--seed", "42"])
    assert vars(positional) == vars(flagged)
    assert positional.seed == 42 and positional.run_num == 1701


@pytest.mark.parametrize("argv", [["paper", "1701"], ["paper", "1701", "anymal_d_flat", "42", "--seed", "42"]])
def test_p15_rejects_missing_or_duplicate_seed(argv, capsys):
    with pytest.raises(SystemExit) as error:
        p15_train.parse_args(argv)
    assert error.value.code == 2
    assert "seed" in capsys.readouterr().err


def test_p15_import_does_not_import_training_or_launch_run(monkeypatch):
    monkeypatch.delitem(sys.modules, "train", raising=False)
    spec = importlib.util.spec_from_file_location("isolated_p15", MODEL_BASED / "p15_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "train" not in sys.modules
    assert callable(module.main)


def test_p15_passes_seed_and_mode_to_training_and_resets_warm_flag(monkeypatch):
    train = ModuleType("train")
    envs = ModuleType("envs")
    envs.BaseEnv = type("BaseEnv", (), {})
    records = []
    def configuration(task):
        return SimpleNamespace(
            environment_config=SimpleNamespace(num_envs=64),
            policy_training_config=SimpleNamespace(num_steps_per_env=24, max_iterations=500),
            policy_algorithm_config=SimpleNamespace(learning_rate=0.0001, entropy_coef=0.0),
            model_architecture_config=SimpleNamespace(resume_path="wm.pt", ensemble_size=5),
        )
    train.resolve_task_config = configuration
    train.run = lambda config: records.append((vars(train.args_cli), config, envs.BaseEnv.warm_start_resets))
    monkeypatch.setitem(sys.modules, "train", train)
    monkeypatch.setitem(sys.modules, "envs", envs)
    p15_train.main(["paper_warm", "1701", "anymal_d_flat", "42"])
    p15_train.main(["shipped", "1702", "--seed", "42"])
    assert [record[0]["seed"] for record in records] == [42, 42]
    assert [record[0]["mode"] for record in records] == ["paper_warm", "shipped"]
    assert [record[2] for record in records] == [True, False]
    assert records[0][1].environment_config.num_envs == 4096
    assert records[0][1].policy_training_config.max_iterations == 2500
    assert records[1][1].environment_config.num_envs == 64
