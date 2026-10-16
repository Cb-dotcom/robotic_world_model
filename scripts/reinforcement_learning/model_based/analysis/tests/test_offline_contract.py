"""Frozen-WM normalization, real imagination-path parity, and seeded policy tests."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

MODEL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_DIR))
from configs import AnymalDFlatConfig, Go2FlatConfig
from normalization_contract import resolve_normalization
from run_provenance import seed_everything
from rsl_rl.modules import SystemDynamicsEnsemble
from train import ModelBasedExperiment


def _stats(state=45, action=12):
    return dict(state_mean=torch.linspace(-3, 3, state), state_std=torch.linspace(0.2, 2, state),
                action_mean=torch.linspace(-0.5, 0.5, action), action_std=torch.linspace(0.5, 1.5, action))


def test_checkpoint_statistics_override_unrelated_legacy_constants():
    expected = _stats()
    c = resolve_normalization({"normalized": True, **expected},
                              AnymalDFlatConfig().data_config.to_dict(), 45, 12)
    assert c.source == "checkpoint"
    for k, v in expected.items():
        assert torch.equal(c.statistics[k], v)
    legacy = resolve_normalization({}, AnymalDFlatConfig().data_config.to_dict(), 45, 12)
    assert legacy.source == "config"
    assert not torch.equal(c.statistics["state_std"], legacy.statistics["state_std"])


@pytest.mark.parametrize("mutation,match", [
    ({"state_mean": None}, "Missing state_mean"),
    ({"state_std": torch.zeros(45)}, "strictly positive"),
    ({"action_std": torch.full((12,), float("nan"))}, "finite"),
    ({"state_mean": torch.zeros(44)}, "expected 45"),
])
def test_invalid_checkpoint_never_falls_back_to_config(mutation, match):
    with pytest.raises(ValueError, match=match):
        resolve_normalization({"normalized": True, **_stats(), **mutation},
                              AnymalDFlatConfig().data_config.to_dict(), 45, 12)


def test_unknown_go2_normalizer_is_rejected_before_loading_data():
    exp = ModelBasedExperiment("go2_flat", "cpu")
    exp.state_dim, exp.action_dim = 45, 12
    with pytest.raises(ValueError, match="dataset recomputation is forbidden"):
        exp.prepare_data(**Go2FlatConfig().data_config.to_dict())


def test_checkpoint_with_ambiguous_statistics_is_rejected():
    with pytest.raises(ValueError, match="without normalized=True"):
        resolve_normalization(_stats(), AnymalDFlatConfig().data_config.to_dict(), 45, 12)


def test_hash_identifies_effective_stats_not_source_container():
    a = resolve_normalization({"normalized": True, **_stats()}, {}, 45, 12)
    b = resolve_normalization({"normalized": True, **{k: v.reshape(1, -1) for k, v in _stats().items()}}, {}, 45, 12)
    assert a.norm_stats_sha256 == b.norm_stats_sha256


def _experiment(tmp_path, legacy):
    cfg = AnymalDFlatConfig() if legacy else Go2FlatConfig()
    ec = cfg.environment_config.to_dict()
    ec.update(num_envs=2, observation_noise=False, event_interval_range=None)
    exp = ModelBasedExperiment(cfg.experiment_config.environment, "cpu")
    exp.prepare_environment(**ec)
    mac = cfg.model_architecture_config.to_dict()
    mac["architecture_config"] = dict(mac["architecture_config"], rnn_hidden_size=16,
        state_mean_shape=[8], state_logstd_shape=[8], contact_shape=[8], termination_shape=[8])
    sd = SystemDynamicsEnsemble(45, 12, 0, 8, 1, "cpu", ensemble_size=5,
        history_horizon=32, architecture_config=mac["architecture_config"])
    checkpoint = {"system_dynamics_state_dict": sd.state_dict(), "iter": 1}
    if not legacy:
        checkpoint.update(normalized=True, **_stats())
    ck = tmp_path / "wm.pt"
    torch.save(checkpoint, ck)
    mac["resume_path"] = str(ck)
    exp.prepare_model(**mac)
    gen = np.random.default_rng(3)
    rows = np.zeros((64, 66), dtype=np.float32)
    rows[:, :57] = gen.normal(2, 0.2, size=(64, 57))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(rows).to_csv(data_dir / "state_action_data_0.csv", header=False, index=False)
    dc = cfg.data_config.to_dict()
    dc.update(dataset_root=str(tmp_path), dataset_folder="data", file_data_size=64, batch_data_size=64,
              num_eval_trajectories=2, num_visualizations=0, len_eval_trajectory=32)
    exp.prepare_data(**dc)
    return exp, rows, checkpoint, cfg


@pytest.mark.parametrize("legacy", [False, True], ids=["refit-checkpoint", "eth-legacy-config"])
def test_real_imagination_path_matches_correctly_preprocessed_model(tmp_path, legacy):
    exp, rows, checkpoint, cfg = _experiment(tmp_path, legacy)
    stats = exp.normalization_contract.statistics
    raw_s = torch.tensor(rows[None, :32, :45]).repeat(2, 1, 1)
    raw_a = torch.tensor(rows[None, :32, 45:57]).repeat(2, 1, 1)
    next_a = torch.full((2, 12), 0.3)
    ns = (raw_s - stats["state_mean"]) / stats["state_std"]
    na = (raw_a - stats["action_mean"]) / stats["action_std"]
    next_na = (next_a - stats["action_mean"]) / stats["action_std"]
    model_ids = torch.zeros(1, 2, 1, dtype=torch.long)
    with torch.inference_mode():
        exp.system_dynamics.reset()
        reference = exp.system_dynamics(ns, torch.cat([na[:, 1:], next_na[:, None]], 1), model_ids)
        expected = reference[0] * stats["state_std"] + stats["state_mean"]
        exp.env.prepare_imagination()
        exp.env.system_dynamics_model_ids = model_ids.clone()
        exp.env.get_imagination_observation(ns, na)
        exp.system_dynamics.reset()
        seen = {}
        original = exp.env._parse_imagination_states
        def capture(raw):
            seen["physical_prediction"] = raw.clone()
            return original(raw)
        exp.env._parse_imagination_states = capture
        exp.env.imagination_step(next_a, ns, na)
    assert torch.allclose(seen["physical_prediction"], expected, atol=1e-5)
    ds_s, ds_a = exp.dataset.normalize(raw_s, raw_a)
    assert torch.equal(ds_s, ns) and torch.equal(ds_a, na)
    back_s, back_a = exp.dataset.denormalize(ds_s, ds_a)
    assert torch.allclose(back_s, raw_s, atol=1e-6)
    assert torch.allclose(back_a, raw_a, atol=1e-6)


def test_seed_controls_actual_policy_initialization_and_start_windows(tmp_path):
    exp, _, _, cfg = _experiment(tmp_path, legacy=False)
    pc = cfg.policy_architecture_config.to_dict()
    pc.update(actor_hidden_dims=[8], critic_hidden_dims=[8])
    results = []
    for seed in (19, 19, 20):
        seed_everything(seed)
        exp.prepare_policy(**pc)
        weights = {k: v.detach().clone() for k, v in exp.actor_critic.state_dict().items()}
        windows = tuple(x.clone() for x in exp.dataset.sample_batch(4))
        results.append((weights, windows))
    assert all(torch.equal(v, results[1][0][k]) for k, v in results[0][0].items())
    assert all(torch.equal(a, b) for a, b in zip(results[0][1], results[1][1]))
    assert any(not torch.equal(v, results[2][0][k]) for k, v in results[0][0].items())


def test_missing_world_model_cannot_train_random_dynamics():
    exp = ModelBasedExperiment("go2_flat", "cpu")
    with pytest.raises(ValueError, match="trained --wm-checkpoint"):
        exp.prepare_model(**Go2FlatConfig().model_architecture_config.to_dict())


def test_extra_csv_boundary_column_is_rejected(tmp_path):
    exp = ModelBasedExperiment("go2_flat", "cpu")
    exp.state_dim, exp.action_dim = 45, 12
    exp.extension_dim, exp.contact_dim, exp.termination_dim = 0, 8, 1
    pd.DataFrame(np.zeros((4, 67))).to_csv(tmp_path / "state_action_data_0.csv", header=False, index=False)
    with pytest.raises(ValueError, match="67 columns; expected 66"):
        exp._load_data(str(tmp_path), ".", file_data_size=4, batch_data_size=4)


def test_seeded_offline_training_smoke_writes_policy_and_effective_contract(tmp_path, monkeypatch):
    """Exercise load -> dataset -> imagination -> PPO -> checkpoint on CPU."""
    import json
    import train as training
    from envs import BaseEnv
    exp, _, _, cfg = _experiment(tmp_path, legacy=False)
    cfg.experiment_name = "offline_runs"
    monkeypatch.chdir(tmp_path)
    cfg.experiment_config.device = "cpu"
    cfg.environment_config.num_envs = 2
    cfg.environment_config.observation_noise = False
    cfg.environment_config.event_interval_range = None
    cfg.model_architecture_config.architecture_config = exp.system_dynamics.architecture_config
    cfg.model_architecture_config.resume_path = str(tmp_path / "wm.pt")
    cfg.data_config.dataset_root = str(tmp_path)
    cfg.data_config.dataset_folder = "data"
    cfg.data_config.file_data_size = cfg.data_config.batch_data_size = 64
    cfg.data_config.num_eval_trajectories = 2
    cfg.data_config.len_eval_trajectory = 32
    cfg.policy_architecture_config.actor_hidden_dims = [8]
    cfg.policy_architecture_config.critic_hidden_dims = [8]
    cfg.policy_training_config.num_steps_per_env = 2
    cfg.policy_training_config.max_iterations = 1
    cfg.policy_algorithm_config.num_learning_epochs = 1
    cfg.policy_algorithm_config.num_mini_batches = 1
    from types import SimpleNamespace
    monkeypatch.setattr(training, "args_cli", SimpleNamespace(
        task="go2_flat", run_num=9001, seed=17, init_policy_checkpoint=None,
    ), raising=False)
    monkeypatch.setattr(BaseEnv, "warm_start_resets", True, raising=False)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    try:
        training.run(cfg)
        files = list((tmp_path / "logs/offline_runs").rglob("run_args.json"))
        assert len(files) == 1
        metadata = json.loads(files[0].read_text())
        assert metadata["seed"] == 17
        assert metadata["normalizer_contract"]["source"] == "checkpoint"
        assert metadata["normalizer_contract"]["norm_stats_sha256"] == exp.normalization_contract.norm_stats_sha256
        assert (files[0].parent / "policy_0.pt").is_file()
    finally:
        training.wandb.finish()
