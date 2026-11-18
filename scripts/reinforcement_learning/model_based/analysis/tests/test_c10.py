import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import c10_optimism_gap as c10


def fake_run(root, name="r1", seed=0, rows=400, num_envs=8, max_len=40):
    """A complete offline run dir: CSV start states, a random 5-head WM, a policy, run_args.json."""
    import train as T
    from run_provenance import _checkpoint_identity, _json_value
    cfg = T.resolve_task_config("anymal_d_flat")
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    cfg.data_config.dataset_root, cfg.data_config.dataset_folder = str(root), "data"
    cfg.data_config.file_data_size = cfg.data_config.batch_data_size = rows
    cfg.data_config.num_eval_trajectories, cfg.data_config.len_eval_trajectory = 2, 50
    cfg.environment_config.num_envs, cfg.environment_config.max_episode_length = num_envs, max_len
    cfg.experiment_config.device = "cpu"
    csv = data / "state_action_data_0.csv"
    if not csv.exists():
        rng = np.random.default_rng(0)
        mean = np.asarray(cfg.data_config.state_data_mean)
        std = np.asarray(cfg.data_config.state_data_std)
        st = mean + std * rng.normal(0, 1, (rows, 45))
        ac = rng.normal(0, .5, (rows, 12))
        aux = np.zeros((rows, cfg.model_architecture_config.contact_dim + cfg.model_architecture_config.termination_dim
                        + cfg.model_architecture_config.extension_dim))
        np.savetxt(csv, np.hstack([st, ac, aux]), delimiter=",")
    torch.manual_seed(seed)
    exp = T.ModelBasedExperiment(**cfg.experiment_config.to_dict())
    exp.prepare_environment(**cfg.environment_config.to_dict())
    from rsl_rl.modules import SystemDynamicsEnsemble
    m = cfg.model_architecture_config
    wm = SystemDynamicsEnsemble(45, 12, m.extension_dim, m.contact_dim, m.termination_dim, "cpu",
                                ensemble_size=m.ensemble_size, history_horizon=m.history_horizon,
                                architecture_config=m.architecture_config)
    run = root / f"2026-01-01_00-00-00_{name}"
    run.mkdir()
    wm_path = run / "wm.pt"
    torch.save({"system_dynamics_state_dict": wm.state_dict(), "iter": 0}, wm_path)
    cfg.model_architecture_config.resume_path = str(wm_path)
    exp.prepare_model(**cfg.model_architecture_config.to_dict())
    pol = cfg.policy_architecture_config.to_dict()
    pol["resume_path"] = None
    exp.prepare_policy(**pol)
    torch.save({"model_state_dict": exp.actor_critic.state_dict(), "iter": 2499}, run / "policy_2499.pt")
    exp.prepare_data(**cfg.data_config.to_dict())
    record = dict(task="anymal_d_flat", run_num=1, seed=1, mode="paper",
                  args=dict(task="anymal_d_flat", run_num=1, seed=1, mode="paper", termination_rule="selected",
                            termination_threshold=0.5),
                  world_model_checkpoint=_checkpoint_identity(wm_path), resolved_config=_json_value(cfg),
                  normalizer_contract=_json_value(exp.normalization_contract.metadata()))
    (run / "run_args.json").write_text(json.dumps(record))
    return run


def test_rebuild_is_exact_and_counts_terminations(tmp_path):
    run = fake_run(tmp_path)
    exp, record, notes = c10.build_experiment(run, num_envs=8)
    assert notes["max_episode_length"] == 40 and notes["wm_sha256"]
    c10.load_policy(exp, run / "policy_2499.pt")
    r = c10.imagined_terminations(exp, steps=100, seed=0, deterministic=True)
    assert r["imagined_s"] == pytest.approx(8 * 100 * 0.02)
    assert r["terminations"] + r["time_outs"] >= 8   # every env ends at least one episode in 100 steps (max 40)
    assert r["rate_per_1000s"] == pytest.approx(1000 * r["terminations"] / r["imagined_s"])
    again = c10.imagined_terminations(exp, steps=100, seed=0, deterministic=True)
    assert again["terminations"] == r["terminations"]   # seeded


def test_config_mismatch_and_wm_hash_are_refused(tmp_path):
    run = fake_run(tmp_path, "r2")
    original = (run / "run_args.json").read_text()
    rec = json.loads(original)
    rec["resolved_config"]["environment_config"]["max_episode_length"] = 33      # recorded values win over defaults
    (run / "run_args.json").write_text(json.dumps(rec))
    _, _, notes = c10.build_experiment(run, num_envs=8)
    assert notes["max_episode_length"] == 33
    rec["resolved_config"]["environment_config"]["not_a_field"] = 3               # a value the code cannot carry
    (run / "run_args.json").write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="cannot be rebuilt"):
        c10.build_experiment(run, num_envs=8)
    rec = json.loads(original)
    rec["world_model_checkpoint"]["sha256"] = "0" * 64
    (run / "run_args.json").write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="world model sha256"):
        c10.build_experiment(run, num_envs=8)
    rec = json.loads(original)
    rec["normalizer_contract"]["norm_stats_sha256"] = "f" * 64
    (run / "run_args.json").write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="normalizer sha256"):
        c10.build_experiment(run, num_envs=8)


def test_real_rates_and_end_to_end(tmp_path):
    run = fake_run(tmp_path, "r4")
    real = tmp_path / "real.json"
    real.write_text(json.dumps({"seconds_per_cell": 12800, "policies": {"p": {"10s": {"falls": 128}, "30s": {"falls": 0}}}}))
    rr = c10.real_rates(json.loads(real.read_text()), "p")
    assert rr["10s"]["rate_per_1000s"] == pytest.approx(10.0)
    lo, hi = rr["10s"]["ci95_per_1000s"]
    assert lo < 10 < hi and rr["30s"]["ci95_per_1000s"][0] == 0
    rows = c10.run([("p", run)], real, tmp_path / "out", num_envs=8, steps=60)
    assert set(rows["p"]["imagined"]) == {"deterministic", "sampled"}
    assert "deterministic_minus_real_10s" in rows["p"]["gap_per_1000s"]
    assert (tmp_path / "out" / "c10_report.txt").read_text().startswith("Plan v6 C10")
