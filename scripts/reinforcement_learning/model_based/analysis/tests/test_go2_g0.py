"""Plan v6 amendment A3 (21 Nov 2026): Go2 round-0 manifest, s2 contract wiring, runtime reward check, s2 probe
and the G0 gate."""
import json
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

import go2_round0_prepare as prep  # noqa: E402
import go2_study  # noqa: E402
import go2_g0_gate as gate  # noqa: E402
from test_c0_battery import random_trace  # noqa: E402
from test_e16 import synthetic_trace  # noqa: E402

CLASS_ATTRS = ("feet_air_time_threshold", "joint_pos_stand_still_scale", "joint_pos_velocity_threshold",
               "default_joint_pos", "soft_joint_pos_limits", "timers_before_rewards", "acceleration_from_state")


@pytest.fixture(autouse=True)
def restore_go2_class():
    from envs.go2_flat import Go2FlatEnv
    saved = {k: Go2FlatEnv.__dict__.get(k) for k in CLASS_ATTRS}
    yield
    for k, v in saved.items():
        setattr(Go2FlatEnv, k, v)


def round0_trace(T=300, N=6, seed=0):
    tr = random_trace(T=T, N=N, falls=((150, 0), (60, 3)), timeouts=((99, 1), (199, 1), (249, 4)), seed=seed)
    rng = np.random.default_rng(seed + 1)
    tr["contact"] = (rng.random((T, N, 8)) < .5).astype(np.float32)
    tr["termination_observation"] = tr["terminated"][..., None].astype(np.float32)
    tr["previous_action"] = np.zeros((T, N, 12), np.float32)
    # episode id in the last torque dim: lets the tests see which episode a start window came from
    ep = np.zeros((T, N), np.float32)
    for n in range(N):
        k = 0
        for t in range(T):
            ep[t, n] = 1000 * n + k
            k += bool(tr["done"][t, n])
    tr["next_state"][..., 44] = ep
    tr["state"][1:, :, 44] = np.where(tr["done"][:-1], tr["state"][1:, :, 44], ep[:-1])
    tr["state"][0, :, 44] = ep[0]
    return tr


def save(path, tr):
    np.savez(path, **tr)
    return path


def test_round0_split_counts_and_outputs(tmp_path):
    tr = round0_trace()
    src = save(tmp_path / "r0.npz", tr)
    r = prep.prepare(src, tmp_path / "out", frac=0.2, seed=101)
    c = r["counts"]
    assert c["episodes"] == int(tr["done"].sum()) + 6 and c["falls"] == 2 and c["time_outs"] == 3
    assert c["withheld"] == int(round(0.2 * c["episodes"])) and c["withheld"] + c["training"] == c["episodes"]
    again = prep.prepare(src, tmp_path / "out2", frac=0.2, seed=101)
    assert (tmp_path / "out" / "withheld.json").read_text() == (tmp_path / "out2" / "withheld.json").read_text()
    held = json.loads((tmp_path / "out" / "withheld.json").read_text())
    held_ids = {(h["env"], h["first"]) for h in held}
    # segments: training episodes >= the fitter window, fitter layout, true terminal flag
    man = (tmp_path / "out" / "segments" / "manifest.csv").read_text().splitlines()[1:]
    falls_in_fit = 0
    for line in man:
        name, steps, rows, nenv = line.split(",")
        n, s = int(name[6:9]), int(name[11:])
        assert (n, s) not in held_ids and int(rows) >= r["seq_len"] and nenv == "1"
        seg = np.loadtxt(tmp_path / "out" / "segments" / f"{name}.csv", delimiter=",", ndmin=2)
        np.testing.assert_allclose(seg[0, :45], tr["next_state"][s, n], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(seg[0, 45:57], tr["action"][s, n], rtol=1e-6, atol=1e-6)
        assert seg[:-1, -1].sum() == 0
        falls_in_fit += int(seg[-1, -1])
    assert falls_in_fit == c["falls_reaching_fitting"] == c["terminal_positive_rows_reaching_fitting"]
    assert c["falls_reaching_fitting"] + c["withheld_falls"] + c["skipped_short_falls"] == 2
    # normalizer = the fitter's own computation on these segments
    from fit_world_model_go2_pilot_segments import compute_normalizer, read_segments
    segs, _, _ = read_segments(tmp_path / "out" / "segments", 45, 12, 0, 8, 1)
    ref = [t.reshape(-1).numpy() for t in compute_normalizer(segs, 45, 12)]
    with np.load(tmp_path / "out" / "go2_norm.npz") as z:
        for k, v in zip(("state_mean", "state_std", "action_mean", "action_std"), ref):
            np.testing.assert_array_equal(z[k], v)


def trainer_windows(flags, length):
    """train.py SystemDynamicsDataset rule: window i is valid if no flagged row lies in [i, i + length - 2]."""
    idx = np.flatnonzero(flags)
    return [i for i in range(len(flags) - length + 1) if not np.any((idx >= i) & (idx < i + length - 1))]


def test_start_windows_never_cross_an_episode(tmp_path):
    tr = round0_trace()
    r = prep.prepare(save(tmp_path / "r0.npz", tr), tmp_path / "out", frac=0.2, seed=101)
    info = json.loads((tmp_path / "out" / "start_data" / "start_data.json").read_text())
    blocks = [np.loadtxt(tmp_path / "out" / "start_data" / f"state_action_data_{i}.csv", delimiter=",", ndmin=2)
              for i in range(info["files"])]
    data = np.concatenate(blocks)                                  # the trainer concatenates every file
    assert len(data) == info["rows"] == r["counts"]["start_rows"]
    assert int(data[:, -1].sum()) == r["counts"]["training"] == info["episodes"]
    held = {1000 * h["env"] + k for h in json.loads((tmp_path / "out" / "withheld.json").read_text())
            for k in [int(tr["next_state"][h["first"], h["env"], 44])]}
    assert not held & set(np.unique(data[:, 44]).astype(int))      # no withheld episode in the start data
    for i in trainer_windows(data[:, -1], 40):
        assert len(np.unique(data[i:i + 40, 44])) == 1             # one episode per start window
    cfg = __import__("train").resolve_task_config("go2_flat")
    rec = go2_study.apply_start_data(cfg, tmp_path / "out" / "start_data")
    assert cfg.data_config.batch_data_size == info["rows"] and cfg.data_config.file_data_size == info["min_file_rows"]
    assert rec["episodes"] == info["episodes"]
    (tmp_path / "out" / "start_data" / f"state_action_data_{info['files']}.csv").write_text("0\n")
    with pytest.raises(ValueError, match="more CSV files"):
        go2_study.apply_start_data(cfg, tmp_path / "out" / "start_data")


def e16_npz(tmp_path, name="e16.npz", **kw):
    tr, meta = synthetic_trace(**kw)
    return save(tmp_path / name, tr), tr, meta


def test_s2_contract_wiring_and_refusals(tmp_path):
    from configs.go2_flat_cfg import S2_REWARD_CONTRACT
    from envs.go2_flat import Go2FlatEnv
    path, tr, meta = e16_npz(tmp_path)
    cfg = __import__("train").resolve_task_config("go2_flat")
    rec = go2_study.apply_s2_contract(cfg, path)
    assert cfg.environment_config.reward_term_weights == {k: v["weight"] for k, v in S2_REWARD_CONTRACT.items()}
    assert "stand_still" not in cfg.environment_config.reward_term_weights
    assert Go2FlatEnv.timers_before_rewards is True and Go2FlatEnv.acceleration_from_state is True
    assert Go2FlatEnv.feet_air_time_threshold == 0.5 and Go2FlatEnv.default_joint_pos == list(meta["default_joint_pos"])
    assert rec["contract_sha256"] == go2_study.contract_sha256() and rec["terms"] == 16
    bad = dict(meta, reward_weights=dict(meta["reward_weights"], energy=-1.0))
    tr2 = dict(tr, metadata=np.array(json.dumps(bad)))
    with pytest.raises(ValueError, match="does not match"):
        go2_study.apply_s2_contract(cfg, save(tmp_path / "bad.npz", tr2))
    import e1_train
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"x")
    with pytest.raises(SystemExit, match="go2_flat task"):
        e1_train.main(["--wm-checkpoint", str(wm), "--reward-contract", "s2", "--asset-trace", str(path),
                       "paper", "1", "anymal_d_flat", "--seed", "1"])
    with pytest.raises(SystemExit, match="needs --asset-trace"):
        e1_train.main(["--wm-checkpoint", str(wm), "--reward-contract", "s2", "paper", "1", "go2_flat", "--seed", "1"])


def test_runtime_reward_check_pass_and_fail(tmp_path):
    import go2_reward_runtime_check as rc
    path, tr, meta = e16_npz(tmp_path, extra_hip=False)
    r = rc.run(path, tmp_path / "ok")
    assert r["verdict"] == "PASS", r["failures"]
    assert all(t["runtime_equals_reconstruction"] for t in r["terms"].values())
    # hip contacts the model cannot see: a listed approximation, reported, not a failure by itself
    path, tr, meta = e16_npz(tmp_path, name="hip.npz", extra_hip=True)
    r = rc.run(path, tmp_path / "hip")
    assert r["terms"]["undesired_contacts"]["approximation"] and r["terms"]["undesired_contacts"]["verdict"] == "DIFFERS"
    assert not any("undesired_contacts" in f for f in r["failures"])
    # a simulator term that the runtime does not reproduce and that is no approximation -> FAIL
    i = meta["reward_names"].index("energy")
    terms = tr["reward_terms"].copy()
    terms[..., i] *= 2.0
    path = save(tmp_path / "bad.npz", dict(tr, reward_terms=terms))
    r = rc.run(path, tmp_path / "bad")
    assert r["verdict"] == "FAIL" and any("energy" in f for f in r["failures"])


def summary_cell(folder, tag, cmd, seed, falls, track, speed, task="Template-Isaac-Velocity-Flat-Unitree-Go2-Start-v0",
                 push=False):
    d = folder / f"{tag}_cmd{cmd}_s{seed}"
    d.mkdir(parents=True)
    (d / "episodes_summary.json").write_text(json.dumps(dict(
        task=task, num_envs=32, num_steps=10000, seed=seed, std=0.5, cmd_resample_s=cmd, push_robot={"enabled": push},
        summary=dict(falls=falls, episodes_completed=640, track_per_step=track, speed_ratio=speed))))


@pytest.mark.parametrize("track, speed, verdict", [
    (0.934500, 0.85, "PASS"),    # bar = 0.9644946 - 0.030 = 0.9344946: rounding s2 to 0.9645 would fail this one
    (0.934490, 0.85, "FAIL"),    # 5e-6 under the unrounded bar
    (0.950000, 0.79, "FAIL"),    # too slow
])
def test_g0_gate_uses_the_unrounded_reference(tmp_path, track, speed, verdict):
    for s, t in zip((42, 43), (0.9644940, 0.9644952)):    # s2 mean 0.9644946 (prints as 0.9645 in eval.log)
        for cmd in (10, 30):
            summary_cell(tmp_path / "s2", "unitree_s2", cmd, s, 3, t, 0.95)
    for s in (42, 43):
        for cmd in (10, 30):
            summary_cell(tmp_path / "g0", "G0", cmd, s, 5, track, speed)
    r = gate.score(tmp_path / "g0", "G0", tmp_path / "s2", "unitree_s2")
    assert r["verdict"] == verdict
    assert r["rule"]["tracking_bar"] == pytest.approx(0.9344946, abs=1e-12)
    assert r["results"][10]["g0"]["rate_per_1000s"] == pytest.approx(1000 * 10 / 12800)


def test_g0_gate_stops_on_a_protocol_difference(tmp_path):
    for s in (42, 43):
        for cmd in (10, 30):
            summary_cell(tmp_path / "s2", "unitree_s2", cmd, s, 3, 0.96, 0.95)
            summary_cell(tmp_path / "g0", "G0", cmd, s, 3, 0.96, 0.95, push=(cmd == 10))
    with pytest.raises(gate.ProtocolMismatch, match="protocol"):
        gate.score(tmp_path / "g0", "G0", tmp_path / "s2", "unitree_s2")


def fake_go2_wm(path, stats, seed=0):
    from configs.go2_flat_cfg import Go2FlatConfig
    from rsl_rl.modules import SystemDynamicsEnsemble
    m = Go2FlatConfig().model_architecture_config
    torch.manual_seed(seed)
    wm = SystemDynamicsEnsemble(45, 12, m.extension_dim, m.contact_dim, m.termination_dim, "cpu",
                                ensemble_size=m.ensemble_size, history_horizon=m.history_horizon,
                                architecture_config=m.architecture_config)
    torch.save({"system_dynamics_state_dict": wm.state_dict(), "iter": 0, "normalized": True,
                **{k: torch.as_tensor(v).float() for k, v in stats.items()}}, path)
    return path


def test_s2_probe_runs_in_the_training_env(tmp_path):
    import go2_imagination_probe as probe
    from policy_loop import ObservationBuilder, Policy, consistent_policy_trace, save_actor
    tr = round0_trace(T=300, N=6)
    pol_path = save_actor(tmp_path / "s2.pt")
    policy = Policy(pol_path)
    ident = dict(state_mean=np.zeros(45, np.float32), state_std=np.ones(45, np.float32),
                 action_mean=np.zeros(12, np.float32), action_std=np.ones(12, np.float32))
    tr = consistent_policy_trace(tr, policy, ObservationBuilder("go2_flat", ident), ident)
    meta = json.loads(str(tr["metadata"]))
    meta.update(checkpoint=str(pol_path), checkpoint_sha256=policy.sha256, task="Template-Isaac-Velocity-Flat-Unitree-Go2-Start-v0")
    tr["metadata"] = np.array(json.dumps(meta))
    phys = save(tmp_path / "r0.npz", tr)
    prep.prepare(phys, tmp_path / "m", frac=0.2, seed=101)
    with np.load(tmp_path / "m" / "go2_norm.npz") as z:
        stats = {k: z[k] for k in z.files}
    wm = fake_go2_wm(tmp_path / "wm.pt", stats)
    asset, _, _ = e16_npz(tmp_path)
    r = probe.run(wm, tmp_path / "m" / "start_data", asset, phys, tmp_path / "probe", envs=8, steps=30, seed=0,
                  device="cpu")
    im, ph = r["imagination"], r["physics"]
    assert r["policy_checks"]["passed"] and im["envs"] == 8 and im["steps"] == 30
    assert len(im["term_reward_per_s"]) == 16 and np.isfinite(im["penalty_per_s"]) and im["penalty_per_s"] <= 0
    assert ph["terminations"] == 2 and ph["seconds"] == pytest.approx(300 * 6 * 0.02)
    assert 0 <= im["thigh_contact_rate"] <= 1 and len(im["foot_contact_rate"]) == 4


def test_trainer_wiring_reaches_the_run_config(tmp_path, monkeypatch):
    """e1_train.py with the A3 options: the config handed to train.run carries the contract, the start data, the
    world model, lambda 1 and the initial policy; nothing is trained (train.run is replaced)."""
    import e1_train
    import train as T
    from configs.go2_flat_cfg import S2_REWARD_CONTRACT
    from envs.go2_flat import Go2FlatEnv
    tr = round0_trace()
    prep.prepare(save(tmp_path / "r0.npz", tr), tmp_path / "m", frac=0.2, seed=101)
    asset, _, _ = e16_npz(tmp_path)
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"x")
    init = tmp_path / "pi0.pt"
    init.write_bytes(b"y")
    seen = {}
    monkeypatch.setattr(T, "run", lambda cfg: seen.setdefault("cfg", cfg))
    e1_train.main(["--wm-checkpoint", str(wm), "--penalty-weight", "-1.0", "--reward-contract", "s2",
                   "--asset-trace", str(asset), "--start-data", str(tmp_path / "m" / "start_data"),
                   "--init-policy-checkpoint", str(init), "paper", "3001", "go2_flat", "--seed", "1000"])
    cfg = seen["cfg"]
    assert cfg.environment_config.reward_term_weights == {k: v["weight"] for k, v in S2_REWARD_CONTRACT.items()}
    assert cfg.environment_config.uncertainty_penalty_weight == -1.0 and cfg.environment_config.num_envs == 4096
    assert cfg.policy_training_config.max_iterations == 2500 and cfg.model_architecture_config.resume_path == str(wm.resolve())
    assert cfg.data_config.dataset_folder == "start_data" and Path(cfg.data_config.dataset_root) == (tmp_path / "m").resolve()
    assert T.args_cli.init_policy_checkpoint == str(init.resolve()) and T.args_cli.seed == 1000
    assert Go2FlatEnv.timers_before_rewards is True


def test_g0_diagnose_collects_the_evidence(tmp_path):
    import go2_g0_diagnose as dg
    from policy_loop import ObservationBuilder, Policy, consistent_policy_trace, save_actor
    tr = round0_trace(T=300, N=6)
    s2 = Policy(save_actor(tmp_path / "s2.pt"))
    ident = dict(state_mean=np.zeros(45, np.float32), state_std=np.ones(45, np.float32),
                 action_mean=np.zeros(12, np.float32), action_std=np.ones(12, np.float32))
    tr = consistent_policy_trace(tr, s2, ObservationBuilder("go2_flat", ident), ident)
    phys = save(tmp_path / "r0.npz", tr)
    prep.prepare(phys, tmp_path / "m", frac=0.2, seed=101)
    with np.load(tmp_path / "m" / "go2_norm.npz") as z:
        stats = {k: z[k] for k in z.files}
    wm = fake_go2_wm(tmp_path / "wm.pt", stats)
    asset, _, _ = e16_npz(tmp_path)
    torch.manual_seed(5)
    g0 = save_actor(tmp_path / "g0.pt")
    for tag, folder, length in (("G0", tmp_path / "ev", 12), ("unitree_s2", tmp_path / "s2ev", 1000)):
        for s in (42, 43):
            d = folder / f"{tag}_cmd10_s{s}"
            d.mkdir(parents=True)
            with (d / "episodes.csv").open("w") as f:
                f.write("env,start,length,fall,censored\n")
                for e in range(10):
                    f.write(f"{e},0,{length},{length < 100},False\n")
                f.write("0,0,3,False,True\n")
    log = tmp_path / "train.log"
    log.write_text("".join(f"  Learning iteration {i}/2500  \n  Mean reward: {i / 100:.2f}\n  Mean episode length: 200.0\n"
                           for i in (0, 100, 500, 1000, 2499)))
    r = dg.run(g0, wm, tmp_path / "m" / "start_data", asset, phys, tmp_path / "ev", tmp_path / "s2ev", log,
               tmp_path / "out", envs=8, steps=20, device="cpu")
    pe = r["physics_episodes"]
    assert pe["g0"]["episodes"] == 20 and pe["g0"]["median_steps"] == 12 and pe["g0"]["share_under_25_steps"] == 1.0
    assert pe["s2"]["falls"] == 0
    sup = r["action_support_on_s2_observations"]
    assert sup["rows"] == 300 * 6 and 0 <= sup["share_outside_s2_range"] <= 1
    assert r["imagination_g0"]["steps"] == 20 and r["imagination_g0"]["actions"]["abs_mean"] > 0
    assert r["training"]["1000"]["mean_reward"] == 10.0 and "last (2499)" in r["training"]
