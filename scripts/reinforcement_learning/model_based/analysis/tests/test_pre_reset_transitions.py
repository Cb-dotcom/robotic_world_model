"""Terminal-state retention, episode boundaries, action alignment and replay tests."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MB))
sys.path.insert(0, str(MB / "analysis"))
from pre_reset_transitions import SCHEMA, TransitionBuffer, array, gather, validate_trace, windows
import analyse_transition_fidelity as fidelity
import run_transition_diagnostic as queue


def sample_trace(T=130, N=2):
    state = np.zeros((T, N, 45), np.float32)
    state[:, :, 6:9] = (0, 0, -1)
    state[:, :, 0] = np.arange(T)[:, None] * .001
    target = state.copy()
    target[:-1] = state[1:]
    done = np.zeros((T, N), bool)
    done[69, 0] = True
    done[99, 1] = True
    terminated = done.copy()
    terminated[99, 1] = False
    time_out = done & ~terminated
    # A terminal next state must differ from the subsequent reset state.
    target[69, 0, 8] = 0.0
    target[99, 1, 0] = 10.0
    meta = {"schema": SCHEMA, "verified_terminal_rows": 2, "step_dt": .02,
            "reward_names": ["track_lin_vel_xy_exp", "dof_acc_l2", "feet_air_time"],
            "reward_weights": {"track_lin_vel_xy_exp": 1.0, "dof_acc_l2": -2.5e-7, "feet_air_time": .5}}
    policy = np.zeros((T, N, 48), np.float32)
    return dict(state=state, next_state=target, action=np.zeros((T, N, 12), np.float32),
                previous_action=np.zeros((T, N, 12), np.float32),
                command=np.zeros((T, N, 3), np.float32), policy_observation=policy,
                current_air_time=np.zeros((T, N, 4), np.float32), last_air_time=np.zeros((T, N, 4), np.float32),
                current_contact_time=np.zeros((T, N, 4), np.float32), last_contact_time=np.zeros((T, N, 4), np.float32),
                contact=np.zeros((T, N, 8), np.float32), termination_observation=terminated[..., None].astype(np.float32),
                reward_terms=np.zeros((T, N, 3), np.float32), done=done, terminated=terminated, time_out=time_out,
                metadata=np.array(json.dumps(meta)))


def test_terminal_copied_before_reset_and_zero_action_is_valid(tmp_path):
    store = TransitionBuffer(1, 2)
    state, fallen, action = torch.zeros(2, 45), torch.ones(2, 45), torch.zeros(2, 12)
    store.begin(state=state, action=action)
    flags = np.ones(2, bool)
    store.end(next_state=fallen, observed_action=action, done=flags,
              terminated=np.array([True, False]), time_out=np.array([False, True]))
    store.verify_before_reset([0, 1], fallen, action)
    fallen.zero_()  # actual simulator reset mutates the source buffers
    action.fill_(99)
    assert (store.data["next_state"] == 1).all()
    assert (store.data["action"] == 0).all()
    path = tmp_path / "trace.npz"
    store.save(path, {})
    with np.load(path, allow_pickle=False) as trace:
        assert validate_trace(trace)["verified_terminal_rows"] == 2
    with pytest.raises(FileExistsError):
        store.save(path, {})


def test_wrong_action_and_changed_terminal_state_rejected():
    store = TransitionBuffer(1, 1)
    store.begin(state=np.zeros((1, 45)), action=np.ones((1, 12)))
    with pytest.raises(ValueError, match="action"):
        store.end(next_state=np.zeros((1, 45)), observed_action=np.zeros((1, 12)),
                  done=np.array([True]), terminated=np.array([True]), time_out=np.array([False]))
    store.end(next_state=np.ones((1, 45)), observed_action=np.ones((1, 12)),
              done=np.array([True]), terminated=np.array([True]), time_out=np.array([False]))
    with pytest.raises(ValueError, match="Terminal state"):
        store.verify_before_reset([0], np.zeros((1, 45)), np.ones((1, 12)))
    with pytest.raises(ValueError, match="not checked"):
        store.save("unused.npz", {})


def test_unmarked_discontinuity_and_legacy_schema_rejected():
    trace = sample_trace()
    validate_trace(trace)
    trace["state"][10, 0, 0] = 10
    with pytest.raises(ValueError, match="discontinuity"):
        validate_trace(trace)
    trace["metadata"] = np.array('{"schema":"legacy"}')
    with pytest.raises(ValueError, match="legacy"):
        validate_trace(trace)


@pytest.mark.parametrize("horizon", [1, 8])
def test_windows_retain_terminal_endpoint_and_do_not_cross_reset(horizon):
    trace = sample_trace()
    origins, labels, counts = windows(trace, 32, horizon)
    assert counts["eligible"]["fall_now"] == 1
    row = origins[labels == "fall_now"][0]
    assert tuple(row) == (69 - horizon + 1, 0)
    # Timeout row is not a physical positive.
    assert not any(n == 1 and t + horizon - 1 == 99 for t, n in origins[labels == "fall_now"])
    for t, n in origins:
        assert not trace["done"][t - 31:t + horizon - 1, n].any()
    assert len(origins) > 0  # every applied action in this trace is truly zero
    other = windows(trace, 32, horizon)
    np.testing.assert_array_equal(origins, other[0])
    assert counts["selected"] == other[2]["selected"]


class ActionIntegrator:
    """Exact dynamics: next component 0 = state component 0 + applied action 0."""
    ensemble_size = 2
    def reset(self):
        pass
    def __call__(self, state, actions, ids):
        value = state[:, -1].clone()
        value[:, 0] += actions[:, -1, 0]
        return value, torch.zeros(len(value)), torch.zeros(len(value)), None, torch.zeros(len(value), 8), torch.zeros(len(value), 1)


@pytest.mark.parametrize("horizon", [1, 8])
def test_replay_action_target_alignment(horizon):
    trace = sample_trace()
    for t in range(130):
        trace["state"][t, :, 0] = t * (t + 1) / 2
        trace["action"][t, :, 0] = t + 1
        trace["next_state"][t, :, 0] = (t + 1) * (t + 2) / 2
    origin = np.array([[40, 0], [45, 1]])
    stats = {"state_mean": torch.zeros(45), "state_std": torch.ones(45),
             "action_mean": torch.zeros(12), "action_std": torch.ones(12)}
    predictions = fidelity.predict(ActionIntegrator(), SimpleNamespace(statistics=stats), trace, origin, horizon)[0]
    targets = gather(trace["next_state"], origin, horizon - 1, horizon)[:, 0]
    for head in predictions:
        np.testing.assert_array_equal(head, targets)


def test_recorder_adapter_captures_fall_and_timeout_before_mutation():
    """Exercise actual callback bodies without importing Isaac Sim on CPU."""
    script = MB.parent / "rsl_rl/record_pre_reset_transitions.py"
    nodes = [node for node in ast.parse(script.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("PreResetRecorder", "timer_fields")]
    store = TransitionBuffer(1, 2)
    class Term:
        def __init__(self, env):
            self._env = env
    scope = {"RecorderTerm": Term, "buffer": store, "policy_observation": torch.zeros(2, 48), "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(script), "exec"), scope)
    state, action = torch.zeros(2, 45), torch.ones(2, 12)
    contact_data = SimpleNamespace(**{k: torch.zeros(2, 4) for k in
        ("current_air_time", "last_air_time", "current_contact_time", "last_contact_time")})
    sensor = SimpleNamespace(data=contact_data, find_bodies=lambda pattern: ([0, 1, 2, 3], []))
    env = SimpleNamespace(num_envs=2, step_dt=.02,
        observation_manager=SimpleNamespace(compute_group=lambda name: state),
        action_manager=SimpleNamespace(action=action, prev_action=torch.zeros(2, 12)),
        command_manager=SimpleNamespace(get_command=lambda name: torch.ones(2, 3)),
        scene={"contact_forces": sensor, "robot": SimpleNamespace(data=SimpleNamespace(joint_acc=torch.zeros(2, 12)))},
        reward_manager=SimpleNamespace(_step_reward=torch.ones(2, 2), active_terms=["a", "b"]),
        reward_buf=torch.full((2,), .04), reset_buf=torch.ones(2, dtype=torch.bool),
        reset_terminated=torch.tensor([True, False]), reset_time_outs=torch.tensor([False, True]),
        episode_length_buf=torch.tensor([12, 1000]))
    term = scope["PreResetRecorder"](env)
    term.record_pre_step()
    state[:, 8] = .7
    env.obs_buf = {"system_state": state, "system_action": action,
                   "system_contact": torch.ones(2, 8), "system_termination": torch.tensor([[1.], [0.]])}
    env.obs_buf["system_termination"][1] = 1
    with pytest.raises(RuntimeError, match="termination label"):
        term.record_post_step()
    env.obs_buf["system_termination"][1] = 0
    term.record_post_step()
    term.record_pre_reset(torch.tensor([0, 1]))
    state.zero_()
    assert store.verified_terminal_rows == 2
    np.testing.assert_allclose(store.data["next_state"][0, :, 8], .7)
    assert store.data["terminated"][0].tolist() == [True, False]
    assert store.data["time_out"][0].tolist() == [False, True]


def test_real_state_reward_reconstruction_uses_velocity_and_previous_action():
    trace = sample_trace()
    trace["command"][:, :, 0] = .5
    trace["next_state"][:, :, 0] = .5
    trace["reward_terms"][:, :, 0] = 1
    # Distinct q and qdot: catches the previously fixed acceleration slice.
    trace["policy_observation"][:, :, 12:24] = 10
    trace["policy_observation"][:, :, 24:36] = 2
    trace["next_state"][:, :, 21:33] = 2
    result = fidelity.reward_reconstruction(trace, np.array([[40, 0], [41, 0]]), json.loads(str(trace["metadata"])))
    assert result["track_lin_vel_xy_exp"]["mean_absolute_error"] == 0
    assert result["dof_acc_l2"]["mean_absolute_error"] == 0


def test_queue_command_has_two_short_recordings_not_training():
    args = queue.command("python", Path("actor.pt"), Path("new.npz"))
    assert args[args.index("--num_steps") + 1] == "5000"
    assert args[args.index("--num_envs") + 1] == "16"
    assert "train.py" not in " ".join(args)
    args = queue.command("python", Path("actor.pt"), Path("smoke.npz"), smoke=True)
    assert args[args.index("--num_steps") + 1] == "2"


@pytest.mark.parametrize("failure", [None, "capacity", "recording"])
def test_queue_end_to_end_order_and_stops(tmp_path, monkeypatch, failure):
    monkeypatch.chdir(tmp_path)
    wm = tmp_path / "assets/models/pretrain_rnn_ens.pt"
    wm.parent.mkdir(parents=True)
    wm.write_bytes(b"frozen WM")
    baseline = tmp_path / "baseline_1601"
    baseline.mkdir()
    ckpt = baseline / "policy_2499.pt"
    ckpt.write_bytes(b"saved actor")
    saved, out = tmp_path / "saved", tmp_path / "fresh"
    saved.mkdir()
    out.mkdir()
    (saved / "manifest.json").write_text(json.dumps({"cold": {"checkpoint_sha256": queue.pair.sha256(ckpt)}}))
    monkeypatch.setattr(queue, "EXPECTED_WM", queue.pair.sha256(wm))
    monkeypatch.setattr(queue.pair, "MB", MB)
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(queue.pair, "run_directory", lambda *a: baseline)
    monkeypatch.setattr(queue.pair, "provenance", lambda *a: {
        "git": {"commit": queue.review_baseline_git.__globals__["BASELINE_COMMIT"], "dirty": False, "status": []},
        "world_model_checkpoint": {"sha256": queue.EXPECTED_WM}})
    monkeypatch.setattr(queue, "gpu_free_gib", lambda: 4 if failure == "capacity" else 20)
    calls = []
    def execute(command, log, repo):
        calls.append(command)
        if failure == "recording":
            raise RuntimeError("Simulator failed")
        if command[1] == str(queue.RECORDER):
            np.savez_compressed(command[command.index("--out") + 1], **sample_trace())
        else:
            (out / "report.json").write_text("{}")
            (out / "report.txt").write_text("Synthetic only")
    monkeypatch.setattr(queue.pair, "execute", execute)
    states = []
    args = SimpleNamespace(out=out, baseline_pair=saved, python="python")
    status = lambda value, **kw: states.append((value, kw))
    if failure:
        with pytest.raises(RuntimeError):
            queue.run(args, status)
        assert len(calls) == (0 if failure == "capacity" else 1)
        assert not (out / "report.json").exists()
    else:
        queue.run(args, status)
        assert len(calls) == 4
        assert [detail.get("arm") for name, detail in states if name == "recording"] == ["smoke", "baseline", "reference"]
        assert states[-1][0] == "complete"
        assert not any("train.py" in " ".join(call) for call in calls)


def test_queue_main_accepts_own_launch_files_and_records_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    for name in ("queue.log", "queue.pid"):
        (out / name).touch()
    monkeypatch.setattr(sys, "argv", ["run_transition_diagnostic.py", "--baseline-pair", "saved", "--out", str(out)])
    def fail(*args):
        raise RuntimeError("Simulator unavailable")
    monkeypatch.setattr(queue, "run", fail)
    assert queue.main() == 1
    assert json.loads((out / "status.json").read_text())["state"] == "failed"
    (out / "old.npz").write_bytes(b"earlier evidence")
    with pytest.raises(SystemExit):
        queue.main()
    assert (out / "old.npz").read_bytes() == b"earlier evidence"
