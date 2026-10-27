import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_fall_paths as replay
from test_pre_reset_transitions import ActionIntegrator, sample_trace


def test_fall_selection_requires_full_history_and_preserves_terminal_target():
    trace = sample_trace()
    origins, counts = replay.fall_origins(trace)
    assert origins.tolist() == [[45, 0]]
    assert counts == dict(actual_falls=1, included=1, short_history_excluded=0)
    trace["done"][40, 0] = True  # explicit reset inside context, even if action is zero
    assert replay.fall_origins(trace)[0].shape == (0, 2)


def test_early_termination_is_not_mistaken_for_missed_endpoint():
    probability = np.zeros((2, 4, 1))
    probability[0, 1, 0] = .8
    # Both endpoint probabilities are zero; head0 nevertheless stopped at step2.
    first, active = replay.first_and_active(probability)
    assert first.tolist() == [[2], [0]]
    assert active[0, :, 0].tolist() == [True, True, False, False]
    states = np.zeros((2, 4, 1, 45))
    states[0, 2:, :, :3] = 100  # error on inactive forced continuation
    outputs = states, np.zeros((2, 4, 1, 8)), probability, np.ones((2, 4, 1))
    result = replay.summarise(outputs, np.zeros((4, 1, 45)), .02)
    assert result["stopped_anywhere_fraction_per_head"] == [1, 0]
    assert result["steps"][-1]["velocity_rmse_active_paths"] == 0
    assert result["steps"][-1]["velocity_rmse_all_forced_paths"] > 0


def test_logged_action_alignment_and_real_imagined_first_step_match():
    trace = sample_trace()
    for t in range(130):
        trace["state"][t, :, 0] = t*(t+1)/2
        trace["action"][t, :, 0] = t+1
        trace["next_state"][t, :, 0] = (t+1)*(t+2)/2
    origins = np.array([[40, 0], [42, 1]])
    stats = dict(state_mean=torch.zeros(45), state_std=torch.ones(45),
                 action_mean=torch.zeros(12), action_std=torch.ones(12))
    contract = SimpleNamespace(statistics=stats)
    path = replay.imagined_path(ActionIntegrator(), contract, trace, origins, 4)
    target = replay.gather(trace["next_state"], origins, 0, 4).transpose(1, 0, 2)
    for head in path[0]:
        np.testing.assert_array_equal(head, target)
    one = replay.predict(ActionIntegrator(), contract, trace, origins, 1)
    for real, imagined in zip(one, path):
        np.testing.assert_array_equal(real, imagined[:, 0])


def test_cpu_main_writes_reports_and_refuses_overwrite(tmp_path, monkeypatch):
    class Model(ActionIntegrator):
        def eval(self):
            return self
    stats = dict(state_mean=torch.zeros(45), state_std=torch.ones(45),
                 action_mean=torch.zeros(12), action_std=torch.ones(12))
    contract = SimpleNamespace(statistics=stats, metadata=lambda: {"synthetic": True})
    monkeypatch.setattr(replay, "load_model", lambda p: (Model(), contract))
    trace = tmp_path / "trace.npz"
    wm = tmp_path / "wm.pt"
    wm.write_bytes(b"synthetic")
    np.savez_compressed(trace, **sample_trace())
    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["replay", "--trace", str(trace), "--wm", str(wm), "--out", str(out)])
    replay.main()
    result = json.loads((out / "paired_fall_replay.json").read_text())
    assert result["counts"]["included"] == 1
    assert len(result["imagined_context"]["steps"]) == 25
    with np.load(out / "paired_fall_replay.npz") as data:
        assert data["imagined_state"].shape == (2, 25, 1, 45)
    original = (out / "paired_fall_replay.json").read_bytes()
    with pytest.raises(SystemExit):
        replay.main()
    assert original == (out / "paired_fall_replay.json").read_bytes()
