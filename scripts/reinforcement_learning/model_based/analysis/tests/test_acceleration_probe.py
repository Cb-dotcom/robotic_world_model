"""CPU-only checks of the one-run probe's comparison and launch safeguards."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import run_acceleration_probe as probe
from test_policy_queue import project  # shared tiny saved-baseline fixture


def test_nohup_launch_accepts_its_log_and_pid_files(tmp_path):
    # Exercise the actual main() entry point and the supplied shell launch.
    # Only replace the GPU orchestration body; startup and freshness checks run.
    code = (
        f"import sys; sys.path.insert(0, {str(Path(probe.__file__).parent)!r}); "
        "import run_acceleration_probe as p; "
        "p.run = lambda args, status: status('cpu_launch_checked'); "
        "raise SystemExit(p.main())"
    )
    out = tmp_path / "out"
    out.mkdir()
    launch = '''
nohup "$1" -c "$2" --baseline-pair "$3" --out "$3" > "$3/queue.log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$3/queue.pid"
wait "$launcher_pid"
'''
    run = subprocess.run(["bash", "-c", launch, "launch", sys.executable, code, str(out)],
                         cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, (out / "queue.log").read_text()
    assert json.loads((out / "status.json").read_text())["state"] == "cpu_launch_checked"


@pytest.mark.parametrize("existing", ["status.json", "train.log", "manifest.json", "corrected_eval_42"])
def test_main_still_rejects_prior_run_artifacts(tmp_path, monkeypatch, existing):
    (tmp_path / existing).write_text("previous result")
    monkeypatch.setattr(sys, "argv", [probe.__file__, "--baseline-pair", str(tmp_path), "--out", str(tmp_path)])
    monkeypatch.setattr(probe, "run", lambda *args: pytest.fail("must reject before starting jobs"))
    with pytest.raises(SystemExit) as exc:
        probe.main()
    assert exc.value.code == 2


def records(falls=10, tracking=.9, speed=.9):
    return [dict(seed=seed, task=probe.pair.TASK, num_envs=32, num_steps=10000,
                 std=.5, step_dt=.02, episode_length_s=20,
                 command_resampling_time_range_s=[10., 10.], push_robot={"enabled": True},
                 summary=dict(falls=falls, counted_steps=100, episodes_completed=50,
                              track_per_step=tracking, speed_ratio=speed,
                              speed_cmd_per_step=1., speed_act_per_step=speed))
            for seed in (42, 43)]


@pytest.mark.parametrize("falls,tracking,speed,passed", [
    (8, .90, .90, True), (9, .90, .90, False),
    (8, .88, .90, False), (8, .90, .79, False),
])
def test_screen_is_material_fall_gain_without_lost_tracking_or_speed(falls, tracking, speed, passed):
    assert probe.result(records(), records(falls, tracking, speed))["engineering_screen_pass"] is passed


def test_changed_evaluation_settings_abort():
    changed = records()
    changed[1]["push_robot"]["enabled"] = False
    with pytest.raises(RuntimeError, match="conditions differ"):
        probe.result(records(), changed)


@pytest.mark.parametrize("key", ["seed", "resolved_config", "normalizer_contract", "world_model_checkpoint"])
def test_changed_training_settings_abort(key):
    old = dict(seed=1, task="anymal_d_flat", mode="paper", resolved_config={},
               normalizer_contract={}, world_model_checkpoint={}, initial_policy_checkpoint=None)
    new = copy.deepcopy(old)
    new[key] = "changed"
    with pytest.raises(RuntimeError, match=key):
        probe.check_training_match(old, new)


@pytest.mark.parametrize("failure", [None, "busy", "duplicate", "train"])
def test_one_training_two_evaluations_or_stop_before_extra_jobs(project, monkeypatch, failure):
    monkeypatch.chdir(project.repo)
    p = project.cold / "run_args.json"
    baseline = json.loads(p.read_text())
    baseline["git"] = {"dirty": False}
    p.write_text(json.dumps(baseline))
    old_ckpt = project.cold / "policy_2499.pt"
    saved = project.repo / "saved_pair"
    saved.mkdir()
    (saved / "manifest.json").write_text(json.dumps({"cold": {"checkpoint_sha256": probe.pair.sha256(old_ckpt)}}))
    project.out.mkdir()
    (project.repo / "assets/data").mkdir(parents=True)
    (project.repo / "assets/data/state_action_data_0.csv").write_text("fixed data")
    if failure == "duplicate":
        (project.cold.parent / "already_1701").mkdir()
    monkeypatch.setattr(probe, "check_source", lambda repo: None)
    monkeypatch.setattr(probe, "gpu_free_gib", lambda: 1 if failure == "busy" else 20)
    monkeypatch.setattr(probe, "read_eval", lambda directory, ckpt, seed:
                        records(8 if "corrected" in directory.name else 10)[seed - 42])
    calls = []
    def execute(command, log, repo):
        calls.append(command)
        if "--training-child" in command:
            if failure == "train":
                raise RuntimeError("training failed")
            new = project.cold.parent / "corrected_1701"
            new.mkdir()
            record = dict(baseline, run_num=1701)
            (new / "run_args.json").write_text(json.dumps(record))
            (new / "policy_2499.pt").write_bytes(b"corrected")
            log.write_text("Training completed. Policy saved to corrected")
        else:
            log.write_text("eval complete")
    monkeypatch.setattr(probe.pair, "execute", execute)
    args = SimpleNamespace(out=project.out, baseline_pair=saved, python=sys.executable)
    states = []
    status = lambda state, **details: states.append(state)
    if failure:
        with pytest.raises(RuntimeError):
            probe.run(args, status)
        assert len(calls) == (1 if failure == "train" else 0)
    else:
        probe.run(args, status)
        assert len(calls) == 3
        assert "--training-child" in calls[0]
        assert [c[c.index("--seed") + 1] for c in calls[1:]] == ["42", "43"]
        assert states[-1] == "complete"
        assert json.loads((project.out / "comparison.json").read_text())["engineering_screen_pass"]
