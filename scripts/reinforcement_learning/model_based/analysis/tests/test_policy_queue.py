"""Exercise the real queue/subprocess path with tiny CPU trainer/evaluator stubs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MB))
import queue_paired_policy as queue  # noqa: E402


@pytest.fixture
def project(tmp_path):
    repo = tmp_path / "repo"
    model_based = repo / queue.MB
    model_based.mkdir(parents=True)
    evaluator = repo / queue.EVALUATOR
    evaluator.parent.mkdir(parents=True)
    cold = repo / "logs/offline/anymal_d_flat/cold_1601"
    cold.mkdir(parents=True)
    wm = repo / "wm.pt"
    wm.write_bytes(b"fixed wm")
    record = {
        "run_num": 1601, "mode": "paper", "seed": 1, "task": "anymal_d_flat",
        "normalizer_contract": {"source": "config", "norm_stats_sha256": queue.NORM_HASH},
        "world_model_checkpoint": {"path": str(wm), "sha256": hashlib.sha256(wm.read_bytes()).hexdigest()},
        "initial_policy_checkpoint": None,
        "resolved_config": {"paper_settings": [4096, 100, 2500]},
    }
    (cold / "run_args.json").write_text(json.dumps(record))
    (cold / "policy_2499.pt").write_bytes(b"cold policy")
    log = repo / "cold.log"
    log.write_text("Training completed. Policy saved to cold\n")
    (model_based / "p15_train.py").write_text(textwrap.dedent('''
        import json, os, pathlib, sys
        if os.environ.get("QUEUE_TEST_FAIL") == "warm":
            raise SystemExit(7)
        mode, run, task = sys.argv[1:4]
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
        root = pathlib.Path("logs/offline/anymal_d_flat")
        d = json.loads((root / "cold_1601/run_args.json").read_text())
        d.update(mode=mode, run_num=int(run), task=task, seed=seed)
        if os.environ.get("QUEUE_TEST_FAIL") == "config":
            d["resolved_config"]["paper_settings"] = [1, 2, 3]
        out = root / ("warm_" + run)
        out.mkdir()
        (out / "run_args.json").write_text(json.dumps(d))
        (out / "policy_2499.pt").write_bytes(b"warm policy")
        print("Training completed. Policy saved to", out)
    '''))
    evaluator.write_text(textwrap.dedent('''
        import argparse, json, os, pathlib
        p = argparse.ArgumentParser()
        p.add_argument("--checkpoint")
        p.add_argument("--out_csv")
        p.add_argument("--seed", type=int)
        p.add_argument("--num_steps", type=int)
        a, other = p.parse_known_args()
        if os.environ.get("QUEUE_TEST_FAIL") == "eval":
            raise SystemExit(8)
        warm = pathlib.Path(a.checkpoint).parent.name.startswith("warm_")
        n = 8 if a.seed == 42 else 12
        track = (.8 if a.seed == 42 else .9) + (.03 if warm else 0)
        cmd = 1.0 if a.seed == 42 else 2.0
        s = dict(counted_steps=n, falls=1 if warm else 4, episodes_completed=10,
                 track_per_step=track, speed_cmd_per_step=cmd,
                 speed_act_per_step=.9*cmd, speed_ratio=.9)
        d = dict(checkpoint=a.checkpoint, task="Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0",
                 num_envs=32, num_steps=a.num_steps, seed=a.seed, std=.5, step_dt=.02,
                 episode_length_s=20, command_resampling_time_range_s=[10., 10.],
                 push_robot=dict(enabled=True), summary=s)
        if os.environ.get("QUEUE_TEST_FAIL") == "conditions" and warm and a.seed == 43:
            d["std"] = .4
        out = pathlib.Path(a.out_csv)
        out.write_text("env,fall\\n0,False\\n")
        out.with_name("episodes_summary.json").write_text(json.dumps(d))
    '''))
    return SimpleNamespace(repo=repo, cold=cold, log=log, out=tmp_path / "out")


def launch(project, **env):
    return subprocess.run([
        sys.executable, str(MB / "queue_paired_policy.py"), "--repo-root", str(project.repo),
        "--out", str(project.out), "--cold-pid", "999999998", "--cold-log", str(project.log),
        "--python", sys.executable,
    ], env=dict(os.environ, **env), capture_output=True, text=True, timeout=15)


def state(project):
    return json.loads((project.out / "status.json").read_text())["state"]


def test_queue_trains_once_evaluates_four_times_and_pools_exactly(project):
    result = launch(project)
    assert result.returncode == 0, result.stderr
    assert state(project) == "complete"
    assert len(list(project.out.glob("*_eval_*"))) == 4
    d = json.loads((project.out / "comparison.json").read_text())
    assert d["evaluation_seeds"] == [42, 43]
    assert d["num_steps"] == 10000
    assert d["cold"]["falls"] == 8 and d["warm"]["falls"] == 2
    assert d["cold"]["track_per_step"] == pytest.approx(.86)  # weighted, not .85
    assert d["warm"]["track_per_step"] == pytest.approx(.89)
    assert d["cold"]["speed_ratio"] == pytest.approx(.9)
    assert d["fall_win"] and d["tracking_win"] and d["all_warm_speed_ratios_ge_0_8"]
    manifest = json.loads((project.out / "manifest.json").read_text())
    assert manifest["cold"]["checkpoint_sha256"] == queue.sha256(project.cold / "policy_2499.pt")
    assert manifest["warm"]["run_args"]["seed"] == 1
    assert "One training-seed pair" in (project.out / "comparison.txt").read_text()


@pytest.mark.parametrize("missing", ["policy_2499.pt", "completion_marker"])
def test_partial_cold_training_never_starts_warm(project, missing):
    if missing == "completion_marker":
        project.log.write_text("interrupted\n")
    else:
        (project.cold / missing).unlink()
    result = launch(project)
    assert result.returncode == 1 and state(project) == "failed"
    assert not list(project.cold.parent.glob("*_1602"))
    assert not (project.out / "comparison.json").exists()


def test_wrong_training_seed_stops_before_warm(project):
    path = project.cold / "run_args.json"
    d = json.loads(path.read_text())
    d["seed"] = 2
    path.write_text(json.dumps(d))
    result = launch(project)
    assert result.returncode == 1 and "expected seed=1" in result.stderr
    assert not list(project.cold.parent.glob("*_1602"))


def test_world_model_change_stops_before_warm(project):
    (project.repo / "wm.pt").write_bytes(b"changed")
    result = launch(project)
    assert result.returncode == 1 and "World-model bytes changed" in result.stderr


def test_existing_warm_run_is_never_overwritten(project):
    existing = project.cold.parent / "previous_1602"
    existing.mkdir()
    result = launch(project)
    assert result.returncode == 1 and "already exists" in result.stderr
    assert not (project.out / "warm_train.log").exists()


@pytest.mark.parametrize("stage", ["warm", "config", "eval", "conditions"])
def test_failures_stop_without_a_successful_comparison(project, stage):
    result = launch(project, QUEUE_TEST_FAIL=stage)
    assert result.returncode == 1 and state(project) == "failed"
    assert not (project.out / "comparison.json").exists()
    if stage in ("warm", "config"):
        assert not list(project.out.glob("*_eval_*"))
    if stage == "eval":
        assert len(list(project.out.glob("*_eval_*"))) == 1


def test_duplicate_queue_is_locked(project):
    lock = project.repo / "logs/p2/warm_run_1602.lock"
    lock.parent.mkdir(parents=True)
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = launch(project)
    assert result.returncode == 1 and state(project) == "failed"
    assert not list(project.cold.parent.glob("*_1602"))


def test_existing_output_is_not_reused(project):
    project.out.mkdir()
    old = project.out / "comparison.json"
    old.write_text("old result")
    result = launch(project)
    assert result.returncode == 2 and old.read_text() == "old result"


def test_wait_timeout_does_not_accept_a_live_trainers_checkpoint(project, monkeypatch):
    monkeypatch.setattr(queue, "trainer_alive", lambda *args: True)
    times = iter([0., 100.])
    monkeypatch.setattr(queue.time, "monotonic", lambda: next(times))
    args = SimpleNamespace(wait_hours=.001, cold_pid=123, cold_run=1601, poll_seconds=1,
                           cold_log=project.log)
    with pytest.raises(RuntimeError, match="timed out"):
        queue.wait_for_cold(args, project.cold)


def test_dead_or_unrelated_pid_is_not_waited_on():
    assert not queue.trainer_alive(999999998, 1601)
    assert not queue.trainer_alive(os.getpid(), 1601)


def test_cpu_scoring_helper_forces_cpu_and_keeps_six_seed_scope(tmp_path):
    trace = tmp_path / "logs/wm_fit/go2_plus_fail_ens5/exploit_trace/2026-06-21_14-30-38_9_policy499_trace.csv"
    trace.parent.mkdir(parents=True)
    trace.write_text("trace fixture\n")
    for seed in range(1, 7):
        model = tmp_path / f"logs/wm_fit/p1_seed_lottery_2026-10-13/own_seed{seed}/model_2000.pt"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"fixture")
    python = tmp_path / "fake_python"
    python.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''
        import json, os, pathlib, sys
        if sys.argv[1].endswith("eval_signals.py"):
            assert sys.argv[sys.argv.index("--device") + 1] == "cpu"
            assert os.environ["OMP_NUM_THREADS"] == "2"
            out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])
            out.mkdir()
            d = dict(device_used="cpu", metrics=dict(epi=dict(pre5=dict(clean=dict(auc=.8)))))
            (out / "summary.json").write_text(json.dumps(d))
        else:
            os.execv(sys.executable, [sys.executable] + sys.argv[1:])
    '''))
    python.chmod(0o755)
    result = subprocess.run(["bash", str(MB / "score_p14_cpu.sh"), str(tmp_path), str(python)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    outputs = list((tmp_path / "logs").glob("p14_cpu.*/six_seed_scores.csv"))
    assert len(outputs) == 1
    assert len(outputs[0].read_text().splitlines()) == 7
    assert "two existing own-normalizer models" in result.stdout
