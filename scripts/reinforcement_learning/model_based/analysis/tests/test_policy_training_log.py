"""Bug 15s (23 Nov 2026): PolicyTraining.log built the per-key episode info but never printed it, so the D7 launcher's
check for "Mean episode startup_reset_fraction" could not pass. These tests use the real log() as the writer and run the
launcher's own parser (the CHECK_STARTUP block of v6_dev_go2_d7.sh) on its output."""
import re
import subprocess
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MB))
import policy_training  # noqa: E402


def runner(monkeypatch):
    pt = object.__new__(policy_training.PolicyTraining)
    pt.tot_timesteps, pt.tot_time, pt.num_steps_per_env, pt.max_iterations = 0, 0.0, 24, 3
    pt.device, pt.log_dict = "cpu", {}
    pt.env = SimpleNamespace(num_envs=8)
    pt.alg = SimpleNamespace(policy=SimpleNamespace(std=torch.ones(12)), learning_rate=1e-3, rnd=None, symmetry=None)
    monkeypatch.setattr(policy_training.wandb, "log", lambda d: None)
    return pt


def locs(it, fractions, rew=True):
    return dict(ep_infos=[{"startup_reset_fraction": f, "uncertainty": 1.0} for f in fractions], collection_time=1.0,
                learn_time=1.0, loss_dict={"surrogate": 0.1}, rewbuffer=deque([1.0] if rew else []),
                lenbuffer=deque([10.0] if rew else []), it=it, tot_iter=3, start_iter=0)


def real_log(monkeypatch, capsys, iterations):
    pt = runner(monkeypatch)
    for it, fr in enumerate(iterations):
        pt.log(locs(it, fr))
    return capsys.readouterr().out


def test_log_prints_episode_keys(monkeypatch, capsys):
    out = real_log(monkeypatch, capsys, [[0.25, 0.35]])
    assert re.search(r"Mean episode startup_reset_fraction:\s+0\.3000", out)
    assert re.search(r"Mean episode uncertainty:\s+1\.0000", out)
    # the existing parsers of "Mean episode length" still read exactly one value per iteration
    assert re.findall(r"Mean episode length:\s*([-+0-9.eE]+)", out) == ["10.00"]


def test_log_prints_episode_keys_before_the_first_completed_episode(monkeypatch, capsys):
    pt = runner(monkeypatch)
    pt.log(locs(0, [0.5], rew=False))
    out = capsys.readouterr().out
    assert re.search(r"Mean episode startup_reset_fraction:\s+0\.5000", out)


def launcher_check():
    text = (MB / "v6_dev_go2_d7.sh").read_text()
    m = re.search(r"<<'CHECK_STARTUP'[^\n]*\n(.*?)\nCHECK_STARTUP\n", text, re.S)
    assert m, "CHECK_STARTUP block not found in the D7 launcher"
    return m.group(1)


def run_check(tmp_path, log_text, q, iters):
    p = tmp_path / "train.log"
    p.write_text(log_text)
    r = subprocess.run([sys.executable, "-", str(p), str(q), str(iters)], input=launcher_check(), text=True,
                       capture_output=True, check=True)
    return [l for l in r.stdout.splitlines() if l.startswith("STARTUP ")][-1]


def test_launcher_parser_reads_the_real_log(monkeypatch, capsys, tmp_path):
    out = real_log(monkeypatch, capsys, [[0.25, 0.35], [0.3], [0.31, 0.29]])
    assert run_check(tmp_path, out, 0.2993, 3) == "STARTUP 3 0.3000 ok"
    assert run_check(tmp_path, out, 0.40, 3).endswith(" bad")       # mean far from q
    assert run_check(tmp_path, out, 0.2993, 10).endswith(" bad")    # too few logged iterations
    assert run_check(tmp_path, "no such lines\n", 0.2993, 3) == "STARTUP 0 nan bad"
