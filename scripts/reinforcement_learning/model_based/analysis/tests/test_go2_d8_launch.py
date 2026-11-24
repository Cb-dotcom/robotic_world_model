"""A4 D8 launcher: fresh folder, clean-tree and same-commit guards, approval gating of GPU stages, status / resume /
retry, gate results vs implementation errors, wall time per stage, and the exact commands of every stage."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MB))
import go2_d8_launch as L  # noqa: E402


def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for rel in L.CODE:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"# {rel}\n")
    (root / ".gitignore").write_text("logs/\nd8/\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"], cwd=root, check=True)
    paths = {}
    for name in L.INPUTS:
        p = root / "logs" / "in" / f"{name}.bin"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode())
        paths[name] = str(p)
    (root / "logs" / "d7eval").mkdir()
    paths["d7_eval"] = str(root / "logs" / "d7eval")
    return root, paths


@pytest.fixture
def no_prefixes(monkeypatch):
    monkeypatch.setattr(L, "INPUTS", {k: (None, None) for k in L.INPUTS})


def init(root, paths, approval="decisions-log 2026-11-26 D8", **kw):
    return L.Launch.init(root / "d8", approval, paths, kw.pop("q_rule", "recompute"), kw.pop("eval_30", False),
                         root=root, **kw)


def test_init_records_head_and_refuses_unsafe_starts(tmp_path, no_prefixes, monkeypatch):
    root, paths = repo(tmp_path)
    man = init(root, paths)
    assert man["head"] == L.git("rev-parse", "HEAD", cwd=root) and man["gpu_allowed"]
    assert (root / "d8" / "inputs.sha256").read_text().count("\n") == len(L.CODE)
    with pytest.raises(L.StageError, match="fresh folder"):
        init(root, paths)
    (root / L.CODE[0]).write_text("changed\n")                               # a tracked change
    with pytest.raises(L.StageError, match="not clean"):
        L.Launch.init(root / "d8b", "x", paths, "d6b", False, root=root)
    subprocess.run(["git", "checkout", "-q", "--", L.CODE[0]], cwd=root, check=True)
    (root / "scripts" / "stray.py").write_text("x\n")                         # an untracked code file
    with pytest.raises(L.StageError, match="not clean"):
        L.Launch.init(root / "d8c", "x", paths, "d6b", False, root=root)
    (root / "scripts" / "stray.py").unlink()
    (root / "logs" / "offline" / "go2_flat" / "2026-12-01_10-00-00_4004").mkdir(parents=True)
    with pytest.raises(L.StageError, match="run IDs already used"):
        L.Launch.init(root / "d8d", "x", paths, "d6b", False, root=root)
    monkeypatch.setattr(L, "INPUTS", dict(L.INPUTS, d2_norm=(None, "a3a5c954")))
    with pytest.raises(L.StageError, match="does not start with a3a5c954"):
        L.Launch.init(root / "d8e", "x", paths, "d6b", False, root=root)


def fake_stages(calls, fail=None, gate=None):
    def make(name):
        def f():
            calls.append(name)
            if name == fail:
                raise L.StageError(f"{name} broke")
            if name == gate:
                raise L.StageGate(f"{name} gate failed")
            return [], f"{name} ok"
        return f
    return {n: make(n) for n, _ in L.STAGES}


def test_run_resume_retry_and_gates(tmp_path, no_prefixes):
    root, paths = repo(tmp_path)
    init(root, paths)
    calls = []
    launch = L.Launch(root / "d8", root=root)
    assert launch.run(stages=fake_stages(calls, fail="fit")) == "error:fit"
    assert calls == ["preflight", "record", "probe_gate", "test_first", "prepare", "fit"]
    st = launch.status("fit")
    assert st["state"] == "error" and "fit broke" in st["note"] and st["wall_s"] >= 0
    # resume: done stages are skipped; the failed stage blocks until it is retried explicitly
    calls.clear()
    assert launch.run(stages=fake_stages(calls)).startswith("blocked:fit")
    assert calls == []
    (root / "d8" / "fit" / "fit.log").write_text("attempt 1 log\n")
    assert launch.run(stages=fake_stages(calls), retry="fit") == "complete"
    assert calls == ["fit", "retrain", "qualify", "train", "match", "eval", "score"]
    assert (root / "d8" / "fit.attempt1" / "fit.log").read_text() == "attempt 1 log\n"     # logs kept
    assert (root / "d8" / "status" / "fit.attempt1.json").exists()
    log = (root / "d8" / "queue.log").read_text()
    assert "[d8 fit] error: fit broke" in log and "[d8 score] done: score ok" in log
    assert all(launch.status(n)["state"] == "done" for n, _ in L.STAGES)


def test_gate_result_ends_the_run_and_is_not_retried(tmp_path, no_prefixes):
    root, paths = repo(tmp_path)
    init(root, paths)
    launch, calls = L.Launch(root / "d8", root=root), []
    assert launch.run(stages=fake_stages(calls, gate="qualify")) == "gate_failed:qualify"
    assert launch.run(stages=fake_stages(calls), retry="qualify") == "gate_failed:qualify"
    assert calls.count("qualify") == 1 and "train" not in calls


def test_gpu_stages_need_an_approval_record(tmp_path, no_prefixes):
    root, paths = repo(tmp_path)
    init(root, paths, approval="none")
    calls = []
    assert L.Launch(root / "d8", root=root).run(stages=fake_stages(calls)) == "needs_approval:record"
    assert calls == ["preflight"]


def test_same_commit_guard_between_stages(tmp_path, no_prefixes):
    root, paths = repo(tmp_path)
    init(root, paths)
    launch, calls = L.Launch(root / "d8", root=root), []
    assert launch.run(stages=fake_stages(calls), until="record") == "stopped"
    (root / L.CODE[3]).write_text("edited between stages\n")
    with pytest.raises(L.StageError, match="tree changed"):
        launch.run(stages=fake_stages(calls))
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "new"], cwd=root, check=True)
    with pytest.raises(L.StageError, match="launch HEAD"):
        launch.run(stages=fake_stages(calls))


def test_done_outputs_are_hash_checked_on_resume(tmp_path, no_prefixes):
    root, paths = repo(tmp_path)
    init(root, paths)
    launch = L.Launch(root / "d8", root=root)
    out = root / "d8" / "preflight" / "x.json"

    def pre():
        out.write_text("{}")
        return [out], "ok"
    st = dict(fake_stages([]), preflight=pre)
    assert launch.run(stages=st, until="preflight") == "stopped"
    assert launch.status("preflight")["outputs"] == {"preflight/x.json": L.sha256(out)}
    out.write_text('{"edited": 1}')
    with pytest.raises(L.StageError, match="outputs changed"):
        launch.run(stages=st)


def _qualify_runner(q1, q3, q4):
    def runner(cmd, log, gpu):
        Path(log).write_text("ok\n")
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        if cmd[1].endswith("go2_d2_qualify.py"):
            rules = dict(Q1=dict(pass_=None, **{"pass": q1}), Q1b={"pass": True}, Q2={"pass": False}, Q3={"pass": q3})
            (out / "d2_qualify.json").write_text(json.dumps(dict(results=dict(rules=rules))))
        else:
            (out / "startup_check.json").write_text(json.dumps(dict(q4_pass=q4)))
        return 0
    return runner


def test_qualify_gates_on_q1_q3_only(tmp_path, no_prefixes):
    root, paths = repo(tmp_path)
    init(root, paths)
    launch = L.Launch(root / "d8", root=root, runner=_qualify_runner(True, True, False))
    (root / "d8" / "qualify").mkdir()
    outs, note = launch.stage_qualify()                                 # Q4 fails: reported, not gating
    assert "Q4" in note and json.loads((root / "d8" / "qualify" / "gate.json").read_text())["reported"]["Q4"] is False
    for q1, q3 in ((False, True), (True, False)):
        sub = tmp_path / f"r{q1}{q3}"
        sub.mkdir()
        r2, p2 = repo(sub)
        init(r2, p2)
        l2 = L.Launch(r2 / "d8", root=r2, runner=_qualify_runner(q1, q3, True))
        (r2 / "d8" / "qualify").mkdir()
        with pytest.raises(L.StageGate, match="stops before PPO"):
            l2.stage_qualify()


def test_commands_follow_the_draft(tmp_path, no_prefixes):
    root, paths = repo(tmp_path)
    init(root, paths, eval_30=True, rules="proposed-2026-11-25", rule_record="decisions-log 2026-11-26")
    c = L.Launch(root / "d8", root=root).commands()
    j = lambda cmd: " ".join(cmd)
    probe, batch = c["record"]
    assert "--seed 916" in j(probe) and "--num_steps 2000" in j(probe) and "--num_envs 64" in j(probe)
    assert "--seed 104" in j(batch) and "--num_steps 5000" in j(batch) and "--friction-scale 0.3" in j(batch)
    fit = j(c["fit"][0])
    for flag in ("--iterations 500", "--lr 0.0001", "--warm_start", f"--reference_wm {paths['d6b_model']}",
                 f"--norm_stats {paths['d2_norm']}", "--seed 2", "--termination_pos_weight 1", "--startup_segments"):
        assert flag in fit, flag
    assert f"--calibration {paths['calibration_914']}" in j(c["retrain"][0])
    f8, u8 = c["train"]
    assert f8[f8.index("--wm-checkpoint") + 1] == paths["d6b_model"] and u8[u8.index("--wm-checkpoint") + 1].endswith("d8s_wm_seed2.pt")
    strip = lambda cmd: [x for i, x in enumerate(cmd) if i not in (cmd.index("--wm-checkpoint") + 1,)]
    assert [x for x in strip(f8) if x not in ("4003",)] == [x for x in strip(u8) if x not in ("4004",)]   # only the model differs
    assert "4003" in f8 and "4004" in u8 and "--max-iterations 300" in j(f8) and "--seed 1000" in j(f8)
    assert f"--init-policy-checkpoint {paths['d7_policy']}" in j(f8) and "--penalty-weight -1.0" in j(f8)
    ev = [j(x) for x in c["eval"]]
    assert sum("cmd10" in x for x in ev) == 4 and sum("cmd30" in x for x in ev) == 8     # F8, U8 at 10 s; all four at 30 s
    assert all("--num_envs 32 --num_steps 10000" in x for x in ev)
    assert "--rules proposed-2026-11-25" in j(c["score"][0]) and "--also-cmd 30" in j(c["score"][0])
    assert "--q-rule recompute" in j(c["prepare"][0])


def test_fit_provenance_is_read_from_the_checkpoint(tmp_path):
    import torch
    p = tmp_path / "wm.pt"
    torch.save(dict(iter=500, normalized=True, provenance=dict(b3_reset_per_head=True, init_mode="warm_start", seed=2,
                                                               startup=dict(windows=10))), p)
    assert L.Launch.fit_provenance(p) == dict(iter=500, b3_reset_per_head=True, init_mode="warm_start", normalized=True,
                                              seed=2, startup_windows_positive=True)
    torch.save(dict(iter=500, normalized=True, provenance=dict(init_mode="scratch", seed=2)), p)
    assert L.Launch.fit_provenance(p)["init_mode"] == "scratch" and L.Launch.fit_provenance(p)["b3_reset_per_head"] is None
