#!/usr/bin/env python3
"""A4 D8 staged launcher (draft 24 Nov 2026; nothing runs on the GPU without a dated approval record).

Run from the robotic_world_model root inside the container. Subcommands:
  init   OUT --approval TEXT|none ...   fresh OUT; checks the clean tree, records HEAD, the code hashes (inputs.sha256),
                                        every input file's sha256 against the expected prefix, that runs 4003 and 4004
                                        are free, and the launch settings (launch_manifest.json). With --approval none
                                        only CPU stages can run.
  plan   OUT                            prints every stage's exact commands (nothing runs)
  run    OUT [--until STAGE] [--retry STAGE]
                                        runs the stages in order, one at a time (GPU stages strictly sequential)
  status OUT                            prints the status table with wall time per stage
Stages (CPU / GPU):
  preflight  CPU  D8 tests (none may be skipped); saved-reference check (D7 and s2 cells; 30 s cells listed if missing)
  record     GPU  D7 at x0.30 friction: probe seed 916 (64 x 2000), batch seed 104 (64 x 5000); read-back required
  probe_gate CPU  probe counts; >= 20 startup falls and >= 30 time-outs, else the run stops (a valid gate result)
  test_first CPU  descriptive: D6b on probe 916 (go2_d6_startup_check, go2_d2_qualify); written before any fit
  prepare    CPU  go2_d8_prepare (D6b data + batch training split; split check on the written files)
  fit        GPU  warm start from D6b, 500 it, lr 1e-4, fit seed 2, D2's normalizer, per-head reset, pos-weight 1,
                  startup windows, uniform windows (D6's minibatch settings otherwise)
  retrain    GPU  option 1 on calibration 914 (w in 1, 10, 30, 100; false stops <= 0.10); no candidate = gate result
  qualify    CPU  Q1 (round-0 withheld) and Q3 (probe 916, full surviving episode) gate PPO; Q1b, Q2, Q4 and the
                  40-step startup readout are reported only (a Q4 failure never stops the run)
  train      GPU  F8 (run 4003, D6b) then U8 (run 4004, updated); D7 init, new optimizer, 300 it, seed 1000, s2 reward,
                  lambda 1, warm + startup resets at share q, the same start bank
  match      CPU  go2_d8_match on both run_args.json (same clean HEAD, differ only in the world model)
  eval       GPU  F8 and U8 at 10 s commands, seeds 42 + 43, 32 x 10,000 (and 30 s for all four policies if set)
  score      CPU  go2_d8_score (components; the readout only with the frozen rule set and its record)
Outcomes per stage: done | gate_failed (a valid failed gate: the run ends, logs kept, no retry) | error (an
implementation or infrastructure failure: logs kept; `run --retry STAGE` moves the stage folder to STAGE.attemptN and
reruns it). /isaac-sim/python.sh turns any non-zero exit into 1 (bug 15q), so outcomes are read from output files.
Before every stage: HEAD = the launch HEAD, no tracked change, code hashes = inputs.sha256.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

MB_REL = "scripts/reinforcement_learning/model_based"
AN_REL = f"{MB_REL}/analysis"
RSL_REL = "scripts/reinforcement_learning/rsl_rl"
TASK = "Template-Isaac-Velocity-Flat-Unitree-Go2-Start-v0"
STAGES = (("preflight", "cpu"), ("record", "gpu"), ("probe_gate", "cpu"), ("test_first", "cpu"), ("prepare", "cpu"),
          ("fit", "gpu"), ("retrain", "gpu"), ("qualify", "cpu"), ("train", "gpu"), ("match", "cpu"), ("eval", "gpu"),
          ("score", "cpu"))
CODE = [f"{MB_REL}/{f}" for f in ("go2_d8_launch.py", "go2_d8_prepare.py", "go2_startup_prepare.py", "go2_round0_prepare.py",
                                    "fit_world_model_go2_pilot_segments.py", "refit/fit_reset_per_head.py", "e1_train.py",
                                    "policy_training.py", "train.py", "p15_train.py", "go2_study.py", "run_provenance.py",
                                    "envs/base.py", "envs/go2_flat.py")] + \
       [f"{AN_REL}/{f}" for f in ("go2_d8_score.py", "go2_d8_match.py", "go2_d8_references.py", "go2_stop_retrain.py",
                                    "go2_d2_qualify.py", "go2_d6_startup_check.py", "go2_g0_replay.py", "go2_startup_hazard.py",
                                    "go2_d3_fall_timing.py", "fall_rate.py", "episode_stats.py", "pre_reset_transitions.py")] + \
       [f"{RSL_REL}/{f}" for f in ("eval_episode_stats.py", "record_pre_reset_transitions.py", "friction_shift.py")]
TESTS = [f"{AN_REL}/tests/{t}" for t in ("test_go2_d8_score.py", "test_go2_d8_prepare.py", "test_go2_d8_match.py",
                                           "test_go2_d8_references.py", "test_go2_d8_launch.py", "test_go2_stop_retrain.py",
                                           "test_go2_startup_prepare.py", "test_go2_startup_resets.py",
                                           "test_go2_d2_qualify.py", "test_go2_d6_startup_check.py", "test_warm_resets.py",
                                           "test_policy_training_log.py")]
# inputs: name -> (default path, expected sha256 prefix; from the D8 draft and the run-log)
INPUTS = {
    "d7_policy": (None, "38e4805e"),
    "d6b_model": ("logs/p2/a4_d6b_1123/option1/run/d6s_wm_seed2.pt", "54208372"),
    "d6b_manifest": ("logs/p2/a4_d6b_1123/data/manifest.json", "73ed93f5"),
    "d2_norm": ("logs/p2/a4_d2_1121/data/go2_norm.npz", "a3a5c954"),
    "calibration_914": ("logs/p2/a4_d1_cal3/probe.npz", "62e20429"),
    "s2_policy": (None, "bb9d1241"),
    "round0": ("logs/p2/go2_round0_1118/round0_s2_seed101.npz", "3e7ce462"),
    "e16_asset": ("logs/p2/v6w2_rest_1118/e16_s2_check.npz", None),
}
SETTINGS = dict(friction=0.30, probe_seed=916, probe_steps=2000, batch_seed=104, batch_steps=5000, envs_record=64,
                fit_seed=2, fit_iterations=500, fit_lr=1e-4, ppo_iterations=300, ppo_seed=1000, runs=dict(F8=4003, U8=4004),
                eval_seeds=[42, 43], eval_envs=32, eval_steps=10000, penalty_weight=-1.0, d7_falls_10s=306, s2_falls_10s=219)


class StageGate(Exception):
    """A valid failed gate: the run ends here; the result is reported."""


class StageError(Exception):
    """An implementation or infrastructure failure."""


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args, cwd="."):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def tree_state(root="."):
    """(HEAD, tracked changes, untracked code files under scripts/)."""
    head = git("rev-parse", "HEAD", cwd=root)
    tracked = [l for l in git("status", "--porcelain", "--untracked-files=no", cwd=root).splitlines() if l.strip()]
    untracked = [l[3:] for l in git("status", "--porcelain", "--untracked-files=all", "--", "scripts", cwd=root).splitlines()
                 if l.startswith("??") and re.search(r"\.(py|sh)$", l)]
    return head, tracked, untracked


class Launch:
    def __init__(self, out, root=".", runner=None, gpu_free_mb=None, sleep=time.sleep):
        self.out, self.root = Path(out), Path(root)
        self.runner = runner or self._run_cmd
        self.gpu_free_mb = gpu_free_mb or self._gpu_free_mb
        self.sleep = sleep
        self.manifest = json.loads((self.out / "launch_manifest.json").read_text()) if (self.out / "launch_manifest.json").exists() else None

    # ------------------------------------------------------------ init
    @staticmethod
    def init(out, approval, paths, q_rule, eval_30, rules=None, rule_record=None, root="."):
        out, root = Path(out), Path(root)
        if out.exists():
            raise StageError(f"{out} exists: a D8 launch always starts in a fresh folder")
        head, tracked, untracked = tree_state(root)
        if tracked or untracked:
            raise StageError(f"the tree is not clean (tracked {tracked[:3]}, untracked code {untracked[:3]})")
        if q_rule not in ("d6b", "recompute"):
            raise StageError("--q-rule must be d6b or recompute (fixed before recording)")
        inputs = {}
        for name, (default, prefix) in INPUTS.items():
            p = Path(paths.get(name) or default or "")
            if not p.is_file():
                raise StageError(f"input {name}: {p} not found")
            digest = sha256(p)
            if prefix and not digest.startswith(prefix):
                raise StageError(f"input {name}: sha256 {digest[:12]} does not start with {prefix}")
            inputs[name] = dict(path=str(p), sha256=digest)
        refs = Path(paths.get("d7_eval") or "logs/p2/a4_d7_1123b/eval")
        if not refs.is_dir():
            raise StageError(f"saved D7 / s2 evaluations not found: {refs}")
        busy = [r for r in SETTINGS["runs"].values() for d in (root / "logs" / "offline" / "go2_flat").glob(f"*_{r}")]
        if busy:
            raise StageError(f"run IDs already used: {sorted(set(busy))}")
        code = {}
        for rel in CODE:
            if not (root / rel).is_file():
                raise StageError(f"code file missing: {rel}")
            code[rel] = sha256(root / rel)
        out.mkdir(parents=True)
        (out / "status").mkdir()
        (out / "inputs.sha256").write_text("".join(f"{v}  {k}\n" for k, v in code.items()))
        man = dict(protocol="A4 D8 launch (draft 24 Nov 2026)", created_utc=now(), head=head, approval=approval,
                   gpu_allowed=approval not in (None, "", "none"), inputs=inputs, d7_eval=str(refs), code_sha256=code,
                   settings=dict(SETTINGS, q_rule=q_rule, eval_30=bool(eval_30), rules=rules, rule_record=rule_record))
        (out / "launch_manifest.json").write_text(json.dumps(man, indent=1))
        return man

    # ------------------------------------------------------------ guards and status
    def guard(self):
        head, tracked, untracked = tree_state(self.root)
        if head != self.manifest["head"]:
            raise StageError(f"HEAD {head[:12]} != launch HEAD {self.manifest['head'][:12]}")
        if tracked or untracked:
            raise StageError(f"the tree changed since launch: {tracked[:3]} {untracked[:3]}")
        for rel, digest in self.manifest["code_sha256"].items():
            if sha256(self.root / rel) != digest:
                raise StageError(f"code file changed since launch: {rel}")

    def status_path(self, stage):
        return self.out / "status" / f"{stage}.json"

    def status(self, stage):
        p = self.status_path(stage)
        return json.loads(p.read_text()) if p.exists() else None

    def _write_status(self, stage, rec):
        self.status_path(stage).write_text(json.dumps(rec, indent=1))
        with (self.out / "queue.log").open("a") as f:
            wall = f" ({rec['wall_s']:.0f} s)" if rec.get("wall_s") is not None else ""
            note = f": {rec['note']}" if rec.get("note") else ""
            f.write(f"{now()} [d8 {stage}] {rec['state']}{note}{wall}\n")

    def _archive(self, stage):
        n = 1
        while (self.out / f"{stage}.attempt{n}").exists() or (self.out / "status" / f"{stage}.attempt{n}.json").exists():
            n += 1
        if (self.out / stage).exists():
            (self.out / stage).rename(self.out / f"{stage}.attempt{n}")
        self.status_path(stage).rename(self.out / "status" / f"{stage}.attempt{n}.json")
        return n

    def outputs_ok(self, rec):
        return all((self.out / rel).is_file() and sha256(self.out / rel) == d for rel, d in rec.get("outputs", {}).items())

    # ------------------------------------------------------------ running
    def run(self, until=None, retry=None, stages=None):
        table = stages or {name: getattr(self, f"stage_{name}") for name, _ in STAGES}
        kinds = dict(STAGES)
        for name, kind in STAGES:
            if name not in table:
                continue
            rec = self.status(name)
            if rec and rec["state"] == "done":
                if not self.outputs_ok(rec):
                    raise StageError(f"{name}: recorded outputs changed since the stage finished")
                if name == until:
                    return "stopped"
                continue
            if rec and rec["state"] == "gate_failed":
                return f"gate_failed:{name}"
            if rec and rec["state"] in ("error", "running"):
                if retry != name:
                    return f"blocked:{name} (state {rec['state']}; rerun with --retry {name})"
                self._archive(name)
            if kinds[name] == "gpu" and not self.manifest["gpu_allowed"]:
                return f"needs_approval:{name}"
            self.guard()
            (self.out / name).mkdir(exist_ok=True)
            t0 = time.time()
            self._write_status(name, dict(stage=name, state="running", started_utc=now(), head=self.manifest["head"]))
            try:
                outputs, note = table[name]()
                state = "done"
            except StageGate as e:
                outputs, note, state = {}, str(e), "gate_failed"
            except StageError as e:
                outputs, note, state = {}, str(e), "error"
            rels = {}
            for p in outputs or []:
                rel = os.path.relpath(Path(p).resolve(), self.out.resolve())
                rels[rel] = sha256(self.out / rel)
            self._write_status(name, dict(stage=name, state=state, started_utc=self.status(name)["started_utc"],
                                          ended_utc=now(), wall_s=time.time() - t0, head=self.manifest["head"],
                                          outputs=rels, note=note))
            if state != "done":
                return f"{state}:{name}"
            if name == until:
                return "stopped"
        return "complete"

    def table(self):
        rows = []
        for name, kind in STAGES:
            r = self.status(name) or {}
            rows.append(f"{name:11s} {kind:3s} {r.get('state', '-'):12s} "
                        f"{(str(round(r['wall_s'])) + ' s') if r.get('wall_s') is not None else '':>9s}  {r.get('note') or ''}")
        return "\n".join(rows)

    # ------------------------------------------------------------ commands
    def py(self):
        return os.environ.get("PY", "/isaac-sim/python.sh")

    def _run_cmd(self, cmd, log, gpu):
        # as the D6 / D7 launchers: the rsl_rl_rwm fork next to this repo, never $RWM/source (namespace shadow)
        pp = os.environ.get("D8_PYTHONPATH", f"/isaac-sim/kit/extscore/omni.client.lib:{(self.root / '..' / 'rsl_rl_rwm').resolve()}")
        env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0" if gpu else "", PYTHONPATH=pp)
        with open(log, "w") as f:
            return subprocess.run(cmd, cwd=self.root, stdout=f, stderr=subprocess.STDOUT, env=env).returncode

    @staticmethod
    def _gpu_free_mb():
        r = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        return int(r.stdout.strip() or 0)

    def gpu_wait(self, need_mb, what):
        while self.gpu_free_mb() < need_mb:
            with (self.out / "queue.log").open("a") as f:
                f.write(f"{now()} [d8] waiting for {need_mb // 1024} GiB GPU before {what}\n")
            self.sleep(300)

    def call(self, stage, tag, cmd, gpu=False, need_mb=8192):
        log = self.out / stage / f"{tag}.log"
        if gpu:
            self.gpu_wait(need_mb, f"{stage}/{tag}")
        code = self.runner([str(c) for c in cmd], log, gpu)
        return code, log

    def inp(self, name):
        return self.manifest["inputs"][name]["path"]

    def S(self, key):
        return self.manifest["settings"][key]

    def commands(self):
        """Every command of every stage, for `plan` and for the tests; paths relative to the repo root."""
        o, PY, S = self.out, self.py(), self.S
        d8, rec = o / "prepare" / "data", o / "record"
        wm, wms = o / "fit" / "d8_wm_seed2.pt", o / "retrain" / "run" / "d8s_wm_seed2.pt"
        d6b_data = str(Path(self.inp("d6b_manifest")).parent)
        common_train = ["--penalty-weight", S("penalty_weight"), "--reward-contract", "s2", "--asset-trace", self.inp("e16_asset"),
                        "--start-data", d8 / "start_data", "--init-policy-checkpoint", self.inp("d7_policy"),
                        "--max-iterations", S("ppo_iterations"), "--startup-resets", d8 / "reset_states.npz",
                        "--startup-share", "{q}"]
        cmds = {
            "preflight": [[PY, "-m", "pytest", "-q", *TESTS],
                          [PY, f"{AN_REL}/go2_d8_references.py", "--ref", self.manifest["d7_eval"], "--friction", S("friction"),
                           "--cmd", "10", "--cmd", "30", "--expect-ckpt", "D7=" + self.manifest["inputs"]["d7_policy"]["sha256"],
                           "--expect-ckpt", "S2=" + self.manifest["inputs"]["s2_policy"]["sha256"], "--expect-falls", f"D7:10={S('d7_falls_10s')}",
                           "--expect-falls", f"S2:10={S('s2_falls_10s')}", "--out", o / "preflight" / "references.json"]],
            "record": [[PY, f"{RSL_REL}/record_pre_reset_transitions.py", "--task", TASK, "--checkpoint", self.inp("d7_policy"),
                        "--num_envs", S("envs_record"), "--num_steps", steps, "--seed", seed, "--cmd_resample_s", "10",
                        "--friction-scale", S("friction"), "--headless", "--out", rec / f"{name}.npz"]
                       for name, seed, steps in (("probe", S("probe_seed"), S("probe_steps")),
                                                 ("batch", S("batch_seed"), S("batch_steps")))],
            "probe_gate": [[PY, f"{MB_REL}/go2_d8_prepare.py", "probe-counts", "--probe", rec / "probe.npz",
                            "--out", o / "probe_gate" / "probe_counts.json"]],
            "test_first": [[PY, f"{AN_REL}/go2_d6_startup_check.py", "--wm", self.inp("d6b_model"), "--norm", self.inp("d2_norm"),
                            "--probe", f"d7_916={rec / 'probe.npz'}", "--out", o / "test_first" / "startup",
                            "--reset-torque", "zero"],
                           [PY, f"{AN_REL}/go2_d2_qualify.py", "--wm", self.inp("d6b_model"), "--norm", self.inp("d2_norm"),
                            "--round0", self.inp("round0"), "--r0-withheld", f"{d6b_data}/r0/withheld.json",
                            "--probe", rec / "probe.npz", "--out", o / "test_first" / "qualify"]],
            "prepare": [[PY, f"{MB_REL}/go2_d8_prepare.py", "prepare", "--d6b-data", d6b_data, "--batch", rec / "batch.npz",
                         "--probe", rec / "probe.npz", "--q-rule", S("q_rule"), "--out", d8]],
            "fit": [[PY, f"{MB_REL}/refit/fit_reset_per_head.py", "--config", "go2_flat", "--data", d8 / "segments",
                     "--output", wm, "--norm_stats", self.inp("d2_norm"), "--seed", S("fit_seed"),
                     "--iterations", S("fit_iterations"), "--num_mini_batches", "20", "--mini_batch_size", "2048",
                     "--lr", S("fit_lr"), "--weight_decay", "0", "--max_grad_norm", "1", "--termination_loss_weight", "1",
                     "--termination_pos_weight", "1", "--lazy_windows", "--save_interval", "500", "--skip_short_segments",
                     "--startup_segments", d8 / "startup.json", "--warm_start", "--reference_wm", self.inp("d6b_model"),
                     "--device", "cuda:0"]],
            "retrain": [[PY, f"{AN_REL}/go2_stop_retrain.py", "--wm", wm, "--norm", self.inp("d2_norm"),
                         "--segments", d8 / "segments", "--calibration", self.inp("calibration_914"),
                         "--out", o / "retrain" / "run", "--device", "cuda:0", "--model-name", "d8s_wm_seed2.pt"]],
            "qualify": [[PY, f"{AN_REL}/go2_d2_qualify.py", "--wm", wms, "--norm", self.inp("d2_norm"), "--round0", self.inp("round0"),
                         "--r0-withheld", f"{d6b_data}/r0/withheld.json", "--probe", rec / "probe.npz",
                         "--out", o / "qualify" / "q"],
                        [PY, f"{AN_REL}/go2_d6_startup_check.py", "--wm", wms, "--norm", self.inp("d2_norm"),
                         "--probe", f"d7_916={rec / 'probe.npz'}", "--withheld",
                         f"r0_withheld={self.inp('round0')}:{d6b_data}/r0/withheld.json", "--out", o / "qualify" / "startup",
                         "--reset-torque", "zero"]],
            "train": [[PY, f"{MB_REL}/e1_train.py", "--wm-checkpoint", model, *common_train, "paper_warm", run, "go2_flat",
                       "--seed", S("ppo_seed")]
                      for model, run in ((self.inp("d6b_model"), S("runs")["F8"]), (wms, S("runs")["U8"]))],
            "match": [[PY, f"{AN_REL}/go2_d8_match.py", "--f8", "{F8_run_args}", "--u8", "{U8_run_args}",
                       "--expect", o / "match" / "expect.json"]],
            "eval": [[PY, f"{RSL_REL}/eval_episode_stats.py", "--task", TASK, "--checkpoint", ck, "--num_envs", S("eval_envs"),
                      "--num_steps", S("eval_steps"), "--seed", s, "--std", ".5", "--cmd_resample_s", cmd,
                      "--friction-scale", S("friction"), "--headless", "--out_csv", o / "eval" / f"{tag}_cmd{cmd}_s{s}" / "episodes.csv"]
                     for cmd in ([10, 30] if S("eval_30") else [10])
                     for tag, ck in ((("F8", "{F8_policy}"), ("U8", "{U8_policy}")) +
                                     ((("D7", self.inp("d7_policy")), ("S2", self.inp("s2_policy"))) if cmd == 30 else ()))
                     for s in S("eval_seeds")],
            "score": [[PY, f"{AN_REL}/go2_d8_score.py", "--new", o / "eval", "--ref", self.manifest["d7_eval"],
                       "--friction", S("friction"), "--cmd", "10"] + (["--also-cmd", "30"] if S("eval_30") else []) +
                      (["--rules", S("rules"), "--rule-record", S("rule_record")] if S("rules") else []) +
                      ["--out", o / "score" / "d8_score.json"]],
        }
        return {k: [[str(x) for x in c] for c in v] for k, v in cmds.items()}

    # ------------------------------------------------------------ stages (outcomes read from files)
    def stage_preflight(self):
        c = self.commands()["preflight"]
        code, log = self.call("preflight", "tests", c[0])
        text = log.read_text()
        if code != 0 or re.search(r"\d+ (failed|error)", text):
            raise StageError(f"tests failed ({log})")
        if re.search(r"\d+ skipped", text):
            raise StageError(f"tests were skipped ({log})")
        self.call("preflight", "references", c[1])
        refs = self.out / "preflight" / "references.json"
        if not refs.is_file():
            raise StageError("the reference check wrote nothing")
        r = json.loads(refs.read_text())
        if not r["ok"] or r["schedules"].get("10", {}).get("status") != "OK":
            raise StageError(f"saved references invalid: {r['problems'][:3]}")
        missing30 = r["schedules"].get("30", {}).get("status") == "MISSING"
        if missing30 and not self.S("eval_30"):
            note = "10 s references OK; 30 s references missing (not evaluated: eval_30 is off)"
        elif missing30:
            note = "10 s references OK; 30 s references missing: the eval stage evaluates D7 and s2 at 30 s"
        else:
            note = "10 s and 30 s references OK"
        return [refs, log], note

    def stage_record(self):
        outs = []
        for cmd, name in zip(self.commands()["record"], ("probe", "batch")):
            npz = self.out / "record" / f"{name}.npz"
            code, log = self.call("record", name, cmd, gpu=True, need_mb=8192)
            if not npz.is_file():
                raise StageError(f"the recorder wrote no {npz} (exit {code}; {log})")
            if "[friction] verified in the simulator" not in log.read_text(errors="replace"):
                raise StageError(f"no friction read-back in {log}")
            os.chmod(npz, 0o444)
            outs += [npz, log]
        return outs, "probe 916 and batch 104 recorded with verified friction"

    def stage_probe_gate(self):
        self.call("probe_gate", "counts", self.commands()["probe_gate"][0])
        p = self.out / "probe_gate" / "probe_counts.json"
        if not p.is_file():
            raise StageError("probe counts not written")
        r = json.loads(p.read_text())
        if not r["minimum"]["ok"]:
            raise StageGate(f"probe minimum not met: {r['minimum']} (stop and report; Charbel decides)")
        return [p], f"probe: {r['counts']['startup_falls']} startup falls, {r['counts']['time_outs']} time-outs"

    def stage_test_first(self):
        c = self.commands()["test_first"]
        self.call("test_first", "startup", c[0])
        self.call("test_first", "qualify", c[1])
        outs = [self.out / "test_first" / "startup" / "startup_check.json", self.out / "test_first" / "qualify" / "d2_qualify.json"]
        for p in outs:
            if not p.is_file():
                raise StageError(f"test first wrote no {p}")
        return outs, "descriptive: D6b on probe 916 (written before the fit)"

    def stage_prepare(self):
        self.call("prepare", "prepare", self.commands()["prepare"][0])
        d = self.out / "prepare" / "data"
        if not (d / "manifest.json").is_file() or not (d / "split_check.json").is_file():
            raise StageError("prepare wrote no manifest or split check")
        m = json.loads((d / "manifest.json").read_text())
        if not m["split_check_ok"]:
            raise StageError(f"held-out rows in the outputs: {json.loads((d / 'split_check.json').read_text())['groups']}")
        return [d / "manifest.json", d / "split_check.json", d / "reset_states.npz", d / "startup.json",
                d / "start_data" / "start_data.json", d / "segments" / "manifest.csv"], \
            f"D7 window share {m['window_census']['d7_share_uniform']:.4f}; q {m['startup_reset_share']:.4f} ({m['q_rule']})"

    def stage_fit(self):
        wm = self.out / "fit" / "d8_wm_seed2.pt"
        code, log = self.call("fit", "fit", self.commands()["fit"][0], gpu=True, need_mb=12288)
        if not wm.is_file():
            raise StageError(f"the fit wrote no model (exit {code}; {log})")
        text = log.read_text(errors="replace")
        nsha = self.manifest["inputs"]["d2_norm"]["sha256"]
        if not re.search(r"normalizer loaded from .*" + nsha, text):
            raise StageError("the fit did not log D2's normalizer hash")
        if "warm-started from reference WM" not in text:
            raise StageError("the fit did not warm start from D6b")
        prov = self.fit_provenance(wm)
        want = dict(iter=self.S("fit_iterations"), b3_reset_per_head=True, init_mode="warm_start", normalized=True,
                    seed=self.S("fit_seed"), startup_windows_positive=True)
        if prov != want:
            raise StageError(f"fit checkpoint provenance {prov} != {want}")
        return [wm, log], f"warm start from D6b, {prov['iter']} it, lr {self.S('fit_lr')}, per-head reset, startup windows"

    @staticmethod
    def fit_provenance(path):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        p = ck.get("provenance", {})
        return dict(iter=ck.get("iter"), b3_reset_per_head=p.get("b3_reset_per_head"), init_mode=p.get("init_mode"),
                    normalized=bool(ck.get("normalized")), seed=p.get("seed"),
                    startup_windows_positive=bool((p.get("startup") or {}).get("windows", 0) > 0))

    def stage_retrain(self):
        self.call("retrain", "retrain", self.commands()["retrain"][0], gpu=True, need_mb=4096)
        rec = self.out / "retrain" / "run" / "option1.json"
        if not rec.is_file():
            raise StageError("stop retraining wrote no result")
        r = json.loads(rec.read_text())
        if r.get("chosen_w") is None:
            raise StageGate("no stop-layer weight meets false stops <= 0.10 on calibration 914: D8 stops (no retuning)")
        return [rec, self.out / "retrain" / "run" / "d8s_wm_seed2.pt"], f"w = {r['chosen_w']:g}"

    def stage_qualify(self):
        c = self.commands()["qualify"]
        self.call("qualify", "q1_q3", c[0])
        self.call("qualify", "startup", c[1])           # Q4 and the 40-step readout: descriptive, never gating
        q = self.out / "qualify" / "q" / "d2_qualify.json"
        if not q.is_file():
            raise StageError("the qualifier wrote nothing")
        res = json.loads(q.read_text())["results"]["rules"]
        q1, q3 = bool(res["Q1"]["pass"]), bool(res["Q3"]["pass"])
        s = self.out / "qualify" / "startup" / "startup_check.json"
        desc = dict(Q1b=res["Q1b"]["pass"], Q2=res["Q2"]["pass"],
                    Q4=json.loads(s.read_text())["q4_pass"] if s.is_file() else "not written")
        (self.out / "qualify" / "gate.json").write_text(json.dumps(dict(Q1=q1, Q3=q3, reported=desc), indent=1))
        if not (q1 and q3):
            raise StageGate(f"Q1 {q1}, Q3 {q3}: D8 stops before PPO (reported: {desc})")
        return [q, self.out / "qualify" / "gate.json"], f"Q1 and Q3 pass; reported {desc}"

    def _q(self):
        return json.loads((self.out / "prepare" / "data" / "manifest.json").read_text())["startup_reset_share"]

    def stage_train(self):
        q, outs = self._q(), []
        for cmd, arm in zip(self.commands()["train"], ("F8", "U8")):
            cmd = [x.replace("{q}", repr(q)) for x in cmd]
            self.call("train", arm, cmd, gpu=True, need_mb=12288)
            log = self.out / "train" / f"{arm}.log"
            text = log.read_text(errors="replace")
            done = re.findall(r"^Training completed\. Policy saved to (logs/offline/go2_flat/[^/]+)\.$", text, re.M)
            run = str(self.S("runs")[arm])
            if len(done) != 1 or not done[0].endswith(f"_{run}"):
                raise StageError(f"{arm}: no unique completion record for run {run}")
            its = [int(i) for i in re.findall(r"Learning iteration (\d+)/\d+", text)]
            if len(its) != self.S("ppo_iterations") or its != list(range(its[0], its[0] + len(its))):
                raise StageError(f"{arm}: expected {self.S('ppo_iterations')} consecutive PPO iterations")
            for pat in (r"\[go2\] reward contract s2: 16 terms", rf"\[d6\] startup resets: share {re.escape(repr(q))}",
                        r"warm_start_resets= True"):
                if not re.search(pat, text):
                    raise StageError(f"{arm}: train log lacks '{pat}'")
            run_dir = self.root / done[0]
            pol = run_dir / f"policy_{its[-1]}.pt"
            if not pol.is_file():
                raise StageError(f"{arm}: final policy {pol} missing")
            (self.out / "train" / f"{arm}.json").write_text(json.dumps(dict(run_dir=str(run_dir), policy=str(pol),
                                                                              policy_sha256=sha256(pol)), indent=1))
            outs += [self.out / "train" / f"{arm}.json", log]
        return outs, "F8 then U8 trained"

    def stage_match(self):
        tr = {a: json.loads((self.out / "train" / f"{a}.json").read_text()) for a in ("F8", "U8")}
        wms = self.out / "retrain" / "run" / "d8s_wm_seed2.pt"
        expect = dict(head=self.manifest["head"], seed=self.S("ppo_seed"), task="go2_flat", mode="paper_warm",
                      init_policy_sha256=self.manifest["inputs"]["d7_policy"]["sha256"],
                      startup_resets_sha256=sha256(self.out / "prepare" / "data" / "reset_states.npz"),
                      startup_reset_share=self._q(), max_iterations=self.S("ppo_iterations"),
                      penalty_weight=self.S("penalty_weight"), wm_f8_sha256=self.manifest["inputs"]["d6b_model"]["sha256"],
                      wm_u8_sha256=sha256(wms))
        (self.out / "match" / "expect.json").write_text(json.dumps(expect, indent=1))
        cmd = [x.replace("{F8_run_args}", str(Path(tr["F8"]["run_dir"]) / "run_args.json"))
                .replace("{U8_run_args}", str(Path(tr["U8"]["run_dir"]) / "run_args.json")) for x in self.commands()["match"][0]]
        self.call("match", "match", cmd)
        text = (self.out / "match" / "match.log").read_text()
        if "MATCH: True" not in text:
            raise StageError(f"the lanes differ beyond the world model ({self.out / 'match' / 'match.log'})")
        return [self.out / "match" / "expect.json", self.out / "match" / "match.log"], "lanes match (same clean HEAD)"

    def stage_eval(self):
        pol = {a: json.loads((self.out / "train" / f"{a}.json").read_text())["policy"] for a in ("F8", "U8")}
        outs = []
        for cmd in self.commands()["eval"]:
            cmd = [x.replace("{F8_policy}", pol["F8"]).replace("{U8_policy}", pol["U8"]) for x in cmd]
            cell = Path(cmd[cmd.index("--out_csv") + 1]).parent
            cell.mkdir(parents=True, exist_ok=False)
            code, log = self.call("eval", cell.name, cmd, gpu=True, need_mb=8192)
            shutil.copyfile(log, cell / "eval.log")
            if "[friction] verified in the simulator" not in log.read_text(errors="replace"):
                raise StageError(f"no friction read-back in {log}")
            if not (cell / "episodes_summary.json").is_file():
                raise StageError(f"no summary for {cell.name} (exit {code})")
            outs += [cell / "episodes_summary.json", cell / "episodes.csv"]
        return outs, f"{len(outs) // 2} cells"

    def stage_score(self):
        self.call("score", "score", self.commands()["score"][0])
        p = self.out / "score" / "d8_score.json"
        if not p.is_file():
            raise StageError(f"the scorer wrote nothing (invalid cell? {self.out / 'score' / 'score.log'})")
        return [p, p.with_suffix(".txt")], (self.out / "score" / "d8_score.txt").read_text().splitlines()[-1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("out", type=Path)
    i.add_argument("--approval", required=True, help="the dated decisions-log entry approving the GPU stages, or 'none'")
    i.add_argument("--q-rule", required=True, choices=("d6b", "recompute"))
    i.add_argument("--eval-30", action="store_true", help="also evaluate all four policies at 30 s commands")
    i.add_argument("--rules", default=None)
    i.add_argument("--rule-record", default=None)
    for name in list(INPUTS) + ["d7_eval"]:
        i.add_argument(f"--{name.replace('_', '-')}", default=None)
    for name in ("plan", "status"):
        sub.add_parser(name).add_argument("out", type=Path)
    r = sub.add_parser("run")
    r.add_argument("out", type=Path)
    r.add_argument("--until", default=None, choices=[s for s, _ in STAGES])
    r.add_argument("--retry", default=None, choices=[s for s, _ in STAGES])
    a = p.parse_args(argv)
    try:
        if a.cmd == "init":
            if (a.rules is None) != (a.rule_record is None):
                p.error("--rules and --rule-record go together")
            paths = {k: getattr(a, k) for k in list(INPUTS) + ["d7_eval"]}
            man = Launch.init(a.out, a.approval, paths, a.q_rule, a.eval_30, a.rules, a.rule_record)
            print(f"[d8] launch folder {a.out}; HEAD {man['head']}; GPU stages {'allowed' if man['gpu_allowed'] else 'NOT allowed (no approval)'}")
            return 0
        L = Launch(a.out)
        if L.manifest is None:
            raise StageError(f"{a.out} has no launch_manifest.json (run init first)")
        if a.cmd == "plan":
            for stage, cmds in L.commands().items():
                for c in cmds:
                    print(f"{stage}: {shlex.join(c)}")
            return 0
        if a.cmd == "status":
            print(L.table())
            return 0
        res = L.run(until=a.until, retry=a.retry)
        print(L.table())
        print(f"[d8] {res}")
        return 0 if res in ("complete", "stopped") else 1
    except StageError as e:
        print(f"[d8] ERROR: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
