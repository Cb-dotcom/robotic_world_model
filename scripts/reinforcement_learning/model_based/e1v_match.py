#!/usr/bin/env python3
"""E1 validation (plan v5.2): check that a new arm differs from the control only where it should.

Compares two run_args.json files written by train.py (write_run_args): the control (1701, arm B)
and a candidate arm. It compares the effective configuration, not recipe names.

Must be equal: seed; the complete resolved_config except the two allowed keys; the normalizer hash;
the training flags in args (mode, termination rule/threshold/uncertainty stop, reward clip, alive
bonus, initial policy), missing older flags read as their defaults.
Allowed to differ: model_architecture_config.resume_path, environment_config.uncertainty_penalty_weight.
Must match the request: candidate world-model sha256 and penalty weight.
Both commits must contain the acceleration-reward fix (checked with git when the object exists).

Exit 0 and "MATCH: True" only if every check passes. Pure Python, CPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ALLOWED = {"model_architecture_config.resume_path", "environment_config.uncertainty_penalty_weight"}
ARG_DEFAULTS = {"mode": None, "termination_rule": "selected", "termination_threshold": 0.5,
                "termination_uncertainty": None, "only_positive_rewards": False, "alive_bonus": 0.0,
                "init_policy_checkpoint": None}
ACC_FIX = "d9d2a66"


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def has_fix(commit, repo):
    """True / False from git; None when the commit object is not in this clone."""
    if not commit:
        return None
    if subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
                      capture_output=True).returncode != 0:
        return None
    return subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", ACC_FIX, commit],
                          capture_output=True).returncode == 0


def compare(control, candidate, expect_wm_sha, expect_penalty, repo, control_fix_note=None):
    problems, notes = [], []
    if control.get("seed") != candidate.get("seed"):
        problems.append(f"seed {control.get('seed')} != {candidate.get('seed')}")
    a, b = flatten(control.get("resolved_config", {})), flatten(candidate.get("resolved_config", {}))
    if not a or not b:
        problems.append("resolved_config missing")
    for key in sorted(set(a) | set(b)):
        if key in ALLOWED:
            continue
        if key not in a or key not in b:
            problems.append(f"config key only in {'candidate' if key in b else 'control'}: {key}")
        elif a[key] != b[key]:
            problems.append(f"config differs: {key}: {a[key]!r} != {b[key]!r}")
    norm = [run.get("normalizer_contract", {}) or {} for run in (control, candidate)]
    if norm[0].get("norm_stats_sha256") is None or norm[0].get("norm_stats_sha256") != norm[1].get("norm_stats_sha256"):
        problems.append(f"normalizer sha differs: {norm[0].get('norm_stats_sha256')} != {norm[1].get('norm_stats_sha256')}")
    for key, default in ARG_DEFAULTS.items():
        x, y = control.get("args", {}).get(key, default), candidate.get("args", {}).get(key, default)
        if x != y:
            problems.append(f"training flag differs: {key}: {x!r} != {y!r}")
    wm = (candidate.get("world_model_checkpoint") or {}).get("sha256")
    if wm != expect_wm_sha:
        problems.append(f"candidate world model sha {wm} != expected {expect_wm_sha}")
    pen = b.get("environment_config.uncertainty_penalty_weight")
    if pen is None or float(pen) != float(expect_penalty):
        problems.append(f"candidate penalty {pen} != expected {expect_penalty}")
    for name, run in (("control", control), ("candidate", candidate)):
        commit = (run.get("git") or {}).get("commit")
        fix = has_fix(commit, repo)
        if fix is False:
            problems.append(f"{name} commit {commit} lacks the acceleration fix {ACC_FIX}")
        elif fix is None:
            if name == "control" and control_fix_note:
                notes.append(f"control commit {commit} not in this clone; {control_fix_note}")
            else:
                problems.append(f"{name} commit {commit} cannot be checked for {ACC_FIX}")
        if (run.get("git") or {}).get("dirty"):
            notes.append(f"{name} worktree was dirty: {(run.get('git') or {}).get('status')}")
    notes.append(f"allowed differences: control penalty {a.get('environment_config.uncertainty_penalty_weight')}, "
                 f"candidate penalty {pen}; candidate world model {wm}")
    return problems, notes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--control", type=Path, required=True, help="run_args.json of the control (1701)")
    p.add_argument("--candidate", type=Path, required=True, help="run_args.json of the new arm")
    p.add_argument("--expect-wm-sha", required=True)
    p.add_argument("--expect-penalty", type=float, required=True)
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--control-fix-note", default=None,
                   help="accepted record if the control commit is absent from this clone")
    args = p.parse_args(argv)
    control, candidate = (json.loads(path.read_text()) for path in (args.control, args.candidate))
    problems, notes = compare(control, candidate, args.expect_wm_sha, args.expect_penalty, args.repo,
                              args.control_fix_note)
    for line in notes:
        print("note:", line)
    for line in problems:
        print("PROBLEM:", line)
    print(f"MATCH: {not problems}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
