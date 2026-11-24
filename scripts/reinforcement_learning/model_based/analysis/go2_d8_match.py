#!/usr/bin/env python3
"""A4 D8: the two PPO lanes (F8 = frozen D6b model, U8 = updated model) must differ only in the world model.

Reads both lanes' run_args.json (run_provenance.write_run_args) and checks:
  source   git.commit == the launch HEAD and git.dirty is False in BOTH lanes (same clean revision)
  request  seed, task, mode, initial policy sha256 (D7), startup reset bank sha256 and share, max iterations and the
           uncertainty penalty weight equal the launch manifest's values
  matched  after flattening, every key is equal between the lanes except the allowed ones: created_utc, run_num
           (top level and in args), the world-model identity (world_model_checkpoint.*, args.wm_checkpoint,
           resolved_config.model_architecture_config.resume_path); start_state_files hashes are compared too
  models   F8's world-model sha256 = D6b's; U8's = the updated model's; the two differ
Exit 0 and "MATCH: True" only if every check passes. Pure Python, CPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ALLOWED_PREFIXES = ("created_utc", "run_num", "args.run_num", "world_model_checkpoint.", "args.wm_checkpoint",
                    "resolved_config.model_architecture_config.resume_path")


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(value, list):
        out = {}
        for i, v in enumerate(value):
            out.update(flatten(v, f"{prefix}[{i}]"))
        return out or {prefix: []}
    return {prefix: value}


def allowed(key):
    return any(key == p or key.startswith(p) for p in ALLOWED_PREFIXES)


def check(f8, u8, expect):
    """f8, u8: run_args dicts; expect: dict(head, seed, task, mode, init_policy_sha256, startup_resets_sha256,
    startup_reset_share, max_iterations, penalty_weight, wm_f8_sha256, wm_u8_sha256)."""
    problems = []
    for name, r in (("F8", f8), ("U8", u8)):
        g = r.get("git") or {}
        if g.get("commit") != expect["head"]:
            problems.append(f"{name}: commit {g.get('commit')} != launch HEAD {expect['head']}")
        if g.get("dirty") is not False:
            problems.append(f"{name}: worktree dirty or unknown ({g.get('status')})")
        a = r.get("args") or {}
        rc = r.get("resolved_config") or {}
        got = dict(seed=r.get("seed"), task=r.get("task"), mode=r.get("mode"),
                   init_policy_sha256=(r.get("initial_policy_checkpoint") or {}).get("sha256"),
                   startup_resets_sha256=a.get("startup_resets_sha256"), startup_reset_share=a.get("startup_reset_share"),
                   max_iterations=(rc.get("policy_training_config") or {}).get("max_iterations"),
                   penalty_weight=(rc.get("environment_config") or {}).get("uncertainty_penalty_weight"))
        for k, v in got.items():
            want = expect[k]
            same = (v is not None and abs(float(v) - float(want)) <= 1e-12) if isinstance(want, float) else v == want
            if not same:
                problems.append(f"{name}: {k} = {v!r}, expected {want!r}")
    a, b = flatten(f8), flatten(u8)
    diff = sorted(k for k in set(a) | set(b) if not allowed(k) and a.get(k, "<absent>") != b.get(k, "<absent>"))
    if diff:
        problems.append(f"lanes differ outside the world model: {diff[:8]}{' ...' if len(diff) > 8 else ''}")
    wf = (f8.get("world_model_checkpoint") or {}).get("sha256")
    wu = (u8.get("world_model_checkpoint") or {}).get("sha256")
    if wf != expect["wm_f8_sha256"]:
        problems.append(f"F8 world model {wf} != D6b {expect['wm_f8_sha256']}")
    if wu != expect["wm_u8_sha256"]:
        problems.append(f"U8 world model {wu} != the updated model {expect['wm_u8_sha256']}")
    if wf == wu:
        problems.append("F8 and U8 use the same world model")
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f8", type=Path, required=True, help="F8's run_args.json")
    p.add_argument("--u8", type=Path, required=True, help="U8's run_args.json")
    p.add_argument("--expect", type=Path, required=True, help="JSON with the launch manifest's expected values")
    a = p.parse_args(argv)
    problems = check(json.loads(a.f8.read_text()), json.loads(a.u8.read_text()), json.loads(a.expect.read_text()))
    for pr in problems:
        print(f"[d8-match] {pr}")
    print(f"MATCH: {not problems}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
