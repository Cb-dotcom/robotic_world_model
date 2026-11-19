#!/usr/bin/env python3
"""E1 policy training: the unchanged p15_train recipe, with a chosen world-model checkpoint.

Usage: e1_train.py --wm-checkpoint PATH [--penalty-weight W] paper RUN_NUM anymal_d_flat --seed SEED
Everything except the world-model path and the penalty weight is p15_train.py as is (paper
recipe, cold resets for mode "paper"). Default penalty weight -1.0 (lambda 1, as p15_train);
0.0 gives the lambda-0 arms of plan v5.2 E1. The checkpoint's own normalizer is used.
Both values land in run_args.json (resolved_config) and are checked by e1v_match.py.

--f1-calibration PATH (plan v6 amendment A1, 19 Nov 2026): turns F1 on (envs/base.py) with the frozen
f1_calibration.json of analysis/f1_calibrate.py: K members, s(d) and the gain g. The calibration must have been
made for the same world model (SHA-256 checked); its own SHA-256 is printed. Nothing else changes, so the run's
run_args.json still matches arm D's; the queue stores the calibration beside it as f1_settings.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def apply_f1_calibration(path, wm, env_cls=None):
    """Turn F1 on from a frozen f1_calibration.json (A1); the calibration must belong to this world model."""
    import hashlib
    import json
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    cal = json.loads(Path(path).read_text())
    if sha(wm) not in cal["wm_sha256"]:
        raise SystemExit(f"F1 calibration {path} was made for another world model")
    if not (cal["gain"] > 0 and len(cal["scale"]) >= 1 and all(x > 0 for x in cal["scale"])):
        raise SystemExit("F1 calibration has a non-positive gain or scale")
    if env_cls is None:
        from envs import BaseEnv as env_cls
    env_cls.f1_members = int(cal["members"])
    env_cls.f1_scale = [float(x) for x in cal["scale"]]
    env_cls.f1_gain = float(cal["gain"])
    print(f"[e1] F1 on: members {env_cls.f1_members}, gain {env_cls.f1_gain:.6g}, {len(env_cls.f1_scale)} depths, "
          f"calibration sha256 {sha(path)}", flush=True)
    return cal


def main(argv=None):
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--wm-checkpoint", required=True, type=Path)
    pre.add_argument("--penalty-weight", type=float, default=-1.0,
                     help="environment_config.uncertainty_penalty_weight (-1.0 = lambda 1; 0.0 = lambda 0)")
    pre.add_argument("--f1-calibration", type=Path, default=None,
                     help="A1: frozen f1_calibration.json; turns the F1 penalty on")
    known, rest = pre.parse_known_args(argv)
    wm = known.wm_checkpoint.resolve()
    if not wm.is_file():
        raise SystemExit(f"World-model checkpoint not found: {wm}")
    if known.f1_calibration is not None:
        apply_f1_calibration(known.f1_calibration, wm)
    import train as T
    import p15_train

    original = T.resolve_task_config

    def resolve_with_wm(task):
        cfg = original(task)
        cfg.model_architecture_config.resume_path = str(wm)
        print(f"[e1] world model resume_path = {wm}", flush=True)
        return cfg

    T.resolve_task_config = resolve_with_wm
    original_run = T.run

    def run_with_penalty(cfg):
        # p15_train sets -1.0 after resolve_task_config; override it last, just before the run.
        cfg.environment_config.uncertainty_penalty_weight = float(known.penalty_weight)
        print(f"[e1] uncertainty_penalty_weight = {known.penalty_weight}", flush=True)
        return original_run(cfg)

    T.run = run_with_penalty
    return p15_train.main(rest)


if __name__ == "__main__":
    main()
