#!/usr/bin/env python3
"""E1 policy training: the unchanged p15_train recipe, with a chosen world-model checkpoint.

Usage: e1_train.py --wm-checkpoint PATH [--penalty-weight W] paper RUN_NUM anymal_d_flat --seed SEED
Everything except the world-model path and the penalty weight is p15_train.py as is (paper
recipe, cold resets for mode "paper"). Default penalty weight -1.0 (lambda 1, as p15_train);
0.0 gives the lambda-0 arms of plan v5.2 E1. The checkpoint's own normalizer is used.
Both values land in run_args.json (resolved_config) and are checked by e1v_match.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv=None):
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--wm-checkpoint", required=True, type=Path)
    pre.add_argument("--penalty-weight", type=float, default=-1.0,
                     help="environment_config.uncertainty_penalty_weight (-1.0 = lambda 1; 0.0 = lambda 0)")
    known, rest = pre.parse_known_args(argv)
    wm = known.wm_checkpoint.resolve()
    if not wm.is_file():
        raise SystemExit(f"World-model checkpoint not found: {wm}")

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
