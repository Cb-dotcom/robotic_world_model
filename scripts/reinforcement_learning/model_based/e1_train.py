#!/usr/bin/env python3
"""E1 policy training: the unchanged p15_train recipe, with a chosen world-model checkpoint.

Usage: e1_train.py --wm-checkpoint PATH paper RUN_NUM anymal_d_flat --seed SEED
Everything except the world-model path is p15_train.py as is (paper recipe, penalty -1,
cold resets for mode "paper"). The checkpoint's own normalizer is used (normalization_contract).
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
    return p15_train.main(rest)


if __name__ == "__main__":
    main()
