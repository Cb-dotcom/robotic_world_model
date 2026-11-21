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

Go2 study (plan v6 amendment A3, 21 Nov 2026), task go2_flat only (go2_study.py holds the shared settings):
--reward-contract s2 --asset-trace NPZ   imagined reward = the s2 contract; default pose and soft joint limits
                                         from the trace's metadata (the E16 trace also checks the contract)
--start-data DIR                         imagination starts from go2_round0_prepare.py's start-data folder
--init-policy-checkpoint PT              PPO starts from this actor/critic (fresh optimizer); later rounds
Each prints one "[go2] ..." line with hashes; go2_settings.json is written into the run folder by the queue.
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
    pre.add_argument("--reward-contract", choices=("s2",), default=None, help="A3: imagined reward contract (go2_flat)")
    pre.add_argument("--asset-trace", type=Path, default=None, help="A3: trace whose metadata gives the asset constants")
    pre.add_argument("--start-data", type=Path, default=None, help="A3: imagination start-data folder (go2_flat)")
    pre.add_argument("--init-policy-checkpoint", type=Path, default=None,
                     help="A3: initial actor/critic (later rounds); the optimizer is new")
    pre.add_argument("--max-iterations", type=int, default=None,
                     help="development checks only (A4): stop PPO after this many iterations (default: the recipe's)")
    known, rest = pre.parse_known_args(argv)
    wm = known.wm_checkpoint.resolve()
    if not wm.is_file():
        raise SystemExit(f"World-model checkpoint not found: {wm}")
    if known.f1_calibration is not None:
        apply_f1_calibration(known.f1_calibration, wm)
    go2 = (known.reward_contract, known.start_data, known.init_policy_checkpoint)
    if any(x is not None for x in go2) and "go2_flat" not in rest:
        raise SystemExit("--reward-contract, --start-data and --init-policy-checkpoint are for the go2_flat task")
    if known.reward_contract is not None and known.asset_trace is None:
        raise SystemExit("--reward-contract s2 needs --asset-trace")
    if known.init_policy_checkpoint is not None and not known.init_policy_checkpoint.is_file():
        raise SystemExit(f"initial policy not found: {known.init_policy_checkpoint}")
    import train as T
    import p15_train

    original = T.resolve_task_config

    def resolve_with_wm(task):
        cfg = original(task)
        cfg.model_architecture_config.resume_path = str(wm)
        print(f"[e1] world model resume_path = {wm}", flush=True)
        if known.reward_contract == "s2":
            import go2_study
            rec = go2_study.apply_s2_contract(cfg, known.asset_trace)
            print(f"[go2] reward contract s2: {rec['terms']} terms, contract sha256 {rec['contract_sha256']}, "
                  f"asset trace sha256 {rec['asset_trace_sha256']}, timers_before_rewards "
                  f"{rec['class_settings']['timers_before_rewards']}, acceleration_from_state "
                  f"{rec['class_settings']['acceleration_from_state']}", flush=True)
        if known.start_data is not None:
            import go2_study
            rec = go2_study.apply_start_data(cfg, known.start_data)
            print(f"[go2] start data {rec['start_data']}: {rec['rows']} rows, {rec['episodes']} episodes, "
                  f"{rec['files']} files, start_data.json sha256 {rec['start_data_json_sha256']}", flush=True)
        return cfg

    T.resolve_task_config = resolve_with_wm
    original_run = T.run

    def run_with_penalty(cfg):
        # p15_train sets -1.0 after resolve_task_config; override it last, just before the run.
        cfg.environment_config.uncertainty_penalty_weight = float(known.penalty_weight)
        print(f"[e1] uncertainty_penalty_weight = {known.penalty_weight}", flush=True)
        if known.max_iterations is not None:
            if known.max_iterations < 1:
                raise SystemExit("--max-iterations must be >= 1")
            cfg.policy_training_config.max_iterations = int(known.max_iterations)
            print(f"[dev] max_iterations = {known.max_iterations} (development check, not the paper recipe)", flush=True)
        if known.init_policy_checkpoint is not None:
            import go2_study
            T.args_cli.init_policy_checkpoint = str(known.init_policy_checkpoint.resolve())
            print(f"[go2] initial policy {T.args_cli.init_policy_checkpoint} sha256 "
                  f"{go2_study.sha256(known.init_policy_checkpoint)}; new optimizer", flush=True)
        return original_run(cfg)

    T.run = run_with_penalty
    return p15_train.main(rest)


if __name__ == "__main__":
    main()
