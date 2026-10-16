#!/usr/bin/env python3
"""P1.5 offline policy training with an explicit, recorded training seed.

Usage (from the robotic_world_model root, WANDB_MODE=offline recommended):
    python scripts/reinforcement_learning/model_based/p15_train.py MODE RUN_NUM [TASK] --seed SEED
    python scripts/reinforcement_learning/model_based/p15_train.py MODE RUN_NUM TASK SEED

MODE: shipped | paper | shipped_warm | paper_warm (TASK default: anymal_d_flat)
  paper  = paper Table S11: 4096 envs, 100 steps/it, 2500 it, lr 1e-3, entropy 5e-3
  *_warm = rebuild GRU memory from sampled history after every imagination reset
All modes use uncertainty_penalty_weight = -1.0 (lambda = 1).
The old unseeded invocation now exits with a clear error; run numbers are not seeds.
"""

import argparse
from pathlib import Path
import sys
import types


def parse_args(argv=None):
    from run_provenance import validate_seed

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("shipped", "paper", "shipped_warm", "paper_warm"))
    parser.add_argument("run_num", type=int)
    parser.add_argument("task", nargs="?", default="anymal_d_flat", choices=("anymal_d_flat", "go2_flat"))
    parser.add_argument("positional_seed", nargs="?", type=validate_seed, metavar="SEED")
    parser.add_argument("--seed", type=validate_seed, help="Explicit training seed (0 through 4294967295).")
    args = parser.parse_args(argv)
    if args.positional_seed is not None and args.seed is not None:
        parser.error("provide a seed once, as positional SEED or --seed SEED")
    if args.seed is None:
        args.seed = args.positional_seed
    if args.seed is None:
        parser.error("an explicit training seed is required; append --seed SEED (run numbers are not seeds)")
    del args.positional_seed
    return args


def main(argv=None):
    # Importing this module (or requesting CLI help) cannot launch an experiment.
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    args = parse_args(argv)
    import train as T
    from envs import BaseEnv

    T.args_cli = types.SimpleNamespace(
        task=args.task, run_num=args.run_num, seed=args.seed, mode=args.mode, init_policy_checkpoint=None,
    )
    cfg = T.resolve_task_config(args.task)
    cfg.environment_config.uncertainty_penalty_weight = -1.0
    if args.mode.startswith("paper"):
        cfg.environment_config.num_envs = 4096
        cfg.policy_training_config.num_steps_per_env = 100
        cfg.policy_training_config.max_iterations = 2500
        cfg.policy_algorithm_config.learning_rate = 1e-3
        cfg.policy_algorithm_config.entropy_coef = 5e-3
    BaseEnv.warm_start_resets = args.mode.endswith("_warm")
    print(
        "[p1.5]", args.mode, args.run_num, args.task,
        "seed=", args.seed,
        cfg.environment_config.num_envs, cfg.policy_training_config.num_steps_per_env,
        cfg.policy_training_config.max_iterations, cfg.policy_algorithm_config.learning_rate,
        cfg.policy_algorithm_config.entropy_coef, cfg.environment_config.uncertainty_penalty_weight,
        cfg.model_architecture_config.resume_path, cfg.model_architecture_config.ensemble_size,
        "warm_start_resets=", BaseEnv.warm_start_resets, flush=True,
    )
    T.run(cfg)


if __name__ == "__main__":
    main()
