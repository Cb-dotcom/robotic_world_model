#!/usr/bin/env python3
"""P1.5 offline policy training on ETH's shipped config, with optional paper settings and B7 fix.

Usage (from the robotic_world_model root, WANDB_MODE=offline recommended):
    python scripts/reinforcement_learning/model_based/p15_train.py MODE RUN_NUM [TASK]
MODE: shipped | paper | shipped_warm | paper_warm   (TASK default: anymal_d_flat)
  paper  = paper Table S11: 4096 envs, 100 steps/it, 2500 it, lr 1e-3, entropy 5e-3
  *_warm = B7 fix: rebuild GRU memory from the full sampled history after every imagination reset
All modes use uncertainty_penalty_weight = -1.0 (lambda = 1).
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402
from envs import BaseEnv  # noqa: E402

mode, run = sys.argv[1], int(sys.argv[2])
task = sys.argv[3] if len(sys.argv) > 3 else "anymal_d_flat"
assert mode in ("shipped", "paper", "shipped_warm", "paper_warm"), mode
T.args_cli = types.SimpleNamespace(task=task, run_num=run, init_policy_checkpoint=None)
cfg = T.resolve_task_config(task)
cfg.environment_config.uncertainty_penalty_weight = -1.0
if mode.startswith("paper"):
    cfg.environment_config.num_envs = 4096
    cfg.policy_training_config.num_steps_per_env = 100
    cfg.policy_training_config.max_iterations = 2500
    cfg.policy_algorithm_config.learning_rate = 1e-3
    cfg.policy_algorithm_config.entropy_coef = 5e-3
BaseEnv.warm_start_resets = mode.endswith("_warm")
print("[p1.5]", mode, run, task, cfg.environment_config.num_envs, cfg.policy_training_config.num_steps_per_env,
      cfg.policy_training_config.max_iterations, cfg.policy_algorithm_config.learning_rate,
      cfg.policy_algorithm_config.entropy_coef, cfg.environment_config.uncertainty_penalty_weight,
      cfg.model_architecture_config.resume_path, cfg.model_architecture_config.ensemble_size,
      "warm_start_resets=", BaseEnv.warm_start_resets, flush=True)
T.run(cfg)
