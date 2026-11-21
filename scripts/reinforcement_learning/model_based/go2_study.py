"""Plan v6 amendment A3 (21 Nov 2026): the settings every Go2 study tool shares, in one place.

The trainer (e1_train.py), the s2 imagination probe and the runtime reward check call these functions, so the
policy is trained in exactly the environment that the checks inspect.

* apply_s2_contract: imagined reward = the s2 reward contract (configs/go2_flat_cfg.py S2_REWARD_CONTRACT):
  weights in reward_term_weights, term parameters and timer order on the env class, the asset's default pose
  and soft joint limits from a recorded trace's metadata (the 18 Nov recorder stores them).
* apply_start_data: imagination start histories from a start-data folder written by go2_round0_prepare.py
  (CSV files whose termination column marks every episode end, so no start window crosses an episode).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def contract_sha256():
    """Hash of the contract content (weights, parameters, decisions), stable across Python sessions."""
    from configs.go2_flat_cfg import S2_EXCLUDED_IMAGINED_TERMS, S2_REWARD_CONTRACT
    blob = json.dumps({"contract": S2_REWARD_CONTRACT, "excluded": S2_EXCLUDED_IMAGINED_TERMS}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def asset_constants(trace_path):
    """default_joint_pos [12] and soft_joint_pos_limits [12, 2] from a recorded trace's metadata."""
    with np.load(trace_path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
    for key in ("default_joint_pos", "soft_joint_pos_limits"):
        if key not in meta:
            raise ValueError(f"{trace_path}: metadata lacks {key} (record with the 18 Nov recorder)")
    pos = [float(x) for x in meta["default_joint_pos"]]
    lim = [[float(a), float(b)] for a, b in meta["soft_joint_pos_limits"]]
    if len(pos) != 12 or len(lim) != 12 or any(a >= b for a, b in lim):
        raise ValueError(f"{trace_path}: default pose or soft limits have the wrong shape or order")
    return pos, lim, meta


def apply_s2_contract(cfg, asset_trace, env_cls=None):
    """Imagined reward = the s2 contract. Changes cfg (weights) and the Go2 env class (parameters). Returns a record."""
    from configs.go2_flat_cfg import (S2_EXCLUDED_IMAGINED_TERMS, S2_REWARD_CONTRACT, apply_s2_reward_contract,
                                      s2_reward_term_weights)
    if env_cls is None:
        from envs.go2_flat import Go2FlatEnv as env_cls
    pos, lim, meta = asset_constants(asset_trace)
    if "reward_names" in meta:                       # a trace of s2's own training reward: the contract must match it
        import sys
        here = Path(__file__).resolve().parent / "analysis"
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        from e16_reward_check import contract_problems
        problems = contract_problems(meta, S2_REWARD_CONTRACT)
        if problems:
            raise ValueError("s2 contract does not match the trace's simulator reward: " + "; ".join(problems))
    weights = s2_reward_term_weights()
    cfg.environment_config.reward_term_weights = dict(weights)
    apply_s2_reward_contract(env_cls, default_joint_pos=pos, soft_joint_pos_limits=lim)
    return dict(contract="s2", contract_sha256=contract_sha256(), terms=len(weights), weights=weights,
                excluded=dict(S2_EXCLUDED_IMAGINED_TERMS), asset_trace=str(asset_trace),
                asset_trace_sha256=sha256(asset_trace), default_joint_pos=pos, soft_joint_pos_limits=lim,
                env_class=f"{env_cls.__module__}.{env_cls.__name__}",
                class_settings=dict(feet_air_time_threshold=env_cls.feet_air_time_threshold,
                                    joint_pos_stand_still_scale=env_cls.joint_pos_stand_still_scale,
                                    joint_pos_velocity_threshold=env_cls.joint_pos_velocity_threshold,
                                    timers_before_rewards=env_cls.timers_before_rewards,
                                    acceleration_from_state=env_cls.acceleration_from_state))


def apply_start_data(cfg, start_dir):
    """Imagination start histories from go2_round0_prepare.py's start-data folder (all of its rows)."""
    start_dir = Path(start_dir).resolve()
    info = json.loads((start_dir / "start_data.json").read_text())
    files = [start_dir / f"state_action_data_{i}.csv" for i in range(int(info["files"]))]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"start data incomplete: {missing[:3]}")
    if (start_dir / f"state_action_data_{len(files)}.csv").exists():
        raise ValueError("start data has more CSV files than start_data.json lists")
    cfg.data_config.dataset_root = str(start_dir.parent)
    cfg.data_config.dataset_folder = start_dir.name
    cfg.data_config.file_data_size = int(info["min_file_rows"])
    cfg.data_config.batch_data_size = int(info["rows"])       # the loader stops once it has read every file
    return dict(start_data=str(start_dir), rows=int(info["rows"]), files=len(files), episodes=int(info["episodes"]),
                start_data_json_sha256=sha256(start_dir / "start_data.json"))
