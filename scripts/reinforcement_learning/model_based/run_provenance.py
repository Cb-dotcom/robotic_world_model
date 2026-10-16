"""Explicit RNG initialization and a durable identity for each offline run.

Seed once, before constructing models, policies, or sampling data. Resetting an
imagination episode must not reseed the process: that would replay the same
random stream rather than produce a paired cold/warm experiment.
"""

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
from pathlib import Path
import random
import subprocess

import numpy as np
import torch


def validate_seed(value):
    """Return an explicit seed supported by Python, NumPy, and PyTorch."""
    if isinstance(value, bool):
        raise ValueError("seed must be an integer in [0, 4294967295]")
    if isinstance(value, str):
        try:
            value = int(value, 10)
        except ValueError as exc:
            raise ValueError("seed must be an integer in [0, 4294967295]") from exc
    if not isinstance(value, Integral) or not 0 <= value <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 4294967295]")
    return int(value)


def seed_everything(seed):
    """Seed all training RNGs; this does not promise deterministic GPU kernels."""
    seed = validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return _json_value(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    raise TypeError(f"Cannot record configuration value of type {type(value).__name__}")


def _checkpoint_identity(checkpoint):
    if checkpoint is None:
        return None
    path = Path(checkpoint).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _git_identity(repo_root):
    root = Path(repo_root).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}
    return {"commit": commit, "dirty": bool(status), "status": status.splitlines()}


def write_run_args(
    log_dir,
    args,
    config,
    *,
    normalizer_contract=None,
    world_model_checkpoint=None,
    policy_checkpoint=None,
    repo_root=None,
):
    """Write ``run_args.json`` after deployment settings have been resolved.

    ``normalizer_contract`` should contain the effective values and their source.
    The complete resolved config is retained in addition to command-line values;
    checkpoint hashes identify the bytes actually used, even if paths are reused.
    Missing supplied checkpoint files fail loudly instead of yielding an
    apparently complete provenance record.
    """
    cli = _json_value(args)
    if cli.get("seed") is None:
        raise ValueError("An explicit training seed is required; provide --seed SEED.")
    seed = validate_seed(cli["seed"])
    resolved_config = _json_value(config)
    if world_model_checkpoint is None:
        world_model_checkpoint = resolved_config.get("model_architecture_config", {}).get("resume_path")
    if policy_checkpoint is None:
        policy_checkpoint = cli.get("init_policy_checkpoint")
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    record = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "task": cli.get("task"),
        "run_num": cli.get("run_num"),
        "mode": cli.get("mode", "train"),
        "args": cli,
        "git": _git_identity(repo_root),
        "world_model_checkpoint": _checkpoint_identity(world_model_checkpoint),
        "initial_policy_checkpoint": _checkpoint_identity(policy_checkpoint),
        "resolved_config": resolved_config,
        "normalizer_contract": _json_value(normalizer_contract),
    }
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_args.json"
    temporary = directory / "run_args.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    return path
