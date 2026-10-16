"""Validate the coordinate system of a frozen world model before policy training.

Refit checkpoints own their statistics. Legacy checkpoints need explicit task
constants. Data collected for imagination starts must never redefine that space.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping

import torch


KEYS = ("state_mean", "state_std", "action_mean", "action_std")
CONFIG_KEYS = ("state_data_mean", "state_data_std", "action_data_mean", "action_data_std")


@dataclass(frozen=True)
class NormalizationContract:
    source: str
    statistics: dict[str, torch.Tensor]
    norm_stats_sha256: str

    def metadata(self) -> dict:
        return {
            "source": self.source,
            "norm_stats_sha256": self.norm_stats_sha256,
            "statistics": {key: self.statistics[key].tolist() for key in KEYS},
        }

    def dataset_kwargs(self) -> dict:
        return {name: self.statistics[key] for key, name in zip(KEYS, CONFIG_KEYS)}


def resolve_normalization(
    checkpoint: Mapping,
    config_statistics: Mapping,
    state_dim: int,
    action_dim: int,
) -> NormalizationContract:
    """Reject incomplete, invalid, or unknown normalization instead of guessing.

    The hash describes the effective float32 statistics used by the model, not
    their source filename. Shapes such as [1, D] are accepted as saved vectors.
    """
    if checkpoint.get("normalized", False):
        source = "checkpoint"
        values = {key: checkpoint.get(key) for key in KEYS}
    else:
        if any(key in checkpoint for key in KEYS):
            raise ValueError("WM checkpoint contains normalization statistics without normalized=True")
        source = "config"
        values = {key: config_statistics.get(name) for key, name in zip(KEYS, CONFIG_KEYS)}

    statistics = {}
    for key in KEYS:
        value = values[key]
        if value is None:
            raise ValueError(
                f"Missing {key} in {source} normalization. Use a normalized WM checkpoint "
                "or explicit legacy task constants; dataset recomputation is forbidden."
            )
        try:
            vector = torch.as_tensor(value, dtype=torch.float32, device="cpu").detach().clone().reshape(-1)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Invalid {source} normalization value {key}") from exc
        expected = state_dim if key.startswith("state_") else action_dim
        if vector.numel() != expected:
            raise ValueError(f"{source} {key} has {vector.numel()} values; expected {expected}")
        if not torch.isfinite(vector).all():
            raise ValueError(f"{source} {key} must contain finite values")
        if key.endswith("_std") and not (vector > 0).all():
            raise ValueError(f"{source} {key} must be strictly positive")
        statistics[key] = vector

    encoded = json.dumps(
        {key: statistics[key].tolist() for key in KEYS},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return NormalizationContract(source, statistics, hashlib.sha256(encoded).hexdigest())
