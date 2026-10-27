"""Simulator-independent storage and window rules for pre-reset transitions.

Each row is (s_t, a_t, s_{t+1}), including the true terminal next state.
No inferred zero-action boundary and no post-reset target is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "pre_reset_transitions_v1"
GROUPS = {"lin_vel_m_s": (0, 3), "ang_vel_rad_s": (3, 6), "gravity": (6, 9),
          "joint_pos_rad": (9, 21), "joint_vel_rad_s": (21, 33), "torque_Nm": (33, 45)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


class TransitionBuffer:
    """Copy at the hooks, so subsequent simulator resets cannot change a row."""

    def __init__(self, steps, envs):
        self.steps, self.envs, self.index = steps, envs, 0
        self.data, self.pending = {}, None
        self.verified_terminal_rows = 0

    def begin(self, **fields):
        if self.pending is not None or self.index >= self.steps:
            raise RuntimeError("Unpaired pre-step or recording capacity exceeded")
        self.pending = {key: array(value) for key, value in fields.items()}

    def end(self, **fields):
        if self.pending is None:
            raise RuntimeError("Post-step without pre-step")
        row = dict(self.pending, **{key: array(value) for key, value in fields.items()})
        if row["state"].shape != (self.envs, 45) or row["next_state"].shape != (self.envs, 45):
            raise ValueError("Expected 45 clean ANYmal state components")
        if row["action"].shape != (self.envs, 12):
            raise ValueError("Expected 12 applied policy-space actions")
        if not np.array_equal(row.pop("observed_action"), row["action"]):
            raise ValueError("Recorded action does not match the pre-reset system_action")
        if not np.array_equal(row["done"], row["terminated"] | row["time_out"]):
            raise ValueError("Inconsistent termination flags")
        for key, value in row.items():
            if value.shape[0] != self.envs or not np.isfinite(value).all():
                raise ValueError(f"Invalid transition field {key}")
            if key not in self.data:
                self.data[key] = np.empty((self.steps, *value.shape), dtype=value.dtype)
            self.data[key][self.index] = value
        self.pending = None
        self.index += 1

    def verify_before_reset(self, env_ids, state, action):
        # Initial wrapper reset occurs before any transition is recorded.
        if self.index == 0:
            return
        ids = array(env_ids).astype(int)
        row = self.index - 1
        if not self.data["done"][row, ids].all():
            raise ValueError("Unexpected reset without a terminal transition")
        if not np.array_equal(self.data["next_state"][row, ids], array(state)[ids]):
            raise ValueError("Terminal state changed before reset")
        if not np.array_equal(self.data["action"][row, ids], array(action)[ids]):
            raise ValueError("Terminal action changed before reset")
        self.verified_terminal_rows += len(ids)

    def save(self, path, metadata):
        if self.pending is not None or self.index != self.steps:
            raise ValueError("Incomplete recording")
        if self.verified_terminal_rows != int(self.data["done"].sum()):
            raise ValueError("Some terminal rows were not checked before reset")
        metadata = dict(metadata, schema=SCHEMA, verified_terminal_rows=self.verified_terminal_rows)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create: never replace evidence from an earlier attempt.
        with path.open("xb") as stream:
            np.savez_compressed(stream, **self.data, metadata=json.dumps(metadata, allow_nan=False))


def validate_trace(trace):
    meta = json.loads(str(trace["metadata"]))
    if meta.get("schema") != SCHEMA:
        raise ValueError("Requires pre-reset schema; legacy post-reset recordings are incompatible")
    state, target, action = trace["state"], trace["next_state"], trace["action"]
    T, N, D = state.shape
    if D != 45 or target.shape != state.shape or action.shape != (T, N, 12):
        raise ValueError("Invalid state/action shapes")
    done = trace["done"]
    if done.shape != (T, N) or not np.array_equal(done, trace["terminated"] | trace["time_out"]):
        raise ValueError("Invalid reset boundaries")
    if meta["verified_terminal_rows"] != int(done.sum()):
        raise ValueError("Missing pre-reset checks")
    for key in trace.files if hasattr(trace, "files") else trace:
        if key != "metadata" and not np.isfinite(trace[key]).all():
            raise ValueError(f"Non-finite recorded values: {key}")
    # With pushes off, adjacent transitions must join except at explicit resets.
    joins = np.max(np.abs(target[:-1] - state[1:]), axis=-1)
    if np.any((joins > 1e-5) & ~done[:-1]):
        raise ValueError("Unmarked state discontinuity; do not replay across it")
    return meta


def windows(trace, history, horizon, per_group=256, seed=0):
    """Stratified diagnostics, not prevalence estimates or independent trials.

    Full history in one episode. The endpoint may be the true terminal row.
    Future actions are logged actions; rollout is not a counterfactual policy test.
    """
    done, term, timeout = trace["done"], trace["terminated"], trace["time_out"]
    T, N = done.shape
    candidates = {name: [] for name in ("fall_now", "pre_fall_1_25", "surviving_25")}
    excluded = {"short_history_origins": 0, "insufficient_future_origins": 0,
                "unlabelled_endpoints": 0}
    for n in range(N):
        start = 0
        for end in [*np.flatnonzero(done[:, n]), T - 1]:
            end = int(end)
            if end < start:
                continue
            is_fall = bool(done[end, n] and term[end, n] and not timeout[end, n])
            excluded["short_history_origins"] += min(history - 1, end - start + 1)
            first, last = start + history - 1, end - horizon + 1
            excluded["insufficient_future_origins"] += max(0, end - max(first - 1, last))
            for t in range(first, last + 1):
                distance = end - (t + horizon - 1)
                if is_fall and distance == 0:
                    group = "fall_now"
                elif is_fall and distance <= 25:
                    group = "pre_fall_1_25"
                elif distance >= 25:
                    group = "surviving_25"
                else:
                    excluded["unlabelled_endpoints"] += 1
                    continue
                candidates[group].append((t, n))
            start = end + 1
    rng = np.random.default_rng(seed)
    selected, labels = [], []
    for group, rows in candidates.items():
        idx = np.sort(rng.choice(len(rows), min(per_group, len(rows)), replace=False))
        selected.extend(rows[i] for i in idx)
        labels.extend([group] * len(idx))
    return np.asarray(selected, dtype=int).reshape(-1, 2), np.asarray(labels), {
        "eligible": {key: len(value) for key, value in candidates.items()},
        "selected": {key: int(np.sum(np.asarray(labels) == key)) for key in candidates},
        "exclusions": excluded,
    }


def gather(values, origins, lo, hi):
    return np.stack([values[t + lo:t + hi, n] for t, n in origins])
