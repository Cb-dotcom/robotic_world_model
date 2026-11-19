"""Policy in the loop for imagined passes (plan v6 C0: "rerun with the policy in the loop instead of recorded
actions"; C10 and the F1 screen reuse the loader).

Policy: the actor of an rsl_rl ActorCritic checkpoint (`model_state_dict`), rebuilt on the CPU from its weights
(Linear layers at even indices, the activation between them, optional empirical observation normalizer), run
deterministically (`act_inference`), clipped as the recorder's runner clipped it (`clip_actions` in the trace
metadata). The recorder used the same inference policy, so the policy must reproduce the recorded actions from
the recorded observations; `policy_checks` tests this and the observation layout before any imagined pass.

Observation: built by the imagination env's own `get_imagination_observation` (no noise), so the closed loop
sees exactly what imagination training shows the policy: state (denormalized), the recorded command of that
row, and the previous action.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

MB = Path(__file__).resolve().parents[1]
if str(MB) not in sys.path:
    sys.path.insert(0, str(MB))

ACTIVATIONS = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "selu": torch.nn.SELU}
# policy observation blocks of the 48-dim imagination observation (anymal_d_flat and go2_flat)
OBS_BLOCKS = {"lin_vel": (0, 3), "ang_vel": (3, 6), "gravity": (6, 9), "command": (9, 12), "joint_pos": (12, 24),
              "joint_vel": (24, 36), "last_action": (36, 48)}
ACTION_TOL = 1e-3          # max |policy(recorded obs) - recorded action|: larger means another policy
OBS_TOL = 1e-4             # mean |rebuilt obs - recorded obs| per block, for every block that carries no noise
NOISE_FREE_BLOCKS = ("command", "last_action")   # never corrupted by observation noise (review 19 Nov)


class PolicyMismatch(RuntimeError):
    """The loaded policy or the rebuilt observation does not reproduce the recording: the closed loop stops."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Policy:
    def __init__(self, path, activation="elu", num_actions=12, clip=None):
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
        if not idx or idx != list(range(0, 2 * len(idx), 2)):
            raise ValueError(f"{path}: actor layers at {idx}; expected Linear layers at 0, 2, 4, ...")
        layers = []
        for i, j in enumerate(idx):
            w, b = sd[f"actor.{j}.weight"].float(), sd[f"actor.{j}.bias"].float()
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            with torch.no_grad():
                lin.weight.copy_(w)
                lin.bias.copy_(b)
            layers.append(lin)
            if i < len(idx) - 1:
                layers.append(ACTIVATIONS[activation]())
        self.net = torch.nn.Sequential(*layers).eval()
        self.obs_dim = int(sd[f"actor.{idx[0]}.weight"].shape[1])
        out = int(sd[f"actor.{idx[-1]}.weight"].shape[0])
        if out not in (num_actions, 2 * num_actions):
            raise ValueError(f"{path}: actor output {out}, expected {num_actions} (or {2 * num_actions})")
        self.state_dependent_std = out == 2 * num_actions
        if "actor_obs_normalizer._mean" in sd:
            self.norm = (sd["actor_obs_normalizer._mean"].float().reshape(1, -1),
                         sd["actor_obs_normalizer._std"].float().reshape(1, -1), 1e-2)   # rsl_rl default eps
        else:
            self.norm = None
        self.clip = None if clip is None else float(clip)
        self.path, self.sha256, self.activation = str(path), sha256(path), activation

    @torch.no_grad()
    def __call__(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32)
        if self.norm is not None:
            mean, std, eps = self.norm
            x = (x - mean) / (std + eps)
        a = self.net(x)
        if self.state_dependent_std:
            a = a.reshape(len(a), 2, -1)[:, 0]
        if self.clip is not None:
            a = a.clamp(-self.clip, self.clip)
        return a

    def describe(self):
        return dict(path=self.path, sha256=self.sha256, activation=self.activation, obs_dim=self.obs_dim,
                    obs_normalizer=self.norm is not None, state_dependent_std=self.state_dependent_std,
                    clip=self.clip)


class _Denormalizer:
    def __init__(self, stats):
        self.s = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in stats.items()}

    def denormalize(self, state, action):
        st = None if state is None else state * self.s["state_std"] + self.s["state_mean"]
        ac = None if action is None else action * self.s["action_std"] + self.s["action_mean"]
        return st, ac


class ObservationBuilder:
    """obs = env.get_imagination_observation for one state, command and previous action (normalized inputs)."""

    def __init__(self, env_name, stats):
        if env_name == "anymal_d_flat":
            from envs.anymal_d_flat import AnymalDFlatEnv as cls
        elif env_name == "go2_flat":
            from envs.go2_flat import Go2FlatEnv as cls
        else:
            raise ValueError(f"unknown env {env_name}")
        self.env = cls.__new__(cls)
        self.env.device, self.env.observation_noise = "cpu", False
        self.env.dataset = _Denormalizer(stats)
        self.env_name = env_name

    @torch.no_grad()
    def __call__(self, state_norm, last_action_norm, command):
        self.env._num_envs = len(state_norm)
        self.env.base_velocity = torch.as_tensor(command, dtype=torch.float32)
        obs = self.env.get_imagination_observation(state_norm[:, None], last_action_norm[:, None])
        return obs["policy"]


def env_name_from_meta(meta):
    task = str(meta.get("task", ""))
    return "go2_flat" if "go2" in task.lower() else "anymal_d_flat"


def resolve_policy(name, meta, overrides, activation="elu"):
    """Checkpoint from --policy NAME=PATH, else the trace metadata (`checkpoint`), with its SHA-256 checked."""
    path = overrides.get(name) or meta.get("checkpoint")
    if not path or not Path(path).is_file():
        raise PolicyMismatch(f"{name}: policy checkpoint not found ({path}); pass --policy {name}=PATH")
    policy = Policy(path, activation=activation, clip=meta.get("clip_actions"))
    expected = meta.get("checkpoint_sha256")
    if expected and policy.sha256 != expected:
        raise PolicyMismatch(f"{name}: {path} sha256 {policy.sha256[:12]} != recorded {str(expected)[:12]}")
    return policy


def policy_checks(name, trace, meta, policy, builder, stats, rows=2048, seed=0):
    """(1) the policy reproduces the recorded actions from the recorded observations; (2) the rebuilt observation
    (state, command, previous action) equals the recorded one block by block: always for the command and the
    previous action (no noise), and for every block when the recorder's policy observation had no noise. With
    noise, the noisy blocks are reported with their correlation (a layout error shows as a low correlation).
    Rows from all envs and steps, sampled with a fixed seed."""
    obs_rec = trace["policy_observation"]
    T, N = obs_rec.shape[:2]
    rng = np.random.default_rng(seed)
    pick = rng.choice(T * N, size=min(rows, T * N), replace=False)
    t, n = pick // N, pick % N
    o = torch.from_numpy(obs_rec[t, n]).float()
    if o.shape[1] != policy.obs_dim:
        raise PolicyMismatch(f"{name}: recorded observation has {o.shape[1]} dims, policy expects {policy.obs_dim}")
    a_rec = torch.from_numpy(trace["action"][t, n]).float()
    action_err = float((policy(o) - a_rec).abs().max())
    s = torch.as_tensor(stats["state_mean"]), torch.as_tensor(stats["state_std"])
    a = torch.as_tensor(stats["action_mean"]), torch.as_tensor(stats["action_std"])
    state_n = (torch.from_numpy(trace["state"][t, n]).float() - s[0]) / s[1]
    prev_n = (torch.from_numpy(trace["previous_action"][t, n]).float() - a[0]) / a[1]
    rebuilt = builder(state_n, prev_n, torch.from_numpy(trace["command"][t, n]).float())
    blocks = {k: float((rebuilt[:, lo:hi] - o[:, lo:hi]).abs().mean()) for k, (lo, hi) in OBS_BLOCKS.items()}

    def corr(a, b):
        a, b = a.flatten().double(), b.flatten().double()
        return float(torch.corrcoef(torch.stack([a, b]))[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")

    corrs = {k: corr(rebuilt[:, lo:hi], o[:, lo:hi]) for k, (lo, hi) in OBS_BLOCKS.items()}
    noisy = bool(meta.get("policy_noise", False))
    checked = list(NOISE_FREE_BLOCKS) if noisy else list(OBS_BLOCKS)
    out = dict(rows=int(len(pick)), max_abs_action_error=action_err, obs_mean_abs_error=blocks, obs_corr=corrs,
               recorded_policy_noise=noisy, obs_blocks_checked=checked, action_tolerance=ACTION_TOL,
               obs_tolerance=OBS_TOL)
    problems = []
    if action_err > ACTION_TOL:
        problems.append(f"policy does not reproduce the recorded actions (max |error| {action_err:.3g})")
    bad = {k: blocks[k] for k in checked if blocks[k] > OBS_TOL}
    if bad:
        problems.append(f"rebuilt observation differs from the recorded one: {json.dumps(bad)}")
    out["passed"] = not problems
    if problems:
        raise PolicyMismatch(f"{name}: " + "; ".join(problems))
    return out


# ---------------------------------------------------------------- helpers for tests and dry runs
def save_actor(path, obs_dim=48, hidden=(32, 32), num_actions=12, seed=0, normalizer=False):
    """Write an rsl_rl-style checkpoint with a random actor (and critic stub) for tests and dry runs."""
    g = torch.Generator().manual_seed(seed)
    sd, dims = {}, [obs_dim, *hidden, num_actions]
    for i in range(len(dims) - 1):
        sd[f"actor.{2 * i}.weight"] = torch.randn(dims[i + 1], dims[i], generator=g) / dims[i] ** 0.5
        sd[f"actor.{2 * i}.bias"] = torch.randn(dims[i + 1], generator=g) * 0.1
    sd["critic.0.weight"], sd["critic.0.bias"] = torch.zeros(1, obs_dim), torch.zeros(1)
    sd["std"] = torch.ones(num_actions)
    if normalizer:
        sd["actor_obs_normalizer._mean"] = torch.randn(1, obs_dim, generator=g) * 0.1
        sd["actor_obs_normalizer._std"] = torch.rand(1, obs_dim, generator=g) + 0.5
        sd["actor_obs_normalizer._var"] = sd["actor_obs_normalizer._std"] ** 2
        sd["actor_obs_normalizer.count"] = torch.tensor(1000)
    torch.save({"model_state_dict": sd, "iter": 0}, path)
    return path


def consistent_policy_trace(trace, policy, builder, stats):
    """Rewrite action, previous_action and policy_observation of a trace dict so that the policy produced the
    actions from noise-free observations (episode starts get a zero previous action). Tests and dry runs only."""
    s_mean, s_std = (torch.as_tensor(stats[k]) for k in ("state_mean", "state_std"))
    a_mean, a_std = (torch.as_tensor(stats[k]) for k in ("action_mean", "action_std"))
    state, command, done = trace["state"], trace["command"], trace["done"]
    T, N = state.shape[:2]
    action = np.zeros((T, N, 12), np.float32)
    prev = np.zeros((T, N, 12), np.float32)
    obs = np.zeros((T, N, policy.obs_dim), np.float32)
    for t in range(T):
        if t > 0:
            prev[t] = np.where(done[t - 1][:, None], 0.0, action[t - 1])
        o = builder((torch.from_numpy(state[t]).float() - s_mean) / s_std,
                    (torch.from_numpy(prev[t]) - a_mean) / a_std, torch.from_numpy(command[t]).float())
        obs[t] = o.numpy()
        action[t] = policy(o).numpy()
    return dict(trace, action=action, previous_action=prev, policy_observation=obs)


def main():
    """Recheck the policies of saved traces (no imagined pass): python policy_loop.py --wm WM --trace NAME=NPZ ..."""
    import argparse
    from c0_battery import load_models
    from pre_reset_transitions import validate_trace
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--wm", required=True, help="world model whose normalizer the closed loop used")
    p.add_argument("--trace", action="append", required=True, metavar="NAME=NPZ")
    p.add_argument("--policy", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--policy-activation", default="elu")
    p.add_argument("--out", type=Path, help="optional JSON output")
    a = p.parse_args()
    _, contract, _ = load_models(a.wm)
    stats = {k: contract.statistics[k] for k in ("state_mean", "state_std", "action_mean", "action_std")}
    overrides = dict(item.partition("=")[::2] for item in a.policy)
    results, failed = {}, []
    for item in a.trace:
        name, _, path = item.partition("=")
        with np.load(path, allow_pickle=False) as z:
            meta = validate_trace(z)
            try:
                policy = resolve_policy(name, meta, overrides, a.policy_activation)
                builder = ObservationBuilder(env_name_from_meta(meta), stats)
                results[name] = dict(policy.describe(), checks=policy_checks(name, z, meta, policy, builder, stats))
            except PolicyMismatch as exc:
                results[name] = dict(error=str(exc))
                failed.append(name)
        print(f"[policy_loop] {name}: {json.dumps(results[name].get('checks', results[name]))}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(results, indent=1))
    if failed:
        raise SystemExit(f"[policy_loop] FAILED: {failed}")
    print("[policy_loop] all checks passed", flush=True)


if __name__ == "__main__":
    main()
