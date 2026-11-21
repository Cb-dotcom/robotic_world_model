#!/usr/bin/env python3
"""A3 follow-up (21 Nov 2026, review decision): read-only replay of the G0 world model on held-out s2 data.

Question: why did the fitted Go2 model stop imagined G0 episodes so often (about one stop every 5 steps)?
G0 stays a valid FAIL (A3 §4). This tool trains nothing, changes no file and needs no GPU.

Data: the E22 withheld s2 episodes of the round-0 recording (never fitted; checked against the E22 rule).
Model: the saved G0 fit (g0_wm_seed1.pt) with the frozen normalizer (checked against the checkpoint).
Stop rule as in imagination (envs/go2_flat._parse_terminations): a head stops when sigmoid(logit) > 0.5.

A. Context (reset problem). State and action FIXED (recorded). Each head's stop score for one recorded
   transition with k recorded steps before it and zero memory, k = 1 (a paper cold reset), 2, 4, 8, 16,
   32 (the fit window and the first imagination step), and with memory carried from the episode start.
   All contexts use the same rows (rows with >= 32 recorded steps of history).
B. Own predictions (rollout problem). From a start row the model gets its own predicted state and the
   RECORDED s2 action, for K steps. Each head drives its own rollout (as model_ids in imagination). Warm start:
   32 recorded steps (zero memory before them); cold start: one recorded state, zero memory (every later
   imagination reset). Reference: the teacher-forced stop score (memory from the episode start) on the same rows.
C. Action (added to the review's design; open item "joint state-action coverage"). Arm A at k = 1 and 32 with
   the last action replaced by the G0 policy's action on the recorded observation: its mean, and its mean plus
   its trained std times standard normal noise (the exploration PPO used).
D. The withheld fall (n = 1): stop scores at the terminal row and over the 50 rows before it.

Not the imagination settings: no imagined velocity kicks, no command resampling, s2's actions (A, B).

Reading rule, fixed before the run (heads averaged):
  reset problem   : stop share at k = 1 >= 0.05 and >= 5 x the stop share at k = 32 (s2 actions);
  rollout problem : warm-start survival to step 25 <= 0.5 while teacher-forced survival on the same windows >= 0.95;
  action problem  : stop share with G0's sampled action >= 0.05 and >= 5 x with s2's action, at k = 1 or k = 32.
Each is reported as "read" or "not read". "Not read" does not exclude the cause for G0's own states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MB = HERE.parent
for p in (str(HERE), str(MB)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pre_reset_transitions import sha256, validate_trace  # noqa: E402

CONTEXTS = (1, 2, 4, 8, 16, 32)
SURVIVAL_AT = (1, 2, 5, 10, 25, 50, 100)
BINS = ((1, 1), (2, 5), (6, 10), (11, 25), (26, 50), (51, 100))
NORM_KEYS = ("state_mean", "state_std", "action_mean", "action_std")


# ----------------------------------------------------------------------------- inputs

def load_model(wm, norm_path, device):
    from replay_wm_events import load_wm_cfg
    sd, H, ck_norm, src = load_wm_cfg(str(wm), "go2_flat", device)
    if src != "checkpoint":
        raise ValueError(f"{wm}: the checkpoint carries no normalizer")
    if sd.architecture_config["type"] != "rnn" or sd.prediction_type != "single":
        raise ValueError("expected the go2_flat GRU model with single-step prediction")
    with np.load(norm_path) as z:
        norm = {k: np.asarray(z[k], np.float32).reshape(-1) for k in NORM_KEYS}
    diff = max(float(np.abs(ck_norm[k] - norm[k]).max()) for k in NORM_KEYS)
    if diff > 1e-5:
        raise ValueError(f"normalizer {norm_path} differs from the checkpoint's (max |diff| {diff:.2e})")
    return sd, int(H), norm, diff


def heldout(trace, withheld_path, frac, seed):
    import go2_round0_prepare as prep
    held = json.loads(Path(withheld_path).read_text())
    eps = prep.episodes(trace["done"])
    expect = sorted((eps[i][0], eps[i][1], eps[i][2]) for i in prep.withheld_set(eps, frac, seed))
    got = sorted((h["env"], h["first"], h["last"]) for h in held)
    if expect != got:
        raise ValueError("withheld.json does not match the E22 rule on this recording")
    return held


class Episode:
    def __init__(self, trace, h, norm):
        n, s, e = h["env"], h["first"], h["last"]
        self.env, self.first, self.kind = n, s, h["kind"]
        self.S = ((trace["next_state"][s:e + 1, n] - norm["state_mean"]) / norm["state_std"]).astype(np.float32)
        self.A = ((trace["action"][s:e + 1, n] - norm["action_mean"]) / norm["action_std"]).astype(np.float32)
        self.y = trace["termination_observation"][s:e + 1, n, 0] > 0.5
        po = trace.get("policy_observation")
        self.obs = None if po is None else po[s:e + 1, n].astype(np.float32)
        self.L = len(self.S)


# ----------------------------------------------------------------------------- model passes

@torch.no_grad()
def all_heads(sd, xs, xa):
    """One library-forward step (same calls, same order) returning every head: means [E,B,45], logits [E,B]."""
    base = sd.state_base(xs, xa)
    means = torch.stack([h(base, xs)[0] for h in sd.state_heads])
    aux = sd.auxiliary_base(xs, xa)
    logits = torch.stack([h(aux, xs)[2] for h in sd.auxiliary_heads]).squeeze(-1)
    return means, logits


@torch.no_grad()
def windowed(sd, eps, rows, k, device, last_action=None, batch=2048):
    """Teacher forced, zero memory, k recorded steps: target row j from states j-k..j-1, actions j-k+1..j.

    last_action: optional [len(rows), 12] normalized actions put at the last position (arm C).
    Returns stop probabilities [E, R] and one-step squared errors (mean over 45 dims) [E, R].
    """
    E = len(sd.state_heads)
    P, ERR = np.empty((E, len(rows)), np.float32), np.empty((E, len(rows)), np.float32)
    for b0 in range(0, len(rows), batch):
        chunk = rows[b0:b0 + batch]
        xs = np.stack([eps[i].S[j - k:j] for i, j in chunk])
        xa = np.stack([eps[i].A[j - k + 1:j + 1] for i, j in chunk])
        if last_action is not None:
            xa[:, -1] = last_action[b0:b0 + len(chunk)]
        tgt = torch.from_numpy(np.stack([eps[i].S[j] for i, j in chunk])).to(device)
        sd.reset()
        means, logits = all_heads(sd, torch.from_numpy(xs).to(device), torch.from_numpy(xa).to(device))
        P[:, b0:b0 + len(chunk)] = torch.sigmoid(logits).cpu().numpy()
        ERR[:, b0:b0 + len(chunk)] = ((means - tgt) ** 2).mean(-1).cpu().numpy()
    sd.reset()
    return P, ERR


@torch.no_grad()
def episode_context(sd, ep, device):
    """Teacher forced with memory from the episode start: [E, L] stop probabilities and errors (NaN at row 0)."""
    E = len(sd.state_heads)
    P = np.full((E, ep.L), np.nan, np.float32)
    ERR = np.full((E, ep.L), np.nan, np.float32)
    if ep.L < 2:
        return P, ERR
    xs = torch.from_numpy(ep.S[:-1]).to(device)
    x = torch.cat([xs, torch.from_numpy(ep.A[1:]).to(device)], dim=-1).unsqueeze(0)
    base, _ = sd.state_base.memory.rnn(x)
    aux, _ = sd.auxiliary_base.memory.rnn(x)
    xs3 = xs.unsqueeze(1)
    means = torch.stack([h(base[0], xs3)[0] for h in sd.state_heads])
    logits = torch.stack([h(aux[0], xs3)[2] for h in sd.auxiliary_heads]).squeeze(-1)
    tgt = torch.from_numpy(ep.S[1:]).to(device)
    P[:, 1:] = torch.sigmoid(logits).cpu().numpy()
    ERR[:, 1:] = ((means - tgt) ** 2).mean(-1).cpu().numpy()
    return P, ERR


@torch.no_grad()
def rollouts(sd, eps, starts, K, H, mode, device):
    """Own-prediction rollouts with recorded actions; head h drives copy h of every start (model_ids = h).

    Step 1 predicts row j0 (warm: states j0-H..j0-1, actions j0-H+1..j0; cold: state j0-1, action j0).
    Step m > 1 feeds the driving head's predicted state for row j0+m-2 and the recorded action j0+m-1.
    Returns stop probabilities and squared errors vs the recorded rows, both [starts, E, K].
    """
    E, n = len(sd.state_heads), len(starts)
    if n == 0:
        return np.zeros((0, E, K), np.float32), np.zeros((0, E, K), np.float32)
    rep = lambda a: torch.from_numpy(np.repeat(a, E, axis=0)).to(device)  # start-major, head-minor rows
    drive = torch.arange(E, device=device).repeat(n)
    cols = torch.arange(n * E, device=device)
    c = H if mode == "warm" else 1
    xs = rep(np.stack([eps[i].S[j - c:j] for i, j in starts]))
    xa = rep(np.stack([eps[i].A[j - c + 1:j + 1] for i, j in starts]))
    P = np.empty((n, E, K), np.float32)
    ERR = np.empty((n, E, K), np.float32)
    sd.reset()
    for m in range(K):
        means, logits = all_heads(sd, xs, xa)
        pred = means[drive, cols]
        tgt = rep(np.stack([eps[i].S[j + m] for i, j in starts]))
        P[:, :, m] = torch.sigmoid(logits[drive, cols]).view(n, E).cpu().numpy()
        ERR[:, :, m] = ((pred - tgt) ** 2).mean(-1).view(n, E).cpu().numpy()
        if m + 1 < K:
            xs = pred.unsqueeze(1)
            xa = rep(np.stack([eps[i].A[j + m + 1:j + m + 2] for i, j in starts]))
    sd.reset()
    return P, ERR


# ----------------------------------------------------------------------------- statistics

def survival(stop, steps):
    """stop [n, E, K] bool -> {step: [E] share of starts with no stop in steps 1..step}."""
    first = np.where(stop.any(-1), stop.argmax(-1) + 1, stop.shape[-1] + 1)       # [n, E]
    return {str(s): [float(x) for x in (first > s).mean(0)] for s in steps if s <= stop.shape[-1]}


def by_bin(x, K):
    return {f"{a}-{b}": [float(v) for v in x[:, :, a - 1:b].mean(axis=(0, 2))] for a, b in BINS if b <= K}


def head_stats(P, ERR):
    return dict(stop_share=[float(x) for x in (P > 0.5).mean(1)], mean_p=[float(x) for x in P.mean(1)],
                p99=[float(x) for x in np.percentile(P, 99, axis=1)], one_step_mse=[float(x) for x in ERR.mean(1)])


def mean(v):
    return float(np.mean(v))


def reading(ctx, roll, act):
    """The reading rule fixed before the run (module docstring); heads averaged."""
    k1, k32 = mean(ctx["1"]["stop_share"]), mean(ctx["32"]["stop_share"])
    out = dict(reset=dict(k1=k1, k32=k32, read=bool(k1 >= 0.05 and k1 >= 5 * k32)))
    w = roll.get("warm")
    if w and w.get("starts", 0) > 0 and "25" in w["survival"]:
        s_roll, s_tf = mean(w["survival"]["25"]), mean(w["teacher_forced_survival"]["25"])
        out["rollout"] = dict(warm_survival_25=s_roll, teacher_forced_survival_25=s_tf,
                              read=bool(s_roll <= 0.5 and s_tf >= 0.95))
    else:
        out["rollout"] = dict(read=None, note="no rollout start or fewer than 25 steps")
    if act:
        flags = {}
        for k in ("1", "32"):
            a, b = mean(act[k]["g0_sampled"]["stop_share"]), mean(act[k]["s2"]["stop_share"])
            flags[k] = dict(g0_sampled=a, s2=b, read=bool(a >= 0.05 and a >= 5 * b))
        out["action"] = dict(**flags, read=any(f["read"] for f in flags.values()))
    else:
        out["action"] = dict(read=None, note="no G0 policy given")
    return out


# ----------------------------------------------------------------------------- run

def g0_actions(g0_policy, eps, rows, norm, seed):
    from policy_loop import Policy
    pol = Policy(g0_policy)
    ck = torch.load(g0_policy, map_location="cpu", weights_only=False)
    sdict = ck.get("model_state_dict", ck)
    if "std" in sdict:
        std = sdict["std"].float().reshape(-1).numpy()
    elif "log_std" in sdict:
        std = np.exp(sdict["log_std"].float().reshape(-1).numpy())
    else:
        std = None
    if any(eps[i].obs is None for i, _ in rows):
        raise ValueError("the recording has no policy_observation; arm C needs it")
    obs = np.stack([eps[i].obs[j] for i, j in rows])
    with torch.no_grad():
        mu = pol(torch.from_numpy(obs)).numpy()
    out = {"g0_mean": (mu - norm["action_mean"]) / norm["action_std"]}
    if std is not None:
        noise = np.random.default_rng(seed).standard_normal(mu.shape).astype(np.float32)
        out["g0_sampled"] = (mu + std * noise - norm["action_mean"]) / norm["action_std"]
    return out, pol.describe(), None if std is None else [float(x) for x in std]


def run(wm, norm_path, physics_trace, withheld, out, g0_policy=None, K=100, start_every=50, frac=0.1,
        withhold_seed=101, seed=0, device="cpu"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    sd, H, norm, norm_diff = load_model(wm, norm_path, device)
    with np.load(physics_trace, allow_pickle=False) as z:
        validate_trace(z)
        trace = {k: z[k] for k in z.files if k != "metadata"}
    held = heldout(trace, withheld, frac, withhold_seed)
    eps = [Episode(trace, h, norm) for h in held]
    rows = [(i, j) for i, ep in enumerate(eps) for j in range(H, ep.L) if not ep.y[j]]
    if not rows:
        raise ValueError(f"no held-out row with {H} recorded steps of history")

    # A: context
    ctx = {}
    for k in CONTEXTS:
        P, ERR = windowed(sd, eps, rows, k, device)
        ctx[str(k)] = head_stats(P, ERR)
    ep_P, ep_ERR = zip(*[episode_context(sd, ep, device) for ep in eps])
    Pe = np.stack([ep_P[i][:, j] for i, j in rows], 1)
    Ee = np.stack([ep_ERR[i][:, j] for i, j in rows], 1)
    ctx["episode"] = head_stats(Pe, Ee)

    # B: own-prediction rollouts
    starts = [(i, j) for i, ep in enumerate(eps) for j in range(H, ep.L - K + 1, start_every)
              if not ep.y[j:j + K].any()]
    excluded = sum(1 for i, ep in enumerate(eps) for j in range(H, ep.L - K + 1, start_every) if ep.y[j:j + K].any())
    tf = np.stack([ep_P[i][:, j:j + K] for i, j in starts]) if starts else np.zeros((0, len(sd.state_heads), K))
    roll = {}
    for mode in ("warm", "cold"):
        P, ERR = rollouts(sd, eps, starts, K, H, mode, device)
        roll[mode] = dict(starts=len(starts), steps=K, survival=survival(P > 0.5, SURVIVAL_AT),
                          teacher_forced_survival=survival(tf > 0.5, SURVIVAL_AT),
                          stop_share_by_step=by_bin(P > 0.5, K), mean_p_by_step=by_bin(P, K),
                          mse_by_step=by_bin(ERR, K), teacher_forced_stop_share=[float(x) for x in
                                                                                 (tf > 0.5).mean(axis=(0, 2))]
                          if len(tf) else None)
    roll["excluded_windows_with_a_fall_row"] = excluded

    # C: action swap
    act, g0_info, g0_std = {}, None, None
    if g0_policy is not None:
        swaps, g0_info, g0_std = g0_actions(g0_policy, eps, rows, norm, seed)
        for k in (1, H):
            arm = {"s2": ctx[str(k)]}
            for name, a in swaps.items():
                P, ERR = windowed(sd, eps, rows, k, device, last_action=a.astype(np.float32))
                arm[name] = head_stats(P, ERR)
            act[str(k)] = arm
        if "g0_sampled" not in swaps:
            act = {k: dict(v, g0_sampled=v["g0_mean"]) for k, v in act.items()}
            g0_std = "missing: g0_sampled = g0_mean"

    # D: the withheld fall(s)
    falls = []
    for i, ep in enumerate(eps):
        for j in np.flatnonzero(ep.y):
            pre = ep_P[i][:, max(1, j - 50):j]
            first = [int(np.argmax(r > 0.5)) - len(r) if (r > 0.5).any() else None for r in pre]
            rec = dict(env=ep.env, first_row=ep.first, terminal_row=int(ep.first + j), steps_into_episode=int(j),
                       p_terminal_episode=[float(x) for x in ep_P[i][:, j]],
                       max_p_50_before=[float(x) for x in pre.max(1)] if pre.size else None,
                       first_stop_before_terminal=first)
            for k in (1, H):
                if j >= k:
                    P, _ = windowed(sd, eps, [(i, int(j))], k, device)
                    rec[f"p_terminal_k{k}"] = [float(x) for x in P[:, 0]]
            falls.append(rec)

    read = reading(ctx, roll, act)
    report = dict(protocol="A3 follow-up: read-only G0 world-model replay on held-out s2 data (21 Nov review)",
                  wm=str(wm), wm_sha256=sha256(wm), normalizer=str(norm_path), normalizer_sha256=sha256(norm_path),
                  normalizer_max_diff_to_checkpoint=norm_diff, physics_trace=str(physics_trace),
                  physics_trace_sha256=sha256(physics_trace), withheld=str(withheld), withheld_sha256=sha256(withheld),
                  heldout_episodes=len(eps), heldout_kinds={k: sum(e.kind == k for e in eps) for k in
                                                            sorted({e.kind for e in eps})},
                  rows=len(rows), history=H, contexts=ctx, rollouts=roll, action_swap=act, g0_policy=g0_info,
                  g0_std=g0_std, falls=falls, reading=read, seed=seed, device=device, wall_s=time.time() - t0,
                  analysis_sha256=sha256(__file__))
    (out / "g0_replay.json").write_text(json.dumps(report, indent=1, default=float))
    text = summary(report)
    (out / "g0_replay.txt").write_text(text)
    print(text, flush=True)
    return report


def summary(r):
    f = lambda v: f"{mean(v):.4f} [{min(v):.4f}, {max(v):.4f}]"
    L = ["A3 follow-up: G0 world-model replay on held-out s2 data (read-only; G0 stays a valid FAIL)",
         f"model {Path(r['wm']).name} ({r['wm_sha256'][:12]}); normalizer max diff to checkpoint "
         f"{r['normalizer_max_diff_to_checkpoint']:.1e}; held-out episodes {r['heldout_episodes']} {r['heldout_kinds']}; "
         f"rows {r['rows']}",
         "A. context, s2 action fixed: stop share mean over heads [min, max] | mean p | one-step mse"]
    for k, v in r["contexts"].items():
        L.append(f"  k={k:>7s}: {f(v['stop_share'])} | {mean(v['mean_p']):.4f} | {mean(v['one_step_mse']):.4f}")
    for mode in ("warm", "cold"):
        v = r["rollouts"][mode]
        sv = " ".join(f"{s}:{mean(x):.3f}" for s, x in v["survival"].items())
        tv = " ".join(f"{s}:{mean(x):.3f}" for s, x in v["teacher_forced_survival"].items())
        me = " ".join(f"{b}:{mean(x):.3f}" for b, x in v["mse_by_step"].items())
        L.append(f"B. {mode} rollouts ({v['starts']} starts x {len(v['survival']['1'])} heads, {v['steps']} steps), "
                 f"s2 actions: survival {sv}")
        L.append(f"   teacher-forced survival, same windows: {tv}")
        L.append(f"   state mse by step: {me}")
    L.append(f"   windows excluded (contain the fall row): {r['rollouts']['excluded_windows_with_a_fall_row']}")
    if r["action_swap"]:
        for k, arm in r["action_swap"].items():
            L.append(f"C. action swap k={k}: stop share s2 {mean(arm['s2']['stop_share']):.4f} | G0 mean "
                     f"{mean(arm['g0_mean']['stop_share']):.4f} | G0 sampled {mean(arm['g0_sampled']['stop_share']):.4f}")
    else:
        L.append("C. action swap: not run (no G0 policy)")
    for fl in r["falls"]:
        L.append(f"D. fall env {fl['env']} at step {fl['steps_into_episode']}: p at terminal row (episode memory) "
                 f"{' '.join(f'{x:.2f}' for x in fl['p_terminal_episode'])}; max p over the 50 rows before "
                 f"{' '.join(f'{x:.2f}' for x in fl['max_p_50_before'] or [])}")
    rd = r["reading"]
    L.append(f"Reading (rule fixed before the run): reset problem {'READ' if rd['reset']['read'] else 'not read'}; "
             f"rollout problem {_flag(rd['rollout']['read'])}; action problem {_flag(rd['action']['read'])}")
    return "\n".join(L) + "\n"


def _flag(v):
    return "n/a" if v is None else ("READ" if v else "not read")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", type=Path, required=True, help="the saved G0 fit (g0_wm_seed1.pt)")
    p.add_argument("--norm", type=Path, required=True, help="the frozen normalizer (data/go2_norm.npz)")
    p.add_argument("--physics", type=Path, required=True, help="s2's round-0 recording")
    p.add_argument("--withheld", type=Path, required=True, help="data/withheld.json")
    p.add_argument("--g0-policy", type=Path, default=None, help="G0 checkpoint for arm C")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--start-every", type=int, default=50)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    run(a.wm, a.norm, a.physics, a.withheld, a.out, a.g0_policy, a.steps, a.start_every, device=a.device)


if __name__ == "__main__":
    main()
