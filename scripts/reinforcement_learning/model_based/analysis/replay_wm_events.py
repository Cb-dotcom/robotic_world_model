#!/usr/bin/env python3
"""Replay real trajectories through a world model around command changes (P1.5 follow-up, 15 Oct 2026).

Question: does ETH's world model see the falls that follow a real command change, or is it
"optimistic" there (predicts survival, low disagreement) while the real robot falls?

Input: an .npz from scripts/reinforcement_learning/rsl_rl/record_real_trajectories.py.
No simulator. Read-only on the checkpoint.

Row conventions (as in the recorder): row t = (state after step t, action applied in step t).
If done[t], row t is the post-reset spawn state, so the finished episode's last real row is t-1
and the new episode's first usable (state, action) pair is row t+1.

Events = mid-episode command changes: |cmd_xy[c] - cmd_xy[c-1]| > 1e-6, rows c-1 and c in the
same episode, with >= H usable history rows. Labels (fixed before any replay):
  fall     the episode ends by a fall within --fall_within steps after c
  survived the episode runs >= --fall_within steps after c without a fall, or ends by time-out
  other    anything else (excluded)
Events whose episode ends (any reason) less than W+1 steps after c are excluded and counted.

Signals per event over rows c+1 .. c+W (teacher forced: each row f predicted from rows f-H..f-1
and actions f-H+1..f, the scorer's pairing):
  epi_mean   mean ensemble disagreement (sum over dims of the std across head means)
  mse_mean   mean one-step squared error of the head-mean prediction (normalized space, mean over 45 dims)
  term_max   max termination probability (sigmoid of the head-mean logit)
Baselines over the same rows: jump = |cmd_xy[c] - cmd_xy[c-1]|; tilt_max = max |g_xy| (real);
cmd_speed = |cmd_xy[c]|.
Open loop from c (context rows c-H+1..c, then the real actions), per head, up to the episode end
or --rollout_k steps: max predicted termination probability and max predicted tilt |g_xy|,
next to the real max tilt over the same rows.

Usage (from scripts/reinforcement_learning/model_based):
  python analysis/replay_wm_events.py --traj T.npz --wm assets/models/pretrain_rnn_ens.pt \
      --config anymal_d_flat --out OUTDIR
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

GXY = slice(6, 8)  # projected gravity x, y (raw units)


# ----------------------------------------------------------------------------- events (numpy only)

def episode_ids(done: np.ndarray) -> np.ndarray:
    """[T, N] -> episode index per row; the row of a done step starts the next episode."""
    return np.cumsum(done.astype(np.int64), axis=0)


def find_events(cmd, done, time_out, H, W, fall_within):
    """Return list of dicts (env, c, label, end, end_is_fall) and a dict of exclusion counts."""
    T, N = done.shape
    ep = episode_ids(done)
    events, excl = [], {"short_history": 0, "too_close_to_end": 0, "other": 0}
    for n in range(N):
        # episode end rows (done steps) for this env
        ends = np.nonzero(done[:, n])[0]
        for c in range(1, T):
            if done[c, n] or done[c - 1, n]:
                continue
            if ep[c, n] != ep[c - 1, n]:
                continue
            if np.linalg.norm(cmd[c, n, :2] - cmd[c - 1, n, :2]) <= 1e-6:
                continue
            # first usable row of this episode
            start = 0 if ep[c, n] == 0 else int(np.nonzero(done[:c, n])[0][-1]) + 1
            if c - H < start:
                excl["short_history"] += 1
                continue
            later = ends[ends > c]
            end = int(later[0]) if len(later) else None  # done step; last real row = end-1
            end_fall = bool(end is not None and not time_out[end, n])
            last_row = (end - 1) if end is not None else T - 1
            if last_row < c + W:
                excl["too_close_to_end"] += 1
                continue
            if end_fall and end - c <= fall_within:
                label = "fall"
            elif end is not None:
                label = "survived"  # time-out, or a fall more than fall_within steps later
            elif T - 1 - c >= fall_within:
                label = "survived"  # still running at the end of the recording
            else:
                label = "other"     # censored too soon to know
            if label == "other":
                excl["other"] += 1
                continue
            events.append({"env": n, "c": c, "label": label, "end": end, "end_is_fall": end_fall,
                           "last_row": last_row})
    return events, excl


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie). NaN if a side is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def bootstrap_auc_ci(pos, neg, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return (float("nan"), float("nan"))
    vals = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(B)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ----------------------------------------------------------------------------- world model

def load_wm_cfg(path, config, device):
    import torch
    from rsl_rl.modules import SystemDynamicsEnsemble
    if config == "anymal_d_flat":
        from configs.anymal_d_flat_cfg import AnymalDFlatConfig as Cfg
    elif config == "go2_flat":
        from configs.go2_flat_cfg import Go2FlatConfig as Cfg
    else:
        raise ValueError(config)
    cfg = Cfg()
    mac = cfg.model_architecture_config
    H = getattr(mac, "history_horizon", 32)
    sd = SystemDynamicsEnsemble(
        45, 12, getattr(mac, "extension_dim", 0), getattr(mac, "contact_dim", 8),
        getattr(mac, "termination_dim", 1), device,
        ensemble_size=getattr(mac, "ensemble_size", 5), history_horizon=H,
        architecture_config=mac.architecture_config, freeze_auxiliary=False,
    ).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd.load_state_dict(ckpt["system_dynamics_state_dict"], strict=True)
    sd.eval()
    if ckpt.get("normalized", False) and "state_mean" in ckpt:
        norm = {k: ckpt[k].detach().cpu().numpy().reshape(-1).astype(np.float32)
                for k in ("state_mean", "state_std", "action_mean", "action_std")}
        src = "checkpoint"
    else:
        dc = cfg.data_config
        norm = {"state_mean": np.array(dc.state_data_mean, np.float32), "state_std": np.array(dc.state_data_std, np.float32),
                "action_mean": np.array(dc.action_data_mean, np.float32), "action_std": np.array(dc.action_data_std, np.float32)}
        src = f"config {config}"
    return sd, H, norm, src


def run_replay(sd, H, norm, traj, events, W, rollout_k, device, batch=256):
    import torch
    S = ((traj["state"] - norm["state_mean"]) / norm["state_std"]).astype(np.float32)
    A = ((traj["action"] - norm["action_mean"]) / norm["action_std"]).astype(np.float32)
    E = len(sd.state_heads)
    raw = traj["state"]

    # --- teacher-forced windows
    win = [(i, e["env"], f) for i, e in enumerate(events) for f in range(e["c"] + 1, e["c"] + W + 1)]
    epi = np.zeros(len(win)); mse = np.zeros(len(win)); term = np.full(len(win), np.nan)
    with torch.no_grad():
        for b0 in range(0, len(win), batch):
            chunk = win[b0:b0 + batch]
            xs = torch.from_numpy(np.stack([S[f - H:f, n] for _, n, f in chunk])).to(device)
            xa = torch.from_numpy(np.stack([A[f - H + 1:f + 1, n] for _, n, f in chunk])).to(device)
            tgt = torch.from_numpy(np.stack([S[f, n] for _, n, f in chunk])).to(device)
            sd.reset()
            mean, _alea, e_, _ext, _con, t_ = sd.forward(xs, xa)
            epi[b0:b0 + len(chunk)] = e_.cpu().numpy().reshape(-1)
            mse[b0:b0 + len(chunk)] = ((mean - tgt) ** 2).mean(dim=1).cpu().numpy()
            if t_ is not None:
                term[b0:b0 + len(chunk)] = torch.sigmoid(t_).cpu().numpy().reshape(-1)
    for i, e in enumerate(events):
        m = np.array([w[0] == i for w in win])
        e["epi_mean"] = float(epi[m].mean()); e["mse_mean"] = float(mse[m].mean())
        e["term_max"] = float(np.nanmax(term[m])) if np.isfinite(term[m]).any() else float("nan")
        n, c = e["env"], e["c"]
        e["jump"] = float(np.linalg.norm(traj["cmd"][c, n, :2] - traj["cmd"][c - 1, n, :2]))
        e["cmd_speed"] = float(np.linalg.norm(traj["cmd"][c, n, :2]))
        e["tilt_max"] = float(np.linalg.norm(raw[c + 1:c + W + 1, n, GXY], axis=-1).max())

    # --- open loop from c, per head
    with torch.no_grad():
        for i, e in enumerate(events):
            n, c = e["env"], e["c"]
            K = int(min(rollout_k, e["last_row"] - c))
            e["rollout_k"] = K
            real_tilt = np.linalg.norm(raw[c + 1:c + K + 1, n, GXY], axis=-1)
            e["real_tilt_max_rollout"] = float(real_tilt.max())
            xs0 = torch.from_numpy(S[c - H + 1:c + 1, n][None]).to(device)
            xa0 = torch.from_numpy(A[c - H + 2:c + 2, n][None]).to(device)  # actions rows c-H+2..c+1
            fut = torch.from_numpy(A[c + 2:c + K + 1, n][None]).to(device)   # rows c+2..c+K
            pt, tl = [], []
            for h in range(E):
                ids = torch.full((1, 1, 1), h, dtype=torch.long, device=device)
                sd.reset()
                xs, xa = xs0, xa0
                tmax, gmax = 0.0, 0.0
                for k in range(K):
                    mean, _a, _e, _x, _c, t_ = sd.forward(xs, xa, ids)
                    g = mean[0, GXY] * torch.from_numpy(norm["state_std"][GXY]).to(device) + \
                        torch.from_numpy(norm["state_mean"][GXY]).to(device)
                    gmax = max(gmax, float(torch.linalg.norm(g)))
                    if t_ is not None:
                        tmax = max(tmax, float(torch.sigmoid(t_).reshape(-1)[0]))
                    if k < K - 1:
                        xs = mean.unsqueeze(1); xa = fut[:, k:k + 1]
                pt.append(tmax); tl.append(gmax)
            e["pred_term_max_heads"] = pt
            e["pred_tilt_max_heads"] = tl
            e["pred_term_max_mean"] = float(np.mean(pt))
            e["pred_term_any_head"] = float(np.max(pt))
            e["pred_tilt_max_mean"] = float(np.mean(tl))
    return events


def summarize(events, excl):
    F = [e for e in events if e["label"] == "fall"]
    Sv = [e for e in events if e["label"] == "survived"]
    out = {"n_fall": len(F), "n_survived": len(Sv), "excluded": excl, "auc": {}, "auc_ci95": {}}
    for k in ("epi_mean", "mse_mean", "term_max", "jump", "tilt_max", "cmd_speed"):
        p, q = [e[k] for e in F], [e[k] for e in Sv]
        out["auc"][k] = auc(p, q)
        out["auc_ci95"][k] = bootstrap_auc_ci(p, q)
    if F:
        out["fall_frac_pred_term_mean_below_0.5"] = float(np.mean([e["pred_term_max_mean"] < 0.5 for e in F]))
        out["fall_frac_pred_term_any_head_below_0.5"] = float(np.mean([e["pred_term_any_head"] < 0.5 for e in F]))
        out["fall_frac_pred_tilt_below_half_real"] = float(np.mean(
            [e["pred_tilt_max_mean"] < 0.5 * e["real_tilt_max_rollout"] for e in F]))
        out["fall_median_pred_term_max_mean"] = float(np.median([e["pred_term_max_mean"] for e in F]))
        out["fall_median_pred_tilt_max"] = float(np.median([e["pred_tilt_max_mean"] for e in F]))
        out["fall_median_real_tilt_max"] = float(np.median([e["real_tilt_max_rollout"] for e in F]))
    if Sv:
        out["surv_median_pred_term_max_mean"] = float(np.median([e["pred_term_max_mean"] for e in Sv]))
        out["surv_median_pred_tilt_max"] = float(np.median([e["pred_tilt_max_mean"] for e in Sv]))
        out["surv_median_real_tilt_max"] = float(np.median([e["real_tilt_max_rollout"] for e in Sv]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traj", required=True)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--config", default="anymal_d_flat", choices=["anymal_d_flat", "go2_flat"])
    ap.add_argument("--window", type=int, default=25, help="W: teacher-forced rows after the change")
    ap.add_argument("--fall_within", type=int, default=100)
    ap.add_argument("--rollout_k", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    z = np.load(a.traj, allow_pickle=False)
    traj = {k: z[k] for k in ("state", "action", "cmd", "done", "time_out")}
    sd, H, norm, src = load_wm_cfg(a.wm, a.config, dev)
    print(f"[replay] wm={a.wm} H={H} heads={len(sd.state_heads)} normalizer from {src} device={dev}")
    events, excl = find_events(traj["cmd"], traj["done"], traj["time_out"], H, a.window, a.fall_within)
    print(f"[replay] events: fall={sum(e['label'] == 'fall' for e in events)} "
          f"survived={sum(e['label'] == 'survived' for e in events)} excluded={excl}")
    events = run_replay(sd, H, norm, traj, events, a.window, a.rollout_k, dev)
    s = summarize(events, excl)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "events.json"), "w") as f:
        json.dump(events, f, indent=1)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    print("================ REPLAY SUMMARY ================")
    print(f"  events: fall={s['n_fall']} survived={s['n_survived']} excluded={s['excluded']}")
    print("  AUC fall vs survived (95% CI), rows c+1..c+W:")
    for k, v in s["auc"].items():
        lo, hi = s["auc_ci95"][k]
        print(f"    {k:10s} {v:.3f}  [{lo:.3f}, {hi:.3f}]")
    for k, v in s.items():
        if k.startswith(("fall_", "surv_")):
            print(f"  {k}: {v:.3f}")
    print("================================================")


if __name__ == "__main__":
    main()
