#!/usr/bin/env python3
"""GRU memory checks on trained world models (doubt register B3 and B7, 13 Oct 2026).

Read-only: loads checkpoints and a trace CSV, writes a text report. No training, no simulator.

B3  Training-time memory leak between heads.
    The fitters call ``sd.reset()`` once per batch and then ``compute_loss(..., bootstrap=True)``.
    Head 0 starts from zero memory. Head k>0 starts its 32-step window from the trunk memory that
    head k-1 left after its 8-step forecast, on a *different* random resample of the batch.
    At scoring and in imagination every head starts from zero memory.
    Measured per head: how far the prediction moves between the two starts ("shift"), compared
    with the normal head disagreement (epi), and the one-step error in both starts.
    Head 0 must show shift = 0 (sanity check).

B7  Imagination cold start.
    ``policy_training.py`` (offline) and ``MBPOOnPolicyRunner.imagine()`` (online) cut the history to
    the last step when ``common_step_counter > 0``. So after every episode reset (offline) and at
    the start of every ``imagine()`` call (online) the model predicts from 1 state with empty memory.
    Measured: k-step open-loop rollout error and disagreement from a cold start (1 state, zero
    memory) and a warm start (32 states), on the same windows with the logged actions.

Usage (from scripts/reinforcement_learning/model_based):
    python analysis/memory_checks.py --trace T.csv --steps_per_env 1000 --out report.txt \
        NAME=CKPT [NAME=CKPT ...]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_signals import load_wm, read_numeric_csv, wm_normalize  # noqa: E402

GROUPS = {"lin_vel": (0, 3), "ang_vel": (3, 6), "gravity": (6, 9), "joint_pos": (9, 21),
          "joint_vel": (21, 33), "torque": (33, 45)}
REPORT_K = (1, 2, 3, 5, 9, 17, 24)  # 1-based rollout steps shown in the B7 table


def clean_window_starts(data, steps_per_env, H, L):
    """Rows f0 such that rows f0-H .. f0+L-1 sit in one env, with no termination flag and no
    all-zero action (unmarked reset row, doubt A2)."""
    n = len(data)
    term = data[:, 65] > 0.5 if data.shape[1] > 65 else np.zeros(n, bool)
    zero_act = np.abs(data[:, 45:57]).sum(1) == 0
    bad = (term | zero_act).astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(bad)])
    f = np.arange(n)
    t = f % steps_per_env
    ok = (t >= H) & (t + L <= steps_per_env)
    f = f[ok]
    ok2 = (cum[f + L] - cum[f - H]) == 0
    return f[ok2]


def gather(arr, starts, lo, hi):
    """arr rows starts+lo .. starts+hi-1 -> (B, hi-lo, D)."""
    idx = starts[:, None] + np.arange(lo, hi)[None, :]
    return arr[idx]


@torch.no_grad()
def head_training_like(sd, head, S, A, H, gen):
    """Copy of SystemDynamicsEnsemble.compute_state_loss forward pass for one head (rnn).
    Starts from whatever memory the trunk holds. Returns the i=0 prediction."""
    F = S.shape[1] - H
    xs = S[:, :H]
    first = None
    for i in range(F):
        xa = A[:, 1:H + 1] if i == 0 else A[:, H + i:H + i + 1]
        mean, std = head(sd.state_base(xs, xa), xs)
        if i == 0:
            first = mean
        if head.output_std:
            noise = (torch.randn_like(mean) if gen is None
                     else torch.randn(mean.shape, generator=gen, device=mean.device, dtype=mean.dtype))
            xs = (noise * std + mean).unsqueeze(1)
        else:
            xs = mean.unsqueeze(1)
    return first


@torch.no_grad()
def check_b3(sd, S, A, H, repeats, seed, say):
    """S, A: (B, H+F, .) tensors; rows f0-H .. f0+F-1. Target of the first prediction: S[:, H]."""
    B = S.shape[0]
    E = len(sd.state_heads)
    target = S[:, H]
    # zero start, all heads, same rows
    sd.reset()
    h = sd.state_base(S[:, :H], A[:, 1:H + 1])
    zero = torch.stack([hd(h, S[:, :H])[0] for hd in sd.state_heads])  # (E, B, 45)
    epi = zero.std(dim=0).sum(dim=1)                                    # (B,)
    epi_mean = epi.mean().item()
    gen = torch.Generator(device=S.device).manual_seed(seed)
    shift = np.zeros((repeats, E)); mse_tr = np.zeros((repeats, E)); mse_z = np.zeros((repeats, E))
    gshift = np.zeros((repeats, E, len(GROUPS)))
    for r in range(repeats):
        sd.reset()
        for k, hd in enumerate(sd.state_heads):
            ids = torch.randint(0, B, (B,), generator=gen, device=S.device)  # bootstrap=True
            tr = head_training_like(sd, hd, S[ids], A[ids], H, gen)
            z = zero[k][ids]
            d = (tr - z).abs()
            shift[r, k] = d.sum(1).mean().item()
            for gi, (a, b) in enumerate(GROUPS.values()):
                gshift[r, k, gi] = d[:, a:b].sum(1).mean().item()
            mse_tr[r, k] = ((tr - target[ids]) ** 2).mean().item()
            mse_z[r, k] = ((z - target[ids]) ** 2).mean().item()
    shift, mse_tr, mse_z, gshift = shift.mean(0), mse_tr.mean(0), mse_z.mean(0), gshift.mean(0)
    say(f"[B3] windows={B} repeats={repeats}  mean epi (zero start, all heads) = {epi_mean:.4f}")
    say("[B3] head | shift = mean sum|train-start - zero-start| | shift/epi | 1-step MSE zero-start | 1-step MSE train-start")
    for k in range(E):
        say(f"[B3]   {k}  | {shift[k]:.4f} | {shift[k] / epi_mean:.3f} | {mse_z[k]:.5f} | {mse_tr[k]:.5f}")
    others = list(range(1, E))
    say(f"[B3] heads 1-{E - 1}: mean shift/epi = {shift[others].mean() / epi_mean:.3f}; "
        f"MSE zero/train = {mse_z[others].mean() / max(mse_tr[others].mean(), 1e-12):.3f}; "
        f"head-0 MSE / heads 1-{E - 1} MSE (zero start) = {mse_z[0] / max(mse_z[others].mean(), 1e-12):.3f}")
    say("[B3] shift by group (heads 1-4 mean): " + ", ".join(
        f"{g}={gshift[others, gi].mean():.4f}" for gi, g in enumerate(GROUPS)))
    if shift[0] > 0.01 * epi_mean:
        say(f"[B3] WARNING: head 0 shift {shift[0]:.2e} > 1% of epi; the emulation does not match training")
    return {"epi_mean": epi_mean, "shift": shift, "mse_zero": mse_z, "mse_train": mse_tr}


@torch.no_grad()
def rollout(sd, S_ctx, A_ctx, A_fut, model_ids, warm):
    """Open-loop rollout like imagination_step. S_ctx: states rows f0-H..f0-1; A_ctx: actions rows
    f0-H+1..f0; A_fut: actions rows f0+1..f0+K-1. Returns preds (K, B, 45) and epi (K, B)."""
    sd.reset()
    xs, xa = (S_ctx, A_ctx) if warm else (S_ctx[:, -1:], A_ctx[:, -1:])
    K = A_fut.shape[1] + 1
    preds, epis = [], []
    for k in range(K):
        mean, _alea, epi, *_ = sd.forward(xs, xa, model_ids)
        preds.append(mean); epis.append(epi)
        if k < K - 1:
            xs = mean.unsqueeze(1)
            xa = A_fut[:, k:k + 1]
    return torch.stack(preds), torch.stack(epis)


@torch.no_grad()
def check_b7(sd, St, At, H, K, seed, say):
    """St, At: (B, H+K, .); rows f0-H .. f0+K-1. Targets: St[:, H:H+K]."""
    B = St.shape[0]
    E = len(sd.state_heads)
    g = torch.Generator(device="cpu").manual_seed(seed)
    model_ids = torch.randint(0, E, (1, B, 1), generator=g).to(St.device)
    S_ctx, A_ctx = St[:, :H], At[:, 1:H + 1]
    A_fut = At[:, H + 1:H + K]
    tgt = St[:, H:H + K].transpose(0, 1)  # (K, B, 45)
    out = {}
    for name, warm in (("warm", True), ("cold", False)):
        p, e = rollout(sd, S_ctx, A_ctx, A_fut, model_ids, warm)
        out[name] = (((p - tgt) ** 2).mean(dim=2).mean(dim=1).cpu().numpy(), e.mean(dim=1).cpu().numpy(),
                     ((p[:, :, 6:9] - tgt[:, :, 6:9]) ** 2).mean(dim=2).mean(dim=1).cpu().numpy())
    mw, ew, gw = out["warm"]; mc, ec, gc = out["cold"]
    say(f"[B7] windows={B} rollout K={K} (open loop, logged actions, one random head per window as in imagination)")
    say("[B7] step | MSE warm | MSE cold | cold/warm | epi warm | epi cold | cold/warm | gravity MSE cold/warm")
    for k1 in REPORT_K:
        if k1 > K:
            continue
        k = k1 - 1
        say(f"[B7]  {k1:3d} | {mw[k]:.5f} | {mc[k]:.5f} | {mc[k] / max(mw[k], 1e-12):6.2f} | "
            f"{ew[k]:.4f} | {ec[k]:.4f} | {ec[k] / max(ew[k], 1e-12):6.2f} | {gc[k] / max(gw[k], 1e-12):6.2f}")
    ratio = mc / np.maximum(mw, 1e-12)
    settle = next((k + 1 for k in range(K) if ratio[k:].max() <= 1.10), None)
    say(f"[B7] mean over {K} steps: MSE cold/warm = {mc.mean() / max(mw.mean(), 1e-12):.2f}; "
        f"epi cold/warm = {ec.mean() / max(ew.mean(), 1e-12):.2f}; "
        f"cold stays within 10% of warm from step {settle if settle else '>' + str(K)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--steps_per_env", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_windows", type=int, default=2048)
    ap.add_argument("--forecast", type=int, default=8, help="fitter forecast horizon (B3)")
    ap.add_argument("--rollout", type=int, default=24, help="imagination steps checked (B7)")
    ap.add_argument("--repeats", type=int, default=3, help="B3 bootstrap repeats")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("models", nargs="+", help="NAME=CKPT")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    data = read_numeric_csv(a.trace)
    first = True
    for spec in a.models:
        name, ck = spec.split("=", 1)
        sd, ckpt, H = load_wm(ck, a.device)
        if first:
            L = max(a.forecast, a.rollout)
            starts = clean_window_starts(data, a.steps_per_env, H, L)
            rng = np.random.default_rng(a.seed)
            if len(starts) > a.n_windows:
                starts = np.sort(rng.choice(starts, a.n_windows, replace=False))
            say(f"trace={a.trace} rows={len(data)} H={H} clean windows used={len(starts)} "
                f"(envs {len(np.unique(starts // a.steps_per_env))})")
            if len(starts) == 0:
                raise SystemExit("no clean windows; check --steps_per_env")
            first = False
        st, ac, msg = wm_normalize(data[:, 0:45].copy(), data[:, 45:57].copy(), ckpt)
        say(f"\n=== {name}  ({ck})  {msg}")
        dev = a.device
        S3 = torch.from_numpy(gather(st, starts, -H, a.forecast)).to(dev)
        A3 = torch.from_numpy(gather(ac, starts, -H, a.forecast)).to(dev)
        check_b3(sd, S3, A3, H, a.repeats, a.seed, say)
        S7 = torch.from_numpy(gather(st, starts, -H, a.rollout)).to(dev)
        A7 = torch.from_numpy(gather(ac, starts, -H, a.rollout)).to(dev)
        check_b7(sd, S7, A7, H, a.rollout, a.seed, say)
        del sd
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[memory_checks] wrote {a.out}")


if __name__ == "__main__":
    main()
