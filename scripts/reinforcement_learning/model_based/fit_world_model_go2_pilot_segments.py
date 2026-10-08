# SPDX-License-Identifier: BSD-3-Clause
"""Boundary-aware offline world-model fit for the Go2 1M pilot dataset.

This trainer uses:
  - clean real termination labels from assets/data/go2_pilot_1m/segments_clean_terms
  - manifest.csv to respect env-major block boundaries
  - no artificial seam labels as termination targets
  - no sequence windows crossing env boundaries or real terminations

It avoids the single-column ambiguity where termination was both a fall label and
a sequence-boundary mask.

Optional (all default to the original behaviour):
  --seed INT          seed python/numpy/torch(+cuda) before model construction.
  --norm_stats PATH   load state/action mean/std from an .npz instead of computing them.
  --lazy_windows      gather training windows on the fly from the flat normalized data
                      instead of materializing a [num_windows, 40, 66] tensor (identical
                      batches; ~12 GB less GPU memory for 1.2M windows).
  manifest.csv may carry a ``num_envs`` column (default 64 when absent): each file is
  env-major with rows == steps * num_envs.
"""

import argparse
import hashlib
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rsl_rl.modules import SystemDynamicsEnsemble

from configs.go2_flat_cfg import Go2FlatConfig
from configs.anymal_d_flat_cfg import AnymalDFlatConfig

_CONFIGS = {"go2_flat": Go2FlatConfig, "anymal_d_flat": AnymalDFlatConfig}


def read_segments(root, state_dim, action_dim, ext_dim, contact_dim, term_dim):
    root = Path(root)
    manifest = pd.read_csv(root / "manifest.csv")
    expected_cols = state_dim + action_dim + ext_dim + contact_dim + term_dim

    has_num_envs = "num_envs" in manifest.columns
    segs = []
    total_rows = 0
    total_terms = 0

    for row in manifest.itertuples(index=False):
        f = root / f"{row.name}.csv"
        steps = int(row.steps)
        rows_expected = int(row.rows)
        num_envs = int(row.num_envs) if has_num_envs else 64

        data = pd.read_csv(f, header=None).values.astype(np.float32)
        if data.shape[1] != expected_cols:
            raise RuntimeError(f"{f}: expected {expected_cols} cols, got {data.shape[1]}")
        if data.shape[0] != rows_expected:
            raise RuntimeError(f"{f}: expected {rows_expected} rows, got {data.shape[0]}")
        if data.shape[0] != steps * num_envs:
            raise RuntimeError(f"{f}: rows {data.shape[0]} != steps*num_envs {steps}*{num_envs}")

        terms = int((data[:, -1] > 0.5).sum())
        print(f"[load] {f.name}: rows={data.shape[0]} steps={steps} envs={num_envs} terms={terms}")

        segs.append((row.name, steps, data, num_envs))
        total_rows += data.shape[0]
        total_terms += terms

    print(f"[load] total_rows={total_rows} total_real_terms={total_terms}")
    return segs, total_rows, total_terms


def compute_normalizer(segs, state_dim, action_dim):
    s_sum = np.zeros((state_dim,), dtype=np.float64)
    s_sumsq = np.zeros((state_dim,), dtype=np.float64)
    a_sum = np.zeros((action_dim,), dtype=np.float64)
    a_sumsq = np.zeros((action_dim,), dtype=np.float64)
    n = 0

    for _, _, data, _ in segs:
        state = data[:, 0:state_dim].astype(np.float64)
        action = data[:, state_dim:state_dim + action_dim].astype(np.float64)
        s_sum += state.sum(axis=0)
        s_sumsq += (state * state).sum(axis=0)
        a_sum += action.sum(axis=0)
        a_sumsq += (action * action).sum(axis=0)
        n += data.shape[0]

    s_mean = s_sum / n
    a_mean = a_sum / n
    s_var = np.maximum(s_sumsq / n - s_mean * s_mean, 1e-12)
    a_var = np.maximum(a_sumsq / n - a_mean * a_mean, 1e-12)

    s_std = np.sqrt(s_var) + 1e-6
    a_std = np.sqrt(a_var) + 1e-6

    return (
        torch.tensor(s_mean, dtype=torch.float32).view(1, 1, -1),
        torch.tensor(s_std, dtype=torch.float32).view(1, 1, -1),
        torch.tensor(a_mean, dtype=torch.float32).view(1, 1, -1),
        torch.tensor(a_std, dtype=torch.float32).view(1, 1, -1),
    )


def load_normalizer(path, state_dim, action_dim):
    """Load state/action mean/std (std already includes the eps) from an .npz.

    Returns the same (1, 1, D) float32 tensors as compute_normalizer.
    """
    with np.load(path, allow_pickle=False) as z:
        stats = {k: np.asarray(z[k], dtype=np.float64).reshape(-1) for k in ("state_mean", "state_std", "action_mean", "action_std")}
    for k, d in (("state_mean", state_dim), ("state_std", state_dim), ("action_mean", action_dim), ("action_std", action_dim)):
        if stats[k].shape != (d,):
            raise RuntimeError(f"{path}: {k} has shape {stats[k].shape}, expected ({d},)")
        if not np.isfinite(stats[k]).all():
            raise RuntimeError(f"{path}: {k} has non-finite values")
    if (stats["state_std"] <= 0).any() or (stats["action_std"] <= 0).any():
        raise RuntimeError(f"{path}: std must be > 0")
    return tuple(torch.tensor(stats[k], dtype=torch.float32).view(1, 1, -1) for k in ("state_mean", "state_std", "action_mean", "action_std"))


class LazyWindows:
    """Window store that gathers [B, seq_len, cols] batches from the flat normalized rows.

    ``starts`` holds the flat row index of every valid window in exactly the order in
    which build_segment_windows concatenates them, so ``lazy[ids]`` equals
    ``materialized[ids]`` element for element.
    """

    def __init__(self, flat, starts, seq_len):
        self.flat = flat
        self.starts = starts
        self.offsets = torch.arange(seq_len, device=flat.device, dtype=torch.long).view(1, -1)
        self.shape = (int(starts.shape[0]), seq_len, int(flat.shape[1]))

    def __getitem__(self, ids):
        return self.flat[self.starts[ids].unsqueeze(1) + self.offsets]


def build_segment_windows(
    segs,
    state_dim,
    action_dim,
    contact_dim,
    term_dim,
    history_horizon,
    forecast_horizon,
    device,
    s_mean,
    s_std,
    a_mean,
    a_std,
    normalize=True,
    lazy=False,
):
    seq_len = history_horizon + forecast_horizon
    windows = []
    flats = []
    row_offset = 0
    valid_total = 0
    skipped_total = 0

    s_mean = s_mean.to(device)
    s_std = s_std.to(device)
    a_mean = a_mean.to(device)
    a_std = a_std.to(device)

    for name, steps, data_np, num_envs in segs:
        x = torch.from_numpy(data_np).to(device)
        x = x.view(num_envs, steps, -1).contiguous()

        if normalize:
            x[:, :, 0:state_dim] = (x[:, :, 0:state_dim] - s_mean) / s_std
            a0 = state_dim
            a1 = state_dim + action_dim
            x[:, :, a0:a1] = (x[:, :, a0:a1] - a_mean) / a_std

        # Window view: [num_envs, W, seq_len, cols]
        W = steps - seq_len + 1
        if W <= 0:
            raise RuntimeError(f"{name}: steps {steps} <= seq_len {seq_len}")

        term = x[:, :, -1]
        # A window starting at i is invalid if a real termination appears before
        # the final target row, i.e. in [i, i+seq_len-2].
        reset_roll = term.unfold(1, seq_len - 1, 1)[:, :W].sum(dim=-1)
        valid = reset_roll <= 0.5

        if lazy:
            # Flat row index of window (env e, start w) = row_offset + e*steps + w; boolean
            # indexing keeps the same (env, start) row-major order as win[valid] below.
            start_grid = row_offset + torch.arange(num_envs, device=device).view(-1, 1) * steps + torch.arange(W, device=device).view(1, -1)
            valid_win = start_grid[valid]
            flats.append(x.view(num_envs * steps, -1))
            row_offset += num_envs * steps
        else:
            win = x.unfold(1, seq_len, 1).permute(0, 1, 3, 2)
            valid_win = win[valid].contiguous()

        skipped = int((~valid).sum().item())
        kept = int(valid_win.shape[0])
        print(f"[windows] {name}: kept={kept} skipped_cross_reset={skipped}")

        windows.append(valid_win)
        valid_total += kept
        skipped_total += skipped

    if lazy:
        all_windows = LazyWindows(torch.cat(flats, dim=0).contiguous(), torch.cat(windows, dim=0).contiguous(), seq_len)
        print(f"[windows] total={valid_total} skipped={skipped_total} lazy window shape={all_windows.shape} flat_rows={tuple(all_windows.flat.shape)}")
        return all_windows

    all_windows = torch.cat(windows, dim=0).contiguous()
    print(f"[windows] total={valid_total} skipped={skipped_total} tensor_shape={tuple(all_windows.shape)}")
    return all_windows


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(path):
    """Commit of the repo containing ``path`` and whether tracked files are modified (None if no git)."""
    try:
        commit = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", path, "status", "--porcelain", "--untracked-files=no"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return {"commit": commit, "dirty": bool(dirty), "dirty_files": dirty.splitlines()[:50]}
    except Exception:
        return {"commit": None, "dirty": None, "dirty_files": []}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser(description="Boundary-aware Go2 pilot world-model fit.")
    p.add_argument("--data", required=True, help="segments_clean_terms directory containing manifest.csv")
    p.add_argument("--output", required=True, help="output WM checkpoint path")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--num_mini_batches", type=int, default=20)
    p.add_argument("--mini_batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--forecast_horizon", type=int, default=None)
    p.add_argument("--termination_loss_weight", type=float, default=1.0)
    p.add_argument("--termination_pos_weight", type=float, default=0.0)
    p.add_argument("--ensemble_size", type=int, default=5)
    p.add_argument("--config", type=str, default="go2_flat", choices=["go2_flat", "anymal_d_flat"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--reference_wm", type=str, default=None)
    p.add_argument("--warm_start", action="store_true")
    p.add_argument("--seed", type=int, default=None, help="seed python/numpy/torch(+cuda) before model construction; default: unseeded")
    p.add_argument("--norm_stats", type=str, default=None, help=".npz with state_mean/state_std/action_mean/action_std (std incl. eps); default: compute from --data")
    p.add_argument("--lazy_windows", action="store_true", help="gather windows on the fly (identical batches, far less memory)")
    args = p.parse_args()
    t_wall_start = time.time()

    if args.seed is not None:
        seed_everything(args.seed)
        print(f"[fit] seeded python/numpy/torch with {args.seed}")

    device = args.device
    cfg = _CONFIGS[args.config]()
    print(f"[fit] config = {args.config}")

    mac = cfg.model_architecture_config
    architecture_config = mac.architecture_config
    history_horizon = getattr(mac, "history_horizon", 32)
    cfg_forecast = getattr(mac, "forecast_horizon", 8)
    forecast_horizon = args.forecast_horizon if args.forecast_horizon is not None else cfg_forecast
    ext_dim = getattr(mac, "extension_dim", 0)
    contact_dim = getattr(mac, "contact_dim", 8)
    term_dim = getattr(mac, "termination_dim", 1)
    action_dim = 12

    # Infer state_dim from the first segment.
    manifest = pd.read_csv(Path(args.data) / "manifest.csv")
    first_name = manifest.iloc[0]["name"]
    first = pd.read_csv(Path(args.data) / f"{first_name}.csv", header=None, nrows=1)
    ncols = len(first.columns)
    state_dim = ncols - action_dim - ext_dim - contact_dim - term_dim

    ensemble_size = getattr(mac, "ensemble_size", None) or getattr(mac, "num_models", None) or args.ensemble_size

    print(f"[fit] state={state_dim} action={action_dim} ext={ext_dim} contact={contact_dim} term={term_dim}")
    print(f"[fit] ensemble={ensemble_size} hist={history_horizon} forecast={forecast_horizon}")
    print(f"[fit] arch={architecture_config}")

    def build():
        return SystemDynamicsEnsemble(
            state_dim,
            action_dim,
            ext_dim,
            contact_dim,
            term_dim,
            device,
            ensemble_size=ensemble_size,
            history_horizon=history_horizon,
            architecture_config=architecture_config,
            freeze_auxiliary=False,
        ).to(device)

    sd = build()

    if args.reference_wm is not None:
        ref = torch.load(args.reference_wm, map_location=device)["system_dynamics_state_dict"]
        if args.warm_start:
            sd.load_state_dict(ref, strict=True)
            print("[fit] warm-started from reference WM.")
        else:
            tmp = build()
            try:
                tmp.load_state_dict(ref, strict=True)
                print("[fit] architecture VERIFIED by strict reference load. Training from scratch.")
            except Exception as e:
                print(f"[fit] WARNING: reference WM strict load failed: {e}")
            del tmp

    optimizer = torch.optim.Adam(sd.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_weights = {
        "state": 1.0,
        "sequence": 1.0,
        "bound": 1.0,
        "kl": 1.0,
        "extension": 1.0,
        "contact": 1.0,
        "termination": args.termination_loss_weight,
    }

    segs, total_rows, total_terms = read_segments(args.data, state_dim, action_dim, ext_dim, contact_dim, term_dim)
    if args.norm_stats is not None:
        s_mean, s_std, a_mean, a_std = load_normalizer(args.norm_stats, state_dim, action_dim)
        norm_sha = sha256_file(args.norm_stats)
        print(f"[fit] normalizer loaded from {args.norm_stats} (sha256 {norm_sha})")
    else:
        s_mean, s_std, a_mean, a_std = compute_normalizer(segs, state_dim, action_dim)

    if args.no_normalize:
        print("[fit] --no_normalize set: training on raw state/action.")
    else:
        print(f"[fit] z-score stats: s_std[:3]={s_std.flatten()[:3].tolist()} a_std[:3]={a_std.flatten()[:3].tolist()}")

    all_windows = build_segment_windows(
        segs,
        state_dim,
        action_dim,
        contact_dim,
        term_dim,
        history_horizon,
        forecast_horizon,
        device,
        s_mean,
        s_std,
        a_mean,
        a_std,
        normalize=(not args.no_normalize),
        lazy=args.lazy_windows,
    )

    Nw = all_windows.shape[0]
    seq_len = history_horizon + forecast_horizon

    pos_w_val = args.termination_pos_weight if args.termination_pos_weight > 0 else (total_rows - total_terms) / max(total_terms, 1)
    pos_w = torch.tensor([pos_w_val], device=device)
    print(f"[fit] transition-level termination pos_weight={pos_w_val:.1f}")

    import types

    def _termination_loss_posweighted(self, termination_pred, termination_target):
        if termination_pred is None or termination_target is None:
            return torch.tensor(0.0, device=self.device)
        if self.prediction_type == "sequence":
            termination_pred = termination_pred[:, -1]
        return nn.BCEWithLogitsLoss(pos_weight=pos_w)(termination_pred, termination_target)

    sd.compute_termination_loss = types.MethodType(_termination_loss_posweighted, sd)

    # Provenance (plain python types only, so torch.load(weights_only=True) still works).
    data_root = Path(args.data)
    provenance = {
        "script": os.path.abspath(__file__),
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "argv": list(sys.argv),
        "args": dict(vars(args)),
        "git": git_info(os.path.dirname(os.path.abspath(__file__))),
        "rsl_rl_system_dynamics": sys.modules[SystemDynamicsEnsemble.__module__].__file__,
        "rsl_rl_git": git_info(os.path.dirname(sys.modules[SystemDynamicsEnsemble.__module__].__file__)),
        "data_dir": os.path.abspath(args.data),
        "manifest_csv": (data_root / "manifest.csv").read_text(),
        "data_files": [
            {"name": name, "steps": int(steps), "num_envs": int(ne), "rows": int(d.shape[0]),
             "realpath": os.path.realpath(data_root / f"{name}.csv"), "sha256": sha256_file(data_root / f"{name}.csv")}
            for name, steps, d, ne in segs
        ],
        "total_rows": int(total_rows),
        "total_real_terms": int(total_terms),
        "num_windows": int(Nw),
        "seq_len": int(seq_len),
        "termination_pos_weight_used": float(pos_w_val),
        "ensemble_size": int(ensemble_size),
        "normalizer_source": args.norm_stats if args.norm_stats is not None else "computed_from_data",
        "seed": args.seed,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__), "pandas": str(pd.__version__)},
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(torch.device(device)) if str(device).startswith("cuda") else None,
        "host": platform.node(),
        "start_time_unix": t_wall_start,
    }
    extra_ckpt = {}
    if args.norm_stats is not None:
        extra_ckpt = {"norm_stats_path": os.path.abspath(args.norm_stats), "norm_stats_sha256": norm_sha}

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    keys = ["state", "sequence", "bound", "kl", "extension", "contact", "termination"]
    t_start = time.time()

    s0 = 0
    a0 = state_dim
    e0 = a0 + action_dim
    c0 = e0 + ext_dim
    t0 = c0 + contact_dim

    for it in range(args.iterations):
        sums = {k: 0.0 for k in keys}
        nb = 0

        for _ in range(args.num_mini_batches):
            ids = torch.randint(0, Nw, (args.mini_batch_size,), device=device)
            batch = all_windows[ids]

            s_b = batch[:, :, s0:a0]
            a_b = batch[:, :, a0:e0]
            ext_b = None
            c_b = batch[:, :, c0:t0]
            term_b = batch[:, :, t0:t0 + term_dim]

            sd.reset()
            (
                state_loss,
                sequence_loss,
                bound_loss,
                kl_loss,
                extension_loss,
                contact_loss,
                termination_loss,
            ) = sd.compute_loss(s_b, a_b, ext_b, c_b, term_b, bootstrap=True)

            def w(weight, loss):
                return weight * loss if loss is not None else 0.0

            loss = (
                w(loss_weights["state"], state_loss)
                + w(loss_weights["sequence"], sequence_loss)
                + w(loss_weights["bound"], bound_loss)
                + w(loss_weights["kl"], kl_loss)
                + w(loss_weights["extension"], extension_loss)
                + w(loss_weights["contact"], contact_loss)
                + w(loss_weights["termination"], termination_loss)
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(sd.parameters(), args.max_grad_norm)
            optimizer.step()

            def val(x):
                return x.item() if (x is not None and hasattr(x, "item")) else 0.0

            sums["state"] += val(state_loss)
            sums["sequence"] += val(sequence_loss)
            sums["bound"] += val(bound_loss)
            sums["kl"] += val(kl_loss)
            sums["extension"] += val(extension_loss)
            sums["contact"] += val(contact_loss)
            sums["termination"] += val(termination_loss)
            nb += 1

        if (it + 1) % args.log_interval == 0:
            m = {k: sums[k] / max(nb, 1) for k in keys}
            print(
                f"[fit] it {it+1}/{args.iterations} "
                f"state={m['state']:.4f} seq={m['sequence']:.4f} "
                f"contact={m['contact']:.4f} term={m['termination']:.4f} "
                f"kl={m['kl']:.4f} ({(time.time()-t_start)/(it+1):.2f}s/it)"
            )

        if (it + 1) % args.save_interval == 0 or (it + 1) == args.iterations:
            torch.save(
                {
                    "system_dynamics_state_dict": sd.state_dict(),
                    "iter": it + 1,
                    "state_mean": s_mean.cpu(),
                    "state_std": s_std.cpu(),
                    "action_mean": a_mean.cpu(),
                    "action_std": a_std.cpu(),
                    "normalized": (not args.no_normalize),
                    "segment_aware": True,
                    "total_rows": total_rows,
                    "total_real_terms": total_terms,
                    "num_windows": Nw,
                    **extra_ckpt,
                    "provenance": {**provenance, "saved_iter": it + 1, "save_time_unix": time.time()},
                },
                args.output,
            )

    if str(device).startswith("cuda"):
        print(f"[fit] cuda peak_allocated_mb={torch.cuda.max_memory_allocated(torch.device(device)) / 2**20:.0f} "
              f"peak_reserved_mb={torch.cuda.max_memory_reserved(torch.device(device)) / 2**20:.0f}")
    print(f"[fit] DONE -> {args.output}  rows={total_rows} real_terms={total_terms} windows={Nw}")


if __name__ == "__main__":
    main()
