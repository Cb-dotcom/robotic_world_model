# SPDX-License-Identifier: BSD-3-Clause
"""Boundary-aware offline world-model fit for the Go2 1M pilot dataset.

This trainer uses:
  - clean real termination labels from assets/data/go2_pilot_1m/segments_clean_terms
  - manifest.csv to respect env-major block boundaries
  - no artificial seam labels as termination targets
  - no sequence windows crossing env boundaries or real terminations

It avoids the single-column ambiguity where termination was both a fall label and
a sequence-boundary mask.
"""

import argparse
import os
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

    segs = []
    total_rows = 0
    total_terms = 0

    for row in manifest.itertuples(index=False):
        f = root / f"{row.name}.csv"
        steps = int(row.steps)
        rows_expected = int(row.rows)

        data = pd.read_csv(f, header=None).values.astype(np.float32)
        if data.shape[1] != expected_cols:
            raise RuntimeError(f"{f}: expected {expected_cols} cols, got {data.shape[1]}")
        if data.shape[0] != rows_expected:
            raise RuntimeError(f"{f}: expected {rows_expected} rows, got {data.shape[0]}")
        if data.shape[0] != steps * 64:
            raise RuntimeError(f"{f}: rows {data.shape[0]} != steps*64 {steps*64}")

        terms = int((data[:, -1] > 0.5).sum())
        print(f"[load] {f.name}: rows={data.shape[0]} steps={steps} terms={terms}")

        segs.append((row.name, steps, data))
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

    for _, _, data in segs:
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
):
    seq_len = history_horizon + forecast_horizon
    windows = []
    valid_total = 0
    skipped_total = 0

    s_mean = s_mean.to(device)
    s_std = s_std.to(device)
    a_mean = a_mean.to(device)
    a_std = a_std.to(device)

    for name, steps, data_np in segs:
        x = torch.from_numpy(data_np).to(device)
        x = x.view(64, steps, -1).contiguous()

        if normalize:
            x[:, :, 0:state_dim] = (x[:, :, 0:state_dim] - s_mean) / s_std
            a0 = state_dim
            a1 = state_dim + action_dim
            x[:, :, a0:a1] = (x[:, :, a0:a1] - a_mean) / a_std

        # Window view: [64, W, seq_len, cols]
        W = steps - seq_len + 1
        if W <= 0:
            raise RuntimeError(f"{name}: steps {steps} <= seq_len {seq_len}")

        term = x[:, :, -1]
        # A window starting at i is invalid if a real termination appears before
        # the final target row, i.e. in [i, i+seq_len-2].
        reset_roll = term.unfold(1, seq_len - 1, 1)[:, :W].sum(dim=-1)
        valid = reset_roll <= 0.5

        win = x.unfold(1, seq_len, 1).permute(0, 1, 3, 2)
        valid_win = win[valid].contiguous()

        skipped = int((~valid).sum().item())
        kept = int(valid_win.shape[0])
        print(f"[windows] {name}: kept={kept} skipped_cross_reset={skipped}")

        windows.append(valid_win)
        valid_total += kept
        skipped_total += skipped

    all_windows = torch.cat(windows, dim=0).contiguous()
    print(f"[windows] total={valid_total} skipped={skipped_total} tensor_shape={tuple(all_windows.shape)}")
    return all_windows


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
    args = p.parse_args()

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
                },
                args.output,
            )

    print(f"[fit] DONE -> {args.output}  rows={total_rows} real_terms={total_terms} windows={Nw}")


if __name__ == "__main__":
    main()
