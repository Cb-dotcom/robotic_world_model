# SPDX-License-Identifier: BSD-3-Clause
"""Standalone OFFLINE world-model fit for Go2 RWM-U.

Trains a SystemDynamicsEnsemble on a curated offline CSV
(state(45)|action(12)|contact(8)|termination(1) = 66 cols), lifting the exact
loss loop from MBPOPPO.update_system_dynamics so the resulting world model loads
straight through the offline pipeline's --wm-checkpoint.

No Isaac / sim needed -- pure PyTorch. Identity normalizer (raw values inserted),
matching the offline pipeline's convention.

Run (container):
  unset PYTHONPATH; export PYTHONPATH=<rsl_rl_rwm>
  cd <robotic_world_model>
  /isaac-sim/python.sh scripts/reinforcement_learning/model_based/fit_world_model.py \
    --data assets/data/go2_noise/state_action_data_noise.csv \
    --output logs/wm_fit/go2_curated_wm.pt \
    --iterations 2000 --termination_loss_weight 5.0 \
    --reference_wm logs/rsl_rl/unitree_go2_flat/2026-06-12_13-39-03_pretrain_ens5/model_2000.pt
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rsl_rl.modules import SystemDynamicsEnsemble
from rsl_rl.storage.replay_buffer import ReplayBuffer

# Pull architecture straight from the offline config so the fitted WM is
# shape-identical to what --wm-checkpoint loads. (Config import is lightweight:
# dataclasses only, no Isaac.)
from configs.go2_flat_cfg import Go2FlatConfig
from configs.anymal_d_flat_cfg import AnymalDFlatConfig

_CONFIGS = {"go2_flat": Go2FlatConfig, "anymal_d_flat": AnymalDFlatConfig}


def main():
    p = argparse.ArgumentParser(description="Offline world-model fit for Go2 RWM-U.")
    p.add_argument("--data", required=True, help="curated CSV (66 cols)")
    p.add_argument("--output", required=True, help="output WM checkpoint path")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--num_mini_batches", type=int, default=20)
    p.add_argument("--mini_batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--forecast_horizon", type=int, default=None,
                   help="override; default = config forecast_horizon (8)")
    p.add_argument("--termination_loss_weight", type=float, default=1.0,
                   help="scales termination loss vs contact/state heads (head-vs-head, NOT pos/neg balance)")
    p.add_argument("--termination_pos_weight", type=float, default=0.0,
                   help="pos_weight INSIDE termination BCE to fix the ~650:1 class imbalance. "
                        "0 = auto (neg/pos from data). This is the lever that stops the head going blind.")
    p.add_argument("--ensemble_size", type=int, default=5,
                   help="fallback if config attr name differs; config value wins if present")
    p.add_argument("--config", type=str, default="go2_flat", choices=["go2_flat", "anymal_d_flat"],
                   help="which robot config to pull dims/arch from (go2_flat or anymal_d_flat)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no_normalize", action="store_true",
                   help="train on RAW state/action (debug only; mismatches the normalized deployment)")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--reference_wm", type=str, default=None,
                   help="ens5 checkpoint: verify architecture match (and warm-start if --warm_start)")
    p.add_argument("--warm_start", action="store_true",
                   help="initialize from --reference_wm instead of from scratch")
    args = p.parse_args()

    device = args.device
    cfg = _CONFIGS[args.config]()
    print(f"[fit] config = {args.config}")
    mac = cfg.model_architecture_config
    # architecture_config dict is essential and cannot be safely hardcoded -> pull from config.
    architecture_config = mac.architecture_config
    # dims: pull from config, fall back to known values if attr name differs.
    history_horizon = getattr(mac, "history_horizon", 32)
    cfg_forecast = getattr(mac, "forecast_horizon", 8)
    forecast_horizon = args.forecast_horizon if args.forecast_horizon is not None else cfg_forecast
    ext_dim = getattr(mac, "extension_dim", 0)
    contact_dim = getattr(mac, "contact_dim", 8)
    term_dim = getattr(mac, "termination_dim", 1)
    # state_dim/action_dim: Go2 and ANYmal are both 45/12; infer state_dim from the CSV
    # width (cols - action - ext - contact - term) so this stays correct if a config differs.
    action_dim = 12
    _ncols = len(pd.read_csv(args.data, header=None, nrows=1).columns)
    state_dim = _ncols - action_dim - ext_dim - contact_dim - term_dim
    print(f"[fit] inferred state_dim={state_dim} from {_ncols}-col CSV (action={action_dim} ext={ext_dim} contact={contact_dim} term={term_dim})")
    ensemble_size = getattr(mac, "ensemble_size", None) or getattr(mac, "num_models", None) or args.ensemble_size

    print(f"[fit] state={state_dim} action={action_dim} ext={ext_dim} contact={contact_dim} term={term_dim}")
    print(f"[fit] ensemble={ensemble_size} hist={history_horizon} forecast={forecast_horizon}")
    print(f"[fit] arch={architecture_config}")

    def build():
        return SystemDynamicsEnsemble(
            state_dim, action_dim, ext_dim, contact_dim, term_dim, device,
            ensemble_size=ensemble_size, history_horizon=history_horizon,
            architecture_config=architecture_config, freeze_auxiliary=False,
        ).to(device)

    sd = build()

    # architecture self-check / optional warm start
    if args.reference_wm is not None:
        ref = torch.load(args.reference_wm, map_location=device)["system_dynamics_state_dict"]
        if args.warm_start:
            sd.load_state_dict(ref, strict=True)
            print("[fit] warm-started from reference WM (architecture verified by strict load).")
        else:
            tmp = build()
            try:
                tmp.load_state_dict(ref, strict=True)
                print("[fit] architecture VERIFIED: reference WM loads strict=True -> "
                      "fitted WM will load through --wm-checkpoint. Training from scratch.")
            except Exception as e:
                print(f"[fit] WARNING: reference WM did not load strict (arch mismatch?): {e}")
            del tmp

    optimizer = torch.optim.Adam(sd.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_weights = {"state": 1.0, "sequence": 1.0, "bound": 1.0, "kl": 1.0,
                    "extension": 1.0, "contact": 1.0, "termination": args.termination_loss_weight}

    # --- load CSV into the replay buffer ---
    data = pd.read_csv(args.data, header=None).values.astype(np.float32)
    N = data.shape[0]
    expected_cols = state_dim + action_dim + ext_dim + contact_dim + term_dim
    assert data.shape[1] == expected_cols, f"expected {expected_cols} cols, got {data.shape[1]}"
    s0, a0 = 0, state_dim
    e0 = a0 + action_dim
    c0 = e0 + ext_dim
    t0 = c0 + contact_dim
    state = torch.from_numpy(data[:, s0:a0]).to(device).view(1, N, state_dim)
    action = torch.from_numpy(data[:, a0:e0]).to(device).view(1, N, action_dim)
    contact = torch.from_numpy(data[:, c0:t0]).to(device).view(1, N, contact_dim)
    termination = torch.from_numpy(data[:, t0:t0 + term_dim]).to(device).view(1, N, term_dim)
    n_falls = int(termination.sum().item())
    print(f"[fit] loaded {N} transitions, {n_falls} terminations ({100.0*n_falls/N:.3f}% positive)")

    # --- CRITICAL: z-score state/action to match the DEPLOYMENT normalizer ---
    # The offline pipeline's Dataset (train.py ~L131/162) computes mean/std from the
    # same CSV and feeds the WM normalized state/action at policy time; contact and
    # termination stay raw (the pipeline only normalizes state/action). Training raw
    # while deploying normalized makes every head misfire -> episode length pins at 1.
    s_mean = state.mean(dim=(0, 1), keepdim=True)
    s_std = state.std(dim=(0, 1), keepdim=True) + 1e-6
    a_mean = action.mean(dim=(0, 1), keepdim=True)
    a_std = action.std(dim=(0, 1), keepdim=True) + 1e-6
    if not args.no_normalize:
        state = (state - s_mean) / s_std
        action = (action - a_mean) / a_std
        print(f"[fit] z-scored state/action to match deployment "
              f"(s_std[:3]={s_std.flatten()[:3].tolist()}, a_std[:3]={a_std.flatten()[:3].tolist()})")
    else:
        print("[fit] --no_normalize set: training on RAW state/action (will mismatch normalized deployment)")

    buf = ReplayBuffer([state_dim, action_dim, ext_dim, contact_dim, term_dim], N, device)
    buf.insert([state, action, None, contact, termination])  # ext slot None (dim 0)
    print(f"[fit] buffer num_transitions={buf.num_transitions}")

    # --- fix termination-head imbalance: inject pos_weight into the termination BCE ---
    # ETH's compute_termination_loss is naive BCEWithLogitsLoss; at ~0.15% positives it
    # collapses to all-negative. Override on this instance only (online code untouched).
    import types
    pos_w_val = args.termination_pos_weight if args.termination_pos_weight > 0 else (N - n_falls) / max(n_falls, 1)
    pos_w = torch.tensor([pos_w_val], device=device)
    print(f"[fit] termination pos_weight = {pos_w_val:.1f} (the anti-blindness lever)")

    def _termination_loss_posweighted(self, termination_pred, termination_target):
        if termination_pred is None or termination_target is None:
            return torch.tensor(0.0, device=self.device)
        if self.prediction_type == "sequence":
            termination_pred = termination_pred[:, -1]
        return nn.BCEWithLogitsLoss(pos_weight=pos_w)(termination_pred, termination_target)

    sd.compute_termination_loss = types.MethodType(_termination_loss_posweighted, sd)

    seq_len = history_horizon + forecast_horizon
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    keys = ["state", "sequence", "bound", "kl", "extension", "contact", "termination"]
    t0 = time.time()
    for it in range(args.iterations):
        gen = buf.mini_batch_generator(seq_len, args.num_mini_batches, args.mini_batch_size)
        sums = {k: 0.0 for k in keys}
        nb = 0
        for s_b, a_b, ext_b, c_b, term_b in gen:
            sd.reset()
            (state_loss, sequence_loss, bound_loss, kl_loss,
             extension_loss, contact_loss, termination_loss) = sd.compute_loss(
                s_b, a_b, ext_b, c_b, term_b, bootstrap=True)

            def w(weight, loss):
                return weight * loss if loss is not None else 0.0

            loss = (w(loss_weights["state"], state_loss)
                    + w(loss_weights["sequence"], sequence_loss)
                    + w(loss_weights["bound"], bound_loss)
                    + w(loss_weights["kl"], kl_loss)
                    + w(loss_weights["extension"], extension_loss)
                    + w(loss_weights["contact"], contact_loss)
                    + w(loss_weights["termination"], termination_loss))
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
            print(f"[fit] it {it+1}/{args.iterations}  state={m['state']:.4f} seq={m['sequence']:.4f} "
                  f"contact={m['contact']:.4f} term={m['termination']:.4f} kl={m['kl']:.4f} "
                  f"({(time.time()-t0)/(it+1):.2f}s/it)")
        if (it + 1) % args.save_interval == 0 or (it + 1) == args.iterations:
            torch.save({"system_dynamics_state_dict": sd.state_dict(), "iter": it + 1,
                        "state_mean": s_mean.cpu(), "state_std": s_std.cpu(),
                        "action_mean": a_mean.cpu(), "action_std": a_std.cpu(),
                        "normalized": (not args.no_normalize)}, args.output)

    print(f"[fit] DONE -> {args.output}  (terminations in data: {n_falls})")


if __name__ == "__main__":
    main()