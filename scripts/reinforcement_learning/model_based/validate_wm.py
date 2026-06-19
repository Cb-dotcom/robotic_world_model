# SPDX-License-Identifier: BSD-3-Clause
"""Step 3: validate the fitted Go2 world model's TERMINATION head.

Does the head fire on states leading to real falls and stay quiet on competent
walking? Builds positive windows (32-step history ending right before each real
fall; segment seams excluded) and negative windows (competent walking from the
clean n00 segment), runs the WM forward (ensemble-mean termination logit for the
next step), and reports mean P(terminate), separation, and ROC-AUC.

  blind head      -> AUC ~0.5, falls and walking indistinguishable
  working head    -> AUC >> 0.5, P(term) high on falls, low on walking

Window alignment matches compute_auxiliary_loss(i=0):
  x_state  = state [f-H : f]      (H steps)
  x_action = action[f-H+1 : f+1]  (H steps, offset +1)
  target   = termination at f

Run:
  unset PYTHONPATH; export PYTHONPATH=<rsl_rl_rwm>
  cd <robotic_world_model>
  /isaac-sim/python.sh scripts/reinforcement_learning/model_based/validate_wm.py \
    --wm logs/wm_fit/go2_curated_wm.pt \
    --data assets/data/go2_noise/state_action_data_noise.csv
"""
import argparse
import numpy as np
import pandas as pd
import torch

from rsl_rl.modules import SystemDynamicsEnsemble
from configs.go2_flat_cfg import Go2FlatConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True, help="fitted WM checkpoint")
    p.add_argument("--data", required=True, help="curated CSV used to fit (for fall/walk windows)")
    p.add_argument("--clean_rows", type=int, default=10000,
                   help="rows of the leading clean n00 segment to draw negatives from")
    p.add_argument("--num_neg", type=int, default=2000)
    p.add_argument("--fall_range", default=None, help="LO,HI rows to restrict positive falls (held-out test)")
    p.add_argument("--neg_range", default=None, help="LO,HI rows to draw negative walking windows from")
    p.add_argument("--seam_lens", type=str, default="10000,20000,35000,65000,90000,115000",
                   help="cumulative segment row counts; their last rows are seam terminations, not real falls")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = args.device

    cfg = Go2FlatConfig()
    mac = cfg.model_architecture_config
    H = getattr(mac, "history_horizon", 32)
    sd = SystemDynamicsEnsemble(
        45, 12, getattr(mac, "extension_dim", 0), getattr(mac, "contact_dim", 8),
        getattr(mac, "termination_dim", 1), device,
        ensemble_size=getattr(mac, "ensemble_size", None) or getattr(mac, "num_models", None) or 5,
        history_horizon=H, architecture_config=mac.architecture_config, freeze_auxiliary=False,
    ).to(device)
    sd.load_state_dict(torch.load(args.wm, map_location=device)["system_dynamics_state_dict"], strict=True)
    sd.eval()

    data = pd.read_csv(args.data, header=None).values.astype(np.float32)
    state_all = np.ascontiguousarray(data[:, 0:45])
    action_all = np.ascontiguousarray(data[:, 45:57])
    term_all = np.ascontiguousarray(data[:, 65])

    seams = set(int(c) - 1 for c in args.seam_lens.split(","))
    fall_idx = [int(f) for f in np.where(term_all > 0.5)[0] if int(f) not in seams and f >= H]
    if args.fall_range:
        lo, hi = map(int, args.fall_range.split(","))
        fall_idx = [f for f in fall_idx if lo <= f < hi]

    def window(f):
        xs = torch.from_numpy(state_all[f - H:f]).to(device).unsqueeze(0)
        xa = torch.from_numpy(action_all[f - H + 1:f + 1]).to(device).unsqueeze(0)
        return xs, xa

    @torch.no_grad()
    def term_prob(f):
        sd.reset()
        xs, xa = window(f)
        out = sd.forward(xs, xa)
        return torch.sigmoid(out[5]).item()  # out[5] = output_terminations (logit)

    pos = np.array([term_prob(f) for f in fall_idx])

    rng = np.random.default_rng(0)
    if args.neg_range:
        nlo, nhi = map(int, args.neg_range.split(",")); nlo = max(nlo, H)
    else:
        nlo, nhi = H, args.clean_rows
    neg_cand = [g for g in range(nlo, nhi) if term_all[g - H:g + 1].sum() == 0]
    neg_sample = rng.choice(neg_cand, size=min(args.num_neg, len(neg_cand)), replace=False)
    neg = np.array([term_prob(int(g)) for g in neg_sample])

    print(f"[val] real falls used: {len(pos)}   negative walking windows: {len(neg)}")
    print(f"[val] P(term) on FALLS:   mean={pos.mean():.3f} median={np.median(pos):.3f} min={pos.min():.3f}")
    print(f"[val] P(term) on WALKING: mean={neg.mean():.3f} median={np.median(neg):.3f} max={neg.max():.3f}")

    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort().astype(np.float64) + 1.0
    n_pos, n_neg = len(pos), len(neg)
    auc = (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    print(f"[val] ROC-AUC (falls vs walking) = {auc:.3f}   (0.5=blind, >0.8=fires correctly)")

    for thr in [0.3, 0.5, 0.7]:
        print(f"[val] thr={thr}: fall-recall={(pos > thr).mean():.3f}  walking-false-alarm={(neg > thr).mean():.3f}")


if __name__ == "__main__":
    main()