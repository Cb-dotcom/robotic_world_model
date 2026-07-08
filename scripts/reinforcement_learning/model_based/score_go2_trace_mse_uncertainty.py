import argparse
import numpy as np
import pandas as pd
import torch

from rsl_rl.modules import SystemDynamicsEnsemble
from configs.go2_flat_cfg import Go2FlatConfig


def stats(name, x):
    x = np.asarray(x, dtype=np.float64)
    print(
        f"{name:18s} n={len(x):5d} "
        f"mean={x.mean():.6f} median={np.median(x):.6f} "
        f"p90={np.quantile(x, 0.90):.6f} p99={np.quantile(x, 0.99):.6f} "
        f"min={x.min():.6f} max={x.max():.6f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True)
    p.add_argument("--trace", required=True)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--steps_per_env", type=int, default=1000)
    p.add_argument("--num_neg", type=int, default=5000)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = args.device
    cfg = Go2FlatConfig()
    mac = cfg.model_architecture_config
    H = getattr(mac, "history_horizon", 32)

    ckpt = torch.load(args.wm, map_location=device)
    sd = SystemDynamicsEnsemble(
        45, 12,
        getattr(mac, "extension_dim", 0),
        getattr(mac, "contact_dim", 8),
        getattr(mac, "termination_dim", 1),
        device,
        ensemble_size=getattr(mac, "ensemble_size", None) or getattr(mac, "num_models", None) or 5,
        history_horizon=H,
        architecture_config=mac.architecture_config,
        freeze_auxiliary=False,
    ).to(device)

    sd.load_state_dict(ckpt["system_dynamics_state_dict"], strict=True)
    sd.eval()

    data = pd.read_csv(args.trace, header=None).values.astype(np.float32)
    assert data.shape[1] == 66, data.shape
    assert data.shape[0] == args.num_envs * args.steps_per_env, data.shape

    state_all = np.ascontiguousarray(data[:, 0:45])
    action_all = np.ascontiguousarray(data[:, 45:57])
    term_all = np.ascontiguousarray(data[:, 65] > 0.5)

    if ckpt.get("normalized", False) and "state_mean" in ckpt:
        sm = ckpt["state_mean"].detach().cpu().numpy().reshape(-1)
        ss = ckpt["state_std"].detach().cpu().numpy().reshape(-1)
        am = ckpt["action_mean"].detach().cpu().numpy().reshape(-1)
        astd = ckpt["action_std"].detach().cpu().numpy().reshape(-1)
        state_all = ((state_all - sm) / ss).astype(np.float32)
        action_all = ((action_all - am) / astd).astype(np.float32)
        print("[mse] normalized inputs/targets to WM training space")
    else:
        print("[mse] WM has no saved normalizer -> using raw inputs")

    fall_idx, pre1_idx, pre5_idx, neg_cand = [], [], [], []

    for e in range(args.num_envs):
        base = e * args.steps_per_env
        seam = base + args.steps_per_env - 1
        for local_t in range(H, args.steps_per_env):
            f = base + local_t
            is_seam = f == seam
            hist_clean = term_all[f - H:f].sum() == 0

            if term_all[f] and (not is_seam) and hist_clean:
                fall_idx.append(f)
                if local_t - 1 >= H and term_all[f - H - 1:f].sum() == 0:
                    pre1_idx.append(f - 1)
                if local_t - 5 >= H and term_all[f - H - 5:f - 4].sum() == 0:
                    pre5_idx.append(f - 5)

            if (not term_all[f]) and hist_clean:
                neg_cand.append(f)

    rng = np.random.default_rng(0)
    neg_idx = rng.choice(neg_cand, size=min(args.num_neg, len(neg_cand)), replace=False)

    print(f"[mse] wm={args.wm}")
    print(f"[mse] trace={args.trace}")
    print(f"[mse] fall={len(fall_idx)} pre1={len(pre1_idx)} pre5={len(pre5_idx)} neg={len(neg_idx)}")

    def window(f):
        xs = torch.from_numpy(state_all[f - H:f]).to(device).unsqueeze(0)
        xa = torch.from_numpy(action_all[f - H + 1:f + 1]).to(device).unsqueeze(0)
        target = torch.from_numpy(state_all[f]).to(device)
        return xs, xa, target

    @torch.no_grad()
    def score_one(f):
        sd.reset()
        xs, xa, target = window(int(f))
        state_pred, alea, epi, ext, contact, term = sd.forward(xs, xa)

        pred = state_pred.detach()
        pred = pred.reshape(-1, pred.shape[-1])[-1]
        if pred.numel() != 45:
            raise RuntimeError(f"unexpected state_pred shape {tuple(state_pred.shape)} -> pred {tuple(pred.shape)}")

        mse = torch.mean((pred - target) ** 2).item()
        rmse = float(np.sqrt(mse))
        epi_v = float(epi.detach().cpu().reshape(-1)[0].item())
        return mse, rmse, epi_v

    def score_many(idx):
        return np.array([score_one(f) for f in idx], dtype=np.float64)

    fall = score_many(fall_idx)
    pre1 = score_many(pre1_idx)
    pre5 = score_many(pre5_idx)
    neg = score_many(neg_idx)

    print("\\n=== normalized next-state MSE ===")
    stats("fall_mse", fall[:, 0])
    stats("pre1_mse", pre1[:, 0])
    stats("pre5_mse", pre5[:, 0])
    stats("walking_mse", neg[:, 0])

    print("\\n=== normalized next-state RMSE ===")
    stats("fall_rmse", fall[:, 1])
    stats("pre1_rmse", pre1[:, 1])
    stats("pre5_rmse", pre5[:, 1])
    stats("walking_rmse", neg[:, 1])

    print("\\n=== epistemic paired with MSE ===")
    stats("fall_epi", fall[:, 2])
    stats("pre1_epi", pre1[:, 2])
    stats("pre5_epi", pre5[:, 2])
    stats("walking_epi", neg[:, 2])

    print("\\n=== simple calibration flags ===")
    print(f"fall_mse_over_walk = {fall[:,0].mean() / max(neg[:,0].mean(), 1e-12):.3f}")
    print(f"pre1_mse_over_walk = {pre1[:,0].mean() / max(neg[:,0].mean(), 1e-12):.3f}")
    print(f"pre5_mse_over_walk = {pre5[:,0].mean() / max(neg[:,0].mean(), 1e-12):.3f}")


if __name__ == "__main__":
    main()
