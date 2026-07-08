import argparse
import numpy as np
import pandas as pd
import torch

from rsl_rl.modules import SystemDynamicsEnsemble
from configs.go2_flat_cfg import Go2FlatConfig


def stats(name, x):
    x = np.asarray(x, dtype=np.float64)
    print(
        f"{name:16s} n={len(x):5d} "
        f"mean={x.mean():.6f} median={np.median(x):.6f} "
        f"p90={np.quantile(x, 0.90):.6f} p99={np.quantile(x, 0.99):.6f} "
        f"min={x.min():.6f} max={x.max():.6f}"
    )


def auc(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = len(pos), len(neg)
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--steps_per_env", type=int, default=1500)
    p.add_argument("--num_neg", type=int, default=5000)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = args.device
    cfg = Go2FlatConfig()
    mac = cfg.model_architecture_config
    H = getattr(mac, "history_horizon", 32)

    ckpt = torch.load(args.wm, map_location=device)
    sd_state = ckpt["system_dynamics_state_dict"]

    sd = SystemDynamicsEnsemble(
        45,
        12,
        getattr(mac, "extension_dim", 0),
        getattr(mac, "contact_dim", 8),
        getattr(mac, "termination_dim", 1),
        device,
        ensemble_size=getattr(mac, "ensemble_size", None) or getattr(mac, "num_models", None) or 5,
        history_horizon=H,
        architecture_config=mac.architecture_config,
        freeze_auxiliary=False,
    ).to(device)

    sd.load_state_dict(sd_state, strict=True)
    sd.eval()

    data = pd.read_csv(args.data, header=None).values.astype(np.float32)
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
        print("[diag] normalized inputs to WM training space")
    else:
        print("[diag] WM has no saved normalizer -> using raw inputs")

    print(f"[diag] wm={args.wm}")
    print(f"[diag] data={args.data}")
    print(f"[diag] rows={len(data)} num_envs={args.num_envs} steps_per_env={args.steps_per_env}")
    print(f"[diag] H={H} raw_terms={int(term_all.sum())}")

    fall_idx = []
    neg_cand = []

    for e in range(args.num_envs):
        base = e * args.steps_per_env
        for local_t in range(H, args.steps_per_env):
            f = base + local_t
            hist_terms = term_all[f - H:f].sum()

            # Positive: current row is a real fall, but history before it is clean.
            if term_all[f] and hist_terms == 0:
                fall_idx.append(f)

            # Negative: history and current row are all non-terminal.
            if (not term_all[f]) and hist_terms == 0:
                neg_cand.append(f)

    rng = np.random.default_rng(0)
    neg_idx = rng.choice(neg_cand, size=min(args.num_neg, len(neg_cand)), replace=False)

    print(f"[diag] valid fall windows={len(fall_idx)}")
    print(f"[diag] negative candidates={len(neg_cand)} sampled={len(neg_idx)}")

    if len(fall_idx) == 0 or len(neg_idx) == 0:
        print("[diag] not enough positives or negatives")
        return

    def window(f):
        xs = torch.from_numpy(state_all[f - H:f]).to(device).unsqueeze(0)
        xa = torch.from_numpy(action_all[f - H + 1:f + 1]).to(device).unsqueeze(0)
        return xs, xa

    @torch.no_grad()
    def score(f):
        sd.reset()
        xs, xa = window(int(f))
        state_pred, alea, epi, ext, contact, term = sd.forward(xs, xa)
        epi_v = float(epi.detach().cpu().reshape(-1)[0].item())
        alea_v = float(alea.detach().cpu().reshape(-1)[0].item())
        term_p = float(torch.sigmoid(term).detach().cpu().reshape(-1)[0].item()) if term is not None else float("nan")
        return epi_v, alea_v, term_p

    pos = np.array([score(f) for f in fall_idx], dtype=np.float64)
    neg = np.array([score(g) for g in neg_idx], dtype=np.float64)

    pos_epi, pos_alea, pos_term = pos[:, 0], pos[:, 1], pos[:, 2]
    neg_epi, neg_alea, neg_term = neg[:, 0], neg[:, 1], neg[:, 2]

    print("\n=== epistemic uncertainty: exact MOPO penalty scalar ===")
    stats("falls", pos_epi)
    stats("walking", neg_epi)
    print(f"ratio mean falls/walking = {pos_epi.mean() / max(neg_epi.mean(), 1e-12):.3f}")
    print(f"ROC-AUC U(falls > walking) = {auc(pos_epi, neg_epi):.3f}")

    print("\n=== aleatoric uncertainty ===")
    stats("falls", pos_alea)
    stats("walking", neg_alea)
    print(f"ratio mean falls/walking = {pos_alea.mean() / max(neg_alea.mean(), 1e-12):.3f}")
    print(f"ROC-AUC alea(falls > walking) = {auc(pos_alea, neg_alea):.3f}")

    print("\n=== termination probability ===")
    stats("falls", pos_term)
    stats("walking", neg_term)
    print(f"ROC-AUC term(falls > walking) = {auc(pos_term, neg_term):.3f}")

    print("\n=== decision hint ===")
    epi_ratio = pos_epi.mean() / max(neg_epi.mean(), 1e-12)
    epi_auc = auc(pos_epi, neg_epi)
    if epi_ratio >= 3.0 and epi_auc >= 0.75:
        print("CHEAP HOLDOUT PASS: epistemic uncertainty separates held-out noisy falls.")
        print("Still run exploit-rollout diagnostic before policy training.")
    elif epi_ratio <= 1.5 or epi_auc < 0.65:
        print("CHEAP HOLDOUT FAIL/FLAT: epistemic uncertainty does not separate held-out falls.")
        print("Do not trust the penalty yet; debug calibration or data.")
    else:
        print("CHEAP HOLDOUT MIXED: partial separation.")
        print("Exploit-rollout diagnostic is mandatory before policy training.")


if __name__ == "__main__":
    main()
