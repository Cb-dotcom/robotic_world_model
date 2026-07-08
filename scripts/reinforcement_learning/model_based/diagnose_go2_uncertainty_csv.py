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
    p.add_argument("--clean_rows", type=int, default=10000)
    p.add_argument("--num_neg", type=int, default=2000)
    p.add_argument("--fall_range", default=None)
    p.add_argument("--neg_range", default=None)
    p.add_argument("--seam_lens", default="10000,20000,35000,65000,90000,115000")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = args.device
    cfg = Go2FlatConfig()
    mac = cfg.model_architecture_config
    H = getattr(mac, "history_horizon", 32)

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

    ckpt = torch.load(args.wm, map_location=device)
    sd.load_state_dict(ckpt["system_dynamics_state_dict"], strict=True)
    sd.eval()

    data = pd.read_csv(args.data, header=None).values.astype(np.float32)
    state_all = np.ascontiguousarray(data[:, 0:45])
    action_all = np.ascontiguousarray(data[:, 45:57])
    term_all = np.ascontiguousarray(data[:, 65])

    if ckpt.get("normalized", False) and "state_mean" in ckpt:
        sm = ckpt["state_mean"].detach().cpu().numpy().reshape(-1)
        ss = ckpt["state_std"].detach().cpu().numpy().reshape(-1)
        am = ckpt["action_mean"].detach().cpu().numpy().reshape(-1)
        astd = ckpt["action_std"].detach().cpu().numpy().reshape(-1)
        state_all = ((state_all - sm) / ss).astype(np.float32)
        action_all = ((action_all - am) / astd).astype(np.float32)
        print(f"[unc] normalized inputs to WM training space")
    else:
        print("[unc] WM has no saved normalizer -> using raw inputs")

    seams = set(int(c) - 1 for c in args.seam_lens.split(","))
    fall_idx = [int(f) for f in np.where(term_all > 0.5)[0] if int(f) not in seams and int(f) >= H]

    if args.fall_range:
        lo, hi = map(int, args.fall_range.split(","))
        fall_idx = [f for f in fall_idx if lo <= f < hi]

    rng = np.random.default_rng(0)

    if args.neg_range:
        nlo, nhi = map(int, args.neg_range.split(","))
        nlo = max(nlo, H)
    else:
        nlo, nhi = H, args.clean_rows

    neg_cand = [g for g in range(nlo, nhi) if term_all[g - H:g + 1].sum() == 0]
    neg_idx = rng.choice(neg_cand, size=min(args.num_neg, len(neg_cand)), replace=False)

    def window(f):
        xs = torch.from_numpy(state_all[f - H:f]).to(device).unsqueeze(0)
        xa = torch.from_numpy(action_all[f - H + 1:f + 1]).to(device).unsqueeze(0)
        return xs, xa

    @torch.no_grad()
    def score(f):
        sd.reset()
        xs, xa = window(int(f))
        state_pred, alea, epi, ext, contact, term = sd.forward(xs, xa)
        epi_v = float(epi.detach().cpu().item())
        alea_v = float(alea.detach().cpu().item())
        term_p = float(torch.sigmoid(term).detach().cpu().item()) if term is not None else float("nan")
        return epi_v, alea_v, term_p

    pos = np.array([score(f) for f in fall_idx], dtype=np.float64)
    neg = np.array([score(g) for g in neg_idx], dtype=np.float64)

    print(f"[unc] fall windows: {len(pos)}   walking windows: {len(neg)}")
    if len(pos) == 0 or len(neg) == 0:
        print("[unc] not enough positives or negatives")
        return

    pos_epi, pos_alea, pos_term = pos[:, 0], pos[:, 1], pos[:, 2]
    neg_epi, neg_alea, neg_term = neg[:, 0], neg[:, 1], neg[:, 2]

    print("\\n=== epistemic uncertainty: exact MOPO penalty scalar ===")
    stats("falls", pos_epi)
    stats("walking", neg_epi)
    print(f"ratio mean falls/walking = {pos_epi.mean() / max(neg_epi.mean(), 1e-12):.3f}")
    print(f"ROC-AUC U(falls > walking) = {auc(pos_epi, neg_epi):.3f}")

    print("\\n=== aleatoric uncertainty ===")
    stats("falls", pos_alea)
    stats("walking", neg_alea)
    print(f"ratio mean falls/walking = {pos_alea.mean() / max(neg_alea.mean(), 1e-12):.3f}")
    print(f"ROC-AUC alea(falls > walking) = {auc(pos_alea, neg_alea):.3f}")

    print("\\n=== termination probability ===")
    stats("falls", pos_term)
    stats("walking", neg_term)
    print(f"ROC-AUC term(falls > walking) = {auc(pos_term, neg_term):.3f}")

    print("\\n=== decision hint ===")
    r = pos_epi.mean() / max(neg_epi.mean(), 1e-12)
    if r >= 3.0 and auc(pos_epi, neg_epi) >= 0.75:
        print("GOOD: epistemic uncertainty is higher on fall windows. Data coverage plan is justified.")
    elif r <= 1.5 or auc(pos_epi, neg_epi) < 0.65:
        print("BAD/FLAT: epistemic uncertainty barely separates falls from walking. Debug fitter/calibration before 6M.")
    else:
        print("MIXED: uncertainty separates somewhat but not strongly. Run exploit-rollout diagnostic before 6M.")


if __name__ == "__main__":
    main()
