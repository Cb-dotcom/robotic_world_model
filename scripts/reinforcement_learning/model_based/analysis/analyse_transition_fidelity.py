#!/usr/bin/env python3
"""CPU replay of true pre-reset transitions. No simulator and no training.

Freeze: H=32, horizons 1/8, up to 256 windows per stratum, sampling seed 0.
Mean-state forecasts per fixed ensemble head; no sampled process noise.
Immediate fall labels are not advance warnings. Conditional logged-action replay
cannot by itself prove why a trained policy fails or select a proven repair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

MB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MB))
from configs.anymal_d_flat_cfg import AnymalDFlatConfig
from normalization_contract import resolve_normalization
from pre_reset_transitions import GROUPS, gather, sha256, validate_trace, windows


def load_model(path):
    from rsl_rl.modules import SystemDynamicsEnsemble
    cfg = AnymalDFlatConfig()
    mac = cfg.model_architecture_config
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SystemDynamicsEnsemble(45, 12, 0, 8, 1, "cpu", ensemble_size=mac.ensemble_size,
                                  history_horizon=mac.history_horizon,
                                  architecture_config=mac.architecture_config).eval()
    model.load_state_dict(ckpt["system_dynamics_state_dict"], strict=True)
    contract = resolve_normalization(ckpt, vars(cfg.data_config), 45, 12)
    return model, contract


def reward_reconstruction(trace, origins, metadata):
    """Use true next state/contact to isolate reward implementation discrepancy.

    This is NOT model-predicted reward. Previous noisy policy observations and
    simulator foot timers are supplied so noise and timer phase are explicit.
    Uncertainty penalty is outside these terms and is not included in the total.
    """
    from envs.anymal_d_flat import AnymalDFlatEnv
    from tensordict import TensorDict
    env = AnymalDFlatEnv.__new__(AnymalDFlatEnv)
    env._num_envs, env.device = len(origins), "cpu"
    env._step_dt = metadata["step_dt"]
    take = lambda name: torch.from_numpy(np.stack([trace[name][t, n] for t, n in origins])).float()
    env.base_velocity, env.obs_last_action = take("command"), take("previous_action")
    env.last_obs = TensorDict({"policy": take("policy_observation")}, batch_size=[len(origins)])
    for name in ("current_air_time", "last_air_time", "current_contact_time", "last_contact_time"):
        setattr(env, name, take(name))
    state, contact = take("next_state"), take("contact")
    parsed = env._parse_imagination_states(state)
    env._compute_imagination_reward_terms(parsed, take("action"), None,
                                          {"thigh_contact": contact[:, :4], "foot_contact": contact[:, 4:]})
    real = take("reward_terms").numpy()
    weights = AnymalDFlatConfig().environment_config.reward_term_weights
    result = {}
    for index, name in enumerate(metadata["reward_names"]):
        if name not in env.imagination_reward_per_step or name not in weights:
            result[name] = {"available": False}
            continue
        estimate = env.imagination_reward_per_step[name].numpy() * weights[name]
        error = estimate - real[:, index]
        result[name] = {"available": True, "real_mean": float(real[:, index].mean()),
                        "reconstructed_mean": float(estimate.mean()),
                        "mean_bias": float(error.mean()), "mean_absolute_error": float(np.abs(error).mean()),
                        "sim_weight": metadata["reward_weights"][name], "offline_weight": weights[name]}
    return result


@torch.no_grad()
def predict(model, norm, trace, origins, horizon):
    """Fixed head per sample; full history once, then recurrent one-step inputs."""
    stats = norm.statistics
    st = (torch.from_numpy(gather(trace["state"], origins, -31, 1)).float() - stats["state_mean"]) / stats["state_std"]
    ac = (torch.from_numpy(gather(trace["action"], origins, -31, horizon)).float() - stats["action_mean"]) / stats["action_std"]
    state, contacts, terms, epis = [], [], [], []
    for head in range(model.ensemble_size):
        model.reset()
        xs, xa = st, ac[:, :32]
        ids = torch.full((1, len(origins), 1), head, dtype=torch.long)
        for k in range(horizon):
            mean, _, epi, _, contact, term = model(xs, xa, ids)
            if k < horizon - 1:
                xs, xa = mean[:, None], ac[:, 32 + k:33 + k]
        state.append((mean * stats["state_std"] + stats["state_mean"]).numpy())
        contacts.append(torch.sigmoid(contact).numpy())
        terms.append(torch.sigmoid(term).squeeze(-1).numpy())
        epis.append(epi.numpy())
    return np.stack(state), np.stack(contacts), np.stack(terms), np.stack(epis)


def grouped_metrics(trace, origins, labels, horizon, predictions):
    predicted, contacts, term, epi = predictions
    targets = gather(trace["next_state"], origins, horizon - 1, horizon)[:, 0]
    true_contact = gather(trace["contact"], origins, horizon - 1, horizon)[:, 0]
    # The head was fitted on system_termination, so use that physical contact label.
    true_term = gather(trace["termination_observation"], origins, horizon - 1, horizon)[:, 0, 0].astype(bool)
    output = {}
    for group in ("fall_now", "pre_fall_1_25", "surviving_25"):
        mask = labels == group
        if not mask.any():
            output[group] = {"windows": 0}
            continue
        err = predicted[:, mask] - targets[mask]
        decision, truth = term[:, mask] > .5, true_term[mask]
        positive, negative = int(truth.sum()), int((~truth).sum())
        output[group] = {
            "windows": int(mask.sum()), "physical_positives": positive, "physical_negatives": negative,
            "rmse_physical_units_per_head_mean": {
                name: float(np.sqrt((err[..., lo:hi] ** 2).mean(axis=(1, 2))).mean())
                for name, (lo, hi) in GROUPS.items()},
            "term_probability_mean": float(term[:, mask].mean()),
            "term_recall_at_deployment_threshold_per_head": (
                decision[:, truth].mean(1).tolist() if positive else None),
            "term_false_positive_rate_per_head": (
                decision[:, ~truth].mean(1).tolist() if negative else None),
            "contact_bit_error_per_head": ((contacts[:, mask] > .5) != (true_contact[mask] > .5)).mean((1, 2)).tolist(),
            "epistemic_sum_std_normalized_mean": float(epi[:, mask].mean()),
            "command_speed_mean": float(np.linalg.norm(gather(trace["command"], origins[mask], 0, 1)[:, 0, :2], axis=-1).mean()),
        }
    return output


def analyse(path, model, contract):
    with np.load(path, allow_pickle=False) as trace:
        meta = validate_trace(trace)
        report = {"metadata": meta, "trace_sha256": sha256(path),
                  "fall_events": int((trace["done"] & ~trace["time_out"]).sum()),
                  "physical_terminations": int(trace["terminated"].sum()),
                  "timeouts": int(trace["time_out"].sum()), "horizons": {}}
        for horizon in (1, 8):
            origins, labels, counts = windows(trace, 32, horizon)
            if not len(origins):
                raise ValueError("No valid replay windows")
            chunks = [predict(model, contract, trace, origins[b:b+64], horizon) for b in range(0, len(origins), 64)]
            predictions = tuple(np.concatenate([chunk[i] for chunk in chunks], axis=1) for i in range(4))
            data = dict(counts, groups=grouped_metrics(trace, origins, labels, horizon, predictions))
            if horizon == 1:
                data["reward_on_true_transitions_by_group"] = {
                    group: reward_reconstruction(trace, origins[labels == group], meta)
                    for group in np.unique(labels)}
            report["horizons"][str(horizon)] = data
        return report


def brief(report):
    lines = ["Transition fidelity diagnostic: recorded actions; full 32-step history; fixed head mean rollouts.",
             "Stratified samples are descriptive. No new training or causal success claim."]
    for arm, result in report["arms"].items():
        lines.append(f"\n{arm}: falls={result['fall_events']}; pre-reset checks={result['metadata']['verified_terminal_rows']}")
        for horizon, values in result["horizons"].items():
            lines.append(f"  horizon {horizon}: eligible={values['eligible']}; selected={values['selected']}")
            for group, metrics in values["groups"].items():
                if metrics["windows"]:
                    errors = metrics["rmse_physical_units_per_head_mean"]
                    lines.append(f"    {group}: velocity RMSE={errors['lin_vel_m_s']:.4f} m/s; gravity RMSE={errors['gravity']:.4f}; "
                                 f"term mean={metrics['term_probability_mean']:.4f}; recall/head={metrics['term_recall_at_deployment_threshold_per_head']}")
        lines.append("  True-transition reward MAE (weighted per-second, surviving_25):")
        rewards = result["horizons"]["1"]["reward_on_true_transitions_by_group"].get("surviving_25", {})
        lines.extend(f"    {name}: {values['mean_absolute_error']:.6f}" for name, values in rewards.items() if values["available"])
    lines.append("\nRead report.json for reward terms near falls, contact errors, per-head thresholds, exclusions and provenance.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--wm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, contract = load_model(args.wm)
    report = {"world_model_sha256": sha256(args.wm), "normalization": contract.metadata(),
              "analysis_sha256": sha256(__file__),
              "scope": "One diagnostic seed; two saved actors; no pushes; logging changes observation RNG stream. "
                       "Stratified, overlapping windows are not independent trials. Fixed-head mean logged-action "
                       "replay is not the stochastic policy rollout. No numerical decision bars are inferred post hoc.",
              "arms": {name: analyse(path, model, contract) for name, path in
                       (("baseline", args.baseline), ("reference", args.reference))}}
    a, b = (report["arms"][name]["metadata"] for name in ("baseline", "reference"))
    for key in ("task", "seed", "num_envs", "num_steps", "step_dt", "push_enabled", "command_resampling_s",
                "policy_noise", "clip_actions", "isaaclab_env_source_sha256", "recorder_sha256", "storage_sha256",
                "termination_names", "reward_names", "reward_weights"):
        if a[key] != b[key]:
            raise ValueError(f"Recording protocols differ: {key}")
    if b["checkpoint_sha256"] != report["world_model_sha256"]:
        raise ValueError("Reference actor is not from the supplied world-model asset")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    summary = brief(report)
    with (args.out / "report.txt").open("x") as stream:
        stream.write(summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
