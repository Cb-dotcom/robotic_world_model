#!/usr/bin/env python3
"""Paired real-context versus imagined-context replay of saved fall episodes.

CPU only. Each arm uses the same recorded actions and true target timestamps.
Count termination ANYWHERE in the path, not just at its endpoint. Forecasts are
forced to continue for diagnosis; rows after a predicted termination are marked
inactive and are not presented as steps the deployed imagination would execute.
This is a warm full-context mean-path diagnostic, not stochastic PPO replay.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyse_transition_fidelity import load_model, predict
from pre_reset_transitions import gather, sha256, validate_trace


def fall_origins(trace, horizon=25, history=32):
    falls = trace["done"] & trace["terminated"] & ~trace["time_out"]
    origins, excluded = [], 0
    for end, n in np.argwhere(falls):
        start = int(end) - horizon + 1
        if start < history - 1 or trace["done"][start-history+1:end, n].any():
            excluded += 1
        else:
            origins.append((start, int(n)))
    return np.asarray(origins, dtype=int).reshape(-1, 2), dict(
        actual_falls=int(falls.sum()), included=len(origins), short_history_excluded=excluded)


@torch.no_grad()
def imagined_path(model, contract, trace, origins, horizon):
    stats = contract.statistics
    states = (torch.from_numpy(gather(trace["state"], origins, -31, 1)).float() - stats["state_mean"]) / stats["state_std"]
    actions = (torch.from_numpy(gather(trace["action"], origins, -31, horizon)).float() - stats["action_mean"]) / stats["action_std"]
    outputs = [[], [], [], []]
    for head in range(model.ensemble_size):
        model.reset()
        xs, xa = states, actions[:, :32]
        ids = torch.full((1, len(origins), 1), head, dtype=torch.long)
        path = [[], [], [], []]
        for k in range(horizon):
            mean, _, epi, _, contact, term = model(xs, xa, ids)
            for accumulator, value in zip(path, (
                mean * stats["state_std"] + stats["state_mean"],
                torch.sigmoid(contact), torch.sigmoid(term).squeeze(-1), epi)):
                accumulator.append(value.numpy())
            xs, xa = mean[:, None], actions[:, 32+k:33+k]
        for accumulator, values in zip(outputs, path):
            accumulator.append(np.stack(values))
    return tuple(np.stack(values) for values in outputs)  # [head, step, event, ...]


def first_and_active(probability):
    decisions = probability > .5
    any_stop = decisions.any(axis=1)
    first = np.where(any_stop, decisions.argmax(axis=1) + 1, 0)
    # Active before this prediction, including the first step predicting a stop.
    active = np.ones_like(decisions, dtype=bool)
    active[:, 1:] = ~np.maximum.accumulate(decisions[:, :-1], axis=1)
    return first, active


def summarise(outputs, targets, step_dt):
    state, contact, term, epi = outputs
    first, active = first_and_active(term)
    lines = []
    for k in range(state.shape[1]):
        error = state[:, k, :, :3] - targets[k, :, :3]
        mask = active[:, k]
        lines.append(dict(
            step=k+1, seconds=(k+1)*step_dt,
            velocity_rmse_all_forced_paths=float(np.sqrt((error**2).mean())),
            velocity_rmse_active_paths=float(np.sqrt((error[mask]**2).mean())) if mask.any() else None,
            real_tilt=float(np.linalg.norm(targets[k, :, 6:8], axis=-1).mean()),
            predicted_tilt_all_forced_paths=float(np.linalg.norm(state[:, k, :, 6:8], axis=-1).mean()),
            uncertainty_mean=float(epi[:, k].mean()),
            termination_probability_mean=float(term[:, k].mean()),
            fraction_active_before_step=float(mask.mean()),
            cumulative_stop_fraction=float(((first > 0) & (first <= k+1)).mean())))
    return dict(first_stop_step_per_head_event=first.tolist(),
                stopped_anywhere_fraction_per_head=(first > 0).mean(axis=1).tolist(), steps=lines)


def run(trace_path, wm_path, out):
    torch.set_num_threads(2)
    model, contract = load_model(wm_path)
    horizon = 25
    with np.load(trace_path, allow_pickle=False) as trace:
        metadata = validate_trace(trace)
        origins, counts = fall_origins(trace, horizon)
        if not len(origins):
            raise ValueError("No fall has the required reset-safe 32+25 context")
        print(f"[paired replay] {counts}; CPU, horizon={horizon}", flush=True)
        # Materialize each source once: NpzFile access otherwise decompresses
        # the full compressed member on every indexing operation.
        data = {key: trace[key] for key in ("state", "action", "next_state")}
        targets = gather(data["next_state"], origins, 0, horizon).transpose(1, 0, 2)
        teacher_origins = np.asarray([(t+k, n) for k in range(horizon) for t, n in origins])
        chunks = []
        for b in range(0, len(teacher_origins), 64):
            chunks.append(predict(model, contract, data, teacher_origins[b:b+64], 1))
            print(f"[paired replay] real-context rows {min(b+64,len(teacher_origins))}/{len(teacher_origins)}", flush=True)
        teacher = tuple(np.concatenate([chunk[i] for chunk in chunks], axis=1).reshape(
            model.ensemble_size, horizon, len(origins), *chunks[0][i].shape[2:]) for i in range(4))
        imagined = imagined_path(model, contract, data, origins, horizon)
        first_step_deltas = []
        for real_first, imagined_first in zip(teacher, imagined):
            # CPU GRU/linear kernels use different reduction orders for the
            # teacher's 64-row batches and the event-count imagined batch.
            first_step_deltas.append(float(np.abs(real_first[:, 0] - imagined_first[:, 0]).max()))
            np.testing.assert_allclose(real_first[:, 0], imagined_first[:, 0], rtol=1e-4, atol=1e-4,
                                       err_msg="Paired paths disagree before any imagined feedback")
        print("[paired replay] imagined paths complete", flush=True)
        report = dict(counts=counts, metadata=metadata, trace_sha256=sha256(trace_path),
                      first_step_max_abs_deltas_state_contact_term_uncertainty=first_step_deltas,
                      wm_sha256=sha256(wm_path), analysis_sha256=sha256(__file__), normalization=contract.metadata(),
                      scope="Same fall episodes, actions, commands and target timestamps in both arms. Teacher forcing reinitializes "
                            "32-step real context for each prediction; imagination consumes the full real context once and then its own "
                            "predicted mean state. No state noise sampling or policy recomputation. Predicted stops are counted cumulatively. "
                            "Forced continuation after a predicted stop is marked inactive. Earlier predicted stops precede physical contact, "
                            "not necessarily a calibrated advance-failure signal. Descriptive one-seed evidence, not causal proof or actual PPO returns.",
                      real_context=summarise(teacher, targets, metadata["step_dt"]),
                      imagined_context=summarise(imagined, targets, metadata["step_dt"]))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "paired_fall_replay.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    with (out / "paired_fall_replay.npz").open("xb") as stream:
        np.savez_compressed(stream, origins=origins, targets=targets,
                            **{f"{arm}_{name}": value for arm, values in (("real",teacher),("imagined",imagined))
                               for name, value in zip(("state","contact_probability","termination_probability","uncertainty"),values)})
    lines = [f"Paired fall replay: {counts}", report["scope"]]
    for arm in ("real_context", "imagined_context"):
        result = report[arm]
        lines.append(f"\n{arm}: stopped anywhere/head={result['stopped_anywhere_fraction_per_head']}")
        for row in result["steps"]:
            if row["step"] in (1,5,10,15,20,25):
                lines.append(f"step={row['step']:2} active={row['fraction_active_before_step']:.3f} "
                             f"cumulative_stop={row['cumulative_stop_fraction']:.3f} "
                             f"uncertainty={row['uncertainty_mean']:.4f} "
                             f"velocity_RMSE_forced={row['velocity_rmse_all_forced_paths']:.4f} "
                             f"term_mean={row['termination_probability_mean']:.4f}")
    text = "\n".join(lines) + "\n"
    with (out / "paired_fall_replay.txt").open("x") as stream:
        stream.write(text)
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--wm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if any((args.out / name).exists() for name in ("paired_fall_replay.json", "paired_fall_replay.npz", "paired_fall_replay.txt")):
        parser.error("Output evidence already exists; use a fresh directory")
    run(args.trace, args.wm, args.out)


if __name__ == "__main__":
    main()
