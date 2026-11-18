#!/usr/bin/env python3
"""Plan v6 E16: check the imagined reward of the Go2 start policy s2 against the simulator, term by term.

Input: a pre-reset trace recorded in `Template-Isaac-Velocity-Flat-Unitree-Go2-Start-Unitree-Check-v0` (s2's
training rewards, full command ranges, no curriculum) with the s2 checkpoint. The trace stores every
simulator reward term (weighted, per second), the true next state, contacts, foot timers, the asset's default
pose and soft joint limits, and each term's weight and parameters.

Method (as reward_reconstruction in analyse_transition_fidelity.py, the feet_slide pattern): the imagination
env's own term code (envs/go2_flat.py) is fed the TRUE next state and contacts of each row, the recorded
command, previous action, action and foot timers; the result x the contract weight is compared with the
simulator's term. This isolates the implementation; it is not a model prediction.

Rules (fixed 18 Nov 2026, before the recording):
* every simulator term must be in S2_REWARD_CONTRACT (configs/go2_flat_cfg.py), with the simulator's weight and
  parameters; otherwise the check stops (the contract is incomplete or stale);
* an included term MATCHES if its mean absolute error <= 10 % of the simulator term's mean absolute value and
  its mean bias <= 5 % of it (a term that is ~0 in both counts as matching); otherwise it DIFFERS and the
  difference is reported (it is not silently corrected).
feet_air_time is computed with both timer orders (rewards before / after the foot-timer update); the report
names the order that matches. air_time_variance always reads the updated timers, as the simulator does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MB = HERE.parent
for p in (str(HERE), str(MB)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pre_reset_transitions import sha256, validate_trace  # noqa: E402

REL_MAE, REL_BIAS, ZERO = 0.10, 0.05, 1e-9
TIMER_TERMS = ("feet_air_time",)            # air_time_variance always reads the updated timers


class ContractError(RuntimeError):
    pass


def contract_problems(meta, contract):
    names = list(meta["reward_names"])
    problems = []
    missing = [n for n in names if n not in contract]
    extra = [n for n in contract if n not in names]
    if missing:
        problems.append(f"simulator terms not in the contract: {missing}")
    if extra:
        problems.append(f"contract terms not in the simulator reward: {extra}")
    for n in names:
        if n not in contract:
            continue
        w_sim, w_c = float(meta["reward_weights"][n]), float(contract[n]["weight"])
        if not np.isclose(w_sim, w_c, rtol=1e-6, atol=0):
            problems.append(f"{n}: contract weight {w_c} != simulator weight {w_sim}")
        for k, v in (contract[n].get("params") or {}).items():
            sim = (meta.get("reward_params") or {}).get(n, {}).get(k)
            if sim is None or not np.isclose(float(sim), float(v)):
                problems.append(f"{n}: contract parameter {k}={v} != simulator {sim}")
    return problems


def reconstruct(trace, meta, contract, timers_before_rewards, rows=None):
    """Imagined term values x contract weight on the true next state of each row; dict name -> [R]."""
    from tensordict import TensorDict
    from envs.go2_flat import Go2FlatEnv
    T, N = trace["state"].shape[:2]
    idx = np.arange(T * N) if rows is None else np.asarray(rows)
    t, n = idx // N, idx % N
    take = lambda key: torch.from_numpy(np.asarray(trace[key])[t, n]).float()
    env = Go2FlatEnv.__new__(Go2FlatEnv)
    env._num_envs, env.device, env._step_dt = len(idx), "cpu", float(meta["step_dt"])
    env.feet_air_time_threshold = float(contract["feet_air_time"]["params"]["threshold"])
    env.joint_pos_stand_still_scale = float(contract["joint_pos"]["params"]["stand_still_scale"])
    env.joint_pos_velocity_threshold = float(contract["joint_pos"]["params"]["velocity_threshold"])
    env.default_joint_pos = meta["default_joint_pos"]
    env.soft_joint_pos_limits = meta["soft_joint_pos_limits"]
    env.timers_before_rewards = bool(timers_before_rewards)
    env.base_velocity, env.obs_last_action = take("command"), take("previous_action")
    prev = torch.zeros(len(idx), 48)
    prev[:, 24:36] = take("state")[:, 21:33]                 # imagination's last_obs joint velocity = previous state
    env.last_obs = TensorDict({"policy": prev}, batch_size=[len(idx)])
    for key in ("current_air_time", "last_air_time", "current_contact_time", "last_contact_time"):
        setattr(env, key, take(key).clone())
    contact = take("contact")
    parsed = env._parse_imagination_states(take("next_state"))
    env._compute_imagination_reward_terms(parsed, take("action"), None,
                                          {"thigh_contact": contact[:, :4], "foot_contact": contact[:, 4:]})
    return {name: (env.imagination_reward_per_step[name] * float(spec["weight"])).numpy()
            for name, spec in contract.items() if spec["decision"] == "include"}


def compare(est, real):
    scale = float(np.abs(real).mean())
    err = est - real
    mae, bias = float(np.abs(err).mean()), float(err.mean())
    if scale < ZERO and float(np.abs(est).mean()) < ZERO:
        verdict, rel_mae, rel_bias = "matches (zero in both)", 0.0, 0.0
    else:
        rel_mae = mae / scale if scale > 0 else float("inf")
        rel_bias = abs(bias) / scale if scale > 0 else float("inf")
        verdict = "matches" if rel_mae <= REL_MAE and rel_bias <= REL_BIAS else "DIFFERS"
    corr = float(np.corrcoef(est, real)[0, 1]) if np.std(est) > 0 and np.std(real) > 0 else float("nan")
    return dict(sim_mean=float(real.mean()), imagined_mean=float(est.mean()), mean_bias=bias, mean_abs_error=mae,
                sim_mean_abs=scale, rel_mae=rel_mae, rel_bias=rel_bias, corr=corr, verdict=verdict)


def run(trace_path, out, contract=None, excluded=None):
    from configs.go2_flat_cfg import S2_EXCLUDED_IMAGINED_TERMS, S2_REWARD_CONTRACT
    contract = contract or S2_REWARD_CONTRACT
    excluded = S2_EXCLUDED_IMAGINED_TERMS if excluded is None else excluded
    out.mkdir(parents=True, exist_ok=False)
    with np.load(trace_path, allow_pickle=False) as z:
        meta = validate_trace(z)
        trace = {k: z[k] for k in z.files if k != "metadata"}
    for key in ("reward_params", "default_joint_pos", "soft_joint_pos_limits", "reward_names", "reward_weights"):
        if key not in meta:
            raise ContractError(f"trace metadata lacks {key}: record with the 18 Nov recorder")
    problems = contract_problems(meta, contract)
    if problems:
        raise ContractError("; ".join(problems))
    real_terms = trace["reward_terms"].reshape(-1, len(meta["reward_names"]))
    variants = {order: reconstruct(trace, meta, contract, order) for order in (False, True)}
    rows = {}
    for i, name in enumerate(meta["reward_names"]):
        spec = contract[name]
        if spec["decision"] != "include":
            rows[name] = dict(decision=spec["decision"], reason=spec.get("reason"))
            continue
        res = {("timers_first" if order else "timers_after"): compare(variants[order][name], real_terms[:, i])
               for order in (False, True)}
        if name in TIMER_TERMS:
            best = min(res, key=lambda k: res[k]["rel_mae"])
            rows[name] = dict(decision="include", source=spec.get("source"), timer_order=best, **res[best],
                              other_order=res["timers_after" if best == "timers_first" else "timers_first"])
        else:
            rows[name] = dict(decision="include", source=spec.get("source"), **res["timers_first"])
    total_sim = real_terms.sum(1)
    total_est = sum(variants[True][n] for n in rows if rows[n]["decision"] == "include")
    report = dict(trace=str(trace_path), trace_sha256=sha256(trace_path), task=meta.get("task"),
                  checkpoint=meta.get("checkpoint"), checkpoint_sha256=meta.get("checkpoint_sha256"),
                  rows_checked=int(len(total_sim)), rules=dict(rel_mae=REL_MAE, rel_bias=REL_BIAS),
                  terms=rows, excluded_imagined_terms=excluded,
                  total=compare(total_est, total_sim), analysis_sha256=sha256(__file__))
    (out / "e16_report.json").write_text(json.dumps(report, indent=1))
    lines = ["Plan v6 E16 imagined-reward check, s2 (true next state and contacts; contract weights)",
             f"trace {Path(trace_path).name}: {len(total_sim)} rows; task {meta.get('task')}",
             f"rule: matches if MAE <= {REL_MAE:.0%} and |bias| <= {REL_BIAS:.0%} of the simulator term's mean |value|",
             "term                 | sim mean    | imagined mean | rel MAE | rel bias | corr  | verdict"]
    for name, r in rows.items():
        if r["decision"] != "include":
            lines.append(f"{name:20s} | excluded: {r.get('reason')}")
            continue
        extra = f" ({r['timer_order']})" if "timer_order" in r else ""
        lines.append(f"{name:20s} | {r['sim_mean']:+.4e} | {r['imagined_mean']:+.4e}   | {r['rel_mae']:6.1%} | "
                     f"{r['rel_bias']:7.1%} | {r['corr']:+.3f} | {r['verdict']}{extra}")
    t = report["total"]
    lines.append(f"{'TOTAL (included)':20s} | {t['sim_mean']:+.4e} | {t['imagined_mean']:+.4e}   | {t['rel_mae']:6.1%} | "
                 f"{t['rel_bias']:7.1%} | {t['corr']:+.3f} | {t['verdict']}")
    for k, v in excluded.items():
        lines.append(f"imagined-only term {k}: {v}")
    text = "\n".join(lines) + "\n"
    (out / "e16_report.txt").write_text(text)
    print(text, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("output folder exists")
    try:
        run(a.trace, a.out)
    except ContractError as exc:
        raise SystemExit(f"[e16] STOPPED: {exc}")


if __name__ == "__main__":
    main()
