#!/usr/bin/env python3
"""Plan v6 amendment A3 §3 (21 Nov 2026): verify the imagined reward the Go2 TRAINER will use, at runtime.

The E16 check (18 Nov) fed the env's term code directly. This check goes through the training path: the config
and env class are set by go2_study.apply_s2_contract (the function e1_train.py calls), the env is constructed as
the trainer constructs it, and the reward comes out of the env's own step aggregation (_post_imagination_step,
weights x terms x dt). Input: the E16 trace (s2 in Start-Unitree-Check-v0: every simulator reward term recorded).

Pass rule (fixed 21 Nov, before the Go2 round-0 fit):
1. the runtime weights equal the contract's included weights exactly, and stand_still is not rewarded;
2. every runtime term equals the E16 reconstruction of the same rows (relative 1e-5): the training wiring adds
   no difference of its own;
3. the total runtime reward matches the simulator total by the E16 rule (MAE <= 10 %, |bias| <= 5 %);
4. every term that is not a listed approximation matches by the E16 rule.
Listed approximations (A3: kept explicit, reported, never silently corrected): dof_acc_l2 (finite difference of
predicted joint velocity), feet_air_time (timers driven by predicted foot contacts), undesired_contacts (thigh
contacts only; hip and calf contacts are not predicted; untested when no contact occurs in the trace).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MB = HERE.parent
for p in (str(HERE), str(MB)):
    if p not in sys.path:
        sys.path.insert(0, p)

import e16_reward_check as e16  # noqa: E402
from pre_reset_transitions import sha256, validate_trace  # noqa: E402

APPROXIMATIONS = {
    "dof_acc_l2": "finite difference of the predicted joint velocity against the previous noise-free state "
                  "(simulator: joint_acc from the physics engine)",
    "feet_air_time": "foot timers driven by the predicted foot contact (simulator: contact sensor timers)",
    "undesired_contacts": "predicted thigh contacts only; hip and calf contacts are not predicted",
}
EXACT = 1e-5


def runtime_terms(trace, meta, cfg, env_cls):
    """Per-term reward per second (weight x term) and the total, from the env's own step aggregation."""
    from tensordict import TensorDict
    T, N = trace["state"].shape[:2]
    R = T * N
    take = lambda key: torch.from_numpy(np.asarray(trace[key]).reshape(R, *np.asarray(trace[key]).shape[2:])).float()
    ec = cfg.environment_config
    dt = float(meta["step_dt"])
    env = env_cls(R, ec.max_episode_length, dt, dict(ec.reward_term_weights), "cpu", 0.0, False, None, None)
    env._init_imagination_reward_buffer()
    env._init_additional_imagination_attributes()
    env.common_step_counter, env.termination_flags, env.extras = 0, None, {}
    env.epistemic_uncertainty = torch.zeros(R)
    env.base_velocity, env.obs_last_action = take("command"), take("previous_action")
    prev = take("state")[:, 21:33]
    obs = torch.zeros(R, env.observation_dim)
    obs[:, 24:36] = prev
    env.last_obs = TensorDict({"policy": obs}, batch_size=[R])
    env.last_joint_vel_state = prev.clone()
    for key in ("current_air_time", "last_air_time", "current_contact_time", "last_contact_time"):
        setattr(env, key, take(key).clone())
    contact = take("contact")
    parsed = env._parse_imagination_states(take("next_state"))
    env._compute_imagination_reward_terms(parsed, take("action"), None,
                                          {"thigh_contact": contact[:, :4], "foot_contact": contact[:, 4:]})
    rewards, dones, _ = env._post_imagination_step()
    terms = {k: (v / dt).numpy() for k, v in env.imagination_episode_sums.items() if k != "uncertainty"}
    return terms, (rewards / dt).numpy(), env


def run(trace_path, out):
    import go2_study
    from configs.go2_flat_cfg import Go2FlatConfig, S2_REWARD_CONTRACT
    from envs.go2_flat import Go2FlatEnv
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with np.load(trace_path, allow_pickle=False) as z:
        meta = validate_trace(z)
        trace = {k: z[k] for k in z.files if k != "metadata"}
    cfg = Go2FlatConfig()
    rec = go2_study.apply_s2_contract(cfg, trace_path)        # the trainer's own call
    included = {k: float(v["weight"]) for k, v in S2_REWARD_CONTRACT.items() if v["decision"] == "include"}
    weights = dict(cfg.environment_config.reward_term_weights)
    failures = []
    if weights != included:
        failures.append(f"runtime weights differ from the contract: {sorted(set(weights.items()) ^ set(included.items()))}")
    if weights.get("stand_still", 0.0) != 0.0:
        failures.append("stand_still is rewarded at runtime")
    terms, total, env = runtime_terms(trace, meta, cfg, Go2FlatEnv)
    recon = e16.reconstruct(trace, meta, S2_REWARD_CONTRACT, timers_before_rewards=True)
    real = trace["reward_terms"].reshape(-1, len(meta["reward_names"]))
    rows = {}
    for i, name in enumerate(meta["reward_names"]):
        if name not in included:
            continue
        diff = float(np.max(np.abs(terms[name] - recon[name])))
        scale = max(float(np.max(np.abs(recon[name]))), 1e-12)
        same = diff <= EXACT * scale
        if not same:
            failures.append(f"{name}: runtime differs from the E16 reconstruction (max |diff| {diff:.3e})")
        cmp = e16.compare(terms[name], real[:, i])
        approx = name in APPROXIMATIONS
        untested = approx and cmp["verdict"].startswith("matches (zero")
        if cmp["verdict"] == "DIFFERS" and not approx:
            failures.append(f"{name}: differs from the simulator and is not a listed approximation")
        rows[name] = dict(runtime_equals_reconstruction=same, max_abs_diff=diff, approximation=approx,
                          approximation_note=APPROXIMATIONS.get(name), untested=untested, **cmp)
    tot = e16.compare(total, real.sum(1))
    if tot["verdict"] != "matches":
        failures.append(f"total runtime reward differs from the simulator total (rel MAE {tot['rel_mae']:.1%}, "
                        f"rel bias {tot['rel_bias']:.1%})")
    report = dict(rule="A3 §3 runtime reward check (4 conditions, fixed 21 Nov)", trace=str(trace_path),
                  trace_sha256=sha256(trace_path), rows=int(len(total)), contract=rec, runtime_weights=weights,
                  terms=rows, total=tot, approximations=APPROXIMATIONS, failures=failures,
                  verdict="PASS" if not failures else "FAIL", analysis_sha256=sha256(__file__))
    (out / "reward_runtime.json").write_text(json.dumps(report, indent=1, default=float))
    lines = ["A3 runtime reward check (training path: go2_study.apply_s2_contract -> env step aggregation)",
             f"trace {Path(trace_path).name}: {len(total)} rows; contract sha256 {rec['contract_sha256'][:12]}; "
             f"{len(weights)} weighted terms",
             "term                 | runtime=recon | sim mean    | runtime mean | rel MAE | rel bias | verdict"]
    for name, r in rows.items():
        tag = " [approximation" + (", untested]" if r["untested"] else "]") if r["approximation"] else ""
        lines.append(f"{name:20s} | {str(r['runtime_equals_reconstruction']):13s} | {r['sim_mean']:+.4e} | "
                     f"{r['imagined_mean']:+.4e}  | {r['rel_mae']:6.1%} | {r['rel_bias']:7.1%} | {r['verdict']}{tag}")
    lines.append(f"{'TOTAL':20s} | {'':13s} | {tot['sim_mean']:+.4e} | {tot['imagined_mean']:+.4e}  | "
                 f"{tot['rel_mae']:6.1%} | {tot['rel_bias']:7.1%} | {tot['verdict']}")
    lines += [f"FAIL: {f}" for f in failures]
    lines.append(f"VERDICT: {report['verdict']}")
    text = "\n".join(lines) + "\n"
    (out / "reward_runtime.txt").write_text(text)
    print(text, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, required=True, help="E16 trace (s2 in Start-Unitree-Check-v0)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    r = run(a.trace, a.out)
    sys.exit(0 if r["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
