#!/usr/bin/env python3
"""Plan v6 amendment A3 §4 step 2 (21 Nov 2026): probe s2 in the fitted Go2 model's imagination. Descriptive.

s2 (the start policy, a simulator policy) acts in imagination built exactly as the trainer builds it
(go2_study: s2 reward contract, round-0 start data; paper-mode cold resets; lambda 1 penalty; observation noise
and imagined events as configured). The same statistics are computed on s2's physics recording (the round-0
trace) with the same code (episode_stats.EpisodeStats: tracking exp(-|cmd - v|^2 / 0.25), speed ratio, falls):
  tracking and speed, foot and thigh contact rates, termination rate per 1000 s,
  the imagined task reward per second and the imagined penalty per second, logged SEPARATELY.
Before the run, s2 must reproduce its recorded actions from observations rebuilt by the imagination code
(policy_loop.policy_checks): this checks the observation layout the trained policies will see.
No pass rule: the numbers describe how the model treats the start policy before any policy is trained in it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MB = HERE.parent
for p in (str(HERE), str(MB)):
    if p not in sys.path:
        sys.path.insert(0, p)

from episode_stats import EpisodeStats  # noqa: E402
from pre_reset_transitions import sha256, validate_trace  # noqa: E402


def build(wm, start_data, asset_trace, num_envs, device):
    import go2_study
    import train as T
    cfg = T.resolve_task_config("go2_flat")
    contract = go2_study.apply_s2_contract(cfg, asset_trace)
    start = go2_study.apply_start_data(cfg, start_data)
    cfg.model_architecture_config.resume_path = str(wm)
    cfg.environment_config.num_envs = int(num_envs)
    cfg.environment_config.uncertainty_penalty_weight = -1.0
    cfg.experiment_config.device = device
    exp = T.ModelBasedExperiment(**cfg.experiment_config.to_dict())
    exp.prepare_environment(**cfg.environment_config.to_dict())
    exp.prepare_model(**cfg.model_architecture_config.to_dict())
    exp.prepare_data(**cfg.data_config.to_dict())
    return exp, cfg, contract, start


def physics_stats(trace, dt):
    T, N = trace["done"].shape
    st = EpisodeStats(N, dt, episode_length_s=20.0, std=0.5)
    for t in range(T):
        st.step(trace["command"][t], trace["next_state"][t, :, :3], trace["done"][t], trace["time_out"][t])
    st.finish()
    s = st.summary()
    falls = int((trace["terminated"] & ~trace["time_out"]).sum())
    seconds = T * N * dt
    c = trace["contact"]
    return dict(track_per_step=s["track_per_step"], speed_ratio=s["speed_ratio"],
                speed_cmd_per_step=s["speed_cmd_per_step"], terminations=falls, seconds=seconds,
                terminations_per_1000s=1000.0 * falls / seconds,
                foot_contact_rate=[float(x) for x in (c[..., 4:8] > 0.5).mean(axis=(0, 1))],
                thigh_contact_rate=float((c[..., 0:4] > 0.5).mean()))


@torch.no_grad()
def imagination_stats(exp, policy, steps, seed):
    from run_provenance import seed_everything
    seed_everything(seed)
    env = exp.env
    cap = {}
    original = env._compute_imagination_reward_terms

    def hook(parsed, action, ext, contacts):
        cap["v"] = parsed["base_lin_vel"].detach().clone()        # before imagined events change it
        cap["cmd"] = env.base_velocity.detach().clone()
        cap["foot"] = contacts["foot_contact"].detach().clone()
        cap["thigh"] = contacts["thigh_contact"].detach().clone()
        return original(parsed, action, ext, contacts)

    env._compute_imagination_reward_terms = hook
    dt, n = env._step_dt, env.num_envs
    st = EpisodeStats(n, dt, episode_length_s=env._max_episode_length * dt, std=0.5)
    w = dict(env.reward_term_weights)
    task_sum = pen_sum = u_sum = 0.0
    term_sums = {k: 0.0 for k in w}
    foot = torch.zeros(4, device=env.device)
    thigh = term = tout = 0.0
    state_history, action_history = env.prepare_imagination()
    rnn = env.system_dynamics.architecture_config["type"] in ("rnn", "rssm")
    for i in range(steps):
        if rnn and env.common_step_counter > 0:
            state_history, action_history = state_history[:, -1:], action_history[:, -1:]
        obs = env.get_imagination_observation(state_history, action_history)
        actions = policy(obs["policy"].cpu()).to(env.device)
        _, _, dones, extras, state_history, action_history, u = env.imagination_step(actions, state_history,
                                                                                    action_history)
        to = extras["time_outs"].bool()
        d = dones.bool()
        st.step(cap["cmd"].cpu().numpy(), cap["v"].cpu().numpy(), d.cpu().numpy(), to.cpu().numpy())
        term += float((d & ~to).sum())
        tout += float(to.sum())
        for k in w:
            v = float((w[k] * env.imagination_reward_per_step[k]).mean())
            term_sums[k] += v
            task_sum += v
        pen = float((env.uncertainty_penalty_weight * u).mean())
        pen_sum += pen
        u_sum += float(u.mean())
        foot += cap["foot"].float().mean(0)
        thigh += float(cap["thigh"].float().mean())
    env._compute_imagination_reward_terms = original
    st.finish()
    s = st.summary()
    seconds = n * steps * dt
    return dict(envs=n, steps=steps, seconds=seconds, track_per_step=s["track_per_step"],
                speed_ratio=s["speed_ratio"], speed_cmd_per_step=s["speed_cmd_per_step"],
                terminations=int(term), time_outs=int(tout), terminations_per_1000s=1000.0 * term / seconds,
                foot_contact_rate=[float(x) / steps for x in foot.cpu()], thigh_contact_rate=thigh / steps,
                task_reward_per_s=task_sum / steps, penalty_per_s=pen_sum / steps,
                mean_disagreement=u_sum / steps, term_reward_per_s={k: v / steps for k, v in term_sums.items()})


def run(wm, start_data, asset_trace, physics_trace, out, envs=256, steps=1000, seed=0, device="cuda:0",
        policy_path=None):
    from policy_loop import ObservationBuilder, policy_checks, resolve_policy
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    t0 = time.time()
    with np.load(physics_trace, allow_pickle=False) as z:
        meta = validate_trace(z)
        trace = {k: z[k] for k in z.files if k != "metadata"}
    exp, cfg, contract, start = build(wm, start_data, asset_trace, envs, device)
    stats = {k: v.reshape(-1).cpu().numpy() for k, v in
             zip(("state_mean", "state_std", "action_mean", "action_std"),
                 (exp.dataset.state_data_mean, exp.dataset.state_data_std,
                  exp.dataset.action_data_mean, exp.dataset.action_data_std))}
    policy = resolve_policy("s2", meta, {"s2": policy_path} if policy_path else {})
    checks = policy_checks("s2", trace, meta, policy, ObservationBuilder("go2_flat", stats), stats)
    imag = imagination_stats(exp, policy, steps, seed)
    phys = physics_stats(trace, float(meta.get("step_dt", 0.02)))
    report = dict(protocol="plan v6 amendment A3 §4 step 2: s2 probe in imagination (descriptive)",
                  wm=str(wm), wm_sha256=sha256(wm), physics_trace=str(physics_trace),
                  physics_trace_sha256=sha256(physics_trace), policy=policy.describe(), policy_checks=checks,
                  contract=dict(contract_sha256=contract["contract_sha256"], terms=contract["terms"]),
                  start_data=start, imagination=imag, physics=phys, seed=seed,
                  imagination_settings=dict(max_episode_length=cfg.environment_config.max_episode_length,
                                            observation_noise=cfg.environment_config.observation_noise,
                                            event_interval_range=cfg.environment_config.event_interval_range,
                                            command_resample_interval_range=cfg.environment_config.command_resample_interval_range,
                                            uncertainty_penalty_weight=-1.0),
                  wall_s=time.time() - t0, analysis_sha256=sha256(__file__))
    (out / "s2_probe.json").write_text(json.dumps(report, indent=1, default=float))
    rows = [("track_per_step", "{:.4f}"), ("speed_ratio", "{:.4f}"), ("speed_cmd_per_step", "{:.3f}"),
            ("terminations_per_1000s", "{:.2f}"), ("thigh_contact_rate", "{:.4f}")]
    lines = ["A3 G0 step 2: s2 in imagination vs s2 in physics (descriptive, same statistics code)",
             f"model {Path(wm).name} ({report['wm_sha256'][:12]}); imagination {imag['envs']} envs x {imag['steps']} "
             f"steps ({imag['seconds']:.0f} s); physics {phys['seconds']:.0f} s",
             f"policy check: max action error {checks['max_abs_action_error']:.2e} (passed)",
             "metric                  | imagination | physics"]
    for k, f in rows:
        lines.append(f"{k:23s} | {f.format(imag[k]):11s} | {f.format(phys[k])}")
    lines.append(f"{'foot_contact_rate':23s} | {' '.join(f'{x:.2f}' for x in imag['foot_contact_rate'])} | "
                 f"{' '.join(f'{x:.2f}' for x in phys['foot_contact_rate'])}")
    lines.append(f"imagined task reward {imag['task_reward_per_s']:+.4f} per s; imagined penalty "
                 f"{imag['penalty_per_s']:+.4f} per s (mean disagreement {imag['mean_disagreement']:.4f}); "
                 f"terminations {imag['terminations']}, time-outs {imag['time_outs']}")
    text = "\n".join(lines) + "\n"
    (out / "s2_probe.txt").write_text(text)
    print(text, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", type=Path, required=True)
    p.add_argument("--start-data", type=Path, required=True)
    p.add_argument("--asset-trace", type=Path, required=True, help="E16 trace (asset constants, contract check)")
    p.add_argument("--physics", type=Path, required=True, help="s2's round-0 physics recording")
    p.add_argument("--policy", type=Path, default=None, help="s2 checkpoint (default: the recording's metadata)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    run(a.wm, a.start_data, a.asset_trace, a.physics, a.out, a.envs, a.steps, a.seed, a.device, a.policy)


if __name__ == "__main__":
    main()
