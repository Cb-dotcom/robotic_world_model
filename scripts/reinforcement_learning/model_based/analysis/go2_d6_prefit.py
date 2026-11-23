#!/usr/bin/env python3
"""A4 D6 pre-fit verifications (22 Nov 2026, Charbel's decision 20:47). Run on the PREPARED data, before the fit.
All must pass, or the fit does not start. CPU. Fixed before the data were prepared.

(a) first transition: every startup segment's row 0 equals the recording's pre-action state and previous action of the
    episode's first step (contact and termination 0), and row 1 equals its first recorded successor with the first
    action; the fitter's own startup windows of history 1 start with exactly these two rows (normalized), so the
    library pairs (reset state, first action) and targets the first successor.
(b) terminal targets: every training fall has its terminal transition as a target: in a startup window when its
    segment (reset row included) has fewer than 40 rows, else in a standard 40-row window; falls without a target
    are counted and fail the check.
(c) sampler: with share 1, the imagination env (envs/base.py, real model, real reset_states.npz) puts a recorded reset
    state (normalized) in the last history row of every reset env, keeps their memory at zero, and its next model
    step equals a fresh start from that reset state.
(d) reset rows are physical: no reset row equals the previous episode's terminal state in the same env (a stale
    observation); the share with an unchanged torque block and the mean |joint velocity| are reported.
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

from pre_reset_transitions import sha256, validate_trace  # noqa: E402

TOL = 1e-5
SEQ = 40


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        validate_trace(z)
        return {k: z[k] for k in z.files if k != "metadata"}


def _close(a, b):
    return bool(np.allclose(a, b, rtol=TOL, atol=TOL))


def check_first_transition(data, traces, norm):
    from fit_world_model_go2_pilot_segments import build_startup_groups, read_segments
    from go2_startup_prepare import reset_state
    rt = json.loads((data / "manifest.json").read_text()).get("reset_torque", "recorded")
    startup = json.loads((data / "startup.json").read_text())["segments"]
    segs, _, _ = read_segments(data / "segments", 45, 12, 0, 8, 1)
    by = {name: d for name, _, d, _ in segs}
    bad = []
    for name in startup:
        tag, env, row = name[:2], int(name[6:9]), int(name[11:])
        tr, x = traces[tag], by[name]
        ok = (_close(x[0, :45], reset_state(tr, env, row, rt)) and _close(x[0, 45:57], tr["previous_action"][row, env])
              and not x[0, 57:].any() and _close(x[1, :45], tr["next_state"][row, env])
              and _close(x[1, 45:57], tr["action"][row, env]))
        if not ok:
            bad.append(name)
    t = lambda k: torch.as_tensor(norm[k])
    groups = build_startup_groups(segs, startup, 45, 12, 32, 8, "cpu", t("state_mean"), t("state_std"),
                                  t("action_mean"), t("action_std"))
    n_h1 = sum(groups[k].shape[0] for k in groups if k[0] == 1)
    # build_startup_groups walks the segments in manifest order: segment i of forecast f is window i of group (1, f)
    keep = set(startup)
    names = [name for name, _, d, _ in segs if name in keep and d.shape[0] >= 2]
    pos, w_bad = {}, 0
    for name in names:
        x = by[name]
        f = min(8, x.shape[0] - 1)
        k = pos.get(f, 0)
        pos[f] = k + 1
        got = groups[(1, f)][torch.tensor([k])][0, :2].numpy()
        want = x[:2].copy()
        want[:, :45] = (want[:, :45] - norm["state_mean"]) / norm["state_std"]
        want[:, 45:57] = (want[:, 45:57] - norm["action_mean"]) / norm["action_std"]
        w_bad += not np.allclose(got, want, rtol=1e-5, atol=1e-5)
    with np.load(data / "reset_states.npz") as z:
        rs_torque_ok = rt == "recorded" or bool(np.all(z["state"][:, 33:45] == 0.0))
    ok = not bad and n_h1 == len(names) and w_bad == 0 and rs_torque_ok
    return dict(ok=ok, reset_torque=rt, reset_states_torque_ok=rs_torque_ok, startup_segments=len(startup), mismatched_rows=bad[:10], mismatched_count=len(bad),
                history1_windows=n_h1, history1_missing=w_bad)


def check_terminal_targets(data, manifest):
    startup = set(json.loads((data / "startup.json").read_text())["segments"])
    missing, n_startup, n_standard = [], 0, 0
    for line in (data / "segments" / "manifest.csv").read_text().splitlines()[1:]:
        name, n = line.split(",")[0], int(line.split(",")[2])
        x = np.loadtxt(data / "segments" / f"{name}.csv", delimiter=",", ndmin=2, usecols=[65])
        if not x[-1] > 0.5:
            continue
        if x[:-1].sum() != 0:
            missing.append(name)
        elif n >= SEQ:
            n_standard += 1                      # the 40-row window ending at the terminal row is valid
        elif name in startup and n >= 2:
            n_startup += 1                       # window (h, n - h) for h = max(1, n - 8) .. n - 1 ends at the terminal row
        else:
            missing.append(name)
    falls = sum(s["counts"]["training_falls"] for s in manifest["sources"].values())
    no_seg = falls - (n_standard + n_startup + len(missing))
    return dict(ok=not missing and no_seg == 0, training_falls=falls, in_standard_windows=n_standard,
                in_startup_windows=n_startup, without_target=missing[:10], without_target_count=len(missing),
                falls_without_segment=no_seg)


@torch.no_grad()
def check_sampler(data, wm, norm_path, n_envs=16):
    import go2_g0_replay as rp
    from envs.base import BaseEnv
    sd, H, norm, _ = rp.load_model(wm, norm_path, "cpu")
    with np.load(data / "reset_states.npz") as z:
        raw = (np.asarray(z["state"], np.float32), np.asarray(z["previous_action"], np.float32))

    class DS:
        def normalize(self, s, a):
            return ((s - torch.as_tensor(norm["state_mean"])) / torch.as_tensor(norm["state_std"]),
                    (a - torch.as_tensor(norm["action_mean"])) / torch.as_tensor(norm["action_std"]))

        def sample_batch(self, n, normalized=True):
            g = torch.Generator().manual_seed(n)
            return torch.randn(n, H, 45, generator=g), torch.randn(n, H, 12, generator=g)

    env = BaseEnv.__new__(BaseEnv)
    env.system_dynamics, env._num_envs, env.device = sd, n_envs, "cpu"
    env.dataset, env.init_dataset, env.init_data_ratio = DS(), DS(), 0.0
    env.common_step_counter, env.warm_start_resets, env.extras = 5, True, {"log": {}}
    env.startup_reset_share, env.startup_reset_raw = 1.0, raw
    g = torch.Generator().manual_seed(0)
    sd.reset()
    sd.forward(torch.randn(n_envs, H, 45, generator=g), torch.randn(n_envs, H, 12, generator=g), None)
    ids = torch.arange(0, n_envs, 2)
    sd.reset_partial(ids)
    sh, ah = env._reset_imagination_history(ids, torch.randn(n_envs, 1, 45, generator=g), torch.randn(n_envs, 1, 12, generator=g))
    ns, na = DS().normalize(torch.as_tensor(raw[0]), torch.as_tensor(raw[1]))
    in_set = all(bool(((ns - sh[e, -1]).abs().amax(-1) < 1e-5).any()) for e in ids.tolist())
    zero = all(bool(torch.all(getattr(sd, n).memory.hidden_states[:, ids] == 0)) for n in ("state_base", "auxiliary_base"))
    a = torch.randn(n_envs, 1, 12, generator=g)
    out = sd.forward(sh, a, None)[0][ids]
    sd.reset()
    ref = sd.forward(sh[ids], a[ids], None)[0]
    same = bool(torch.allclose(out, ref, atol=1e-5))
    return dict(ok=in_set and zero and same, reset_states=int(len(raw[0])), last_row_is_a_reset_state=in_set,
                zero_memory=zero, next_step_equals_fresh_start=same)


def check_reset_rows(traces):
    import go2_round0_prepare as prep
    out, ok = {}, True
    for tag, tr in traces.items():
        stale = torque = n = 0
        jv = []
        for env, s, e, fin in prep.episodes(tr["done"]):
            if int(tr["episode_step"][s, env]) != 1:
                continue
            n += 1
            jv.append(float(np.abs(tr["state"][s, env, 21:33]).mean()))
            if s > 0:
                prev = tr["next_state"][s - 1, env]
                stale += bool(np.array_equal(tr["state"][s, env], prev))
                torque += bool(np.array_equal(tr["state"][s, env, 33:45], prev[33:45]))
        out[tag] = dict(reset_rows=n, equal_to_previous_terminal=stale, torque_unchanged=torque,
                        mean_abs_joint_vel=float(np.mean(jv)) if jv else float("nan"))
        ok &= stale == 0 and n > 0
    return dict(ok=ok, **out)


def run(data, round0, shift, wm, norm_path, out):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((data / "manifest.json").read_text())
    for tag, path in (("r0", round0), ("sh", shift)):
        if sha256(path) != manifest["sources"][tag]["sha256"]:
            raise ValueError(f"{path}: not the recording the data were prepared from")
    traces = {"r0": _load(round0), "sh": _load(shift)}
    with np.load(norm_path) as z:
        norm = {k: np.asarray(z[k], np.float32).reshape(-1) for k in ("state_mean", "state_std", "action_mean", "action_std")}
    res = dict(a_first_transition=check_first_transition(data, traces, norm),
               b_terminal_targets=check_terminal_targets(data, manifest),
               c_sampler=check_sampler(data, wm, norm_path),
               d_reset_rows=check_reset_rows(traces))
    res["pass"] = all(v["ok"] for v in res.values())
    res.update(data=str(data), manifest_sha256=sha256(data / "manifest.json"), wm_for_sampler=str(wm),
               wm_sha256=sha256(wm), normalizer_sha256=sha256(norm_path), analysis_sha256=sha256(__file__))
    (out / "prefit.json").write_text(json.dumps(res, indent=1, default=float))
    a, b, c, d = (res[k] for k in ("a_first_transition", "b_terminal_targets", "c_sampler", "d_reset_rows"))
    lines = [
        "A4 D6 pre-fit verifications (prepared data, before the fit)",
        f"(a) first transition (reset-row torque {a['reset_torque']}): {a['startup_segments']} startup segments, mismatches {a['mismatched_count']}, history-1 "
        f"windows {a['history1_windows']} (missing {a['history1_missing']}) -> {'PASS' if a['ok'] else 'FAIL'}",
        f"(b) terminal targets: {b['training_falls']} training falls: {b['in_standard_windows']} in standard windows, "
        f"{b['in_startup_windows']} in startup windows, {b['without_target_count']} without a target, "
        f"{b['falls_without_segment']} without a segment -> {'PASS' if b['ok'] else 'FAIL'}",
        f"(c) sampler: {c['reset_states']} reset states; last row a reset state {c['last_row_is_a_reset_state']}, zero "
        f"memory {c['zero_memory']}, next step = fresh start {c['next_step_equals_fresh_start']} -> "
        f"{'PASS' if c['ok'] else 'FAIL'}",
        "(d) reset rows: " + "; ".join(f"{t}: {d[t]['reset_rows']} rows, {d[t]['equal_to_previous_terminal']} equal to the "
                                       f"previous terminal state, torque unchanged {d[t]['torque_unchanged']}, mean |joint vel| "
                                       f"{d[t]['mean_abs_joint_vel']:.3f}" for t in ("r0", "sh"))
        + f" -> {'PASS' if d['ok'] else 'FAIL'}",
        f"PRE-FIT: {'PASS' if res['pass'] else 'FAIL'}"]
    (out / "prefit.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--round0", type=Path, required=True)
    p.add_argument("--shift", type=Path, required=True)
    p.add_argument("--wm", type=Path, required=True, help="any qualified Go2 model (the sampler check needs a real GRU)")
    p.add_argument("--norm", type=Path, required=True, help="D2's frozen normalizer")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    return 0 if run(a.data, a.round0, a.shift, a.wm, a.norm, a.out)["pass"] else 5


if __name__ == "__main__":
    sys.exit(main())
