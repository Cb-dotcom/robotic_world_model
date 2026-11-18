#!/usr/bin/env python3
"""Plan v6 C6: normalizer saturation. Do the GRU gates saturate on fall paths more than on walking? CPU.

For each window of a c0_battery run (same origins, same traces), the real input sequence (rows t-31 ..
t+25: normalized state and action, the model's own normalization) runs through the GRU of the state trunk
and of the auxiliary (contact / termination) trunk, layer by layer. The gate pre-activations are
recomputed from the weights (PyTorch order r, z, n):
    r = W_ir x + b_ir + W_hr h + b_hr,  z = W_iz x + b_iz + W_hz h + b_hz,  n = W_in x + b_in + r_gate * (W_hn h + b_hn)
and a unit is saturated when |pre-activation| > 3. Reported per depth d (row t + d): saturated share per
gate and layer, and the normalized input magnitude (gravity dims, all dims). Pre-fall rows = d 16..25.
Rule (plan §2): saturation association supported if saturation on pre-fall rows >= 2x walking (matched
non-fall windows, same rows; ratio CI from an episode-cluster bootstrap of matched pairs) with the ratio CI
excluding 1, and H concentrates in gravity (gravity H(25)
CI above every other component's H, from the run's c0 report). An association only; the widened-normalizer
refit is the intervention (plan §11 #7).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from c0_battery import COMPONENTS, HISTORY, HORIZON, ci, cluster_bootstrap, json_safe, load_models, pair_bootstrap
from pre_reset_transitions import gather, validate_trace

LIMIT = 3.0
PRE_FALL = slice(16, 26)


@torch.no_grad()
def gru_gate_saturation(rnn, x, limit=LIMIT):
    """Manual pass of a batch-first torch.nn.GRU. x [B, T, in].
    Returns outputs [B, T, H] of the last layer and saturated shares [layers, 3 gates, B, T]."""
    if not isinstance(rnn, torch.nn.GRU) or not rnn.batch_first:
        raise TypeError("expects a batch-first torch.nn.GRU")
    B, T, _ = x.shape
    H = rnn.hidden_size
    shares = torch.zeros(rnn.num_layers, 3, B, T)
    inp = x
    for layer in range(rnn.num_layers):
        w_ih, w_hh = getattr(rnn, f"weight_ih_l{layer}"), getattr(rnn, f"weight_hh_l{layer}")
        b_ih, b_hh = getattr(rnn, f"bias_ih_l{layer}"), getattr(rnn, f"bias_hh_l{layer}")
        h = torch.zeros(B, H)
        outs = []
        gi_all = inp @ w_ih.T + b_ih                                                     # [B, T, 3H]
        for t in range(T):
            gi, gh = gi_all[:, t], h @ w_hh.T + b_hh
            r_pre = gi[:, :H] + gh[:, :H]
            z_pre = gi[:, H:2 * H] + gh[:, H:2 * H]
            r, z = torch.sigmoid(r_pre), torch.sigmoid(z_pre)
            n_pre = gi[:, 2 * H:] + r * gh[:, 2 * H:]
            h = (1 - z) * torch.tanh(n_pre) + z * h
            outs.append(h)
            for g, pre in enumerate((r_pre, z_pre, n_pre)):
                shares[layer, g, :, t] = (pre.abs() > limit).float().mean(-1)
        inp = torch.stack(outs, 1)
    return inp, shares


def trunks(model):
    """The GRUs of the state and auxiliary trunks of a SystemDynamicsEnsemble."""
    out = {}
    for name in ("state_base", "auxiliary_base"):
        base = getattr(model, name, None)
        rnn = getattr(getattr(base, "memory", None), "rnn", None)
        if rnn is None:
            raise TypeError(f"{name} has no GRU memory (only the rnn architecture is supported)")
        out[name] = rnn
    return out


def window_inputs(contract, data, origins):
    """Normalized [state, action] rows t-31 .. t+25 for each window: [W, 57, 57]."""
    st = contract.statistics
    s = (torch.from_numpy(gather(data["state"], origins, -(HISTORY - 1), HORIZON)).float() - st["state_mean"]) / st["state_std"]
    a = (torch.from_numpy(gather(data["action"], origins, -(HISTORY - 1), HORIZON)).float() - st["action_mean"]) / st["action_std"]
    return torch.cat([s, a], -1), s


def analyse(model, contract, run_dir, traces, n_boot=10_000, seed=0):
    with np.load(Path(run_dir) / "c0_windows.npz", allow_pickle=False) as z:
        windows = {k: z[k] for k in z.files if k.endswith("/origins") or k.endswith("/clusters")}
    rnns = trunks(model)
    per = {kind: {"sat": [], "grav": [], "all": [], "clusters": [], "strata": []} for kind in ("fall", "nonfall")}
    for name, path in traces:
        with np.load(path, allow_pickle=False) as trace:
            validate_trace(trace)
            data = {k: trace[k] for k in ("state", "action")}
        for kind in ("fall", "nonfall"):
            origins = windows[f"{name}/{kind}/origins"]
            x, s = window_inputs(contract, data, origins)
            sat = []
            for trunk, rnn in rnns.items():
                _, shares = gru_gate_saturation(rnn, x)
                sat.append(shares[:, :, :, HISTORY - 1:].numpy())                    # rows t .. t+25 -> d 0..25
            per[kind]["sat"].append(np.stack(sat))                                     # [2 trunks, L, 3, W, 26]
            mag = s[:, HISTORY - 1:].abs().numpy()                                     # [W, 26, 45]
            per[kind]["grav"].append(mag[..., 6:9].mean(-1))
            per[kind]["all"].append(mag.mean(-1))
            per[kind]["clusters"].append(windows[f"{name}/{kind}/clusters"])
            per[kind]["strata"].append([name] * len(origins))
    res, boots, paired = {}, {}, {}
    for kind, v in per.items():
        sat = np.concatenate(v["sat"], axis=3)                                          # [2, L, 3, W, 26]
        grav, allm = np.concatenate(v["grav"]), np.concatenate(v["all"])                # [W, 26]
        cl, st = np.concatenate(v["clusters"]), np.concatenate(v["strata"])
        pre = sat[..., PRE_FALL].mean(-1)                                               # [2, L, 3, W]
        overall = pre.mean(axis=(1, 2))                                                 # [2 trunks, W]
        point, boot = cluster_bootstrap(np.column_stack([overall.T, grav[:, PRE_FALL].mean(1)]), cl, st, n_boot, seed)
        boots[kind] = boot
        paired[kind] = (overall.T, st, cl)
        res[kind] = dict(
            windows=int(len(cl)),
            pre_fall_saturation={"state_trunk": float(point[0]), "auxiliary_trunk": float(point[1])},
            pre_fall_gravity_input_abs_z=float(point[2]),
            by_depth={f"d{d}": dict(state_trunk=float(sat[0, ..., d].mean()), auxiliary_trunk=float(sat[1, ..., d].mean()),
                                    gravity_abs_z=float(grav[:, d].mean()), all_abs_z=float(allm[:, d].mean()))
                      for d in (0, 8, 16, 25)},
            by_gate_layer={f"{trunk}/layer{l}/{g}": float(pre[ti, l, gi].mean())
                           for ti, trunk in enumerate(("state", "auxiliary")) for l in range(sat.shape[1])
                           for gi, g in enumerate(("reset", "update", "candidate"))})
    ratio = {}
    # matched pairs resampled by episode component (review 18 Nov, two rounds), stratified by source
    pf, pn = pair_bootstrap(paired["fall"][0], paired["nonfall"][0], paired["fall"][1], paired["fall"][2],
                            paired["nonfall"][2], n_boot, seed)
    for ti, trunk in enumerate(("state_trunk", "auxiliary_trunk")):
        f, n = res["fall"]["pre_fall_saturation"][trunk], res["nonfall"]["pre_fall_saturation"][trunk]
        rb = pf[:, ti] / np.where(pn[:, ti] == 0, np.nan, pn[:, ti])
        ratio[trunk] = dict(ratio=float(f / n) if n > 0 else None, ratio_ci=ci(rb[np.isfinite(rb)]) if np.isfinite(rb).any() else None)
    res["fall_over_walking"] = ratio
    report = json.loads((Path(run_dir) / "c0_report.json").read_text())["report"]["pooled"]["fall"]["rows"]
    h = {c: (report[f"d25/{c}"]["H"], report[f"d25/{c}"]["H_ci"]) for c in COMPONENTS if c != "sum"}
    others = max(v[0] for c, v in h.items() if c != "gravity")
    res["h_components_d25"] = {c: dict(H=v[0], H_ci=v[1]) for c, v in h.items()}
    res["h_concentrates_in_gravity"] = bool(h["gravity"][1][0] > others)
    st_ratio = ratio["state_trunk"]
    met = st_ratio["ratio"] is not None and st_ratio["ratio"] >= 2 and st_ratio["ratio_ci"] is not None and st_ratio["ratio_ci"][0] > 1
    if met and res["h_concentrates_in_gravity"]:
        verdict = "supported"
    elif st_ratio["ratio_ci"] is not None and st_ratio["ratio_ci"][1] < 2:
        verdict = "not supported"
    else:
        verdict = "inconclusive"
    res["verdict"] = verdict
    res["rule"] = ("state-trunk saturation on pre-fall rows >= 2x matched walking (ratio CI > 1) AND gravity H(25) CI "
                   "above every other component's H")
    return res


def text(name, r):
    s, f, n = r["fall_over_walking"]["state_trunk"], r["fall"], r["nonfall"]
    fmt = lambda x: "n/a" if x is None else f"{x:.2f}"
    lines = [f"== {name}: C6 saturation association -> {r['verdict'].upper()}",
             f"  pre-fall saturated share (|pre| > {LIMIT}): state trunk falls {f['pre_fall_saturation']['state_trunk']:.4f} vs "
             f"walking {n['pre_fall_saturation']['state_trunk']:.4f} (ratio {fmt(s['ratio'])}, CI "
             f"{'n/a' if s['ratio_ci'] is None else '[' + ', '.join(f'{v:.2f}' for v in s['ratio_ci']) + ']'}); auxiliary trunk "
             f"{f['pre_fall_saturation']['auxiliary_trunk']:.4f} vs {n['pre_fall_saturation']['auxiliary_trunk']:.4f}",
             f"  pre-fall gravity |z|: falls {f['pre_fall_gravity_input_abs_z']:.2f} vs walking {n['pre_fall_gravity_input_abs_z']:.2f}",
             "  H(25) by component: " + ", ".join(f"{c} {v['H']:.2f}" for c, v in r["h_components_d25"].items())
             + f"; concentrates in gravity: {r['h_concentrates_in_gravity']}",
             "  falls by depth (state/aux saturation, gravity |z|): " + "; ".join(
                 f"{d} {v['state_trunk']:.4f}/{v['auxiliary_trunk']:.4f}, {v['gravity_abs_z']:.2f}" for d, v in f["by_depth"].items())]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", required=True, help="one world-model checkpoint (rnn architecture)")
    p.add_argument("--run", type=Path, required=True, help="c0_battery folder of the same model (origins, H components)")
    p.add_argument("--trace", action="append", required=True, metavar="NAME=PATH", help="the run's traces, same names")
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("output folder exists; use a fresh one")
    if "," in a.wm:
        p.error("one checkpoint only")
    traces = []
    for item in a.trace:
        name, _, path = item.partition("=")
        if not name or not path:
            p.error(f"bad --trace {item!r}")
        traces.append((name, Path(path)))
    torch.set_num_threads(4)
    model, contract, _ = load_models(a.wm)
    res = analyse(model, contract, a.run, traces, a.bootstrap, a.seed)
    a.out.mkdir(parents=True)
    (a.out / "c6_report.json").write_text(json.dumps(json_safe(dict(wm=a.wm, run=str(a.run), result=res)), indent=1, allow_nan=False))
    t = text(a.wm, res)
    (a.out / "c6_report.txt").write_text(t)
    print(t)


if __name__ == "__main__":
    main()
