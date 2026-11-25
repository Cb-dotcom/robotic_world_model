#!/usr/bin/env python3
"""C0-M (E29; draft rules 24 Nov 2026, "draft-cpu-rules-c0m-coverage" §1): how much of the registered C0 deficit
comes from the memory schedule, and how much from the state source? CPU only, no simulator, no training.

The registered C0 compares two readings of the same windows:
  u_ctx(d)   memory restarted for every target: 32 recorded rows ending at the row before the target;
  u_imag(d)  memory restarted once per window: 32 recorded rows, then d imagined states fed back (each head drives
             its own path with its own mean; recorded actions).
They differ in two ways: the states fed after d = 0 (recorded vs imagined) AND the memory schedule (32 rows vs 32 + d
rows). This script adds a third reading on the same windows and recorded actions:
  u_car(d)   memory restarted once per window, as u_imag: the 32 recorded context rows, then the RECORDED state and
             action at every step (the recorder's `state` field, the same rows the u_ctx arm reads).
u_car and u_imag share the memory schedule; u_car and u_ctx share the states. Spread convention as C0: std over
heads (ddof 1) of the head means, summed over the 45 normalized dims; per window the imagined reading is the mean
over driving heads of log u_imag.

Decomposition, per window, then the C0 mean over windows (every window weight 1, pooled over sources):
  H(d) = H_mem(d) + H_src(d),  H_mem = mean log u_ctx - mean log u_car,  H_src = mean log u_car - mean log u_imag.
Intervals: the registered episode-cluster pair bootstrap (pair components, stratified by source, seed 0, 10,000
draws); the same draws for every statistic (the draws depend only on the pair structure and the seed).

Gates, in this order, before any H_mem / H_src value is computed. A failed gate stops the run (exit 3), and no
reading is written:
  1. reproduction: c0_supplement reproduces the registered d = 25 numbers exactly from the saved spreads
     (H 1.241 / 0.476, ratio 2.61 [2.14, 3.28]; c0_supplement.check_registered, rtol 1e-9); the model and every
     trace must have the sha256 recorded in the registered c0_report.json;
  2. depth-0 parity: u_car(0) = u_ctx(0) = u_imag(h, 0) for every window and head (relative difference <= 1e-5,
     against the SAVED registered spreads);
  3. known answers: a fake model whose spread depends only on the last input gives H_mem = 0; a fake model whose
     spread depends only on the memory length gives H_src = 0 (known_answer_checks; also in the tests).

Readings (the rule text is a DRAFT until Charbel freezes it in A4 §3; --rule-record must name that record):
  primary   ANYmal, d = 25, the C0 rule applied to H_src (draft §1):
              SURVIVES       H_src,fall / H_src,nonfall >= 1.5, ratio CI lower bound > 1, H_src,fall CI lower > 0;
              NOT SUPPORTED  ratio CI upper < 1.5, or H_src,fall CI upper <= 0;
              INCONCLUSIVE   every other case.
            Checked FIRST (Charbel, 25 Nov 2026 01:02, decision 5): if the H_src,nonfall CI includes 0 the ratio has
            an unstable denominator and the verdict is INCONCLUSIVE, before any ratio rule is applied. The paired
            difference dH_src = H_src,fall - H_src,nonfall is always reported with its CI.
  secondary d = 16; per source; absolute geometric-mean disagreement of each arm; the prediction error of each arm
            (normalized units, mean over dims; base velocity in m/s); the share of H in H_src.
The registered C0 result and its files are read only; this script writes a fresh folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c0_battery as c0  # noqa: E402
import c0_supplement as sup  # noqa: E402
from pre_reset_transitions import sha256  # noqa: E402

DEPTHS = (25, 16)
PRIMARY_DEPTH = 25
PARITY_RTOL = 1e-5
DATED = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} \S")
RATIO_BAR = 1.5


class GateFailed(RuntimeError):
    pass


# ---------------------------------------------------------------- the new reading
@torch.no_grad()
def carried_pass(model, contract, data, origins, horizon=c0.HORIZON, keep_means=False):
    """u_car: memory restarted once per window; 32 recorded context rows, then the recorded state and action of
    every step. Returns spread [H, W, 45], se_norm / se_vel [H, W] (ensemble mean vs the recorded next state), and
    the head means [H, heads, W, 45] when keep_means."""
    stats = contract.statistics
    states = c0._normalized(contract, data, origins, -31, horizon, "state")       # rows t-31 .. t+H-1
    actions = c0._normalized(contract, data, origins, -31, horizon, "action")
    target = (torch.from_numpy(c0.gather(data["next_state"], origins, 0, horizon)).float()
              - stats["state_mean"]) / stats["state_std"]                          # [W, H, 45]
    model.reset()
    xs, xa = states[:, :c0.HISTORY], actions[:, :c0.HISTORY]
    spread, se_n, se_v, kept = [], [], [], []
    for k in range(horizon):
        means, _, _, _ = c0.forward_all(model, xs, xa)
        spread.append(c0.head_spread(means).numpy())
        n_err, v_err = c0._errors(means.mean(0), target[:, k], stats)
        se_n.append(n_err)
        se_v.append(v_err)
        if keep_means:
            kept.append(means.numpy())
        if k + 1 < horizon:                     # the recorded state of the row the imagined arm would feed
            xs = states[:, c0.HISTORY + k:c0.HISTORY + k + 1]
            xa = actions[:, c0.HISTORY + k:c0.HISTORY + k + 1]
    model.reset()
    out = dict(spread=np.stack(spread), se_norm=np.stack(se_n), se_vel=np.stack(se_v))
    if keep_means:
        out["means"] = np.stack(kept)
    return out


def window_logs(u_ctx, u_car, u_imag):
    """float64 per-window logs of the summed spreads at every depth: log u_ctx [H, W], log u_car [H, W], mean over
    driving heads of log u_imag [H, W]."""
    a = u_ctx.astype(np.float64).sum(-1)
    b = u_car.astype(np.float64).sum(-1)
    c = u_imag.astype(np.float64).sum(-1)
    if (a <= 0).any() or (b <= 0).any() or (c <= 0).any():
        raise c0.ZeroDisagreement("zero summed spread in a C0-M reading")
    return np.log(a), np.log(b), np.log(c).mean(0)


def depth0_parity(u_ctx, u_car, u_imag):
    """Largest relative difference at d = 0 between the three readings (summed spread)."""
    a = u_ctx[0].astype(np.float64).sum(-1)
    b = u_car[0].astype(np.float64).sum(-1)
    c = u_imag[:, 0].astype(np.float64).sum(-1)
    return float(max(np.max(np.abs(b - a) / a), np.max(np.abs(c - a[None]) / a[None])))


# ---------------------------------------------------------------- statistics
def decompose(logs, strata, clusters, n_boot=10_000, seed=0, depths=DEPTHS):
    """logs: {kind: (log_ctx, log_car, log_imag), each [H, W]}; strata [W]; clusters {kind: [W]}.
    Returns per depth: point values and CIs of H, H_mem, H_src for falls and non-falls; paired differences; the ratio
    H_src,fall / H_src,nonfall with CI; the share of H in H_src."""
    res = {}
    for d in depths:
        vals = {k: np.stack([logs[k][0][d], logs[k][1][d], logs[k][2][d]], axis=1) for k in ("fall", "nonfall")}
        bf, bn = c0.pair_bootstrap(vals["fall"], vals["nonfall"], strata, clusters["fall"], clusters["nonfall"],
                                   n_boot, seed)
        row = {}
        for kind, b in (("fall", bf), ("nonfall", bn)):
            p = vals[kind].mean(0)
            h, hm, hs = p[0] - p[2], p[0] - p[1], p[1] - p[2]
            row[kind] = dict(windows=int(len(vals[kind])), H=float(h), H_ci=c0.ci(b[:, 0] - b[:, 2]),
                             H_mem=float(hm), H_mem_ci=c0.ci(b[:, 0] - b[:, 1]),
                             H_src=float(hs), H_src_ci=c0.ci(b[:, 1] - b[:, 2]),
                             share_src=float(hs / h) if h != 0 else float("nan"),
                             geomean_u_ctx=float(np.exp(p[0])), geomean_u_car=float(np.exp(p[1])),
                             geomean_u_imag=float(np.exp(p[2])), identity_error=float(h - (hm + hs)))
        f_src, n_src = bf[:, 1] - bf[:, 2], bn[:, 1] - bn[:, 2]
        f_mem, n_mem = bf[:, 0] - bf[:, 1], bn[:, 0] - bn[:, 1]
        hsf, hsn = row["fall"]["H_src"], row["nonfall"]["H_src"]
        with np.errstate(divide="ignore", invalid="ignore"):
            rb = f_src / n_src
        row["src_ratio"] = dict(ratio=float(hsf / hsn) if hsn != 0 else float("nan"),
                                ratio_ci=c0.ci(rb[np.isfinite(rb)]) if np.isfinite(rb).any() else [float("nan")] * 2,
                                nonfinite_draws=int((~np.isfinite(rb)).sum()))
        row["difference"] = dict(dH_src=float(hsf - hsn), dH_src_ci=c0.ci(f_src - n_src),
                                 dH_mem=float(row["fall"]["H_mem"] - row["nonfall"]["H_mem"]),
                                 dH_mem_ci=c0.ci(f_mem - n_mem),
                                 dH=float(row["fall"]["H"] - row["nonfall"]["H"]))
        lo, hi = row["nonfall"]["H_src_ci"]
        row["unstable_denominator"] = bool(lo <= 0 <= hi)
        res[f"d{d}"] = row
    return res


def src_verdict(row):
    """The C0 rule on H_src (draft §1 with decision 5 of 25 Nov 2026): an unstable denominator (the H_src,nonfall CI
    includes 0) gives INCONCLUSIVE before any ratio rule is read."""
    if row["unstable_denominator"]:
        return "INCONCLUSIVE"
    r, (rlo, rhi) = row["src_ratio"]["ratio"], row["src_ratio"]["ratio_ci"]
    flo, fhi = row["fall"]["H_src_ci"]
    if rhi < RATIO_BAR or fhi <= 0:
        return "NOT SUPPORTED"
    if r >= RATIO_BAR and rlo > 1 and flo > 0:
        return "SURVIVES"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------- known-answer models (gate 3 and the tests)
class _KAHead(torch.nn.Module):
    def __init__(self, h, fn):
        super().__init__()
        self.h, self.fn = h, fn

    def forward(self, base, xs):
        mean = self.fn(base, self.h)
        return mean, torch.ones_like(mean)


class _KAAux(torch.nn.Module):
    def forward(self, aux, xs):
        B, T = xs.shape[:2]
        return None, torch.zeros(B, T, 8), torch.full((B, T, 1), -5.0)


class KnownAnswerModel:
    """A stand-in for SystemDynamicsEnsemble with the attributes c0_battery._forward_one reads.
    kind "last_input": head h's mean = (1 + 0.1 h) x the last input state -> the spread depends only on the last input.
    kind "memory_length": head h's mean = (1 + 0.1 h) x log(1 + n) in every dim, n = rows read since reset -> the
    spread depends only on the memory length."""

    prediction_type = "sequence"

    def __init__(self, kind, heads=5):
        if kind not in ("last_input", "memory_length"):
            raise ValueError(kind)
        self.kind, self.ensemble_size, self.n = kind, heads, 0
        if kind == "last_input":
            fn = lambda base, h: base * (1.0 + 0.1 * h) + 0.01 * h
        else:
            fn = lambda base, h: base * (1.0 + 0.1 * h)
        self.state_heads = [_KAHead(h, fn) for h in range(heads)]
        self.auxiliary_heads = [_KAAux() for _ in range(heads)]

    def reset(self):
        self.n = 0

    def state_base(self, xs, xa):
        B, T, D = xs.shape
        if self.kind == "last_input":
            return xs.clone()
        pos = torch.arange(self.n + 1, self.n + T + 1, dtype=torch.float32)
        self.n += T
        return torch.log1p(pos).view(1, T, 1).expand(B, T, D).clone()

    def auxiliary_base(self, xs, xa):
        return xs


def _ka_data(seed=0, T=120, N=3):
    g = np.random.default_rng(seed)
    data = dict(state=g.normal(0, 1, (T, N, 45)).astype(np.float32), action=g.normal(0, 1, (T, N, 12)).astype(np.float32))
    data["next_state"] = np.concatenate([data["state"][1:], g.normal(0, 1, (1, N, 45)).astype(np.float32)])
    data["state"] = np.abs(data["state"]) + 0.5                # positive spread for the last-input model
    data["next_state"] = np.abs(data["next_state"]) + 0.5
    origins = np.array([(40, 0), (50, 1), (60, 2), (70, 0)], dtype=int)
    stats = dict(state_mean=torch.zeros(45), state_std=torch.ones(45), action_mean=torch.zeros(12),
                 action_std=torch.ones(12))
    return data, origins, SimpleNamespace(statistics=stats)


def known_answer_checks(horizon=c0.HORIZON, tol=1e-6):
    """Gate 3. Returns {"last_input": max |H_mem(d)|, "memory_length": max |H_src(d)|, "passed": bool} with the
    other component reported, so a test that passes trivially (everything zero) is visible."""
    data, origins, contract = _ka_data()
    out = {}
    for kind in ("last_input", "memory_length"):
        m = KnownAnswerModel(kind)
        u_ctx = c0.real_context_pass(m, contract, data, origins, horizon)["spread"]
        u_car = carried_pass(m, contract, data, origins, horizon)["spread"]
        u_imag = c0.imagined_pass(m, contract, data, origins, horizon)["spread"]
        a, b, c = window_logs(u_ctx, u_car, u_imag)
        out[kind] = dict(max_abs_H_mem=float(np.abs((a - b).mean(1)).max()),
                         max_abs_H_src=float(np.abs((b - c).mean(1)).max()),
                         depth0_parity=depth0_parity(u_ctx, u_car, u_imag))
    out["passed"] = bool(out["last_input"]["max_abs_H_mem"] <= tol and out["memory_length"]["max_abs_H_src"] <= tol
                         and out["last_input"]["max_abs_H_src"] > 1e-3 and out["memory_length"]["max_abs_H_mem"] > 1e-3)
    return out


# ---------------------------------------------------------------- run on the registered windows
def _registered(report_path):
    rep = json.loads(Path(report_path).read_text())
    meta = rep["meta"]
    if meta.get("feedback", "head") != "head" or meta.get("actions", "recorded") != "recorded" \
            or int(meta.get("imag_context", c0.HISTORY)) != c0.HISTORY or int(meta.get("horizon", c0.HORIZON)) != c0.HORIZON:
        raise GateFailed("the report is not a registered-protocol C0 run (head feedback, recorded actions, 32-row "
                         "context, horizon 26)")
    return rep


def run(windows, report, out, wm=None, trace_overrides=None, n_boot=10_000, seed=0, rule_record=None):
    """windows / report: the registered c0_windows.npz and c0_report.json. wm: the model (default: the report's).
    trace_overrides: {source: path} when the traces moved (the sha256 must still match)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rep = _registered(report)
    meta = rep["meta"]
    gates = {}
    # gate 1: the registered numbers from the saved spreads, and the inputs by hash
    sources = sup.load_sources(windows)
    chk = sup.check_registered(sup.analyse(sources, n_boot, seed), report)
    gates["reproduction"] = chk
    wm_paths = [Path(p) for p in (str(wm).split(",") if wm else meta["wm"])]
    wm_sha = [sha256(p) for p in wm_paths]
    gates["model_hash"] = dict(expected=meta["wm_sha256"], found=wm_sha, ok=wm_sha == list(meta["wm_sha256"]))
    traces = {}
    for name, info in meta["inputs"].items():
        path = Path((trace_overrides or {}).get(name, info["path"]))
        traces[name] = path
        gates.setdefault("trace_hashes", {})[name] = dict(expected=info["sha256"], found=sha256(path),
                                                          ok=sha256(path) == info["sha256"])
    hash_ok = gates["model_hash"]["ok"] and all(v["ok"] for v in gates["trace_hashes"].values())
    if not chk["reproduced"] or not hash_ok:
        _stop(out, gates, "gate 1 (reproduction of the registered d25 numbers, input hashes)")
    # gate 3 first: it needs no data
    gates["known_answers"] = known_answer_checks()
    if not gates["known_answers"]["passed"]:
        _stop(out, gates, "gate 3 (known-answer models)")
    # the carried reading on the saved windows
    model, contract, _ = c0.load_models(",".join(str(p) for p in wm_paths))
    for key in ("state_mean", "state_std"):
        with np.load(windows, allow_pickle=False) as z:
            saved = z[f"meta/{key}"]
        if not np.array_equal(saved, contract.statistics[key].numpy()):
            _stop(out, gates, f"gate 1 (the model's normalizer {key} differs from the registered run's)")
    logs, clusters, strata, parity, errors = {}, {"fall": [], "nonfall": []}, [], {}, {}
    arrays = {}
    with np.load(windows, allow_pickle=False) as z:
        per = {k: [] for k in ("fall", "nonfall")}
        for name in sources:
            with np.load(traces[name], allow_pickle=False) as tz:
                data = {k: tz[k] for k in ("state", "action", "next_state")}
            for kind in ("fall", "nonfall"):
                origins = z[f"{name}/{kind}/origins"]
                u_ctx, u_imag = z[f"{name}/{kind}/u_real"], z[f"{name}/{kind}/u_imag"]
                car = carried_pass(model, contract, data, origins)
                parity[f"{name}/{kind}"] = depth0_parity(u_ctx, car["spread"], u_imag)
                join = np.abs(c0.gather(data["state"], origins, 1, c0.HORIZON)
                              - c0.gather(data["next_state"], origins, 0, c0.HORIZON - 1)).max()
                errors[f"{name}/{kind}"] = dict(
                    carried_norm=car["se_norm"].mean(1).tolist(), carried_vel=car["se_vel"].mean(1).tolist(),
                    ctx_norm=z[f"{name}/{kind}/se_real_norm"].mean(1).tolist(),
                    ctx_vel=z[f"{name}/{kind}/se_real_vel"].mean(1).tolist(),
                    imag_norm=z[f"{name}/{kind}/se_imag_norm"].mean((0, 2)).tolist(),
                    imag_vel=z[f"{name}/{kind}/se_imag_vel"].mean((0, 2)).tolist(),
                    state_join_max_abs=float(join))
                per[kind].append(window_logs(u_ctx, car["spread"], u_imag))
                clusters[kind].append(z[f"{name}/{kind}/clusters"].astype(str))
                arrays[f"{name}/{kind}/u_car"] = car["spread"].astype(np.float32)
                arrays[f"{name}/{kind}/se_car_norm"] = car["se_norm"]
                arrays[f"{name}/{kind}/se_car_vel"] = car["se_vel"]
            strata += [name] * len(z[f"{name}/fall/origins"])
    gates["depth0_parity"] = dict(max_rel=max(parity.values()), per=parity, rtol=PARITY_RTOL,
                                  ok=max(parity.values()) <= PARITY_RTOL)
    with (out / "c0m_windows.npz").open("xb") as f:
        np.savez_compressed(f, **arrays)
    if not gates["depth0_parity"]["ok"]:
        _stop(out, gates, "gate 2 (depth-0 parity)")
    # readings: only after every gate passed
    for kind in ("fall", "nonfall"):
        logs[kind] = tuple(np.concatenate([p[i] for p in per[kind]], axis=1) for i in range(3))
        clusters[kind] = np.concatenate(clusters[kind])
    strata = np.asarray(strata)
    pooled = decompose(logs, strata, clusters, n_boot, seed)
    by_source, off = {}, 0
    for name in sources:
        w = len(sources[name]["fall"][2])
        sl = slice(off, off + w)
        by_source[name] = decompose({k: tuple(x[:, sl] for x in logs[k]) for k in logs}, strata[sl],
                                    {k: clusters[k][sl] for k in clusters}, n_boot, seed)
        off += w
    reading = None
    if rule_record is not None:
        reading = dict(endpoint=f"d{PRIMARY_DEPTH}", rule_record=rule_record,
                       verdict=src_verdict(pooled[f"d{PRIMARY_DEPTH}"]),
                       paired_dH_src=pooled[f"d{PRIMARY_DEPTH}"]["difference"]["dH_src"],
                       paired_dH_src_ci=pooled[f"d{PRIMARY_DEPTH}"]["difference"]["dH_src_ci"])
    payload = dict(protocol="C0-M (E29): memory schedule held fixed; draft rules 24 Nov 2026", gates=gates,
                   pooled=pooled, by_source=by_source, errors_by_depth=errors, reading=reading,
                   inputs=dict(windows=str(windows), windows_sha256=sha256(windows), report=str(report),
                               report_sha256=sha256(report), wm=[str(p) for p in wm_paths], wm_sha256=wm_sha,
                               traces={k: str(v) for k, v in traces.items()}),
                   n_boot=n_boot, seed=seed, analysis_sha256=sha256(__file__))
    (out / "c0m.json").write_text(json.dumps(c0.json_safe(payload), indent=1))
    text = text_report(payload)
    (out / "c0m.txt").write_text(text)
    print(text, flush=True)
    return payload


def _stop(out, gates, what):
    (out / "c0m_gates.json").write_text(json.dumps(c0.json_safe(gates), indent=1))
    raise GateFailed(f"{what} failed; nothing was read (gates in {out / 'c0m_gates.json'})")


def text_report(p):
    g = p["gates"]
    lines = ["C0-M: memory schedule held fixed (E29). Retrospective: these windows were seen before (exploratory).",
             f"gates: reproduction {g['reproduction']['reproduced']} (max rel {g['reproduction']['max_rel_diff']:.1e}); "
             f"inputs by sha256 {g['model_hash']['ok'] and all(v['ok'] for v in g['trace_hashes'].values())}; "
             f"depth-0 parity max rel {g['depth0_parity']['max_rel']:.1e} (<= {PARITY_RTOL:g}); known answers "
             f"{g['known_answers']['passed']}"]
    for d in DEPTHS:
        r = p["pooled"][f"d{d}"]
        lines.append(f"d{d}{' (primary)' if d == PRIMARY_DEPTH else ' (secondary)'}:")
        for kind in ("fall", "nonfall"):
            x = r[kind]
            lines.append(f"  {kind:7s} ({x['windows']} windows): H {x['H']:.3f} {_ci(x['H_ci'])} = H_mem {x['H_mem']:.3f} "
                         f"{_ci(x['H_mem_ci'])} + H_src {x['H_src']:.3f} {_ci(x['H_src_ci'])}; share in H_src "
                         f"{x['share_src']:.2f}; geometric means u_ctx {x['geomean_u_ctx']:.3f}, u_car "
                         f"{x['geomean_u_car']:.3f}, u_imag {x['geomean_u_imag']:.3f}")
        s = r["src_ratio"]
        lines.append(f"  H_src fall / non-fall {s['ratio']:.3f} {_ci(s['ratio_ci'])}"
                     f"{'  UNSTABLE DENOMINATOR (non-fall H_src CI includes 0)' if r['unstable_denominator'] else ''}")
        df = r["difference"]
        lines.append(f"  paired dH_src {df['dH_src']:.3f} {_ci(df['dH_src_ci'])}; dH_mem {df['dH_mem']:.3f} "
                     f"{_ci(df['dH_mem_ci'])}; dH {df['dH']:.3f}")
    if p["reading"]:
        lines.append(f"READING (d{PRIMARY_DEPTH}, record {p['reading']['rule_record']}): {p['reading']['verdict']}; "
                     f"paired dH_src {p['reading']['paired_dH_src']:.3f} {_ci(p['reading']['paired_dH_src_ci'])}")
    else:
        lines.append("READING: not made (no frozen rule record given); components only")
    return "\n".join(lines) + "\n"


def _ci(x):
    return f"[{x[0]:.3f}, {x[1]:.3f}]"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", type=Path, required=True, help="registered c0_windows.npz (rerun 2, 18 Nov)")
    p.add_argument("--report", type=Path, required=True, help="its c0_report.json")
    p.add_argument("--out", type=Path, required=True, help="fresh folder")
    p.add_argument("--wm", default=None, help="model path(s); default: the paths in the report (hash-checked)")
    p.add_argument("--trace", action="append", default=[], metavar="SOURCE=PATH", help="moved trace (hash-checked)")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rule-record", default=None,
                   help="the dated A4 §3 / decisions-log entry that froze the rule, e.g. '2026-11-25 01:02 CPU rules'")
    a = p.parse_args(argv)
    if a.rule_record is not None and not DATED.match(a.rule_record):
        p.error("--rule-record must start with the record's date and time: 'YYYY-MM-DD HH:MM ...'")
    over = dict(s.split("=", 1) for s in a.trace)
    try:
        run(a.windows, a.report, a.out, a.wm, over, a.n_boot, a.seed, a.rule_record)
    except GateFailed as e:
        print(f"[c0m] STOP: {e}", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
