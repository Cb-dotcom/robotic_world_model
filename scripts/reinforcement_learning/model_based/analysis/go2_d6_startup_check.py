#!/usr/bin/env python3
"""A4 D6 Q4: the startup check on fresh episodes (22 Nov 2026, Charbel's decision 20:47). Read-only, CPU.
Fixed before the D6 model exists.

Episodes: finished episodes that began at a physical reset (recorded episode_step == 1 at the first row) of
  - s2's probe 913 (0.30 friction; never fitted) and the round-0 withheld episodes (nominal; never fitted): Q4 rule;
  - optional descriptive recordings (D3's probe 915): reported, no rule.
Start: the recorder's pre-action reset state, ZERO memory, the recorded actions (the convention of D6's startup windows
and startup resets). Step m (1-based) predicts episode row m - 1, through 40 steps or the terminal transition.
Two arms on the same episodes:
  free      head h drives its own copy with its own predicted state (as imagination); head h's stop output on copy h
  recorded  the recorded state is fed at every step (memory from the reset); every head reads the same input
Measured:
  fidelity     pooled RMSE over (episode, head) of the base-velocity xy error (m/s) and the tilt error (deg) at step 25
               (free arm; episodes with >= 25 rows), as D2's Q1b; step 10 reported
  false stops  share of (time-out episode, head) pairs with a stop (p > 0.5) at steps 1..40 (free arm; recorded arm
               reported)
  startup falls  falls with <= 25 rows (the terminal transition at step <= 25), scored through that transition:
               caught = a stop at one of the last 11 transitions; first stop: the earliest stop at any step, reported as
               steps before the terminal one, and the share that stopped before the last 11 transitions; falls with
               26..40 rows reported the same way
Rule (Q4): on each rule source, velocity RMSE at step 25 <= 0.25 m/s, tilt RMSE at step 25 <= 5 deg and false stops
<= 0.10, each on >= 20 episodes (fewer -> the source cannot pass). Q4 passes if both rule sources pass.
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

import go2_g0_replay as rp  # noqa: E402
from go2_d2_qualify import GRAV, VXY, load, tilt_deg  # noqa: E402
from pre_reset_transitions import sha256  # noqa: E402

K, FID_STEP, LAST, MIN_N = 40, 25, 11, 20
BARS = dict(vxy=0.25, tilt=5.0, false_stop=0.10)


class StartEp:
    def __init__(self, trace, n, s, e, kind, norm):
        nz = lambda x, m, sd: ((x - norm[m]) / norm[sd]).astype(np.float32)
        self.R = nz(trace["state"][s, n], "state_mean", "state_std")
        self.S = nz(trace["next_state"][s:e + 1, n], "state_mean", "state_std")
        self.A = nz(trace["action"][s:e + 1, n], "action_mean", "action_std")
        self.raw = trace["next_state"][s:e + 1, n].astype(np.float32)
        self.L, self.kind, self.env, self.first = e - s + 1, kind, n, s


def start_episodes(trace, norm, only=None):
    """Finished episodes that began at a physical reset; `only` = {(env, first)} restricts to a withheld list."""
    import go2_round0_prepare as prep
    out = []
    for n, s, e, fin in prep.episodes(trace["done"]):
        if not fin or int(trace["episode_step"][s, n]) != 1 or (only is not None and (n, s) not in only):
            continue
        kind = "fall" if (trace["terminated"][e, n] and not trace["time_out"][e, n]) else "time_out"
        out.append(StartEp(trace, n, s, e, kind, norm))
    return out


@torch.no_grad()
def startup_rollouts(sd, eps, device="cpu", steps=K):
    """Both arms from the reset state with zero memory. Returns p_free [n, E, steps], p_rec [n, E, steps],
    pred_free [n, E, steps, 45] (normalized; rows beyond an episode's end are padding and must be ignored)."""
    E, n = len(sd.state_heads), len(eps)
    A = np.zeros((n, steps, 12), np.float32)
    S = np.zeros((n, steps, 45), np.float32)
    for i, ep in enumerate(eps):
        m = min(ep.L, steps)
        A[i, :m], S[i, :m] = ep.A[:m], ep.S[:m]
    R = np.stack([ep.R for ep in eps])[:, None]
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    p_free, p_rec = np.zeros((n, E, steps)), np.zeros((n, E, steps))
    pred_free = np.zeros((n, E, steps, 45), np.float32)
    # recorded arm
    sd.reset()
    xs, xa = T(R), T(A[:, 0:1])
    for m in range(steps):
        _, logits = rp.all_heads(sd, xs, xa)
        p_rec[:, :, m] = torch.sigmoid(logits).cpu().numpy().T
        if m + 1 < steps:
            xs, xa = T(S[:, m:m + 1]), T(A[:, m + 1:m + 2])
    # free arm: head h drives copy h
    sd.reset()
    rep = lambda a: np.repeat(a, E, axis=0)
    drive, cols = torch.arange(E, device=device).repeat(n), torch.arange(n * E, device=device)
    xs, xa = T(rep(R)), T(rep(A[:, 0:1]))
    for m in range(steps):
        means, logits = rp.all_heads(sd, xs, xa)
        own = means[drive, cols]
        p_free[:, :, m] = torch.sigmoid(logits[drive, cols]).view(n, E).cpu().numpy()
        pred_free[:, :, m] = own.view(n, E, -1).cpu().numpy()
        if m + 1 < steps:
            xs, xa = own.unsqueeze(1), T(rep(A[:, m + 1:m + 2]))
    sd.reset()
    return p_free, p_rec, pred_free


def fidelity(pred, eps, step, norm):
    idx = [i for i, ep in enumerate(eps) if ep.L >= step]
    if not idx:
        return dict(n=0, vxy=float("nan"), tilt=float("nan"))
    p = pred[idx, :, step - 1] * norm["state_std"] + norm["state_mean"]
    t = np.stack([eps[i].raw[step - 1] for i in idx])[:, None]
    v = float(np.sqrt(np.mean(np.sum((p[..., VXY] - t[..., VXY]) ** 2, -1))))
    a = float(np.sqrt(np.mean((tilt_deg(p[..., GRAV]) - tilt_deg(t[..., GRAV])) ** 2)))
    return dict(n=len(idx), vxy=v, tilt=a)


def fall_scores(p, eps, lo, hi):
    """Falls with lo < L <= hi: caught (last 11 transitions), first stop lead, early share; per (fall, head)."""
    idx = [i for i, ep in enumerate(eps) if ep.kind == "fall" and lo < ep.L <= hi]
    caught, early, leads = [], [], []
    for i in idx:
        L = eps[i].L
        st = p[i, :, :L] > 0.5                                     # [E, L]
        caught.append(st[:, max(0, L - LAST):L].any(-1))
        first = np.where(st.any(-1), st.argmax(-1), -1)
        for f in first:
            if f >= 0:
                leads.append(L - 1 - int(f))
        early.append((first >= 0) & (first < L - LAST))
    if not idx:
        return dict(n=0, caught=float("nan"), stopped_before_last_11=float("nan"), median_lead=float("nan"))
    return dict(n=len(idx), caught=float(np.mean(caught)), stopped_before_last_11=float(np.mean(early)),
                median_lead=float(np.median(leads)) if leads else float("nan"), leads=leads)


def false_stops(p, eps):
    idx = [i for i, ep in enumerate(eps) if ep.kind == "time_out" and ep.L >= K]
    if not idx:
        return dict(n=0, share=float("nan"))
    return dict(n=len(idx), share=float(np.mean((p[idx] > 0.5).any(-1))))


def score(sd, eps, norm, device):
    pf, pr, pred = startup_rollouts(sd, eps, device)
    return dict(episodes=len(eps), falls=sum(e.kind == "fall" for e in eps),
                fidelity_step25=fidelity(pred, eps, FID_STEP, norm), fidelity_step10=fidelity(pred, eps, 10, norm),
                false_stop_free=false_stops(pf, eps), false_stop_recorded=false_stops(pr, eps),
                startup_falls_free=fall_scores(pf, eps, 0, FID_STEP), startup_falls_recorded=fall_scores(pr, eps, 0, FID_STEP),
                falls_26_40_free=fall_scores(pf, eps, FID_STEP, K), falls_26_40_recorded=fall_scores(pr, eps, FID_STEP, K))


def verdict(r):
    f, s = r["fidelity_step25"], r["false_stop_free"]
    enough = f["n"] >= MIN_N and s["n"] >= MIN_N
    ok = dict(vxy=f["vxy"] <= BARS["vxy"], tilt=f["tilt"] <= BARS["tilt"], false_stop=s["share"] <= BARS["false_stop"],
              enough_episodes=enough)
    return dict(**{k: bool(v) for k, v in ok.items()}, passed=bool(all(ok.values())))


def run(wm, norm_path, rule_sources, descriptive, out, device="cpu"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    torch.set_num_threads(8)
    sd, _, norm, diff = rp.load_model(wm, norm_path, device)
    res, meta = {}, {}
    for name, (path, only) in list(rule_sources.items()) + list(descriptive.items()):
        tr, m = load(path)
        eps = start_episodes(tr, norm, only)
        res[name] = score(sd, eps, norm, device)
        if name in rule_sources:
            res[name]["verdict"] = verdict(res[name])
        meta[name] = dict(path=str(path), sha256=sha256(path), friction=(m.get("friction_shift") or {}).get("factor"),
                          restricted_to_withheld=only is not None)
    q4 = all(res[n]["verdict"]["passed"] for n in rule_sources)
    rep = dict(protocol="A4 D6 Q4 startup check (rules fixed before the D6 model existed)", wm=str(wm), wm_sha256=sha256(wm),
               normalizer_sha256=sha256(norm_path), normalizer_max_diff=diff, sources=meta, results=res, bars=BARS,
               min_episodes=MIN_N, q4_pass=q4, analysis_sha256=sha256(__file__), wall_s=time.time() - t0)
    (out / "startup_check.json").write_text(json.dumps(rep, indent=1, default=float))
    t = text(rep)
    (out / "startup_check.txt").write_text(t)
    print(t, end="", flush=True)
    return rep


def text(rep):
    L = [f"A4 D6 Q4 startup check (from the physical reset, zero memory, recorded actions); model {rep['wm_sha256'][:12]}"]
    for name, r in rep["results"].items():
        f25, f10 = r["fidelity_step25"], r["fidelity_step10"]
        L.append(f"[{name}] {r['episodes']} episodes from a reset ({r['falls']} falls)")
        L.append(f"  free rollout: step 25 vxy RMSE {f25['vxy']:.3f} m/s, tilt RMSE {f25['tilt']:.2f} deg (n {f25['n']}); "
                 f"step 10 {f10['vxy']:.3f} m/s, {f10['tilt']:.2f} deg")
        L.append(f"  false stops in 40 steps: free {r['false_stop_free']['share']:.3f}, recorded "
                 f"{r['false_stop_recorded']['share']:.3f} (n {r['false_stop_free']['n']})")
        for key, lab in (("startup_falls", "falls <= 25 rows"), ("falls_26_40", "falls 26-40 rows")):
            a, b = r[f"{key}_free"], r[f"{key}_recorded"]
            L.append(f"  {lab}: n {a['n']}; caught free {a['caught']:.3f}, recorded {b['caught']:.3f}; stopped before the "
                     f"last 11 free {a['stopped_before_last_11']:.3f}, recorded {b['stopped_before_last_11']:.3f}; median "
                     f"first-stop lead free {a['median_lead']}, recorded {b['median_lead']} steps")
        if "verdict" in r:
            v = r["verdict"]
            L.append(f"  Q4 {name}: vxy {v['vxy']}, tilt {v['tilt']}, false stops {v['false_stop']}, enough episodes "
                     f"{v['enough_episodes']} -> {'PASS' if v['passed'] else 'FAIL'}")
    L.append(f"Q4: {'PASS' if rep['q4_pass'] else 'FAIL'}")
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", type=Path, required=True)
    p.add_argument("--norm", type=Path, required=True)
    p.add_argument("--probe", action="append", default=[], help="rule source name=path (all its reset episodes)")
    p.add_argument("--withheld", action="append", default=[],
                   help="rule source name=trace:withheld.json (only the listed withheld episodes)")
    p.add_argument("--descriptive", action="append", default=[], help="name=path, reported without a rule")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    rule = {s.split("=", 1)[0]: (Path(s.split("=", 1)[1]), None) for s in a.probe}
    for s in a.withheld:
        name, rest = s.split("=", 1)
        trace, held = rest.split(":", 1)
        only = {(h["env"], h["first"]) for h in json.loads(Path(held).read_text())}
        rule[name] = (Path(trace), only)
    desc = {s.split("=", 1)[0]: (Path(s.split("=", 1)[1]), None) for s in a.descriptive}
    if not rule:
        raise SystemExit("at least one rule source")
    return 0 if run(a.wm, a.norm, rule, desc, a.out)["q4_pass"] else 7


if __name__ == "__main__":
    sys.exit(main())
