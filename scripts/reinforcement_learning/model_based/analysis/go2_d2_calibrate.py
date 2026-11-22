#!/usr/bin/env python3
"""A4 D2 option B (Charbel, 22 Nov 2026): recalibrate only the stop output of the D2 model, then re-qualify once.

Calibration data: the E22 withheld episodes of the D1 fitting batch (0.30 friction; never fitted; not the probe).
For one constant c added to every head's termination logit (stop when logit + c > 0, i.e. p > 0.5):
  detection(c)  = share of (validation fall, head) pairs with a stop in the last 11 rows (memory from the episode start);
  false_stop(c) = share of (validation surviving episode, head) pairs with a stop on any row.
Rule (fixed before the calibration data are read; A4 §3): c = the smallest value >= 0 with detection(c) >= 0.5.
If false_stop(c) > 0.10 at that c, no valid c exists and D2 closes as FAIL. Otherwise the calibrated model (bias of
each head's last termination layer + c) is written and Q1-Q3 are run ONCE on it with the unchanged qualifier.
Both shares only grow with c, so the smallest c that reaches the detection bar gives the fewest false stops.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MB = HERE.parent
for p in (str(HERE), str(MB)):
    if p not in sys.path:
        sys.path.insert(0, p)

import go2_g0_replay as rp  # noqa: E402
from pre_reset_transitions import sha256, validate_trace  # noqa: E402

DETECT, FALSE_STOP, LAST = 0.5, 0.10, 11


def need(logits_rows):
    """Smallest offset that makes a head fire on any of these rows: fire if max(logit) + c > 0."""
    m = np.nanmax(logits_rows) if np.isfinite(logits_rows).any() else -np.inf
    return -m


def shares(fall_need, surv_need, c):
    """fall_need [falls, E], surv_need [surviving, E]: c needed per pair; a pair fires if c > need."""
    det = float((c > fall_need).mean()) if fall_need.size else float("nan")
    fs = float((c > surv_need).mean()) if surv_need.size else float("nan")
    return det, fs


def choose(fall_need, surv_need):
    """The rule: smallest c >= 0 with detection >= 0.5; valid only if false stops <= 0.10 there."""
    det0, fs0 = shares(fall_need, surv_need, 0.0)
    if fall_need.size == 0 or surv_need.size == 0:
        return dict(offset=float("nan"), detection=det0, false_stop=fs0, detection_at_0=det0, false_stop_at_0=fs0,
                    valid=False, note="no validation falls or no surviving episodes")
    if det0 >= DETECT:
        c = 0.0
    else:
        k = int(np.ceil(DETECT * fall_need.size))          # pairs that must fire
        c = max(0.0, float(np.sort(fall_need.reshape(-1))[k - 1]) + 1e-3)   # margin for float32 rounding of bias + c
    det, fs = shares(fall_need, surv_need, c)
    return dict(offset=c, detection=det, false_stop=fs, detection_at_0=det0, false_stop_at_0=fs0,
                valid=bool(det >= DETECT and fs <= FALSE_STOP))


def apply_offset(ckpt_in, ckpt_out, c):
    ck = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    sdict = ck["system_dynamics_state_dict"]
    last = {}
    for k in sdict:
        m = re.fullmatch(r"auxiliary_heads\.(\d+)\.termination_layers\.(\d+)\.bias", k)
        if m:
            h, i = int(m.group(1)), int(m.group(2))
            last[h] = max(last.get(h, -1), i)
    if not last:
        raise ValueError("no termination layers found in the checkpoint")
    for h, i in last.items():
        sdict[f"auxiliary_heads.{h}.termination_layers.{i}.bias"] = sdict[f"auxiliary_heads.{h}.termination_layers.{i}.bias"] + c
    ck.setdefault("provenance", {})
    ck["provenance"] = dict(ck["provenance"], termination_logit_offset=float(c), calibrated_from_sha256=sha256(ckpt_in),
                            calibration="A4 D2 option B (22 Nov 2026)")
    with Path(ckpt_out).open("xb") as f:
        torch.save(ck, f)
    return sorted(last)


def run(wm, norm_path, shift_trace, shift_withheld, out, device="cpu"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sd, H, norm, diff = rp.load_model(wm, norm_path, device)
    with np.load(shift_trace, allow_pickle=False) as z:
        meta = validate_trace(z)
        tr = {k: z[k] for k in z.files if k != "metadata"}
    held = rp.heldout(tr, shift_withheld, 0.1, 101)
    fall_need, surv_need, kinds = [], [], {}
    for h in held:
        kinds[h["kind"]] = kinds.get(h["kind"], 0) + 1
        if h["kind"] not in ("fall", "time_out"):
            continue
        ep = rp.Episode(tr, h, norm)
        _, _, LG = rp.episode_context(sd, ep, device, return_logits=True)
        if h["kind"] == "fall":
            fall_need.append([need(LG[e, max(1, ep.L - LAST):ep.L]) for e in range(LG.shape[0])])
        else:
            surv_need.append([need(LG[e, 1:]) for e in range(LG.shape[0])])
    fall_need, surv_need = np.array(fall_need, np.float64), np.array(surv_need, np.float64)
    res = choose(fall_need, surv_need)
    rec = dict(protocol="A4 D2 option B: one termination-logit offset chosen on the withheld D1 fitting-batch episodes",
               wm=str(wm), wm_sha256=sha256(wm), shift_trace_sha256=sha256(shift_trace),
               friction=(meta.get("friction_shift") or {}).get("factor"), validation_kinds=kinds,
               validation_falls=int(len(fall_need)), validation_surviving=int(len(surv_need)), **res,
               per_head_detection_at_offset=[float(x) for x in (res["offset"] > fall_need).mean(0)] if fall_need.size else None,
               per_head_false_stop_at_offset=[float(x) for x in (res["offset"] > surv_need).mean(0)] if surv_need.size else None,
               analysis_sha256=sha256(__file__))
    if res["valid"]:
        heads = apply_offset(wm, out / "d2c_wm_seed2.pt", res["offset"])
        rec["calibrated_model"] = str(out / "d2c_wm_seed2.pt")
        rec["calibrated_model_sha256"] = sha256(out / "d2c_wm_seed2.pt")
        rec["heads_changed"] = heads
    (out / "calibration.json").write_text(json.dumps(rec, indent=1))
    print(f"[d2-cal] validation (0.30 friction, never fitted): {rec['validation_falls']} falls, "
          f"{rec['validation_surviving']} surviving episodes; at offset 0: detection {res['detection_at_0']:.3f}, "
          f"false stops {res['false_stop_at_0']:.3f}", flush=True)
    print(f"[d2-cal] chosen offset {res['offset']:+.4f}: detection {res['detection']:.3f}, false stops "
          f"{res['false_stop']:.3f} -> {'VALID' if res['valid'] else 'NO VALID OFFSET (D2 closes as FAIL)'}", flush=True)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", type=Path, required=True)
    p.add_argument("--norm", type=Path, required=True)
    p.add_argument("--shift", type=Path, required=True, help="the D1 fitting batch")
    p.add_argument("--shift-withheld", type=Path, required=True, help="data/shift/withheld.json")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    rec = run(a.wm, a.norm, a.shift, a.shift_withheld, a.out)
    return 0 if rec["valid"] else 5


if __name__ == "__main__":
    sys.exit(main())
