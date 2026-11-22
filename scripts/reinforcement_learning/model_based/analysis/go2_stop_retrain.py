#!/usr/bin/env python3
"""A4 option 1 (Charbel, 22 Nov 2026): retrain ONLY the stop (termination) layers of the D2 model.

Frozen: both GRU trunks, the state heads, the contact heads. Each head's termination MLP (256 -> 128 -> 1) is
fine-tuned from its current weights on the features of its own frozen auxiliary trunk, teacher forced with memory
from the start of each training segment (the D2 training data: data/segments; never the probes or the calibration
recording). Loss: BCE with logits and pos_weight w; Adam lr 1e-3; 2000 steps of 4096 rows sampled uniformly; seed =
head index (all fixed before the run).
Candidates w in (1, 10, 30, 100). Selection on the independent calibration recording (0.30 friction, seed 914):
detection = share of (fall, head) pairs with a stop (logit > 0) in the last 11 rows; false stops = share of
(surviving episode, head) pairs with a stop on any row; memory from the episode start. Rule (A4 §3): the candidate
with the highest detection among those with false stops <= 0.10 (tie: the smaller w). None -> option 1 fails.
The chosen layers are written into a copy of the D2 checkpoint; Q1-Q3 then run ONCE on the fresh probe (seed 913)
with the unchanged qualifier.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
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
from go2_d2_qualify import probe_episodes  # noqa: E402
from pre_reset_transitions import sha256, validate_trace  # noqa: E402

WEIGHTS = (1.0, 10.0, 30.0, 100.0)
STEPS, BATCH, LR, LAST = 2000, 4096, 1e-3, 11
DETECT_MIN_NOTE, FALSE_STOP_MAX = 0.5, 0.10


@torch.no_grad()
def features(sd, S, A, device):
    """Auxiliary-trunk features [L-1, hidden] for target rows 1..L-1 (memory from row 0)."""
    x = torch.cat([torch.from_numpy(S[:-1]), torch.from_numpy(A[1:])], dim=-1).unsqueeze(0).to(device)
    out, _ = sd.auxiliary_base.memory.rnn(x)
    return out[0]


def training_set(sd, seg_root, norm, device):
    from fit_world_model_go2_pilot_segments import read_segments
    segs, _, _ = read_segments(seg_root, 45, 12, 0, 8, 1)
    F, Y = [], []
    for _, _, data, _ in segs:
        if len(data) < 2:
            continue
        S = ((data[:, :45] - norm["state_mean"]) / norm["state_std"]).astype(np.float32)
        A = ((data[:, 45:57] - norm["action_mean"]) / norm["action_std"]).astype(np.float32)
        F.append(features(sd, S, A, device))
        Y.append(torch.from_numpy((data[1:, -1] > 0.5).astype(np.float32)).to(device))
    return torch.cat(F), torch.cat(Y)


def train(layers, F, Y, w, seed, steps=STEPS, batch=BATCH, lr=LR):
    layers = copy.deepcopy(layers).train()
    for p in layers.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(layers.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(w), device=F.device))
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(len(F), (batch,), generator=g).to(F.device)
        loss = loss_fn(layers(F[idx]).squeeze(-1), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return layers.eval()


@torch.no_grad()
def score(layer_sets, eps, sd, device):
    """layer_sets: per head; returns detection and false-stop shares (mean over pairs) and per head."""
    det, fs = [], []
    for ep in eps:
        if ep.L < 2:
            continue
        f = features(sd, ep.S, ep.A, device)
        lg = torch.stack([L(f).squeeze(-1) for L in layer_sets]).cpu().numpy()      # [E, L-1] for rows 1..L-1
        if ep.kind == "fall":
            det.append((lg[:, max(0, lg.shape[1] - LAST):] > 0).any(1))
        elif ep.kind == "time_out":
            fs.append((lg > 0).any(1))
    det, fs = np.array(det), np.array(fs)
    return dict(detection=float(det.mean()), false_stop=float(fs.mean()), falls=int(len(det)), surviving=int(len(fs)),
                detection_per_head=[float(x) for x in det.mean(0)], false_stop_per_head=[float(x) for x in fs.mean(0)])


def choose(results):
    ok = [(r["detection"], -r["w"], r) for r in results if r["false_stop"] <= FALSE_STOP_MAX]
    return max(ok, key=lambda t: (t[0], t[1]))[2] if ok else None


def write_checkpoint(wm, out, layer_sets, w, provenance):
    ck = torch.load(wm, map_location="cpu", weights_only=False)
    sdict = ck["system_dynamics_state_dict"]
    for h, L in enumerate(layer_sets):
        new = {f"auxiliary_heads.{h}.termination_layers.{k}": v.detach().cpu() for k, v in L.state_dict().items()}
        missing = [k for k in new if k not in sdict]
        if missing:
            raise KeyError(f"checkpoint lacks {missing[:2]}")
        sdict.update(new)
    ck["provenance"] = dict(ck.get("provenance", {}), **provenance, stop_retrain_pos_weight=float(w))
    with Path(out).open("xb") as f:
        torch.save(ck, f)


def run(wm, norm_path, seg_root, calibration, out, device="cpu", steps=STEPS, weights=WEIGHTS):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    torch.manual_seed(0)
    sd, H, norm, _ = rp.load_model(wm, norm_path, device)
    for p in sd.parameters():
        p.requires_grad_(False)
    F, Y = training_set(sd, seg_root, norm, device)
    with np.load(calibration, allow_pickle=False) as z:
        meta = validate_trace(z)
        tr = {k: z[k] for k in z.files if k != "metadata"}
    cal = probe_episodes(tr, norm)
    heads = [h.termination_layers for h in sd.auxiliary_heads]
    results, trained = [], {}
    base = score(heads, cal, sd, device)
    for w in weights:
        sets = [train(L, F, Y, w, seed=h, steps=steps) for h, L in enumerate(heads)]
        r = dict(w=w, **score(sets, cal, sd, device))
        results.append(r)
        trained[w] = sets
        print(f"[option1] w={w:g}: detection {r['detection']:.3f}, false stops {r['false_stop']:.3f} "
              f"({r['falls']} falls, {r['surviving']} surviving; calibration seed 914)", flush=True)
    best = choose(results)
    rec = dict(protocol="A4 option 1: stop-layer retraining on the frozen auxiliary trunk", wm_sha256=sha256(wm),
               calibration_sha256=sha256(calibration), calibration_friction=(meta.get("friction_shift") or {}).get("factor"),
               training_rows=int(len(Y)), training_positive_rows=int(Y.sum().item()), steps=steps, batch=BATCH, lr=LR,
               unchanged_model_on_calibration=base, candidates=results, chosen_w=None if best is None else best["w"],
               analysis_sha256=sha256(__file__))
    if best is not None:
        path = out / "d2s_wm_seed2.pt"
        write_checkpoint(wm, path, trained[best["w"]], best["w"],
                         dict(stop_retrain="A4 option 1 (22 Nov 2026)", stop_retrain_from_sha256=sha256(wm),
                              stop_retrain_calibration_sha256=sha256(calibration)))
        rec["model"], rec["model_sha256"] = str(path), sha256(path)
    rec["wall_s"] = time.time() - t0
    (out / "option1.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"[option1] unchanged model on the calibration set: detection {base['detection']:.3f}, false stops "
          f"{base['false_stop']:.3f}", flush=True)
    print(f"[option1] chosen w: {rec['chosen_w']} -> " + ("model written" if best else "NO CANDIDATE (option 1 fails)"),
          flush=True)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wm", type=Path, required=True)
    p.add_argument("--norm", type=Path, required=True)
    p.add_argument("--segments", type=Path, required=True, help="the D2 training segments (data/segments)")
    p.add_argument("--calibration", type=Path, required=True, help="the independent calibration recording (seed 914)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    rec = run(a.wm, a.norm, a.segments, a.calibration, a.out, a.device)
    return 0 if rec["chosen_w"] is not None else 6


if __name__ == "__main__":
    sys.exit(main())
