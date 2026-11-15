#!/usr/bin/env python3
"""Plan v6 C6: a normalizer with a widened projected-gravity std; every other value unchanged.

ETH's gravity std is 0.02/0.02/0.04 (mean 0/0/-1): a 60 deg tilt gives z = 43 (x or y) and 12.5 (z).
Rule (fixed before the fit, decisions-log 15 Nov): the plan fall threshold (tilt --tilt_deg, default 60)
maps to |z| = --z (default 3) on each gravity component:
    std_x = std_y = sin(tilt) / z,   std_z = (1 - cos(tilt)) / z
With the defaults: 0.2887, 0.2887, 0.1667. Means stay 0/0/-1 (checked). Clipping was not chosen: a clip
also bounds the targets, so the model could not represent a tilt beyond about 6 deg in gravity.

Input and output: .npz with state_mean, state_std, action_mean, action_std (the fitter's --norm_stats
format). Also writes <out>.json with old and new gravity values and both sha256. CPU only, no torch.

  make_widened_norm.py --src <e1v>/data/eth_norm.npz --out <dir>/widegrav_norm.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

GRAVITY = (6, 7, 8)  # AnymalDFlatConfig.data_config.state_idx_dict "$g$"
KEYS = ("state_mean", "state_std", "action_mean", "action_std")


def widened_gravity_std(tilt_deg=60.0, z=3.0):
    t = math.radians(tilt_deg)
    return np.array([math.sin(t) / z, math.sin(t) / z, (1.0 - math.cos(t)) / z])


def widen(stats, tilt_deg=60.0, z=3.0):
    out = {k: np.array(stats[k], dtype=np.float32, copy=True) for k in KEYS}
    mean = out["state_mean"][list(GRAVITY)]
    if not np.allclose(mean, [0.0, 0.0, -1.0]):
        raise ValueError(f"gravity mean {mean.tolist()} is not 0/0/-1: not the ETH layout")
    out["state_std"][list(GRAVITY)] = widened_gravity_std(tilt_deg, z).astype(np.float32)
    return out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tilt_deg", type=float, default=60.0)
    p.add_argument("--z", type=float, default=3.0)
    args = p.parse_args(argv)
    if args.out.exists():
        p.error("--out must not exist")
    with np.load(args.src, allow_pickle=False) as z:
        stats = {k: np.asarray(z[k]) for k in KEYS}
    new = widen(stats, args.tilt_deg, args.z)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "xb") as stream:
        np.savez(stream, **new)
    card = {"src": str(args.src.resolve()), "src_sha256": sha256(args.src), "out_sha256": sha256(args.out),
            "rule": f"tilt {args.tilt_deg} deg -> |z| = {args.z} per gravity component",
            "gravity_index": list(GRAVITY),
            "gravity_std_old": stats["state_std"].reshape(-1)[list(GRAVITY)].tolist(),
            "gravity_std_new": new["state_std"][list(GRAVITY)].tolist()}
    Path(str(args.out) + ".json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
