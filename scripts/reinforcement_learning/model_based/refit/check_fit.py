#!/usr/bin/env python3
"""Check a refit checkpoint before a queue uses it. Exit 0 and "FIT CHECK PASS" only if all asked checks hold.

Always: iter == --iters and provenance b3_reset_per_head is True (the per-head reset wrapper ran).
Optional: --state_loss (provenance), --norm_sha256 (checkpoint norm_stats_sha256), --data_dir
(provenance data_dir, compared as resolved paths), --real_terms (checkpoint total_real_terms),
--heads (provenance ensemble_size).

  check_fit.py --ckpt OUT/nll/wm.pt --iters 500 --state_loss gaussian_nll --heads 5
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


def problems_for(ck, args):
    prov = ck.get("provenance") or {}
    out = []
    if ck.get("iter") != args.iters:
        out.append(f"iter {ck.get('iter')} != {args.iters}")
    if prov.get("b3_reset_per_head") is not True:
        out.append("provenance b3_reset_per_head is not True")
    if args.state_loss is not None and prov.get("state_loss", "mse") != args.state_loss:
        out.append(f"state_loss {prov.get('state_loss', 'mse')!r} != {args.state_loss!r}")
    if args.norm_sha256 is not None and ck.get("norm_stats_sha256") != args.norm_sha256:
        out.append(f"norm_stats_sha256 {ck.get('norm_stats_sha256')} != {args.norm_sha256}")
    if args.data_dir is not None and os.path.realpath(str(prov.get("data_dir"))) != os.path.realpath(args.data_dir):
        out.append(f"data_dir {prov.get('data_dir')} != {args.data_dir}")
    if args.real_terms is not None and ck.get("total_real_terms") != args.real_terms:
        out.append(f"total_real_terms {ck.get('total_real_terms')} != {args.real_terms}")
    if args.heads is not None and prov.get("ensemble_size") != args.heads:
        out.append(f"ensemble_size {prov.get('ensemble_size')} != {args.heads}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--iters", type=int, required=True)
    p.add_argument("--state_loss", default=None)
    p.add_argument("--norm_sha256", default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--real_terms", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    args = p.parse_args(argv)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    problems = problems_for(ck, args)
    for line in problems:
        print(f"[check_fit] {args.ckpt}: {line}")
    print(f"FIT CHECK PASS: {not problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
