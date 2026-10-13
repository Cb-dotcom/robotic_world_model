#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Check the reset-row hypothesis of code review 1.2 (2026-10-08) on 66-column CSVs.

IsaacLab resets a terminated env before it computes observations, and the reset zeroes
the last action. If that is what our CSVs contain, every interior termination row has an
all-zero action and a spawn-like state (zero velocities, zero joint_pos_rel/joint_vel,
gravity ~ (0, 0, -1)). This script prints, per file:

* interior terminations (block seams excluded when --steps_per_env is given),
* max |action| and max |base vel|, |joint_pos_rel|, |joint_vel| on those rows,
* mean gravity on those rows,
* the number of non-terminal rows that look like a spawn (zero action and zero joint_vel):
  unmarked resets (time-outs or missed falls).

Usage: python analysis/check_reset_rows.py FILE_OR_GLOB [...] [--steps_per_env 1000]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def check(path: str, steps_per_env: int) -> None:
    d = pd.read_csv(path, header=None)
    if d.shape[1] != 66:
        print(f"{path}: skip ({d.shape[1]} columns)")
        return
    d = d.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    term = d[:, 65] > 0.5
    seam = np.zeros(len(d), dtype=bool)
    if steps_per_env > 0 and len(d) % steps_per_env == 0:
        seam[steps_per_env - 1::steps_per_env] = True
    fi = np.where(term & ~seam)[0]
    act0 = np.abs(d[:, 45:57]).max(axis=1) == 0
    qd0 = np.abs(d[:, 21:33]).max(axis=1) == 0
    spawn_like = act0 & qd0
    name = os.path.relpath(path)
    if len(fi) == 0:
        print(f"{name}: rows={len(d)} interior_terms=0 unmarked_spawn_rows={int((spawn_like & ~term).sum())}")
        return
    print(f"{name}: rows={len(d)} interior_terms={len(fi)} "
          f"max|action|@term={np.abs(d[fi, 45:57]).max():.6g} "
          f"zero_action_frac@term={act0[fi].mean():.3f} "
          f"max|v,w|@term={np.abs(d[fi, 0:6]).max():.4g} "
          f"max|q,qd|@term={np.abs(d[fi, 9:33]).max():.4g} "
          f"gravity@term={np.round(d[fi, 6:9].mean(axis=0), 3).tolist()} "
          f"zero_action_rows_total={int(act0.sum())} "
          f"unmarked_spawn_rows={int((spawn_like & ~term).sum())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--steps_per_env", type=int, default=0, help="block length; 0 = do not mark seams")
    a = ap.parse_args()
    for pat in a.paths:
        for p in sorted(glob.glob(pat)) or [pat]:
            if os.path.isfile(p):
                check(p, a.steps_per_env)


if __name__ == "__main__":
    main()
