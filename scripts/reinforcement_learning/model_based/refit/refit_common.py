# SPDX-License-Identifier: BSD-3-Clause
"""Small helpers shared by the P1 re-fit data tools (make_curated_segments, make_subsample,
make_shared_norm). Pure numpy/pandas; no torch.

Column layout of every 66-column CSV (headerless, float):
    state 0:45, action 45:57, contact 57:65, termination 65 (> 0.5 means terminal)

"Pilot format" directory = ``manifest.csv`` (columns name, steps, rows[, num_envs]) plus one
``<name>.csv`` per manifest row; each file is env-major with rows == steps * num_envs
(num_envs defaults to 64 when the column is absent, as in the original pilot fitter).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import time
from typing import Dict, List

import numpy as np
import pandas as pd

NUM_COLS = 66
STATE = slice(0, 45)
ACTION = slice(45, 57)
TERM_COL = 65
DEFAULT_NUM_ENVS = 64


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(path: str) -> Dict[str, object]:
    try:
        commit = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                         text=True).strip()
        dirty = subprocess.check_output(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                                        stderr=subprocess.DEVNULL, text=True).strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except Exception:
        return {"commit": None, "dirty": None}


def provenance(script: str, args) -> Dict[str, object]:
    return {"script": os.path.abspath(script), "script_sha256": sha256_file(os.path.abspath(script)),
            "args": vars(args), "git": git_info(os.path.dirname(os.path.abspath(script))),
            "created_unix": time.time(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def read_lines(path: str) -> List[str]:
    """Non-empty lines of a text CSV, without line terminators (data rows are copied verbatim)."""
    with open(path, "r") as fh:
        return [ln for ln in fh.read().splitlines() if ln.strip() != ""]


def parse_lines(lines: List[str], what: str = "") -> np.ndarray:
    """Parse CSV lines exactly like the fitter (pd.read_csv(header=None)) -> float64 array."""
    if not lines:
        return np.zeros((0, NUM_COLS), dtype=np.float64)
    df = pd.read_csv(io.StringIO("\n".join(lines) + "\n"), header=None)
    x = df.to_numpy(dtype=np.float64)
    if x.shape[1] != NUM_COLS:
        raise RuntimeError(f"{what}: expected {NUM_COLS} columns, got {x.shape[1]}")
    if not np.isfinite(x).all():
        raise RuntimeError(f"{what}: non-finite values (header row or corrupt data?)")
    return x


def read_numeric(path: str) -> np.ndarray:
    return parse_lines(read_lines(path), path)


def read_manifest(root: str) -> pd.DataFrame:
    m = pd.read_csv(os.path.join(root, "manifest.csv"))
    for c in ("name", "steps", "rows"):
        if c not in m.columns:
            raise RuntimeError(f"{root}/manifest.csv: missing column {c!r} (has {list(m.columns)})")
    if "num_envs" not in m.columns:
        m["num_envs"] = DEFAULT_NUM_ENVS
    m["name"] = m["name"].astype(str)
    for c in ("steps", "rows", "num_envs"):
        m[c] = m[c].astype(int)
    return m


def zero_action_mask(x: np.ndarray, tol: float = 0.0) -> np.ndarray:
    return np.abs(x[:, ACTION]).max(axis=1) <= tol


def prepare_out_dir(out: str, force: bool) -> None:
    if os.path.exists(out) and os.listdir(out):
        if not force:
            raise SystemExit(f"output directory {out} exists and is not empty (use --force to overwrite)")
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)


def write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")


def write_json(path: str, obj) -> None:
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        return o

    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=1)
        fh.write("\n")
