"""A4 D3 milestone gate (22 Nov 2026)."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import go2_d3_gate as g  # noqa: E402
from go2_g0_gate import ProtocolMismatch  # noqa: E402


def cellfile(root, tag, seed, falls, speed, track, factor=0.3, seed_proto=None):
    d = root / f"{tag}_cmd10_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "episodes_summary.json").write_text(json.dumps(dict(
        task="Go2-Start-v0", num_envs=32, num_steps=10000, seed=seed if seed_proto is None else seed_proto, std=0.5,
        cmd_resample_s=10, friction_shift=dict(factor=factor, verified=True),
        summary=dict(falls=falls, episodes_completed=100, track_per_step=track, speed_ratio=speed))))


def make(root, d3=(10, 0.9, 0.95), s2=(10, 0.93, 0.96), d3_factor=0.3):
    for s in (42, 43):
        cellfile(root, "D3", s, d3[0], d3[1], d3[2], d3_factor)
        cellfile(root, "S2", s, s2[0], s2[1], s2[2])


def test_rule(tmp_path):
    make(tmp_path / "a")
    r = g.score(tmp_path / "a", 0.3)
    assert r["verdict"] == "MILESTONE MET" and r["rule"]["fall_bar"] == pytest.approx(1.25 * 20 + 3)
    make(tmp_path / "b", d3=(15, 0.9, 0.95))                       # 30 falls > 28
    assert g.score(tmp_path / "b", 0.3)["checks"]["falls"] is False
    make(tmp_path / "c", d3=(10, 0.7, 0.95))
    assert g.score(tmp_path / "c", 0.3)["checks"]["speed"] is False
    make(tmp_path / "d", d3=(10, 0.9, 0.92))
    assert g.score(tmp_path / "d", 0.3)["checks"]["tracking"] is False


def test_mismatches(tmp_path):
    make(tmp_path / "e", d3_factor=0.45)
    with pytest.raises(ProtocolMismatch, match="friction"):
        g.score(tmp_path / "e", 0.3)
    make(tmp_path / "f")
    cellfile(tmp_path / "f", "D3", 42, 10, 0.9, 0.95, seed_proto=7)
    with pytest.raises(ProtocolMismatch):
        g.score(tmp_path / "f", 0.3)
