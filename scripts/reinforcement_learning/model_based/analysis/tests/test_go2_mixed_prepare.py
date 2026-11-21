"""A4 D2 (21 Nov 2026): the mixed-data manifest (round-0 nominal + D1 shift batch)."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

MB = Path(__file__).resolve().parents[2]
for p in (str(MB), str(MB / "analysis"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("rsl_rl.modules")

import go2_mixed_prepare as mix  # noqa: E402
import go2_round0_prepare as prep  # noqa: E402
from test_go2_g0 import round0_trace, save  # noqa: E402


def test_mixed_manifest_pools_training_data_only(tmp_path):
    r0 = save(tmp_path / "r0.npz", round0_trace(seed=0))
    sh = save(tmp_path / "sh.npz", round0_trace(seed=7))
    a3 = prep.prepare(r0, tmp_path / "a3", frac=0.1, seed=101)
    r = mix.prepare_mixed(r0, sh, tmp_path / "mix", a3_withheld=tmp_path / "a3" / "withheld.json")
    out = tmp_path / "mix"
    man = (out / "segments" / "manifest.csv").read_text().splitlines()[1:]
    n0 = len((out / "r0" / "segments" / "manifest.csv").read_text().splitlines()) - 1
    n1 = len((out / "shift" / "segments" / "manifest.csv").read_text().splitlines()) - 1
    assert len(man) == n0 + n1 == r["segments"] and all(m.startswith(("r0_", "sh_")) for m in man)
    for m in man:
        assert (out / "segments" / f"{m.split(',')[0]}.csv").is_file()
    assert r["falls_reaching_fitting"] == r["sources"]["round0"]["counts"]["falls_reaching_fitting"] + \
        r["sources"]["shift"]["counts"]["falls_reaching_fitting"]
    from fit_world_model_go2_pilot_segments import compute_normalizer, read_segments
    segs, _, _ = read_segments(out / "segments", 45, 12, 0, 8, 1)
    ref = [t.reshape(-1).numpy() for t in compute_normalizer(segs, 45, 12)]
    with np.load(out / "go2_norm.npz") as z:
        for k, v in zip(("state_mean", "state_std", "action_mean", "action_std"), ref):
            np.testing.assert_allclose(z[k], v)
    sd = json.loads((out / "start_data" / "start_data.json").read_text())
    assert sd["files"] == len(list((out / "start_data").glob("state_action_data_*.csv")))
    assert sd["rows"] == r["sources"]["round0"]["counts"]["start_rows"] + r["sources"]["shift"]["counts"]["start_rows"]
    # withheld episodes never reach the pooled segments
    for tag, src in (("r0", "r0"), ("sh", "shift")):
        held = {(h["env"], h["first"]) for h in json.loads((out / src / "withheld.json").read_text())}
        for m in man:
            if m.startswith(tag + "_"):
                name = m.split(",")[0][len(tag) + 1:]
                assert (int(name[6:9]), int(name[11:])) not in held
    with pytest.raises(FileExistsError):
        mix.prepare_mixed(r0, sh, out)
    (tmp_path / "other.json").write_text("[]")
    with pytest.raises(ValueError, match="differs from A3"):
        mix.prepare_mixed(r0, sh, tmp_path / "mix2", a3_withheld=tmp_path / "other.json")
