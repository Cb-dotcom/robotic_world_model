"""CPU tests for the fitter's partial warm start (plan v6 C8 and C1c). Needs rsl_rl (rsl_rl_rwm) importable."""
from pathlib import Path
import sys

import pytest
import torch

MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MB))
rsl = pytest.importorskip("rsl_rl.modules")
import fit_world_model_go2_pilot_segments as F  # noqa: E402
from configs.anymal_d_flat_cfg import AnymalDFlatConfig  # noqa: E402

ARCH = AnymalDFlatConfig().model_architecture_config.architecture_config


def build(heads, seed):
    torch.manual_seed(seed)
    return rsl.SystemDynamicsEnsemble(45, 12, 0, 8, 1, "cpu", ensemble_size=heads, history_horizon=32,
                                      architecture_config=ARCH)


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_trunk_only_twenty_heads():
    ref, model = build(5, 0), build(20, 1)
    fresh = {k: v.clone() for k, v in model.state_dict().items()}
    F.partial_warm_start(model, ref.state_dict())
    got, r = model.state_dict(), ref.state_dict()
    for k in got:
        if k.startswith(F.BASE_PREFIXES):
            assert torch.equal(got[k], r[k])
        else:
            assert torch.equal(got[k], fresh[k])  # heads keep their seeded fresh init
    heads = [{k: v for k, v in got.items() if k.startswith(f"state_heads.{i}.")} for i in range(2)]
    assert not all(torch.equal(a, b) for a, b in zip(heads[0].values(), heads[1].values()))


def test_member_copy_into_single_head_model():
    ref, model = build(5, 0), build(1, 2)
    F.partial_warm_start(model, ref.state_dict(), member=3)
    got, r = model.state_dict(), ref.state_dict()
    for k, v in got.items():
        if k.startswith(F.BASE_PREFIXES):
            assert torch.equal(v, r[k])
        else:
            prefix, rest = k.split(".0.", 1)
            assert torch.equal(v, r[f"{prefix}.3.{rest}"])


def test_member_needs_single_head_and_existing_member():
    ref = build(5, 0).state_dict()
    with pytest.raises(ValueError):
        F.partial_warm_start(build(2, 1), ref, member=0)
    with pytest.raises(ValueError):
        F.partial_warm_start(build(1, 1), ref, member=7)
