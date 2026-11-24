"""A4 option 1 (22 Nov 2026): stop-layer retraining, selection rule, checkpoint edit, end-to-end."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

MB = Path(__file__).resolve().parents[2]
for p in (str(MB), str(MB / "analysis"), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("rsl_rl.modules")

import go2_round0_prepare as prep  # noqa: E402
import go2_stop_retrain as st  # noqa: E402
import go2_g0_replay as rp  # noqa: E402
from test_go2_g0 import fake_go2_wm, restore_go2_class, round0_trace, save  # noqa: E402,F401


def test_choose_rule():
    r = lambda w, d, f: dict(w=w, detection=d, false_stop=f)
    assert st.choose([r(1, .4, .05), r(10, .6, .08), r(30, .8, .2)])["w"] == 10
    assert st.choose([r(1, .4, .05), r(10, .4, .05)])["w"] == 1
    assert st.choose([r(1, .9, .2)]) is None


def test_end_to_end_and_checkpoint(tmp_path):
    r0 = save(tmp_path / "r0.npz", round0_trace(seed=0))
    prep.prepare(r0, tmp_path / "m", frac=0.1, seed=101)
    cal = round0_trace(seed=9)
    meta = json.loads(str(cal["metadata"])); meta["friction_shift"] = {"factor": 0.3, "verified": True}
    cal["metadata"] = np.array(json.dumps(meta))
    calp = save(tmp_path / "cal.npz", cal)
    with np.load(tmp_path / "m" / "go2_norm.npz") as z:
        stats = {k: z[k] for k in z.files}
    wm = fake_go2_wm(tmp_path / "wm.pt", stats)
    rec = st.run(wm, tmp_path / "m" / "go2_norm.npz", tmp_path / "m" / "segments", calp, tmp_path / "out",
                 steps=20, weights=(1.0, 100.0))
    assert rec["training_positive_rows"] >= 1 and len(rec["candidates"]) == 2
    if rec["chosen_w"] is None:
        return
    sd0, _, norm, _ = rp.load_model(wm, tmp_path / "m" / "go2_norm.npz", "cpu")
    sd1, _, _, _ = rp.load_model(rec["model"], tmp_path / "m" / "go2_norm.npz", "cpu")
    for a, b in zip(sd0.state_heads.parameters(), sd1.state_heads.parameters()):      # state side unchanged
        assert torch.equal(a, b)
    for a, b in zip(sd0.auxiliary_base.parameters(), sd1.auxiliary_base.parameters()):  # trunk unchanged
        assert torch.equal(a, b)
    for h0, h1 in zip(sd0.auxiliary_heads, sd1.auxiliary_heads):
        for a, b in zip(h0.contact_layers.parameters(), h1.contact_layers.parameters()):
            assert torch.equal(a, b)
    # the written stop layers are the ones that were scored: re-score from the checkpoint
    eps = st.probe_episodes({k: v for k, v in np.load(calp).items() if k != "metadata"}, norm)
    again = st.score([h.termination_layers for h in sd1.auxiliary_heads], eps, sd1, "cpu")
    chosen = [c for c in rec["candidates"] if c["w"] == rec["chosen_w"]][0]
    assert again["detection"] == chosen["detection"] and again["false_stop"] == chosen["false_stop"]
    ck = torch.load(rec["model"], weights_only=False)
    assert ck["provenance"]["stop_retrain_pos_weight"] == rec["chosen_w"]


def test_write_checkpoint_replaces_only_the_stop_layers(tmp_path):
    stats = dict(state_mean=np.zeros(45, np.float32), state_std=np.ones(45, np.float32),
                 action_mean=np.zeros(12, np.float32), action_std=np.ones(12, np.float32))
    np.savez(tmp_path / "n.npz", **stats)
    wm = fake_go2_wm(tmp_path / "wm.pt", stats)
    sd0, _, _, _ = rp.load_model(wm, tmp_path / "n.npz", "cpu")
    F = torch.randn(500, sd0.auxiliary_heads[0].termination_layers[0].in_features)
    Y = (torch.rand(500) < 0.05).float()
    sets = [st.train(h.termination_layers, F, Y, 30.0, seed=i, steps=5, batch=64) for i, h in enumerate(sd0.auxiliary_heads)]
    st.write_checkpoint(wm, tmp_path / "s.pt", sets, 30.0, dict(note="test"))
    sd1, _, _, _ = rp.load_model(tmp_path / "s.pt", tmp_path / "n.npz", "cpu")
    for L, h in zip(sets, sd1.auxiliary_heads):
        torch.testing.assert_close(L(F), h.termination_layers(F))
    for a, b in zip(sd0.auxiliary_base.parameters(), sd1.auxiliary_base.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(sd0.state_heads.parameters(), sd1.state_heads.parameters()):
        assert torch.equal(a, b)
    assert not torch.equal(sd0.auxiliary_heads[0].termination_layers[0].weight, sd1.auxiliary_heads[0].termination_layers[0].weight)
    with pytest.raises(FileExistsError):
        st.write_checkpoint(wm, tmp_path / "s.pt", sets, 30.0, {})


def _option1_inputs(tmp_path):
    r0 = save(tmp_path / "r0.npz", round0_trace(seed=0))
    prep.prepare(r0, tmp_path / "m", frac=0.1, seed=101)
    cal = round0_trace(seed=9)
    meta = json.loads(str(cal["metadata"])); meta["friction_shift"] = {"factor": 0.3, "verified": True}
    cal["metadata"] = np.array(json.dumps(meta))
    calp = save(tmp_path / "cal.npz", cal)
    with np.load(tmp_path / "m" / "go2_norm.npz") as z:
        stats = {k: z[k] for k in z.files}
    wm = fake_go2_wm(tmp_path / "wm.pt", stats)
    return wm, tmp_path / "m" / "go2_norm.npz", tmp_path / "m" / "segments", calp


def test_success_path_selects_writes_reloads_and_rescores(tmp_path, monkeypatch):
    """Ticket 1.3 (24 Nov): the success path always runs here (every candidate eligible), so the test cannot pass
    by returning early. Only the termination layers differ between the input and the written checkpoint."""
    from pre_reset_transitions import sha256
    wm, norm, seg, calp = _option1_inputs(tmp_path)
    before = sha256(wm)
    monkeypatch.setattr(st, "FALSE_STOP_MAX", 1.0)
    rec = st.run(wm, norm, seg, calp, tmp_path / "out", steps=30, weights=(1.0, 10.0, 100.0))
    assert rec["chosen_w"] is not None and Path(rec["model"]).is_file()
    best = max(rec["candidates"], key=lambda c: (c["detection"], -c["w"]))
    assert rec["chosen_w"] == best["w"]                                  # the rule, with every candidate eligible
    assert sha256(wm) == before                                          # the input checkpoint is never edited
    ck0 = torch.load(wm, weights_only=False)
    ck1 = torch.load(rec["model"], weights_only=False)
    s0, s1 = ck0["system_dynamics_state_dict"], ck1["system_dynamics_state_dict"]
    assert set(s0) == set(s1)
    changed = {k for k in s0 if not torch.equal(s0[k], s1[k])}
    assert changed and all(".termination_layers." in k and k.startswith("auxiliary_heads.") for k in changed)
    heads = {k.split(".")[1] for k in changed}
    assert heads == {str(h) for h in range(len(heads))} and len(heads) >= 1   # one block per head, from head 0
    for k in set(ck0) - {"system_dynamics_state_dict", "provenance"}:   # everything else in the file is untouched
        a, b = ck0[k], ck1[k]
        assert (torch.equal(a, b) if torch.is_tensor(a) else a == b), k
    prov = ck1["provenance"]
    assert prov["stop_retrain_pos_weight"] == rec["chosen_w"] and prov["stop_retrain_from_sha256"] == before
    # reload through the production loader and re-score on the calibration recording
    sd1, _, normd, _ = rp.load_model(rec["model"], norm, "cpu")
    eps = st.probe_episodes({k: v for k, v in np.load(calp).items() if k != "metadata"}, normd)
    again = st.score([h.termination_layers for h in sd1.auxiliary_heads], eps, sd1, "cpu")
    chosen = [c for c in rec["candidates"] if c["w"] == rec["chosen_w"]][0]
    assert again["detection"] == chosen["detection"] and again["false_stop"] == chosen["false_stop"]
    assert again["detection_per_head"] == chosen["detection_per_head"]
    saved = json.loads((tmp_path / "out" / "option1.json").read_text())
    assert saved["model_sha256"] == sha256(rec["model"]) and saved["chosen_w"] == rec["chosen_w"]


def test_no_eligible_weight_writes_no_model(tmp_path, monkeypatch):
    wm, norm, seg, calp = _option1_inputs(tmp_path)
    monkeypatch.setattr(st, "FALSE_STOP_MAX", -1.0)                      # no candidate can pass
    full_run = st.run
    monkeypatch.setattr(st, "run", lambda *a, **k: full_run(*a, **dict(k, steps=10)))   # short training; same CLI path
    code = st.main(["--wm", str(wm), "--norm", str(norm), "--segments", str(seg), "--calibration", str(calp),
                    "--out", str(tmp_path / "out")])
    assert code == 6
    saved = json.loads((tmp_path / "out" / "option1.json").read_text())
    assert saved["chosen_w"] is None and "model" not in saved and len(saved["candidates"]) == 4
    assert not list((tmp_path / "out").glob("*.pt"))
