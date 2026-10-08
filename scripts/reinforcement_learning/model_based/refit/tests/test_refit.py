# SPDX-License-Identifier: BSD-3-Clause
"""CPU-only synthetic tests for the P1 re-fit tooling.

Run from anywhere:  python -m pytest scripts/reinforcement_learning/model_based/refit/tests -q
Needs torch, pandas and rsl_rl (path from $RSL_RL_PATH, default the local clone).
"""
import glob
import json
import os
import shutil
import stat
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REFIT_DIR = os.path.dirname(HERE)
MB_DIR = os.path.dirname(REFIT_DIR)
ANALYSIS_DIR = os.path.join(MB_DIR, "analysis")
REPO_DIR = os.path.abspath(os.path.join(MB_DIR, "..", "..", ".."))
FITTER = os.path.join(MB_DIR, "fit_world_model_go2_pilot_segments.py")
FITTER_REL = "scripts/reinforcement_learning/model_based/fit_world_model_go2_pilot_segments.py"
BASE_COMMIT = "c5bcfe5"
RSL_RL_PATH = os.environ.get("RSL_RL_PATH", "/home/claude/cb-dotcom/rsl_rl_rwm")
for p in (REFIT_DIR, MB_DIR, RSL_RL_PATH):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

import fit_world_model_go2_pilot_segments as fitmod  # noqa: E402
import make_curated_segments as mcs  # noqa: E402
import make_shared_norm as msn  # noqa: E402
import make_subsample as msub  # noqa: E402
import refit_common as rc  # noqa: E402

SEQ_LEN = 40  # history 32 + forecast 8 (Go2FlatConfig)
TINY_FIT = ["--iterations", "2", "--num_mini_batches", "2", "--mini_batch_size", "16", "--device", "cpu",
            "--log_interval", "1", "--save_interval", "1"]
SUBPROC_ENV = {**os.environ, "PYTHONPATH": os.pathsep.join([MB_DIR, RSL_RL_PATH]), "OMP_NUM_THREADS": "1"}


# ======================================================================================
# synthetic data helpers
# ======================================================================================
def make_rows(n, rng, terms=(), zero_action=()):
    x = np.zeros((n, 66))
    x[:, 0:45] = rng.normal(size=(n, 45))
    x[:, 45:57] = rng.normal(size=(n, 12))
    x[:, 57:65] = (rng.random((n, 8)) > 0.5).astype(float)
    for r in zero_action:
        x[r, 45:57] = 0.0
    for r in terms:
        x[r, 65] = 1.0
    return x


def write_csv(path, x):
    np.savetxt(path, x, delimiter=",", fmt="%.7g")


def make_env_major(num_envs, steps, rng, fall_at=()):
    """fall_at: list of (env, t) -> termination with zero action (reset-row semantics)."""
    x = make_rows(num_envs * steps, rng)
    for e, t in fall_at:
        r = e * steps + t
        x[r, 45:57] = 0.0
        x[r, 65] = 1.0
    return x


def make_pilot_dir(root, files, num_envs_col=True):
    """files: list of (name, steps, num_envs, data)."""
    os.makedirs(root, exist_ok=True)
    lines = ["name,steps,rows,num_envs" if num_envs_col else "name,steps,rows"]
    for name, steps, ne, x in files:
        write_csv(os.path.join(root, f"{name}.csv"), x)
        lines.append(f"{name},{steps},{len(x)},{ne}" if num_envs_col else f"{name},{steps},{len(x)}")
    with open(os.path.join(root, "manifest.csv"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return root


def pilot64_dir(tmp_path, name="pilot64"):
    rng = np.random.default_rng(1)
    files = [("fa", 50, 64, make_env_major(64, 50, rng, fall_at=[(3, 45), (10, 20)])),
             ("fb", 45, 64, make_env_major(64, 45, rng, fall_at=[(0, 44), (63, 41)]))]
    return make_pilot_dir(str(tmp_path / name), files, num_envs_col=False)


WRAPPER = r"""
import random, runpy, sys
import numpy as np, torch
torch.set_num_threads(1)
s = int(sys.argv[1])
random.seed(s); np.random.seed(s); torch.manual_seed(s)
script = sys.argv[2]
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
"""


def run_fitter(script, args, ext_seed=1234, check=True):
    cmd = [sys.executable, "-c", WRAPPER, str(ext_seed), script] + list(args)
    p = subprocess.run(cmd, env=SUBPROC_ENV, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise AssertionError(f"fitter failed ({p.returncode}):\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return p


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)  # also proves weights_only compatibility


def assert_same_state_dict(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def same_state_dict(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


NORM_KEYS = ("state_mean", "state_std", "action_mean", "action_std")


@pytest.fixture(scope="module")
def original_fitter(tmp_path_factory):
    src = subprocess.run(["git", "-C", REPO_DIR, "show", f"{BASE_COMMIT}:{FITTER_REL}"], capture_output=True, text=True,
                         check=True).stdout
    path = tmp_path_factory.mktemp("orig") / "fit_world_model_go2_pilot_segments_orig.py"
    path.write_text(src)
    return str(path)


# ======================================================================================
# 1. fitter: default behaviour unchanged, seed, lazy windows, norm_stats, num_envs
# ======================================================================================
def test_default_args_identical_to_original(tmp_path, original_fitter):
    """Modified fitter with June-style args == original fitter (same external seeds, CPU, 1 thread)."""
    data = pilot64_dir(tmp_path)
    o_old, o_new = str(tmp_path / "old.pt"), str(tmp_path / "new.pt")
    p_old = run_fitter(original_fitter, ["--data", data, "--output", o_old] + TINY_FIT)
    p_new = run_fitter(FITTER, ["--data", data, "--output", o_new] + TINY_FIT)
    old, new = load(o_old), load(o_new)
    assert_same_state_dict(old["system_dynamics_state_dict"], new["system_dynamics_state_dict"])
    for k in old:
        if k != "system_dynamics_state_dict":
            assert k in new
            if torch.is_tensor(old[k]):
                assert torch.equal(old[k], new[k]), k
            else:
                assert old[k] == new[k], k
    assert set(new) - set(old) == {"provenance"}  # no norm_stats_* keys without --norm_stats
    # identical window lines and losses in the logs

    def pick(out):  # drop the timing suffix "(x.xxs/it)"
        return [ln.split(" (")[0] for ln in out.splitlines() if ln.startswith(("[windows]", "[fit] it", "[fit] transition"))]

    assert pick(p_old.stdout) == pick(p_new.stdout)
    pv = new["provenance"]
    assert pv["total_rows"] == new["total_rows"] and pv["num_windows"] == new["num_windows"]
    assert pv["seed"] is None and pv["normalizer_source"] == "computed_from_data"
    assert "name,steps,rows" in pv["manifest_csv"] and len(pv["data_files"]) == 2
    assert pv["git"]["commit"]  # running inside the worktree


def test_window_count_and_skips(tmp_path):
    """Window bookkeeping on the 64-env fixture: a termination in rows [i, i+38] invalidates window i."""
    data = pilot64_dir(tmp_path)
    segs, rows, terms = fitmod.read_segments(data, 45, 12, 0, 8, 1)
    assert rows == 64 * 95 and terms == 4
    z = [torch.zeros(1, 1, 45), torch.ones(1, 1, 45), torch.zeros(1, 1, 12), torch.ones(1, 1, 12)]
    w = fitmod.build_segment_windows(segs, 45, 12, 8, 1, 32, 8, "cpu", *z, normalize=False)
    # fa: W=11 starts per env; fall t=45 (env 3) kills starts 7..10 (4), fall t=20 (env 10) kills 0..10 (11).
    # fb: W=6 starts per env; fall t=44 (env 0, last row) kills none (allowed as final target), t=41 kills 3..5 (3).
    assert w.shape[0] == 64 * 11 - 4 - 11 + 64 * 6 - 0 - 3


def test_seed_reproducible_and_overrides_external_seed(tmp_path):
    data = pilot64_dir(tmp_path)
    a, b, c = (str(tmp_path / f"{n}.pt") for n in "abc")
    run_fitter(FITTER, ["--data", data, "--output", a, "--seed", "0"] + TINY_FIT, ext_seed=1)
    run_fitter(FITTER, ["--data", data, "--output", b, "--seed", "0"] + TINY_FIT, ext_seed=2)
    run_fitter(FITTER, ["--data", data, "--output", c, "--seed", "1"] + TINY_FIT, ext_seed=1)
    A, B, C = load(a), load(b), load(c)
    assert_same_state_dict(A["system_dynamics_state_dict"], B["system_dynamics_state_dict"])
    assert not same_state_dict(A["system_dynamics_state_dict"], C["system_dynamics_state_dict"])
    assert A["provenance"]["seed"] == 0 and C["provenance"]["seed"] == 1


def test_seed_gives_identical_init_across_datasets(tmp_path):
    """With --seed the initial weights do not depend on the data (lr 0 keeps them)."""
    d1 = pilot64_dir(tmp_path, "d1")
    rng = np.random.default_rng(7)
    d2 = make_pilot_dir(str(tmp_path / "d2"), [("s0", 60, 1, make_rows(60, rng)), ("s1", 70, 1, make_rows(70, rng))])
    a, b = str(tmp_path / "a.pt"), str(tmp_path / "b.pt")
    run_fitter(FITTER, ["--data", d1, "--output", a, "--seed", "0", "--lr", "0"] + TINY_FIT, ext_seed=1)
    run_fitter(FITTER, ["--data", d2, "--output", b, "--seed", "0", "--lr", "0"] + TINY_FIT, ext_seed=2)
    assert_same_state_dict(load(a)["system_dynamics_state_dict"], load(b)["system_dynamics_state_dict"])


def test_lazy_windows_equal_materialized(tmp_path):
    rng = np.random.default_rng(3)
    files = [("a", 50, 64, make_env_major(64, 50, rng, fall_at=[(1, 40), (5, 10)])),
             ("b", 45, 3, make_env_major(3, 45, rng, fall_at=[(2, 44)])),
             ("c", 80, 1, make_rows(80, rng, terms=[60], zero_action=[60]))]
    d = make_pilot_dir(str(tmp_path / "mix"), files)
    segs, _, _ = fitmod.read_segments(d, 45, 12, 0, 8, 1)
    stats = fitmod.compute_normalizer(segs, 45, 12)
    segs2 = [(n, s, x.copy(), e) for n, s, x, e in segs]  # in-place normalization on CPU shares memory
    eager = fitmod.build_segment_windows(segs, 45, 12, 8, 1, 32, 8, "cpu", *stats)
    lazy = fitmod.build_segment_windows(segs2, 45, 12, 8, 1, 32, 8, "cpu", *stats, lazy=True)
    assert lazy.shape == tuple(eager.shape)
    ids = torch.arange(eager.shape[0])
    assert torch.equal(lazy[ids], eager)
    ids = torch.randint(0, eager.shape[0], (257,))
    assert torch.equal(lazy[ids], eager[ids])


def test_lazy_windows_fit_identical(tmp_path):
    data = pilot64_dir(tmp_path)
    a, b = str(tmp_path / "a.pt"), str(tmp_path / "b.pt")
    run_fitter(FITTER, ["--data", data, "--output", a, "--seed", "0"] + TINY_FIT)
    run_fitter(FITTER, ["--data", data, "--output", b, "--seed", "0", "--lazy_windows"] + TINY_FIT)
    A, B = load(a), load(b)
    assert_same_state_dict(A["system_dynamics_state_dict"], B["system_dynamics_state_dict"])
    assert A["num_windows"] == B["num_windows"]


def test_norm_stats_saved_and_used(tmp_path):
    data = pilot64_dir(tmp_path)
    rng = np.random.default_rng(5)
    st = {"state_mean": rng.normal(size=45), "state_std": rng.random(45) + 0.5,
          "action_mean": rng.normal(size=12), "action_std": rng.random(12) + 0.5}
    npz = str(tmp_path / "norm.npz")
    np.savez(npz, **st)
    out = str(tmp_path / "n.pt")
    run_fitter(FITTER, ["--data", data, "--output", out, "--seed", "0", "--norm_stats", npz] + TINY_FIT)
    ck = load(out)
    for k in NORM_KEYS:
        assert ck[k].shape == (1, 1, len(st[k]))
        assert torch.equal(ck[k], torch.tensor(st[k], dtype=torch.float32).view(1, 1, -1)), k
    assert ck["norm_stats_path"] == os.path.abspath(npz)
    assert ck["norm_stats_sha256"] == rc.sha256_file(npz)
    assert ck["provenance"]["normalizer_source"] == npz
    # the windows the model is trained on are normalized with exactly these stats
    segs, _, _ = fitmod.read_segments(data, 45, 12, 0, 8, 1)
    raw = np.concatenate([x for _, _, x, _ in segs]).copy()
    loaded = fitmod.load_normalizer(npz, 45, 12)
    lazy = fitmod.build_segment_windows(segs, 45, 12, 8, 1, 32, 8, "cpu", *loaded, lazy=True)
    flat = lazy.flat.numpy()
    exp_s = ((torch.from_numpy(raw[:, :45]) - loaded[0].view(1, -1)) / loaded[1].view(1, -1)).numpy()
    exp_a = ((torch.from_numpy(raw[:, 45:57]) - loaded[2].view(1, -1)) / loaded[3].view(1, -1)).numpy()
    np.testing.assert_array_equal(flat[:, :45], exp_s)
    np.testing.assert_array_equal(flat[:, 45:57], exp_a)
    np.testing.assert_array_equal(flat[:, 57:], raw[:, 57:])


def test_norm_stats_equal_to_own_stats_gives_identical_fit(tmp_path):
    """make_shared_norm on the fit's own data reproduces compute_normalizer bit-for-bit (float32),
    so --norm_stats <that file> yields the identical model as no --norm_stats."""
    data = pilot64_dir(tmp_path)
    npz = str(tmp_path / "own.npz")
    assert msn.main(["--inputs", data, "--out", npz]) == 0
    a, b = str(tmp_path / "a.pt"), str(tmp_path / "b.pt")
    run_fitter(FITTER, ["--data", data, "--output", a, "--seed", "0"] + TINY_FIT)
    run_fitter(FITTER, ["--data", data, "--output", b, "--seed", "0", "--norm_stats", npz] + TINY_FIT)
    A, B = load(a), load(b)
    for k in NORM_KEYS:
        assert torch.equal(A[k], B[k]), k
    assert_same_state_dict(A["system_dynamics_state_dict"], B["system_dynamics_state_dict"])


def test_norm_stats_rejects_bad_file(tmp_path):
    npz = str(tmp_path / "bad.npz")
    np.savez(npz, state_mean=np.zeros(44), state_std=np.ones(45), action_mean=np.zeros(12), action_std=np.ones(12))
    with pytest.raises(RuntimeError, match="state_mean"):
        fitmod.load_normalizer(npz, 45, 12)
    np.savez(npz, state_mean=np.zeros(45), state_std=np.zeros(45), action_mean=np.zeros(12), action_std=np.ones(12))
    with pytest.raises(RuntimeError, match="std"):
        fitmod.load_normalizer(npz, 45, 12)


def test_num_envs_1_windows_never_cross_segment_ends(tmp_path):
    rng = np.random.default_rng(11)
    lens = [60, 75, 41, 90]
    files = []
    for i, n in enumerate(lens):
        x = make_rows(n, rng)
        x[:, 0] = i  # segment id marker (raw, normalize=False)
        x[:, 1] = np.arange(n)  # row index marker
        files.append((f"s{i}", n, 1, x))
    files[3][3][50, 65] = 1.0  # interior termination
    files[3][3][50, 45:57] = 0.0
    files[0][3][59, 65] = 1.0  # termination on the last row: allowed as final target
    d = make_pilot_dir(str(tmp_path / "single"), files)
    segs, rows, terms = fitmod.read_segments(d, 45, 12, 0, 8, 1)
    assert rows == sum(lens) and terms == 2 and all(s[3] == 1 for s in segs)
    z = [torch.zeros(1, 1, 45), torch.ones(1, 1, 45), torch.zeros(1, 1, 12), torch.ones(1, 1, 12)]
    for lazy in (False, True):
        segs_c = [(n, s, x.copy(), e) for n, s, x, e in segs]
        w = fitmod.build_segment_windows(segs_c, 45, 12, 8, 1, 32, 8, "cpu", *z, normalize=False, lazy=lazy)
        W = w[torch.arange(w.shape[0])].numpy()
        assert (W[:, :, 0] == W[:, :1, 0]).all()  # one segment per window
        assert (np.diff(W[:, :, 1], axis=1) == 1).all()  # consecutive rows
        assert (W[:, :-1, 65] == 0).all()  # no termination before the final target row
        # expected count: sum(n - 39) minus windows i with i <= 50 <= i + 38 in s3 (i in 12..50 -> 39)
        assert W.shape[0] == sum(n - SEQ_LEN + 1 for n in lens) - 39
    out = str(tmp_path / "s.pt")
    run_fitter(FITTER, ["--data", d, "--output", out, "--seed", "0", "--lazy_windows"] + TINY_FIT)
    assert load(out)["total_rows"] == sum(lens)


def test_num_envs_column_rows_check(tmp_path):
    rng = np.random.default_rng(2)
    d = make_pilot_dir(str(tmp_path / "bad"), [("a", 50, 3, make_rows(150, rng))])
    with open(os.path.join(d, "manifest.csv"), "w") as fh:
        fh.write("name,steps,rows,num_envs\na,50,150,2\n")
    with pytest.raises(RuntimeError, match="steps\\*num_envs"):
        fitmod.read_segments(d, 45, 12, 0, 8, 1)
    # a 3-env file: windows never cross env blocks
    x = make_rows(150, rng)
    x[:, 0] = np.repeat(np.arange(3), 50)
    x[:, 1] = np.tile(np.arange(50), 3)
    d2 = make_pilot_dir(str(tmp_path / "three"), [("a", 50, 3, x)])
    segs, _, _ = fitmod.read_segments(d2, 45, 12, 0, 8, 1)
    z = [torch.zeros(1, 1, 45), torch.ones(1, 1, 45), torch.zeros(1, 1, 12), torch.ones(1, 1, 12)]
    w = fitmod.build_segment_windows(segs, 45, 12, 8, 1, 32, 8, "cpu", *z, normalize=False).numpy()
    assert w.shape[0] == 3 * 11 and (w[:, :, 0] == w[:, :1, 0]).all() and (np.diff(w[:, :, 1], axis=1) == 1).all()


# ======================================================================================
# 2. make_curated_segments
# ======================================================================================
def curated_fixture(tmp_path, perturb_concat=False):
    rng = np.random.default_rng(21)
    segs = {
        "seg_a": make_rows(60, rng, terms=[30, 59], zero_action=[30]),  # interior fall + forced seam (non-zero action)
        "seg_b": make_rows(50, rng, terms=[49], zero_action=[49]),  # real fall on the last row -> kept
        "seg_c": make_rows(45, rng),  # no label on the last row
        "seg_d": make_rows(55, rng, terms=[20, 54]),  # interior term with NON-zero action (anomaly) + seam
    }
    src = tmp_path / "src"
    src.mkdir()
    paths = []
    for n, x in segs.items():
        write_csv(str(src / f"{n}.csv"), x)
        paths.append(str(src / f"{n}.csv"))
    concat = np.concatenate([x.copy() for x in segs.values()])
    ends = np.cumsum([len(x) for x in segs.values()]) - 1
    concat[ends, 65] = 1.0  # forced seams in the concatenated file
    if perturb_concat:
        concat[7, 3] += 0.5
    write_csv(str(tmp_path / "concat.csv"), concat)
    return segs, paths, str(tmp_path / "concat.csv")


def test_curated_seam_vs_real_fall(tmp_path, capsys):
    segs, paths, concat = curated_fixture(tmp_path)
    out = str(tmp_path / "out")
    assert mcs.main(["--segs", *paths, "--concat", concat, "--out", out]) == 0
    m = pd.read_csv(os.path.join(out, "manifest.csv"))
    assert list(m.columns) == ["name", "steps", "rows", "num_envs"]
    assert m["name"].tolist() == list(segs) and (m["num_envs"] == 1).all()
    assert m["rows"].tolist() == [len(x) for x in segs.values()] and (m["steps"] == m["rows"]).all()
    cards = json.load(open(os.path.join(out, "cards.json")))
    f = {c["name"]: c for c in cards["files"]}
    assert f["seg_a"]["last_row_status"] == "seam_removed" and f["seg_a"]["seams_removed"] == 1 and f["seg_a"]["real_terms"] == 1
    assert f["seg_b"]["last_row_status"] == "real_fall_kept" and f["seg_b"]["real_terms"] == 1
    assert f["seg_c"]["last_row_status"] == "no_term" and f["seg_c"]["real_terms"] == 0
    assert f["seg_d"]["seams_removed"] == 1 and f["seg_d"]["interior_terms_nonzero_action"] == 1
    assert cards["totals"] == {**cards["totals"], "rows": 210, "real_terms": 3, "seams_removed": 2, "terms_raw": 5}
    for n, x in segs.items():
        y = rc.read_numeric(os.path.join(out, f"{n}.csv"))
        exp = rc.parse_lines(rc.read_lines(str(tmp_path / "src" / f"{n}.csv")))
        if n in ("seg_a", "seg_d"):
            exp[-1, 65] = 0.0
        np.testing.assert_array_equal(y, exp)
        # verbatim copy of every line except the relabelled one
        a, b = rc.read_lines(str(tmp_path / "src" / f"{n}.csv")), rc.read_lines(os.path.join(out, f"{n}.csv"))
        assert a[:-1] == b[:-1]
    chk = cards["concat_check"]
    assert chk["ok"] and chk["rows_not_exact"] == 0
    assert chk["n_term_label_diffs"] == 1 and chk["term_label_diffs"][0]["segment"] == "seg_c"
    assert chk["term_label_diffs"][0]["is_segment_last_row"]
    assert "NON-zero action" in capsys.readouterr().out
    # the fitter reads the directory (num_envs=1)
    segs_f, rows, terms = fitmod.read_segments(out, 45, 12, 0, 8, 1)
    assert rows == 210 and terms == 3


def test_curated_concat_mismatch_warns_but_succeeds(tmp_path, capsys):
    _, paths, concat = curated_fixture(tmp_path, perturb_concat=True)
    out = str(tmp_path / "out")
    assert mcs.main(["--segs", *paths, "--concat", concat, "--out", out]) == 0
    chk = json.load(open(os.path.join(out, "cards.json")))["concat_check"]
    assert not chk["ok"] and chk["rows_not_close"] == 1
    assert "WARNING concat check" in capsys.readouterr().out
    # refuses to overwrite without --force
    with pytest.raises(SystemExit):
        mcs.main(["--segs", *paths, "--out", out])
    assert mcs.main(["--segs", *paths, "--out", out, "--force"]) == 0


def test_relabel_keeps_style():
    assert mcs.relabel_seam_line("1.5,2,1") == "1.5,2,0"
    assert mcs.relabel_seam_line("1.5,2,1.0") == "1.5,2,0.0"
    assert mcs.relabel_seam_line("1.5,2,1.000000e+00") == "1.5,2,0.0"


# ======================================================================================
# 3. make_subsample
# ======================================================================================
def plusfail_fixture(tmp_path):
    """3 env-major files (different steps), symlinked into a pilot dir like segments_plus_fail_train_flat."""
    rng = np.random.default_rng(31)
    real = tmp_path / "real"
    real.mkdir()
    files = []
    for name, steps, ne, falls in (("f800", 50, 64, [(1, 45), (9, 47), (9, 20)]), ("f1200", 60, 64, [(2, 50)]),
                                   ("f1500", 45, 32, [(31, 44)])):
        x = make_env_major(ne, steps, rng, fall_at=falls)
        x[:, 2] = np.repeat(np.arange(ne), steps)  # env id marker
        write_csv(str(real / f"{name}.csv"), x)
        files.append((name, steps, ne, x))
    d = tmp_path / "plusfail"
    d.mkdir()
    with open(d / "manifest.csv", "w") as fh:
        fh.write("name,steps,rows,num_envs\n" + "".join(f"{n},{s},{len(x)},{ne}\n" for n, s, ne, x in files))
    for n, *_ in files:
        os.symlink(real / f"{n}.csv", d / f"{n}.csv")
    return str(d), files


def test_subsample_blocks_proportional_deterministic(tmp_path):
    src, files = plusfail_fixture(tmp_path)
    total = sum(len(x) for *_, x in files)
    target = 900
    o1, o2, o3 = (str(tmp_path / n) for n in ("o1", "o2", "o3"))
    assert msub.main(["--src", src, "--out", o1, "--target_rows", str(target), "--seed", "0"]) == 0
    assert msub.main(["--src", src, "--out", o2, "--target_rows", str(target), "--seed", "0"]) == 0
    assert msub.main(["--src", src, "--out", o3, "--target_rows", str(target), "--seed", "5"]) == 0
    c1 = json.load(open(os.path.join(o1, "cards.json")))
    c2 = json.load(open(os.path.join(o2, "cards.json")))
    c3 = json.load(open(os.path.join(o3, "cards.json")))
    assert [f["env_ids_kept"] for f in c1["files"]] == [f["env_ids_kept"] for f in c2["files"]]
    assert [f["env_ids_kept"] for f in c1["files"]] != [f["env_ids_kept"] for f in c3["files"]]
    for n, *_ in files:
        assert open(os.path.join(o1, f"{n}.csv")).read() == open(os.path.join(o2, f"{n}.csv")).read()
    t = c1["totals"]
    assert abs(t["rows"] - target) / target <= 0.02
    m = pd.read_csv(os.path.join(o1, "manifest.csv"))
    exp_terms = 0
    for (name, steps, ne, x), card in zip(files, c1["files"]):
        k = card["blocks_kept"]
        quota = target * len(x) / total / steps
        assert abs(k - quota) < 1.0 + 1e-9  # proportional to the nearest block
        envs = card["env_ids_kept"]
        assert len(envs) == len(set(envs)) == k and all(0 <= e < ne for e in envs)
        y = rc.read_numeric(os.path.join(o1, f"{name}.csv"))
        src_x = rc.read_numeric(os.path.join(src, f"{name}.csv"))
        assert len(y) == k * steps
        for j, e in enumerate(envs):  # whole blocks, verbatim, in env order
            np.testing.assert_array_equal(y[j * steps:(j + 1) * steps], src_x[e * steps:(e + 1) * steps])
        row = m[m["name"] == name].iloc[0]
        assert (row["steps"], row["rows"], row["num_envs"]) == (steps, k * steps, k)
        terms = int((y[:, 65] > 0.5).sum())
        assert card["real_terms"] == terms
        exp_terms += terms
    assert t["real_terms"] == exp_terms and t["blocks"] == sum(f["blocks_kept"] for f in c1["files"])
    segs, rows, terms = fitmod.read_segments(o1, 45, 12, 0, 8, 1)
    assert rows == t["rows"] and terms == t["real_terms"]


def test_subsample_terms_counted_when_present(tmp_path):
    src, files = plusfail_fixture(tmp_path)
    # keep everything -> all source terms must be counted
    total = sum(len(x) for *_, x in files)
    out = str(tmp_path / "all")
    assert msub.main(["--src", src, "--out", out, "--target_rows", str(total)]) == 0
    c = json.load(open(os.path.join(out, "cards.json")))
    assert c["totals"]["rows"] == total and c["totals"]["real_terms"] == 5 == c["totals"]["source_terms"]


def test_subsample_out_of_tolerance_fails(tmp_path):
    src, _ = plusfail_fixture(tmp_path)
    with pytest.raises(SystemExit):
        msub.main(["--src", src, "--out", str(tmp_path / "x"), "--target_rows", "20"])  # one block is 45-60 rows


def test_allocate_blocks_real_manifest():
    """June +fail manifest (train_2000.log): 18 files, 64 envs, steps 800-1500 -> 115,250 rows (+0.22 %)."""
    steps = [800, 800, 800, 1200, 1200, 1200, 1200, 1200, 1200, 1000, 1000, 1000, 1000, 1025, 1000, 1500, 1500, 1500]
    k = msub.allocate_blocks([64 * s for s in steps], steps, [64] * 18, 115000)
    tot = sum(a * b for a, b in zip(k, steps))
    assert abs(tot - 115000) / 115000 < 0.02 and set(k) <= {5, 6}


# ======================================================================================
# 4. make_shared_norm
# ======================================================================================
def test_shared_norm_matches_numpy(tmp_path):
    rng = np.random.default_rng(41)
    d1 = make_pilot_dir(str(tmp_path / "d1"), [("a", 50, 2, make_rows(100, rng, zero_action=[3, 4])),
                                               ("b", 60, 1, make_rows(60, rng))])
    d2 = make_pilot_dir(str(tmp_path / "d2"), [("c", 45, 4, make_rows(180, rng, terms=[7, 89], zero_action=[7]))],
                        num_envs_col=False)
    x_extra = make_rows(33, rng) * 3 + 1
    extra = tmp_path / "glob"
    extra.mkdir()
    write_csv(str(extra / "e.csv"), x_extra)
    shutil.copy(os.path.join(d2, "manifest.csv"), extra / "manifest.csv")  # must be skipped (3 columns)
    out = str(tmp_path / "norm.npz")
    assert msn.main(["--inputs", d1, d2, str(extra / "*.csv"), os.path.join(d1, "a.csv"), "--out", out,
                     "--chunksize", "37"]) == 0
    allx = np.concatenate([rc.read_numeric(p).astype(np.float32).astype(np.float64)
                           for p in (os.path.join(d1, "a.csv"), os.path.join(d1, "b.csv"), os.path.join(d2, "c.csv"),
                                     str(extra / "e.csv"))])
    z = np.load(out, allow_pickle=False)
    np.testing.assert_allclose(z["state_mean"], allx[:, :45].mean(0), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(z["state_std"], allx[:, :45].std(0) + 1e-6, rtol=1e-10)
    np.testing.assert_allclose(z["action_mean"], allx[:, 45:57].mean(0), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(z["action_std"], allx[:, 45:57].std(0) + 1e-6, rtol=1e-10)
    meta = json.loads(str(z["meta_json"]))
    assert int(z["n_rows"]) == meta["n_rows"] == len(allx) == 373  # duplicate a.csv counted once
    assert meta["reset_like_zero_action_rows"] == 3
    b2 = meta["by_input"][d2]
    assert b2["terms"] == 2 and b2["terms_zero_action"] == 1 and b2["terms_on_block_last_row"] == 1  # row 89 = env 1 end
    assert os.path.isfile(out + ".json")
    # equals the fitter's own normalizer on a single directory
    segs, _, _ = fitmod.read_segments(d1, 45, 12, 0, 8, 1)
    own = fitmod.compute_normalizer(segs, 45, 12)
    out1 = str(tmp_path / "d1.npz")
    msn.main(["--inputs", d1, "--out", out1])
    loaded = fitmod.load_normalizer(out1, 45, 12)
    for a, b in zip(own, loaded):
        assert torch.equal(a, b)


# ======================================================================================
# 5. shell scripts: syntax + end-to-end dry run on a fake repo layout
# ======================================================================================
@pytest.mark.parametrize("script", ["run_p1_refit.sh", "score_p1_refit.sh"])
def test_shell_syntax(script):
    subprocess.run(["bash", "-n", os.path.join(REFIT_DIR, script)], check=True)


def fake_repo(root):
    rng = np.random.default_rng(51)
    nd = root / "assets" / "data" / "go2_noise"
    nd.mkdir(parents=True)
    lens = {"seg_n00": 60, "seg_n02": 60, "seg_n04": 75, "seg_n08": 90, "seg_n10": 80, "seg_n12": 85}  # 450 rows
    segs = []
    for i, (n, L) in enumerate(lens.items()):
        falls = [L // 2] if i >= 3 else []
        x = make_rows(L, rng, terms=falls, zero_action=falls)
        x[-1, 65] = 1.0 if i % 2 == 0 else 0.0  # some seg files carry the forced seam label, some not
        write_csv(str(nd / f"{n}.csv"), x)
        segs.append(x.copy())
    concat = np.concatenate(segs)
    concat[np.cumsum(list(lens.values())) - 1, 65] = 1.0
    write_csv(str(nd / "state_action_data_0.csv"), concat)
    real = root / "assets" / "data" / "go2_pilot_1m" / "segments_clean_terms_real"
    real.mkdir(parents=True)
    pf = root / "assets" / "data" / "go2_pilot_1m" / "segments_plus_fail_train_flat"
    pf.mkdir(parents=True)
    man = ["name,steps,rows"]
    for n, falls in (("bad_x_n00", [(0, 44), (5, 30)]), ("mixed_y_n08", [(7, 41)]), ("ckpt_z_n08", [])):
        x = make_env_major(64, 45, rng, fall_at=falls)
        write_csv(str(real / f"{n}.csv"), x)
        os.symlink(real / f"{n}.csv", pf / f"{n}.csv")
        man.append(f"{n},45,{len(x)}")
    (pf / "manifest.csv").write_text("\n".join(man) + "\n")
    tr = root / "logs" / "wm_fit" / "go2_plus_fail_ens5" / "exploit_trace"
    tr.mkdir(parents=True)
    t9 = make_env_major(4, 100, rng, fall_at=[(0, 70), (2, 85), (3, 60)])
    t6 = make_env_major(4, 120, rng, fall_at=[(1, 90), (3, 100), (0, 75)])
    write_csv(str(tr / "2026-06-21_14-30-38_9_policy499_trace.csv"), t9)
    write_csv(str(tr / "2026-06-21_14-22-41_6_policy499_trace_2000.csv"), t6)


def shell_env(root, **kw):
    env = {**os.environ, "PY": sys.executable, "ROOT": str(root), "RSL": RSL_RL_PATH, "DEVICE": "cpu", "ITERATIONS": "3",
           "NUM_MINI_BATCHES": "1", "MINI_BATCH_SIZE": "8", "LOG_INTERVAL": "1", "SAVE_INTERVAL": "2", "WAIT": "1",
           "OMP_NUM_THREADS": "1"}
    env.update({k: str(v) for k, v in kw.items()})
    return env


def run_sh(script, env, *args):
    return subprocess.run(["bash", os.path.join("refit", script), *args], cwd=MB_DIR, env=env, capture_output=True,
                          text=True)


def check_fit_outputs(out, root):
    norm = os.path.join(out, "data", "shared_norm.npz")
    z = np.load(norm, allow_pickle=False)
    cards = json.load(open(os.path.join(out, "data", "curated_segments", "cards.json")))
    assert cards["totals"]["rows"] == 450 and cards["totals"]["real_terms"] == 3 and cards["totals"]["seams_removed"] == 3
    assert cards["concat_check"]["ok"]
    sub = json.load(open(os.path.join(out, "data", "plus_fail_115k", "cards.json")))
    assert sub["totals"]["rows"] == 450 and sub["totals"]["blocks"] == 10
    ck = {}
    for r in ("R1_curated", "R2_plusfail", "R3_plusfail_115k"):
        assert open(os.path.join(out, r, "EXIT_CODE")).read().strip() == "0"
        log = open(os.path.join(out, r, "stdout.log")).read()
        assert "[fit] DONE" in log and "[fit] seeded" in log and "lazy window shape" in log
        job = open(os.path.join(out, r, "job.sh")).read()
        assert "--seed 0" in job and "--norm_stats" in job and "--lazy_windows" in job and "--termination_pos_weight 0.0" in job
        c = load(os.path.join(out, r, "model_3.pt"))
        assert c["iter"] == 3 and c["norm_stats_sha256"] == rc.sha256_file(norm)
        for k in NORM_KEYS:
            assert torch.equal(c[k], torch.tensor(z[k], dtype=torch.float32).view(1, 1, -1))
        ck[r] = c
    assert ck["R1_curated"]["total_rows"] == 450 and ck["R3_plusfail_115k"]["total_rows"] == 450
    assert ck["R2_plusfail"]["total_rows"] == 3 * 64 * 45 and ck["R2_plusfail"]["total_real_terms"] == 3
    assert int(z["n_rows"]) == 450 + 3 * 64 * 45
    assert os.path.isfile(os.path.join(out, "STATUS.txt"))
    exp = open(os.path.join(out, "data", "expectations.txt")).read()
    assert "ok    curated concat check ok: True" in exp and "CHECK curated rows: 450 (expected 115000)" in exp


@pytest.fixture(scope="module")
def fake_layout(tmp_path_factory):
    root = tmp_path_factory.mktemp("fakerepo")
    fake_repo(root)
    return root


def test_run_refuses_low_gpu_then_sequential(fake_layout, tmp_path):
    """Fake nvidia-smi reporting 5000 MiB free: parallel launch refused (exit 3) after the probe,
    SEQUENTIAL=1 then runs the three fits one after another."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "nvidia-smi"
    fake.write_text("#!/usr/bin/env bash\n"
                    "case \"$*\" in *noheader*) echo 5000;; *query-compute-apps*) echo 'pid, used_memory';; "
                    "*) echo 'index, name, memory.total [MiB], memory.used [MiB], memory.free [MiB]'; "
                    "echo '0, FAKE, 49140 MiB, 44140 MiB, 5000 MiB';; esac\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = shell_env(fake_layout, TAG="seqtest", GPU_CHECK=1, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}")
    p = run_sh("run_p1_refit.sh", env)
    assert p.returncode == 3, p.stdout[-3000:] + p.stderr[-3000:]
    assert "REFUSING to start all three in parallel: free 5000 MiB < 12288 MiB" in p.stdout
    out = os.path.join(fake_layout, "logs", "wm_fit", "seqtest")
    assert "--iterations 2 " in open(os.path.join(out, "probe", "probe_cmd.txt")).read()
    assert "[fit] DONE" in open(os.path.join(out, "probe", "probe.log")).read()
    p = run_sh("run_p1_refit.sh", {**env, "SEQUENTIAL": "1"})
    assert p.returncode == 1 and "exists; refusing" in p.stdout  # protects an existing OUT
    p = run_sh("run_p1_refit.sh", {**env, "SEQUENTIAL": "1", "FORCE": "1"})
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-3000:]
    assert "sequential runner pid=" in open(os.path.join(out, "STATUS.txt")).read()
    check_fit_outputs(out, fake_layout)
    assert glob.glob(out + ".bak.*")


def test_run_parallel_then_score_end_to_end(fake_layout):
    env = shell_env(fake_layout, TAG="partest")
    p = run_sh("run_p1_refit.sh", env)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-3000:]
    out = os.path.join(fake_layout, "logs", "wm_fit", "partest")
    check_fit_outputs(out, fake_layout)
    st = run_sh("run_p1_refit.sh", env, "status")
    assert st.returncode == 0 and st.stdout.count("finished exit=0") == 3
    senv = shell_env(fake_layout, TAG="partest", NUM_ENVS=4, STEPS_T9=100, STEPS_T6=120, NUM_NEG=40, BOOT=20,
                     OUT=os.path.join(fake_layout, "logs", "p1_eval_refit"))
    s = run_sh("score_p1_refit.sh", senv)
    assert s.returncode == 0, s.stdout[-4000:] + s.stderr[-3000:]
    ev = os.path.join(fake_layout, "logs", "p1_eval_refit")
    assert "checkpoint check: OK" in open(os.path.join(ev, "checkpoints.txt")).read()
    runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(ev, "*", "summary.json")))
    assert runs == sorted(f"{r}_{t}" for r in ("R1_curated", "R2_plusfail", "R3_plusfail_115k") for t in ("t9", "t6"))
    for r in runs:
        js = json.load(open(os.path.join(ev, r, "summary.json")))
        assert {"epi", "term_logit", "knn10"} <= set(js["signals"])
        assert js["counts"]["n_pre5"] >= 1
        files = js["knn"]["files"]
        if r.startswith("R1"):
            assert all("curated_segments" in f["path"] for f in files) and len(files) == 6
        if r.startswith("R3"):
            assert all("plus_fail_115k" in f["path"] for f in files) and len(files) == 3
    assert os.path.isfile(os.path.join(ev, "comparison.txt"))


def test_score_refuses_unfinished(fake_layout):
    env = shell_env(fake_layout, TAG="partest_missing", NUM_ENVS=4, STEPS_T9=100, STEPS_T6=120)
    s = run_sh("score_p1_refit.sh", env)
    assert s.returncode != 0 and "missing checkpoint" in s.stdout
