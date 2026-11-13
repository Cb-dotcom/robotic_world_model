"""CPU tests for e1v_match.py (E1 validation matching check, plan v5.2)."""
import json
from pathlib import Path
import subprocess
import sys

MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MB))
import e1v_match as M  # noqa: E402

REPO = MB.parents[2]
FIXED = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def run_args(penalty=-1.0, wm_sha="eth", seed=1, commit=FIXED):
    return {"seed": seed, "args": {"mode": "paper", "seed": seed, "run_num": 1701},
            "git": {"commit": commit, "dirty": False},
            "world_model_checkpoint": {"path": "x", "sha256": wm_sha},
            "normalizer_contract": {"norm_stats_sha256": "3d21"},
            "resolved_config": {"environment_config": {"num_envs": 4096, "uncertainty_penalty_weight": penalty},
                                "model_architecture_config": {"resume_path": "eth.pt", "ensemble_size": 5},
                                "policy_algorithm_config": {"learning_rate": 1e-3, "entropy_coef": 5e-3}}}


def test_only_allowed_differences_pass():
    control = run_args()
    cand = run_args(penalty=0.0, wm_sha="new")
    cand["resolved_config"]["model_architecture_config"]["resume_path"] = "new.pt"
    cand["args"].update(run_num=2003, termination_rule="selected", alive_bonus=0.0)  # newer defaults
    problems, _ = M.compare(control, cand, "new", 0.0, REPO)
    assert problems == []


def test_recipe_difference_fails():
    cand = run_args()
    cand["resolved_config"]["policy_algorithm_config"]["entropy_coef"] = 1e-2
    problems, _ = M.compare(run_args(), cand, "eth", -1.0, REPO)
    assert any("entropy_coef" in p for p in problems)


def test_wrong_penalty_seed_wm_normalizer_and_flags_fail():
    cand = run_args(penalty=-1.0, wm_sha="other", seed=2)
    cand["normalizer_contract"]["norm_stats_sha256"] = "zzz"
    cand["args"]["mode"] = "paper_warm"
    cand["args"]["alive_bonus"] = 0.1
    problems, _ = M.compare(run_args(), cand, "eth", 0.0, REPO)
    text = "\n".join(problems)
    for word in ("seed", "world model", "penalty", "normalizer", "mode", "alive_bonus"):
        assert word in text


def test_commit_without_acceleration_fix_fails():
    old = "0cebd87d4386b42ba23e6449d7ce3402b792c2a5"  # 1601's commit, before d9d2a66
    if subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", old + "^{commit}"]).returncode != 0:
        return  # object not in this clone
    problems, _ = M.compare(run_args(commit=old), run_args(), "eth", -1.0, REPO)
    assert any("lacks the acceleration fix" in p for p in problems)


def test_cli_exit_codes(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(run_args()))
    b.write_text(json.dumps(run_args()))
    assert M.main(["--control", str(a), "--candidate", str(b), "--expect-wm-sha", "eth",
                   "--expect-penalty", "-1", "--repo", str(REPO)]) == 0
    assert M.main(["--control", str(a), "--candidate", str(b), "--expect-wm-sha", "eth",
                   "--expect-penalty", "0", "--repo", str(REPO)]) == 1


def test_replication_seed():
    control, cand = run_args(seed=1), run_args(seed=2)
    assert M.compare(control, cand, "eth", -1.0, REPO)[0]  # plain mode: seeds must match
    assert M.compare(control, cand, "eth", -1.0, REPO, expect_seed=2)[0] == []
    assert any("expected 3" in p for p in M.compare(control, cand, "eth", -1.0, REPO, expect_seed=3)[0])
