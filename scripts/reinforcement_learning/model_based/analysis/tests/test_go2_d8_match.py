"""A4 D8 lane match: the two PPO lanes differ only in the world model, on the launch commit, clean."""
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import go2_d8_match as m  # noqa: E402

HEAD = "a" * 40


def lane(wm_sha, run):
    return {"schema_version": 1, "created_utc": f"2026-12-0{run[-1]}", "seed": 1000, "task": "go2_flat",
            "run_num": run, "mode": "paper_warm",
            "args": {"run_num": run, "seed": 1000, "startup_resets": "/x/reset_states.npz",
                     "startup_resets_sha256": "r" * 64, "startup_reset_share": 0.12},
            "git": {"commit": HEAD, "dirty": False, "status": []},
            "world_model_checkpoint": {"path": f"/wm/{wm_sha[:4]}.pt", "sha256": wm_sha},
            "initial_policy_checkpoint": {"path": "/p/policy_4298.pt", "sha256": "p" * 64},
            "resolved_config": {"model_architecture_config": {"resume_path": f"/wm/{wm_sha[:4]}.pt"},
                                "policy_training_config": {"max_iterations": 300},
                                "environment_config": {"uncertainty_penalty_weight": -1.0},
                                "data_config": {"dataset_root": "/d8/start"}},
            "start_state_files": [{"path": "/d8/start/state_action_data_0.csv", "sha256": "s" * 64}],
            "runtime": {"gpu": "RTX 6000 Ada", "torch": "2.7.0"}}


EXPECT = dict(head=HEAD, seed=1000, task="go2_flat", mode="paper_warm", init_policy_sha256="p" * 64,
              startup_resets_sha256="r" * 64, startup_reset_share=0.12, max_iterations=300, penalty_weight=-1.0,
              wm_f8_sha256="f" * 64, wm_u8_sha256="u" * 64)


def test_matched_lanes_pass():
    assert m.check(lane("f" * 64, "4003"), lane("u" * 64, "4004"), EXPECT) == []


def test_each_difference_is_caught():
    f, u = lane("f" * 64, "4003"), lane("u" * 64, "4004")
    cases = []
    x = copy.deepcopy(u); x["git"]["dirty"] = True; cases.append((x, "dirty"))
    x = copy.deepcopy(u); x["git"]["commit"] = "b" * 40; cases.append((x, "launch HEAD"))
    x = copy.deepcopy(u); x["args"]["startup_reset_share"] = 0.2; cases.append((x, "startup_reset_share"))
    x = copy.deepcopy(u); x["start_state_files"][0]["sha256"] = "t" * 64; cases.append((x, "outside the world model"))
    x = copy.deepcopy(u); x["resolved_config"]["policy_training_config"]["max_iterations"] = 299; cases.append((x, "max_iterations"))
    x = copy.deepcopy(u); x["world_model_checkpoint"]["sha256"] = "f" * 64; cases.append((x, "same world model"))
    x = copy.deepcopy(u); x["runtime"]["gpu"] = "other"; cases.append((x, "outside the world model"))
    for bad, text in cases:
        probs = m.check(f, bad, EXPECT)
        assert any(text in p for p in probs), (text, probs)


def test_cli(tmp_path):
    for name, r in (("f.json", lane("f" * 64, "4003")), ("u.json", lane("u" * 64, "4004")), ("e.json", EXPECT)):
        (tmp_path / name).write_text(json.dumps(r))
    assert m.main(["--f8", str(tmp_path / "f.json"), "--u8", str(tmp_path / "u.json"), "--expect", str(tmp_path / "e.json")]) == 0
    assert m.main(["--f8", str(tmp_path / "f.json"), "--u8", str(tmp_path / "f.json"), "--expect", str(tmp_path / "e.json")]) == 1
