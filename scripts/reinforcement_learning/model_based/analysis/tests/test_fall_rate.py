"""CPU tests for falls per 1000 s with the exact Poisson CI (plan v6 §10)."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fall_rate as F  # noqa: E402
from episode_stats import EpisodeStats  # noqa: E402

# Garwood 95 % limits from standard tables
TABLE = {0: (0.0, 3.6889), 1: (0.0253, 5.5716), 5: (1.6235, 11.6683), 10: (4.7954, 18.3904), 100: (81.3639, 121.6268)}


@pytest.mark.parametrize("k", sorted(TABLE))
def test_exact_interval_matches_tables(k):
    lo, hi = F.poisson_ci(k)
    assert lo == pytest.approx(TABLE[k][0], abs=2e-4) and hi == pytest.approx(TABLE[k][1], abs=2e-4)


def test_rate_scales_with_exposure_and_pools_by_sum():
    rate, lo, hi = F.falls_per_1000s(17, 12800)  # June walker, 10 s commands, seeds 42+43
    assert rate == pytest.approx(17 / 12.8) and lo < rate < hi
    with pytest.raises(ValueError):
        F.falls_per_1000s(1, 0)
    with pytest.raises(ValueError):
        F.poisson_ci(-1)


def test_falls_from_eval_log(tmp_path):
    log = tmp_path / "eval.log"
    log.write_text("[eval_ep] falls: 99 (not the block)\n================ EPISODE STATS ================\n"
                   "  episodes_completed: 641\n  falls: 9\n  falls_within_100_steps_of_cmd_change: 3\n")
    assert F.falls_from_log(log) == 9
    assert F.main(["--logs", str(log), str(log), "--exposure_per_log_s", "6400"]) == 0


def test_episode_stats_summary_reports_rate():
    stats = EpisodeStats(2, 0.02, 20.0)
    s = stats.summary()  # no steps yet: exposure 0, rate nan
    assert s["exposure_s"] == 0.0 and np.isnan(s["falls_per_1000s"])
