#!/usr/bin/env python3
"""Falls per 1000 s with an exact (Garwood) Poisson confidence interval. Pure Python, no scipy.

Plan v6 §10: compute the paper's fall rate in code. Convention (run-log, E1): exposure = envs x steps x
step_dt of the evaluation, e.g. 32 x 10000 x 0.02 s = 6400 s per seed, 12,800 s for seeds 42 + 43 pooled.
Pooling adds falls and exposure; it never averages rates.

  fall_rate.py --falls 17 --exposure_s 12800
  fall_rate.py --logs eval/june_cmd10_s42/eval.log eval/june_cmd10_s43/eval.log --exposure_per_log_s 6400
"""
from __future__ import annotations

import argparse
import math
import re
import sys


def _log_cdf(k, lam):
    """log P(X <= k) for X ~ Poisson(lam), k >= 0."""
    if lam == 0.0:
        return 0.0
    terms = [-lam + i * math.log(lam) - math.lgamma(i + 1) for i in range(k + 1)]
    top = max(terms)
    return top + math.log(sum(math.exp(t - top) for t in terms))


def _solve(f, lo, hi, iters=200):
    """Root of a decreasing function f on [lo, hi] by bisection."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def poisson_ci(k, conf=0.95):
    """Exact two-sided interval for the mean of a Poisson count k (Garwood 1936)."""
    if k < 0 or int(k) != k:
        raise ValueError("k must be a non-negative integer")
    k = int(k)
    alpha = 1.0 - conf
    top = k + 20.0 + 20.0 * math.sqrt(k + 1.0)
    # upper: P(X <= k; lam) = alpha/2 (decreasing in lam)
    upper = _solve(lambda lam: math.exp(_log_cdf(k, lam)) - alpha / 2, 0.0, top)
    # lower: P(X >= k; lam) = alpha/2; P(X >= k) increases with lam, so solve alpha/2 - P(X >= k)
    lower = 0.0 if k == 0 else _solve(lambda lam: alpha / 2 - (1.0 - math.exp(_log_cdf(k - 1, lam))), 0.0, top)
    return lower, upper


def falls_per_1000s(falls, exposure_s, conf=0.95):
    """(rate, lower, upper) in falls per 1000 s of exposure."""
    if exposure_s <= 0:
        raise ValueError("exposure_s must be > 0")
    lo, hi = poisson_ci(falls, conf)
    scale = 1000.0 / exposure_s
    return falls * scale, lo * scale, hi * scale


def falls_from_log(path):
    """The 'falls: N' line of the EPISODE STATS block of an eval_episode_stats.py log."""
    text = open(path).read()
    block = text.split("EPISODE STATS", 1)
    if len(block) != 2:
        raise ValueError(f"{path}: no EPISODE STATS block")
    match = re.search(r"^\s*falls:\s*(\d+)\s*$", block[1], flags=re.MULTILINE)
    if not match:
        raise ValueError(f"{path}: no 'falls:' line")
    return int(match.group(1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--falls", type=int)
    p.add_argument("--exposure_s", type=float)
    p.add_argument("--logs", nargs="+", help="eval_episode_stats.py logs to pool")
    p.add_argument("--exposure_per_log_s", type=float, default=6400.0, help="32 envs x 10000 steps x 0.02 s")
    p.add_argument("--conf", type=float, default=0.95)
    args = p.parse_args(argv)
    if args.logs:
        falls = sum(falls_from_log(path) for path in args.logs)
        exposure = args.exposure_per_log_s * len(args.logs)
    elif args.falls is not None and args.exposure_s is not None:
        falls, exposure = args.falls, args.exposure_s
    else:
        p.error("give --falls and --exposure_s, or --logs")
    rate, lo, hi = falls_per_1000s(falls, exposure, args.conf)
    print(f"falls {falls} in {exposure:.0f} s: {rate:.3f} per 1000 s, exact Poisson {100 * args.conf:.0f} % CI "
          f"[{lo:.3f}, {hi:.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
