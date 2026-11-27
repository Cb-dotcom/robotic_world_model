# Real Go2 tools (ticket T11, 25 Nov 2026)

Off-robot analysis of lab Go2 EDU bags (ROS 2 Foxy, sqlite3). Nothing here drives the robot.
Needs only `numpy` and `rosbags` (no ROS install, no GPU). Message types are built in
(`go2_bag.py`); a wrong layout stops the script, never returns shifted numbers.

| Script | What it gives |
|---|---|
| `t11_v_check.py` | V test: tape displacement vs `/sportmodestate` position, its integrated body velocity, and every `nav_msgs/Odometry` topic (robot_odom, Point-LIO). Ratio per run and source; commanded speed; yaw drift. One `V GATE` line (decisions-log 18 Nov): **PASS/FAIL** (exit 0) only from Point-LIO `/state_estimation_at_scan`, when the listed rows are exactly 3 runs at commanded 0.30 and 3 at 0.45 m/s (one run = one CSV row = one walk; each row and each bag in exactly one speed group; no walk listed twice), every run on the 5 m line, bar 5 % per run, inclusive (|ratio − 1| ≤ 0.05); **INVALID** (exit 1) if a listed run is missing or errored, Point-LIO has no ratio in a run, or a run has no readable commanded speed (`/api/sport/request` Move x); otherwise **DIAGNOSTIC ONLY: reasons** (exit 2), e.g. too few or extra runs, another speed, a bag with several walks and no leg, Unitree sources only, another line length, `--max-rel-err`, `--speed-tol-mps` or `--line-tol-m` changed. Unitree sources never decide. |
| `t11_log_check.py` | Logger: rates, gaps > 20 / 30 / 100 ms, all-topic stops, `/lowstate` tick losses vs late delivery, 50 Hz policy emulation (flagged steps, stale steps that reuse the previous sample or tick, segments, usable 58-step windows without flagged or stale steps), max motor temperature, `/lowcmd` kp/kd. |
| `t11_h2_actuator.py` | H2: logged targets through Isaac Lab's DCMotor (25 / 0.5, 23.5 N m, 30 rad/s). Per joint: lag, gain, offset, RMSE in N m, position offset, clip counts. Stops (exit 2) when gains are not 25 / 0.5. |

Tests (synthetic bags with known answers): `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q scripts/real_go2/tests` (the variable stops ROS 2 Humble's `launch_testing` plugin).

## V test runs file

`runs.csv` beside the bags, one line per walk, readings of the same robot point every time (18 Nov rule: the front-left foot) in metres (along the tape, and sideways from the tape line, left +; empty lateral = 0). The official verdict expects 5 m walks:
```
bag,tape_start_m,tape_end_m,note,lat_start_m,lat_end_m,leg
w045_1,0.00,4.98,out 0.45,0.00,-0.08,1
w045_1,4.98,0.03,back 0.45,-0.08,0.03,2
```
The robot must stand still at least 0.5 s before and after each walk (still-pose window).

One run = one row = one walk with its own tape readings and commanded speed (median Move x over the walk). The two legs above are two runs at 0.45 m/s. List exactly the prescribed 6 runs: extra rows make the verdict DIAGNOSTIC ONLY. A bag listed more than once, or with more than one walk of ≥ `--min-walk-s`, needs a distinct `leg` on every row.

Tolerances, **PROPOSED, to be frozen by a dated decisions-log row before collection** (not in the 18 Nov rule): commanded speed ± 0.05 m/s (`V_SPEED_TOL_MPS`), tape distance 5 m ± 0.5 m (`V_LINE_TOL_M`). Both are written in `v_check.json` (`tolerances`) and on the `V GATE` line; a CLI override gives DIAGNOSTIC ONLY.
