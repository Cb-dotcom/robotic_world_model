# Real Go2 tools (ticket T11, 25 Nov 2026)

Off-robot analysis of lab Go2 EDU bags (ROS 2 Foxy, sqlite3). Nothing here drives the robot.
Needs only `numpy` and `rosbags` (no ROS install, no GPU). Message types are built in
(`go2_bag.py`); a wrong layout stops the script, never returns shifted numbers.

| Script | What it gives |
|---|---|
| `t11_v_check.py` | V test: tape displacement vs `/sportmodestate` position, its integrated body velocity, and every `nav_msgs/Odometry` topic (robot_odom, Point-LIO). Ratio per run and source; commanded speed; yaw drift. One `V GATE` line (decisions-log 18 Nov): **PASS/FAIL** (exit 0) only from Point-LIO `/state_estimation_at_scan`, with ≥ 3 bags at each commanded 0.30 and 0.45 m/s, every run 5 m ± 0.5 m on the tape, bar 5 % per run; **INVALID** (exit 1) if a listed run is missing or errored, or Point-LIO has no ratio in a run; otherwise **DIAGNOSTIC ONLY: reasons** (exit 2), e.g. too few runs, Unitree sources only, another line length, `--max-rel-err` ≠ 0.05. Unitree sources never decide. |
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
