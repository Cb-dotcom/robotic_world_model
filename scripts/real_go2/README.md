# Real Go2 tools (ticket T11, 25 Nov 2026)

Off-robot analysis of lab Go2 EDU bags (ROS 2 Foxy, sqlite3). Nothing here drives the robot.
Needs only `numpy` and `rosbags` (no ROS install, no GPU). Message types are built in
(`go2_bag.py`); a wrong layout stops the script, never returns shifted numbers.

| Script | What it gives |
|---|---|
| `t11_v_check.py` | V test: tape displacement vs `/sportmodestate` position, its integrated body velocity, and every `nav_msgs/Odometry` topic (robot_odom, Point-LIO). Ratio per run and source; commanded speed; yaw drift. PASS/FAIL only with `--max-rel-err` (the bar is Charbel's decision). |
| `t11_log_check.py` | Logger: rates, gaps > 20 / 30 / 100 ms, all-topic stops, `/lowstate` tick losses vs late delivery, 50 Hz policy emulation (flagged steps, segments, usable 58-step windows), max motor temperature, `/lowcmd` kp/kd. |
| `t11_h2_actuator.py` | H2: logged targets through Isaac Lab's DCMotor (25 / 0.5, 23.5 N m, 30 rad/s). Per joint: lag, gain, offset, RMSE in N m, position offset, clip counts. Stops (exit 2) when gains are not 25 / 0.5. |

Tests (synthetic bags with known answers): `python -m pytest -q scripts/real_go2/tests`.

## V test runs file

`runs.csv` beside the bags, one line per walk, tape readings of the head front edge in metres:
```
bag,tape_start_m,tape_end_m,note
w045_1,0.20,5.13,teleop 0.45
```
The robot must stand still at least 0.5 s before and after each walk (still-pose window).
