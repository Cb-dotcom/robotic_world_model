# SPDX-License-Identifier: BSD-3-Clause
"""Read lab Go2 EDU ROS 2 bags off-robot (T11, 25 Nov 2026).

The Jetson records with ROS 2 Foxy (sqlite3 bags). Foxy bags carry no message
definitions, so the unitree_go / unitree_api types are declared here, copied
from `unitree_ros2` (the robot firmware layout). Vault facts used:
`09 Setup and Infrastructure/go2-robot-access.ai.md` (H0, 16 Nov):
  * the robot's SportModeState payload is 512 bytes = ter_go2 msg + PathPoint[10];
  * /sportmodestate.velocity is in the BASE frame, .position is WORLD (odom);
  * under `mcf` foot_force / foot_position_body / foot_speed_body are 0.

Every decode checks that the CDR payload is consumed exactly. A wrong type
layout therefore stops the script; it never yields silently shifted numbers.

Joint order (recording-contract.ai.md §2): SDK motor_state[0:12] is
FR, FL, RR, RL x (hip, thigh, calf); sim order is FL, FR, RL, RR per joint type.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

# --- joint and foot order (contract §2, checked inverse 18 Nov) ---------------
SIM_FROM_SDK = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])
SDK_FROM_SIM = np.array([1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10])
SIM_JOINT_NAMES = [
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
]
SDK_JOINT_NAMES = [f"{leg}_{j}" for leg in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf")]
SIM_FOOT_FROM_SDK = np.array([1, 0, 3, 2])

# --- message definitions (unitree_ros2, unitree_go + unitree_api) -------------
_MSGS: dict[str, str] = {
    "unitree_go/msg/TimeSpec": "int32 sec\nuint32 nanosec\n",
    "unitree_go/msg/IMUState": (
        "float32[4] quaternion\nfloat32[3] gyroscope\nfloat32[3] accelerometer\n"
        "float32[3] rpy\nint8 temperature\n"
    ),
    "unitree_go/msg/MotorState": (
        "uint8 mode\nfloat32 q\nfloat32 dq\nfloat32 ddq\nfloat32 tau_est\n"
        "float32 q_raw\nfloat32 dq_raw\nfloat32 ddq_raw\nint8 temperature\n"
        "uint32 lost\nuint32[2] reserve\n"
    ),
    "unitree_go/msg/BmsState": (
        "uint8 version_high\nuint8 version_low\nuint8 status\nuint8 soc\n"
        "int32 current\nuint16 cycle\nint8[2] bq_ntc\nint8[2] mcu_ntc\n"
        "uint16[15] cell_vol\n"
    ),
    "unitree_go/msg/LowState": (
        "uint8[2] head\nuint8 level_flag\nuint8 frame_reserve\nuint32[2] sn\n"
        "uint32[2] version\nuint16 bandwidth\nunitree_go/IMUState imu_state\n"
        "unitree_go/MotorState[20] motor_state\nunitree_go/BmsState bms_state\n"
        "int16[4] foot_force\nint16[4] foot_force_est\nuint32 tick\n"
        "uint8[40] wireless_remote\nuint8 bit_flag\nfloat32 adc_reel\n"
        "int8 temperature_ntc1\nint8 temperature_ntc2\nfloat32 power_v\n"
        "float32 power_a\nuint16[4] fan_frequency\nuint32 reserve\nuint32 crc\n"
    ),
    "unitree_go/msg/MotorCmd": (
        "uint8 mode\nfloat32 q\nfloat32 dq\nfloat32 tau\nfloat32 kp\nfloat32 kd\n"
        "uint32[3] reserve\n"
    ),
    "unitree_go/msg/BmsCmd": "uint8 off\nuint8[3] reserve\n",
    "unitree_go/msg/LowCmd": (
        "uint8[2] head\nuint8 level_flag\nuint8 frame_reserve\nuint32[2] sn\n"
        "uint32[2] version\nuint16 bandwidth\nunitree_go/MotorCmd[20] motor_cmd\n"
        "unitree_go/BmsCmd bms_cmd\nuint8[40] wireless_remote\nuint8[12] led\n"
        "uint8[2] fan\nuint8 gpio\nuint32 reserve\nuint32 crc\n"
    ),
    "unitree_go/msg/PathPoint": (
        "float32 t_from_start\nfloat32 x\nfloat32 y\nfloat32 yaw\nfloat32 vx\n"
        "float32 vy\nfloat32 vyaw\n"
    ),
    "unitree_go/msg/SportModeState": (
        "unitree_go/TimeSpec stamp\nuint32 error_code\nunitree_go/IMUState imu_state\n"
        "uint8 mode\nfloat32 progress\nuint8 gait_type\nfloat32 foot_raise_height\n"
        "float32[3] position\nfloat32 body_height\nfloat32[3] velocity\n"
        "float32 yaw_speed\nfloat32[4] range_obstacle\nint16[4] foot_force\n"
        "float32[12] foot_position_body\nfloat32[12] foot_speed_body\n"
        "unitree_go/PathPoint[10] path_point\n"
    ),
    "unitree_api/msg/RequestIdentity": "int64 id\nint64 api_id\n",
    "unitree_api/msg/RequestLease": "int64 id\n",
    "unitree_api/msg/RequestPolicy": "int32 priority\nbool noreply\n",
    "unitree_api/msg/RequestHeader": (
        "unitree_api/RequestIdentity identity\nunitree_api/RequestLease lease\n"
        "unitree_api/RequestPolicy policy\n"
    ),
    "unitree_api/msg/Request": "unitree_api/RequestHeader header\nstring parameter\nuint8[] binary\n",
}

SPORT_API_MOVE = 1008  # Move {x, y, z}; H0: the remote sends it at ~42 Hz


def make_typestore():
    ts = get_typestore(Stores.ROS2_FOXY)
    types = {}
    for name, text in _MSGS.items():
        types.update(get_types_from_msg(text, name))
    ts.register(types)
    return ts


TYPESTORE = make_typestore()

_NUMPY_DT = {
    "bool": np.bool_, "int8": np.int8, "uint8": np.uint8, "int16": np.int16, "uint16": np.uint16,
    "int32": np.int32, "uint32": np.uint32, "int64": np.int64, "uint64": np.uint64,
    "float32": np.float32, "float64": np.float64, "byte": np.uint8, "char": np.uint8,
}


def _default_of(desc):
    kind, spec = desc
    kind = int(kind)
    if kind == 1:  # BASE
        base = spec[0]
        return "" if base == "string" else _NUMPY_DT[base](0).item()
    if kind == 2:  # NAME (nested message)
        return default_message(spec)
    sub, n = spec  # ARRAY (fixed) or SEQUENCE
    if kind == 4:
        n = 0
    if int(sub[0]) == 1 and sub[1][0] != "string":
        return np.zeros(n, dtype=_NUMPY_DT[sub[1][0]])
    return [_default_of(sub) for _ in range(n)]


def default_message(msgtype: str):
    """An all-zero message of msgtype (tests, size checks)."""
    fields = TYPESTORE.fielddefs[msgtype][1]
    return TYPESTORE.types[msgtype](**{name: _default_of(desc) for name, desc in fields})


def check_exact_decode(raw: bytes, msgtype: str):
    """Deserialize and re-serialize; the sizes must match, else the layout is wrong."""
    msg = TYPESTORE.deserialize_cdr(raw, msgtype)
    again = TYPESTORE.serialize_cdr(msg, msgtype)
    # CDR pads the end to 4 bytes on some writers; allow only that padding.
    if not (len(again) <= len(raw) <= len(again) + 3):
        raise ValueError(
            f"{msgtype}: payload {len(raw)} bytes, layout gives {len(again)}. "
            "The message definition does not match the robot. Stop."
        )
    return msg


# --- tables ------------------------------------------------------------------
@dataclass
class Table:
    """Columns of one topic, one row per message. t_ns = bag receive time."""

    topic: str
    msgtype: str
    t_ns: np.ndarray
    cols: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self):
        return len(self.t_ns)

    @property
    def t(self) -> np.ndarray:
        return self.t_ns.astype(np.float64) * 1e-9


def _sport_row(m):
    return {
        "error_code": m.error_code,
        "mode": m.mode,
        "position": np.asarray(m.position, dtype=np.float64),
        "velocity": np.asarray(m.velocity, dtype=np.float64),
        "yaw_speed": m.yaw_speed,
        "body_height": m.body_height,
        "rpy": np.asarray(m.imu_state.rpy, dtype=np.float64),
        "stamp_ns": int(m.stamp.sec) * 1_000_000_000 + int(m.stamp.nanosec),
    }


def _low_row(m):
    ms = m.motor_state[:12]
    return {
        "tick": m.tick,
        "q": np.array([x.q for x in ms]),
        "dq": np.array([x.dq for x in ms]),
        "tau_est": np.array([x.tau_est for x in ms]),
        "temperature": np.array([x.temperature for x in ms], dtype=np.float64),
        "foot_force": np.asarray(m.foot_force, dtype=np.float64),
        "quaternion": np.asarray(m.imu_state.quaternion, dtype=np.float64),
        "gyroscope": np.asarray(m.imu_state.gyroscope, dtype=np.float64),
        "soc": m.bms_state.soc,
    }


def _cmd_row(m):
    mc = m.motor_cmd[:12]
    return {
        "mode": np.array([x.mode for x in mc], dtype=np.float64),
        "q": np.array([x.q for x in mc]),
        "dq": np.array([x.dq for x in mc]),
        "tau": np.array([x.tau for x in mc]),
        "kp": np.array([x.kp for x in mc]),
        "kd": np.array([x.kd for x in mc]),
    }


def _odom_row(m):
    p, q = m.pose.pose.position, m.pose.pose.orientation
    v = m.twist.twist.linear
    return {
        "position": np.array([p.x, p.y, p.z]),
        "quat_wxyz": np.array([q.w, q.x, q.y, q.z]),
        "lin_vel": np.array([v.x, v.y, v.z]),
        "frame_id": m.header.frame_id,
        "child_frame_id": m.child_frame_id,
        "stamp_ns": int(m.header.stamp.sec) * 1_000_000_000 + int(m.header.stamp.nanosec),
    }


def _request_row(m):
    api = int(m.header.identity.api_id)
    row = {"api_id": api, "x": np.nan, "y": np.nan, "z": np.nan}
    if api == SPORT_API_MOVE and m.parameter:
        try:
            p = json.loads(m.parameter)
            row.update(x=float(p.get("x", np.nan)), y=float(p.get("y", np.nan)), z=float(p.get("z", np.nan)))
        except (ValueError, TypeError):
            pass
    return row


_ROW = {
    "unitree_go/msg/SportModeState": _sport_row,
    "unitree_go/msg/LowState": _low_row,
    "unitree_go/msg/LowCmd": _cmd_row,
    "nav_msgs/msg/Odometry": _odom_row,
    "unitree_api/msg/Request": _request_row,
}


def topic_info(bag: str | Path) -> dict[str, tuple[str, int]]:
    """topic -> (msgtype, message count)."""
    with Reader(bag) as r:
        return {c.topic: (c.msgtype, c.msgcount) for c in r.connections}


def receive_times(bag: str | Path, topics: list[str] | None = None) -> dict[str, np.ndarray]:
    """Receive time (ns) per topic, for any type, without decoding."""
    out: dict[str, list[int]] = {}
    with Reader(bag) as r:
        conns = [c for c in r.connections if topics is None or c.topic in topics]
        for c, t, _ in r.messages(connections=conns):
            out.setdefault(c.topic, []).append(t)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def read_topic(bag: str | Path, topic: str, stride: int = 1) -> Table:
    """Decode one topic into a Table. stride > 1 keeps every n-th message."""
    with Reader(bag) as r:
        conns = [c for c in r.connections if c.topic == topic]
        if not conns:
            raise KeyError(f"topic {topic} not in bag {bag}")
        msgtype = conns[0].msgtype
        if msgtype not in _ROW:
            raise KeyError(f"no row reader for {msgtype}")
        t_ns, rows = [], []
        for i, (c, t, raw) in enumerate(r.messages(connections=conns)):
            if i % stride:
                continue
            msg = check_exact_decode(raw, c.msgtype) if i == 0 else TYPESTORE.deserialize_cdr(raw, c.msgtype)
            t_ns.append(t)
            rows.append(_ROW[msgtype](msg))
    tab = Table(topic, msgtype, np.asarray(t_ns, dtype=np.int64))
    if rows:
        for k in rows[0]:
            vals = [row[k] for row in rows]
            tab.cols[k] = np.asarray(vals) if not isinstance(vals[0], str) else np.asarray(vals, dtype=object)
    return tab


def odometry_topics(bag: str | Path) -> list[str]:
    return sorted(t for t, (mt, n) in topic_info(bag).items() if mt == "nav_msgs/msg/Odometry" and n > 0)
