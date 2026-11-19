# SPDX-License-Identifier: BSD-3-Clause
"""
Offline RWM-U + MOPO-PPO config for Unitree Go2 flat-terrain velocity tracking.

Values verified against:
  - config/go2/flat_env_cfg.py  (UnitreeGo2FlatEnvCfg_PRETRAIN resolved weights)
  - config/go2/envs/go2_manager_based_mbrl_env.py  (emitted imagination reward keys)

reward_term_weights mirrors EXACTLY the 11 keys emitted by the online env's
imagination_reward_per_step, with Go2 PRETRAIN/FINETUNE weights.
  - feet_slide (-0.25 in the real env) is computed in imagination by leg kinematics
    (envs/go2_kinematics.py) from the state and the predicted foot contact, weight -0.25.
    rsl_rl/validate_go2_feet_slide.py matched the simulator exactly on 11 Nov 2026.
  - undesired_contacts is computed in the env but NOT emitted on the Go2 path
    (Go2 uses stand_still as its collision proxy). Omitted.
  - dof_pos_limits is emitted but is a zero tensor in imagination -> weight 0.0,
    key kept so the dict matches the emitted reward exactly.

Dims (Go2): WM state_dim=45, action_dim=12, contact_dim=8 (4 thigh + 4 foot),
termination_dim=1, extension_dim=0; policy observation_dim=48.
"""
from dataclasses import dataclass, field
from typing import Dict, List

from .base_cfg import BaseConfig


@dataclass
class Go2FlatConfig(BaseConfig):
    experiment_name: str = "offline"   # top-level; train.py reads it for log dir + wandb project

    @dataclass
    class ExperimentConfig(BaseConfig.ExperimentConfig):
        environment: str = "go2_flat"  # dispatch tag -> resolve_environment_cls (add branch in train.py)

    @dataclass
    class EnvironmentConfig(BaseConfig.EnvironmentConfig):
        # keys & order mirror the env's emitted imagination_reward_per_step
        reward_term_weights: Dict[str, float] = field(default_factory=lambda: {
            "track_lin_vel_xy_exp":  1.0,
            "track_ang_vel_z_exp":   0.5,
            "lin_vel_z_l2":         -2.0,
            "ang_vel_xy_l2":        -0.05,
            "dof_torques_l2":       -2.5e-5,
            "dof_acc_l2":           -2.5e-7,
            "action_rate_l2":       -0.01,
            "feet_air_time":         0.25,    # _PRETRAIN override (threshold 0.25 applied in env)
            "stand_still":          -1.0,     # Go2 collision proxy (paper w_c)
            "flat_orientation_l2":  -2.5,     # _PRETRAIN override
            "dof_pos_limits":        0.0,     # off; zero tensor in imagination
            "feet_slide":           -0.25,    # kinematic (envs/go2_kinematics.py), = real env weight; check
                                              # passed 11 Nov 2026: 31,968 env-steps, URDF COM offset,
                                              # speed RMSE 0.0000 m/s, term error 0.0%, gate agreement 1.0
        })
        # MOPO penalty: paper-2 default. Online calibration (pen0 exploits,
        # pen1 over-regularizes, pen0.25 best) informs tuning if we sweep.
        uncertainty_penalty_weight: float = -1.0
        command_resample_interval_range: List[int] | None = field(default_factory=lambda: [100, 120])
        event_interval_range: List[int] = field(default_factory=lambda: [48, 96])

    @dataclass
    class DataConfig(BaseConfig.DataConfig):
        dataset_root: str = "assets"
        dataset_folder: str = "data/go2_noise"
        batch_data_size: int = 10000
        state_idx_dict: Dict[str, List[int]] = field(default_factory=lambda: {
            r"$v$\n$[m/s]$":         [0, 1, 2],
            r"$\omega$\n$[rad/s]$":  [3, 4, 5],
            r"$g$\n$[1]$":           [6, 7, 8],
            r"$q$\n$[rad]$":         [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            r"$\dot{q}$\n$[rad/s]$": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            r"$\tau$\n$[Nm]$":       [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
        })
        # A refit WM checkpoint must provide its trained normalization statistics.
        # Missing checkpoint stats and missing explicit legacy constants abort training.
        state_data_mean:  List[float] | None = None
        state_data_std:   List[float] | None = None
        action_data_mean: List[float] | None = None
        action_data_std:  List[float] | None = None

    @dataclass
    class ModelArchitectureConfig(BaseConfig.ModelArchitectureConfig):
        history_horizon: int = 32
        forecast_horizon: int = 8
        ensemble_size: int = 5
        contact_dim: int = 8
        termination_dim: int = 1
        architecture_config: Dict[str, object] = field(default_factory=lambda: {
            "type": "rnn",
            "rnn_type": "gru",
            "rnn_num_layers": 2,
            "rnn_hidden_size": 256,
            "state_mean_shape":   [128],
            "state_logstd_shape": [128],
            "extension_shape":    [128],
            "contact_shape":      [128],
            "termination_shape":  [128],
        })
        resume_path: str | None = None     # --wm-checkpoint must name a trained Go2 WM

    @dataclass
    class PolicyArchitectureConfig(BaseConfig.PolicyArchitectureConfig):
        observation_dim: int = 48
        action_dim: int = 12
        resume_path: str | None = None

    @dataclass
    class PolicyAlgorithmConfig(BaseConfig.PolicyAlgorithmConfig):
        learning_rate: float = 1.0e-4
        entropy_coef: float = 0.0001

    @dataclass
    class PolicyTrainingConfig(BaseConfig.PolicyTrainingConfig):
        save_interval: int = 50
        max_iterations: int = 500

    experiment_config:          ExperimentConfig         = field(default_factory=ExperimentConfig)
    environment_config:         EnvironmentConfig        = field(default_factory=EnvironmentConfig)
    data_config:                DataConfig               = field(default_factory=DataConfig)
    model_architecture_config:  ModelArchitectureConfig  = field(default_factory=ModelArchitectureConfig)
    policy_architecture_config: PolicyArchitectureConfig = field(default_factory=PolicyArchitectureConfig)
    policy_algorithm_config:    PolicyAlgorithmConfig    = field(default_factory=PolicyAlgorithmConfig)
    policy_training_config:     PolicyTrainingConfig     = field(default_factory=PolicyTrainingConfig)


# ---------------------------------------------------------------------------------------------------------------
# E16 imagined-reward contract for the Go2 start policy s2 (plan v6 §4 "reward contract", gate 4 Dec 2026).
# STATUS: PROPOSED 18 Nov 2026 by Claude; Charbel decides before Go2 round 0 (by 4 Dec). Every term of s2's
# training reward (Go2-Start-Unitree-v0 = UnitreeGo2FlatEnvCfg_START_UNITREE) is listed: included terms use the
# simulator's weight and parameters and are checked against the simulator term on recorded data
# (analysis/e16_reward_check.py); excluded terms carry the reason. Key = simulator term name = imagined key.
# ---------------------------------------------------------------------------------------------------------------
S2_REWARD_CONTRACT = {
    "track_lin_vel_xy_exp": dict(decision="include", weight=1.5, source="state lin vel + command"),
    "track_ang_vel_z_exp":  dict(decision="include", weight=0.75, source="state ang vel + command"),
    "lin_vel_z_l2":         dict(decision="include", weight=-2.0, source="state"),
    "ang_vel_xy_l2":        dict(decision="include", weight=-0.05, source="state"),
    "joint_vel":            dict(decision="include", weight=-0.001, source="state joint vel (joint_vel_l2)"),
    "dof_acc_l2":           dict(decision="include", weight=-2.5e-7,
                                 source="finite difference of predicted joint vel against the previous noise-free "
                                        "state (simulator: joint_acc; E16 18 Nov: about half its size)"),
    "dof_torques_l2":       dict(decision="include", weight=-2e-4, source="state torque (applied torque)"),
    "action_rate_l2":       dict(decision="include", weight=-0.1, source="action and previous action"),
    "dof_pos_limits":       dict(decision="include", weight=-10.0,
                                 source="state joint pos + default pose + soft limits from the recorded asset"),
    "energy":               dict(decision="include", weight=-2e-5, source="state joint vel x torque"),
    "flat_orientation_l2":  dict(decision="include", weight=-2.5, source="state gravity"),
    "joint_pos":            dict(decision="include", weight=-0.7,
                                 source="state joint pos (relative to default) + command + base speed; x5 standing",
                                 params=dict(stand_still_scale=5.0, velocity_threshold=0.3)),
    "feet_air_time":        dict(decision="include", weight=0.1, params=dict(threshold=0.5),
                                 source="foot timers driven by the predicted foot contact"),
    "air_time_variance":    dict(decision="include", weight=-1.0,
                                 source="foot timers driven by the predicted foot contact"),
    "feet_slide":           dict(decision="include", weight=-0.1,
                                 source="leg kinematics + predicted foot contact (checked 11 Nov at -0.25)"),
    "undesired_contacts":   dict(decision="include", weight=-1.0,
                                 source="predicted THIGH contacts only; hip and calf contacts are not predicted by "
                                        "the world model (the check measures the missing part)"),
}
S2_EXCLUDED_IMAGINED_TERMS = {
    "stand_still": "removed from s2's reward (START_UNITREE sets it to None); weight 0 in imagination",
}


def s2_reward_term_weights():
    """reward_term_weights for imagination under the s2 contract (included terms only)."""
    return {k: v["weight"] for k, v in S2_REWARD_CONTRACT.items() if v["decision"] == "include"}


def apply_s2_reward_contract(env_cls, default_joint_pos=None, soft_joint_pos_limits=None):
    """Set the Go2 env's term parameters for the s2 contract (class attributes; imagination instances inherit)."""
    env_cls.feet_air_time_threshold = S2_REWARD_CONTRACT["feet_air_time"]["params"]["threshold"]
    env_cls.joint_pos_stand_still_scale = S2_REWARD_CONTRACT["joint_pos"]["params"]["stand_still_scale"]
    env_cls.joint_pos_velocity_threshold = S2_REWARD_CONTRACT["joint_pos"]["params"]["velocity_threshold"]
    env_cls.timers_before_rewards = True
    env_cls.acceleration_from_state = True      # physical acceleration, not observation noise (review 19 Nov)
    if default_joint_pos is not None:
        env_cls.default_joint_pos = list(default_joint_pos)
    if soft_joint_pos_limits is not None:
        env_cls.soft_joint_pos_limits = [list(x) for x in soft_joint_pos_limits]
    return env_cls
