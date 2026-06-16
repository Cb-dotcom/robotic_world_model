# SPDX-License-Identifier: BSD-3-Clause
"""
Offline RWM-U + MOPO-PPO config for Unitree Go2 flat-terrain velocity tracking.

Values verified against:
  - config/go2/flat_env_cfg.py  (UnitreeGo2FlatEnvCfg_PRETRAIN resolved weights)
  - config/go2/envs/go2_manager_based_mbrl_env.py  (emitted imagination reward keys)

reward_term_weights mirrors EXACTLY the 11 keys emitted by the online env's
imagination_reward_per_step, with Go2 PRETRAIN/FINETUNE weights.
  - feet_slide (-0.25 in the real env) is NOT computed in imagination: the 45-dim
    imagined state lacks foot velocity (needs FK). Omitted in v1; add later via FK
    as a separately-evaluated extension.
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
        })
        # MOPO penalty: paper-2 default. Online calibration (pen0 exploits,
        # pen1 over-regularizes, pen0.25 best) informs tuning if we sweep.
        uncertainty_penalty_weight: float = -1.0
        command_resample_interval_range: List[int] | None = field(default_factory=lambda: [100, 120])
        event_interval_range: List[int] = field(default_factory=lambda: [48, 96])

    @dataclass
    class DataConfig(BaseConfig.DataConfig):
        dataset_root: str = "assets"
        dataset_folder: str = "data/go2"
        batch_data_size: int = 10000
        state_idx_dict: Dict[str, List[int]] = field(default_factory=lambda: {
            r"$v$\n$[m/s]$":         [0, 1, 2],
            r"$\omega$\n$[rad/s]$":  [3, 4, 5],
            r"$g$\n$[1]$":           [6, 7, 8],
            r"$q$\n$[rad]$":         [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            r"$\dot{q}$\n$[rad/s]$": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            r"$\tau$\n$[Nm]$":       [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
        })
        # None -> trainer auto-computes from the collected Go2 dataset
        # (SystemDynamicsDataset uses provided values when not None, else data mean/std)
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
        resume_path: str | None = None     # no Go2 RWM-U checkpoint yet -> train from scratch

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
