# SPDX-License-Identifier: BSD-3-Clause
# Go2 RWM PPO runner — mirrors AnymalDFlatPPOPretrainRunnerCfg structure
# but extends the upstream Isaac Lab Go2 runner.

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)
from mbrl.rl.rsl_rl import (
    RslRlSystemDynamicsCfg,
    RslRlNormalizerCfg,
    RslRlMbrlImaginationCfg,
    RslRlMbrlPpoAlgorithmCfg,
)


# ---------------------------------------------------------------------------
# Pretrain runner — MBPO-on-policy with world-model training.
# Mirrors AnymalDFlatPPOPretrainRunnerCfg.
# ---------------------------------------------------------------------------
@configclass
class UnitreeGo2FlatPPOPretrainRunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    class_name: str = "MBPOOnPolicyRunner"

    system_dynamics = RslRlSystemDynamicsCfg(
        ensemble_size=5,
        history_horizon=32,
        architecture_config={
            "type": "rnn",
            "rnn_type": "gru",
            "rnn_num_layers": 2,
            "rnn_hidden_size": 256,
            "state_mean_shape": [128],
            "state_logstd_shape": [128],
            "extension_shape": [128],
            "contact_shape": [128],
            "termination_shape": [128],
        },
        freeze_auxiliary=False,
    )

    # NOTE: state/action normalizer values are PLACEHOLDERS (zeros / ones).
    # They were hand-tuned for ANYmal-D's specific joint order and pose
    # distribution. Recompute from Go2 rollouts before serious training.
    # See docs/go2-transfer/go2-inventory.md "Implications" section.
    imagination = RslRlMbrlImaginationCfg(
        num_envs=0,
        num_steps_per_env=0,
        max_episode_length=0,
        command_resample_interval_range=None,
        uncertainty_penalty_weight=-0.0,
        state_normalizer=RslRlNormalizerCfg(
            mean=[0.0] * 45,
            std=[1.0] * 45,
        ),
        action_normalizer=RslRlNormalizerCfg(
            mean=[0.0] * 12,
            std=[1.0] * 12,
        ),
    )

    algorithm = RslRlMbrlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        policy_learning_rate=1.0e-3,
        system_dynamics_learning_rate=1.0e-3,
        system_dynamics_weight_decay=0.0,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        system_dynamics_forecast_horizon=8,
        system_dynamics_loss_weights={
            "state": 1.0,
            "sequence": 1.0,
            "bound": 1.0,
            "kl": 0.1,
            "extension": 1.0,
            "contact": 1.0,
            "termination": 1.0,
        },
        system_dynamics_num_mini_batches=20,
        system_dynamics_mini_batch_size=5000,
        system_dynamics_replay_buffer_size=1000,
        system_dynamics_num_eval_trajectories=100,
        system_dynamics_len_eval_trajectory=400,
        system_dynamics_eval_traj_noise_scale=[0.1, 0.2, 0.4, 0.5, 0.8],
    )

    run_name = "pretrain_ens5"
    load_system_dynamics = False
    system_dynamics_load_path = None
    system_dynamics_warmup_iterations = 0
    system_dynamics_num_visualizations = 4

    # State indices for visualization plots (45-dim system_state)
    # Layout: 3 lin_vel + 3 ang_vel + 3 gravity + 12 q + 12 q_dot + 12 tau
    system_dynamics_state_idx_dict = {
        r"$v$\n$[m/s]$": [0, 1, 2],
        r"$\omega$\n$[rad/s]$": [3, 4, 5],
        r"$g$\n$[1]$": [6, 7, 8],
        r"$q$\n$[rad]$": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        r"$\dot{q}$\n$[rad/s]$": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        r"$\tau$\n$[Nm]$": [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
    }
    pca_obs_buf_size = 10000

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2000


# ---------------------------------------------------------------------------
# Baseline runner — vanilla PPO (no world model). Mirrors RWM paper PPO
# hyperparameters exactly. Used for fair PPO vs RWM vs RWM-U comparison.
# ---------------------------------------------------------------------------
@configclass
class UnitreeGo2FlatPPOBaselineRunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    # Standard on-policy PPO runner (not MBPO).
    # No class_name override here -> uses parent's class (OnPolicyRunner).

    experiment_name = "unitree_go2_flat"
    run_name = "baseline"

    def __post_init__(self):
        super().__post_init__()

        # Match RWM paper PPO hyperparameters
        # (algorithm sub-cfg lives on the parent runner cfg)
        self.algorithm.value_loss_coef = 1.0
        self.algorithm.use_clipped_value_loss = True
        self.algorithm.clip_param = 0.2
        self.algorithm.entropy_coef = 0.005
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4
        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.schedule = "adaptive"
        self.algorithm.gamma = 0.99
        self.algorithm.lam = 0.95
        self.algorithm.desired_kl = 0.01
        self.algorithm.max_grad_norm = 1.0

        # Policy network sized per RWM ANYmal-D
        self.policy.init_noise_std = 1.0
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
        self.policy.activation = "elu"

        # Trajectory length and total iterations matching RWM paper
        self.num_steps_per_env = 24
        self.save_interval = 50
        self.max_iterations = 2000

# ---------------------------------------------------------------------------
# Finetune runner — MBPO with imagination ON. Loads the Pretrain world model
# + policy, warms the model, trains the policy on imagined rollouts.
# Mirrors AnymalDFlatPPOFinetuneRunnerCfg.
# ---------------------------------------------------------------------------
@configclass
class UnitreeGo2FlatPPOFinetuneRunnerCfg(UnitreeGo2FlatPPOPretrainRunnerCfg):
    resume = True
    load_system_dynamics = True
    load_run = "2026-06-12_13-39-03_pretrain_ens5"
    system_dynamics_load_path = "logs/rsl_rl/unitree_go2_flat/2026-06-12_13-39-03_pretrain_ens5/model_2000.pt"
    system_dynamics_warmup_iterations = 100
    run_name = "finetune_ens5_pen025_h256"

    def __post_init__(self):
        super().__post_init__()
        self.imagination.num_envs = 8192
        self.imagination.num_steps_per_env = 24
        self.imagination.max_episode_length = 256
        self.imagination.command_resample_interval_range = [100, 120]
        self.imagination.uncertainty_penalty_weight = -0.25
        # Normalizer kept identity on purpose: matches the Pretrain world model (trained with identity). Recompute ONLY with a fresh Pretrain.
@configclass
class UnitreeGo2FlatPPOVisualizeRunnerCfg(UnitreeGo2FlatPPOPretrainRunnerCfg):
    resume = True
    load_system_dynamics = True
    system_dynamics_load_path = "logs/rsl_rl/unitree_go2_flat/2026-06-11_12-35-49_pretrain/model_2000.pt"
    run_name = "visualize"
