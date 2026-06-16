# SPDX-License-Identifier: BSD-3-Clause
# Go2 RWM env config — mirrors the ANYmal-D RWM config structure
# (mbrl.tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg)
# but extends the upstream Isaac Lab Go2 rough env.

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ObservationsCfg,
    RewardsCfg,
)

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

import mbrl.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_slide as isaac_feet_slide
from mbrl.mbrl.envs.mdp.commands import UniformVelocityCommand_Visualize, SampleUniformVelocityCommand


# ---------------------------------------------------------------------------
# Rewards (mirror RewardsCfg_TRAIN from RWM ANYmal-D)
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg_TRAIN(RewardsCfg):
    stand_still = RewTerm(
        func=mdp.joint_pos_stand_still,
        weight=-1.0,
        params={"command_name": "base_velocity", "threshold": 0.05},
    )
    feet_slide = RewTerm(
        func=isaac_feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )


# ---------------------------------------------------------------------------
# Base Flat env — mirror AnymalDFlatEnvCfg
#
# Paper-faithful reward weights from RWM paper Table S6 (ANYmal-D column),
# applied to Go2 as the closest morphological template (both are quadrupeds).
# This explicitly overrides EVERY reward weight to avoid silent inheritance
# from upstream Isaac Lab Go2 configs, which inflate tracking weights 1.5x
# and drive PPO to a degenerate scoot policy.
#
# Reference: Li, Krause, Hutter, "Robotic World Model" (arXiv:2501.10100), Table S6.
# ---------------------------------------------------------------------------
@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):

    rewards: RewardsCfg_TRAIN = RewardsCfg_TRAIN()

    def __post_init__(self):
        # post init of parent (Go2 rough env)
        super().__post_init__()

        # Paper Table S6 (ANYmal-D column) — full explicit override.
        # Do NOT rely on upstream Go2 defaults for any of these.
        self.rewards.track_lin_vel_xy_exp.weight = 1.0      # paper w_vxy
        self.rewards.track_ang_vel_z_exp.weight = 0.5       # paper w_ωz
        self.rewards.lin_vel_z_l2.weight = -2.0             # paper w_vz
        self.rewards.ang_vel_xy_l2.weight = -0.05           # paper w_ωxy
        self.rewards.dof_torques_l2.weight = -2.5e-5        # paper w_qτ
        self.rewards.dof_acc_l2.weight = -2.5e-7            # paper w_q̈
        self.rewards.action_rate_l2.weight = -0.01          # paper w_ȧ
        self.rewards.feet_air_time.weight = 0.5             # paper w_fa
        # Go2 morphology calibration: lower swing-time threshold from 0.5s
        # (ANYmal-D default, ~50kg robot with slow natural cadence) to 0.25s,
        # which is achievable for Go2 (~15kg, faster cadence) during velocity
        # tracking. Without this, feet_air_time stays negative for any
        # reasonable gait on Go2. Paper precedent: G1 had w_fa = 0.0 because
        # the term was morphologically incompatible; our case is milder
        # (recalibrate, not remove).
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.flat_orientation_l2.weight = -5.0      # paper w_g
        self.rewards.dof_pos_limits.weight = 0.0            # not in paper, off
        self.rewards.stand_still.weight = -1.0              # paper w_c (collision proxy)

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # No height scanner (Open Item 5: strip for flat, matches ANYmal-D flat)
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # No terrain curriculum
        self.curriculum.terrain_levels = None


# ---------------------------------------------------------------------------
# Init env — mirror AnymalDFlatEnvCfg_INIT
# (Init actually uses rough terrain for warm-up; upstream RWM convention.)
# ---------------------------------------------------------------------------
@configclass
class UnitreeGo2FlatEnvCfg_INIT(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Revert rewards to Init defaults
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.feet_air_time.weight = 0.125

        # Revert to generated rough terrain, easy difficulty, no curriculum
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.0)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 1.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0


# ---------------------------------------------------------------------------
# Pretrain observations — mirror ObservationsCfg_PRETRAIN
# Adds system_state (45), system_action (12), system_contact (8),
# system_termination (1) observation groups for the world-model heads.
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg_PRETRAIN(ObservationsCfg):

    @configclass
    class SystemStateCfg(ObsGroup):

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        joint_torque = ObsTerm(func=mdp.joint_effort)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemActionCfg(ObsGroup):

        pred_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemExtensionCfg(ObsGroup):
        pass

    @configclass
    class SystemContactCfg(ObsGroup):

        # Go2 body names are lowercase: .*_thigh, .*_foot
        # ANYmal-D used ALL CAPS .*THIGH, .*FOOT — different convention.
        thigh_contact = ObsTerm(
            func=mdp.body_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"),
                "threshold": 1.0,
            },
        )
        foot_contact = ObsTerm(
            func=mdp.body_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 1.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemTerminationCfg(ObsGroup):

        # Go2 base body name is "base" (same as ANYmal-D)
        base_contact = ObsTerm(
            func=mdp.body_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
                "threshold": 1.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    system_state: SystemStateCfg = SystemStateCfg()
    system_action: SystemActionCfg = SystemActionCfg()
    # system_extension: SystemExtensionCfg = SystemExtensionCfg()
    system_contact: SystemContactCfg = SystemContactCfg()
    system_termination: SystemTerminationCfg = SystemTerminationCfg()


# ---------------------------------------------------------------------------
# Pretrain env — mirror AnymalDFlatEnvCfg_PRETRAIN
# ---------------------------------------------------------------------------
@configclass
class UnitreeGo2FlatEnvCfg_PRETRAIN(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Go2 reward fix: match stock Isaac Lab Go2 flat reward tuning.
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.flat_orientation_l2.weight = -2.5


    observations: ObservationsCfg_PRETRAIN = ObservationsCfg_PRETRAIN()


# ---------------------------------------------------------------------------
# Baseline env — flat terrain, RWM-style rewards, ONLY the policy
# observation group (no world-model groups). This is the comparison
# baseline against which RWM/RWM-U are reported.
# ---------------------------------------------------------------------------
@configclass
class UnitreeGo2FlatEnvCfg_BASELINE(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Go2 reward fix: match stock Isaac Lab Go2 flat reward tuning.
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.flat_orientation_l2.weight = -2.5

    # No observation override — keep only the default 'policy' group from
    # the parent ObservationsCfg. No system_* groups because there is no
    # world model in the baseline.
    pass

# ---------------------------------------------------------------------------
# Finetune env — mirror AnymalDFlatEnvCfg_FINETUNE: small real-env scene (10)
# + uniform commands; bulk of training is the 8192 imagined envs (runner cfg).
# ---------------------------------------------------------------------------
@configclass
class UnitreeGo2FlatEnvCfg_FINETUNE(UnitreeGo2FlatEnvCfg_PRETRAIN):
    def __post_init__(self):
        super().__post_init__()

        # Go2 reward fix: match stock Isaac Lab Go2 flat reward tuning.
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.flat_orientation_l2.weight = -2.5
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.class_type = SampleUniformVelocityCommand

@configclass
class UnitreeGo2FlatEnvCfg_VISUALIZE(UnitreeGo2FlatEnvCfg_PRETRAIN):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for visualize
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # disable randomization for visualize
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # override commands
        self.commands.base_velocity.class_type = UniformVelocityCommand_Visualize
        self.commands.base_velocity.resampling_time_range = (2.0, 2.0)
        # override randomization
        self.events.reset_base.func = mdp.reset_root_state_uniform_visualize
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (1.57, 1.57)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.func = mdp.reset_joints_by_scale_visualize
