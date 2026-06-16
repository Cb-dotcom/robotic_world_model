import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Isaac-Velocity-Flat-Unitree-Go2-Init-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg_INIT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Flat-Unitree-Go2-Pretrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg_PRETRAIN",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPOPretrainRunnerCfg",
    },
)

# PPO baseline (no world model). Same flat env as Pretrain/Finetune,
# RWM-paper-aligned PPO hyperparameters. This is the comparison
# baseline against which RWM and RWM-U Go2 results are reported.
gym.register(
    id="Template-Isaac-Velocity-Flat-Unitree-Go2-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg_BASELINE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPOBaselineRunnerCfg",
    },
)
gym.register(
    id="Template-Isaac-Velocity-Flat-Unitree-Go2-Finetune-v0",
    entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.go2.envs.go2_manager_based_mbrl_env:UnitreeGo2ManagerBasedMBRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg_FINETUNE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPOFinetuneRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Flat-Unitree-Go2-Visualize-v0",
    entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.go2.envs.go2_manager_based_visualize_env:UnitreeGo2ManagerBasedVisualizeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg_VISUALIZE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPOVisualizeRunnerCfg",
    },
)