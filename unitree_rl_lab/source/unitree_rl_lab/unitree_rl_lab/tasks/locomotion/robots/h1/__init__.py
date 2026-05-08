"""Package initializer for the unitree_rl_lab.unitree_rl_lab.tasks.locomotion.robots.h1 namespace."""

import gymnasium as gym

gym.register(
    id="Unitree-H1-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rl_base_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rl_base_ppo_cfg:BasePPORunnerCfg",
    },
)
