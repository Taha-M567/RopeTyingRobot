"""RopeReach-SO100: Phase 1 rope manipulation task."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RopeReach-SO100-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ropeuntyingrobot_env_cfg:RopeReachEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:RopeReachPPORunnerCfg"
        ),
    },
)
