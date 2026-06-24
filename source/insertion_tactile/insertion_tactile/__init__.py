"""Registers Isaac-Insertion-Franka-Tactile-v0 as a gymnasium plugin.

Mirrors lift_franka: register() touches no isaaclab code (string entry points only),
resolved lazily at gym.make after the Isaac Sim app boots. Safe to import on any machine —
the heavy env cfg is referenced by string, not imported here."""


def register():
    from gymnasium.envs.registration import register as gym_register, registry

    if "Isaac-Insertion-Franka-Tactile-v0" in registry:
        return

    gym_register(
        id="Isaac-Insertion-Franka-Tactile-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":
                "insertion_tactile.insertion_env_cfg:FrankaInsertionTactileEnvCfg",
        },
    )
