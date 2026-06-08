"""Registers Isaac-Reach-Franka-Custom-v0 as a gymnasium plugin.

`register()` is wired as a ``gymnasium.envs`` entry point (see pyproject.toml),
so gymnasium calls it automatically on ``import gymnasium``. That import happens
*early* (transitively, when ``isaaclab.app`` is imported) — before the Isaac Sim
app launches and therefore before ``isaaclab`` itself is importable. So
``register()`` must touch no ``isaaclab`` code: it stores only string entry
points, which gymnasium resolves lazily at ``gym.make`` time, by which point the
app is up. This keeps registration robust regardless of when gymnasium loads us.
"""


def register():
    """Register the custom Reach task. Idempotent; imports nothing heavy.

    gymnasium calls this from inside ``gymnasium/envs/__init__.py`` while the
    top-level ``gymnasium`` module is still initializing, so ``gymnasium.register``
    isn't bound yet. Import the function/registry from the submodule directly,
    which is fully loaded by the time plugins run.
    """
    from gymnasium.envs.registration import register as gym_register, registry

    if "Isaac-Reach-Franka-Custom-v0" in registry:
        return

    gym_register(
        id="Isaac-Reach-Franka-Custom-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            # String entry points only — resolved lazily after the app boots.
            "env_cfg_entry_point": "reach_franka.reach_env_cfg:FrankaReachCustomEnvCfg",
            # Reuse the built-in Franka rsl_rl PPO runner cfg.
            # NOTE: the class is FrankaReachPPORunnerCfg in v2.1.0 (not ReachPPORunnerCfg).
            "rsl_rl_cfg_entry_point": (
                "isaaclab_tasks.manager_based.manipulation.reach.config.franka."
                "agents.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg"
            ),
        },
    )
