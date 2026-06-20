"""Registers Isaac-Lift-Cube-Franka-Custom-v0 as a gymnasium plugin.

`register()` is wired as a ``gymnasium.envs`` entry point (see pyproject.toml),
so gymnasium calls it automatically on ``import gymnasium``. That import happens
*early* (transitively, when ``isaaclab.app`` is imported) — before the Isaac Sim
app launches and therefore before ``isaaclab`` itself is importable. So
``register()`` must touch no ``isaaclab`` code: it stores only string entry
points, which gymnasium resolves lazily at ``gym.make`` time, by which point the
app is up. This keeps registration robust regardless of when gymnasium loads us.

Mirrors the reach_franka external task; see docs/learning/11-lift-task-isaaclab.html.
"""


def register():
    """Register the custom Lift task. Idempotent; imports nothing heavy.

    gymnasium calls this from inside ``gymnasium/envs/__init__.py`` while the
    top-level ``gymnasium`` module is still initializing, so ``gymnasium.register``
    isn't bound yet. Import the function/registry from the submodule directly,
    which is fully loaded by the time plugins run.
    """
    from gymnasium.envs.registration import register as gym_register, registry

    if "Isaac-Lift-Cube-Franka-Custom-v0" in registry:
        return

    gym_register(
        id="Isaac-Lift-Cube-Franka-Custom-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            # String entry points only — resolved lazily after the app boots.
            "env_cfg_entry_point": "lift_franka.lift_env_cfg:FrankaCubeLiftCustomEnvCfg",
            # Reuse the built-in Franka Lift rsl_rl PPO runner cfg
            # (verified class name in Isaac Lab v2.1.0: LiftCubePPORunnerCfg).
            "rsl_rl_cfg_entry_point": (
                "isaaclab_tasks.manager_based.manipulation.lift.config.franka."
                "agents.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg"
            ),
        },
    )
