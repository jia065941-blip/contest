"""Runtime construction for optional red learning motion policies.

The policy package itself remains importable without PyTorch.  PyTorch is
only required when a PPO or MAPPO checkpoint is explicitly selected.
"""

from __future__ import annotations

from pathlib import Path

from .red_policy import RandomMaskedPolicy, SharedPolicy


LEARNING_MOTION_CHOICES = (
    "random_masked",
    "ppo_baseline",
    "ppo_custom",
    "ppo",
    "mappo",
    "unified_mappo",
)


def build_learning_motion_policy(
    name: str,
    *,
    seed: int | None = None,
    model_path: str | None = None,
    max_steps: int = 1000,
    training: bool = False,
    observation_dim: int = 85,
) -> SharedPolicy:
    """Build one shared inference policy for all red platforms.

    In inference a checkpoint is mandatory, so a randomly initialised network
    is never mistaken for a useful strategy.  Training may start from a fresh
    network or resume from an existing checkpoint. ``random_masked`` is a
    dependency-free wiring smoke test, not a performance baseline.
    """

    policy_name = name.strip().lower().replace("-", "_")
    if policy_name == "random_masked":
        return RandomMaskedPolicy(seed=seed)
    if policy_name not in LEARNING_MOTION_CHOICES:
        valid_names = ", ".join(LEARNING_MOTION_CHOICES)
        raise ValueError(f"Unknown learning motion policy '{name}'. Valid policies: {valid_names}")
    if not training and not model_path:
        raise ValueError(f"RED_LEARNING_MODEL is required for learning motion policy '{policy_name}'")
    if model_path and not Path(model_path).is_file() and not training:
        raise FileNotFoundError(f"Learning checkpoint does not exist: {model_path}")

    try:
        if policy_name == "ppo_baseline":
            from .ppo_baseline_policy import (
                PPOBaselineConfig,
                PPOBaselineSharedPolicy,
            )

            policy = PPOBaselineSharedPolicy(PPOBaselineConfig(
                seed=0 if seed is None else seed,
                observation_dim=observation_dim,
            ))
        elif policy_name in {"ppo", "ppo_custom"}:
            from .ppo_policy import PPOCustomConfig, PPOCustomSharedPolicy

            policy = PPOCustomSharedPolicy(PPOCustomConfig(
                seed=0 if seed is None else seed,
                observation_dim=observation_dim,
            ))
        elif policy_name == "mappo":
            from .mappo_policy import MAPPOConfig, MAPPOSharedPolicy

            policy = MAPPOSharedPolicy(
                MAPPOConfig(
                    seed=0 if seed is None else seed,
                    max_steps=max_steps,
                    observation_dim=observation_dim,
                )
            )
        else:
            from .unified_mappo_policy import UnifiedMAPPOSharedPolicy

            policy = UnifiedMAPPOSharedPolicy(
                seed=0 if seed is None else seed,
                max_steps=max_steps,
                observation_dim=observation_dim,
                training=training,
                model_path=model_path,
            )
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise RuntimeError(
                "PPO/MAPPO requires PyTorch in the Python runtime. "
                "Use random_masked for wiring checks or install a compatible PyTorch build."
            ) from error
        raise

    if policy_name != "unified_mappo" and model_path and Path(model_path).is_file():
        policy.load(model_path)
    policy.set_training(training)
    return policy
