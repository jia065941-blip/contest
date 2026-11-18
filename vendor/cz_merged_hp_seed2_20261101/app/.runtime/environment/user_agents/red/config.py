"""Dependency-light configuration and CLI validation for the red examples."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .baselines import RED_POLICY_CHOICES


LEARNING_MOTION_CHOICES = ("random_masked", "ppo", "mappo")
RED_MOTION_POLICY_CHOICES = ("straight", "reactive_evasion", *LEARNING_MOTION_CHOICES)


@dataclass(frozen=True)
class RedAgentConfig:
    """Example choices owned by user_agents, independent of the environment."""

    policy_name: str = "r0_random"
    motion_policy: str = "reactive_evasion"
    seed: int = 1
    max_steps: int = 1000
    model_path: str | None = None
    training: bool = False

    def __post_init__(self) -> None:
        if self.policy_name not in RED_POLICY_CHOICES:
            raise ValueError(f"Unsupported red baseline '{self.policy_name}'")
        if self.motion_policy not in RED_MOTION_POLICY_CHOICES:
            raise ValueError(f"Unsupported red motion policy: {self.motion_policy}")

    @classmethod
    def from_arguments(cls, args: argparse.Namespace) -> RedAgentConfig:
        return cls(
            policy_name=args.red_policy,
            motion_policy=args.red_motion_policy,
            seed=args.seed,
            max_steps=args.max_steps,
            model_path=str(args.red_learning_model) if args.red_learning_model else None,
            training=args.red_learning_train,
        )


def add_red_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the existing red example switches without loading algorithms."""

    red = parser.add_argument_group("Red agents")
    red.add_argument("--red-policy", choices=RED_POLICY_CHOICES, default="r0_random")
    red.add_argument("--red-motion-policy", choices=RED_MOTION_POLICY_CHOICES,
                     default="reactive_evasion")
    red.add_argument("--red-learning-model", type=Path,
                     help="PPO/MAPPO checkpoint; required for inference and training output")
    red.add_argument("--red-learning-train", action="store_true",
                     help="Run the user_agents PPO/MAPPO training entry point")


def validate_red_arguments(args: argparse.Namespace) -> None:
    """Validate example combinations after caller-relative paths are resolved."""

    if args.red_policy == "r9_hierarchical_learning" and args.red_motion_policy not in LEARNING_MOTION_CHOICES:
        raise ValueError("r9_hierarchical_learning requires --red-motion-policy random_masked, ppo, or mappo")
    if args.red_learning_train and args.red_motion_policy not in {"ppo", "mappo"}:
        raise ValueError("--red-learning-train requires --red-motion-policy ppo or mappo")
    if args.red_motion_policy in {"ppo", "mappo"}:
        if args.red_learning_model is None:
            raise ValueError("--red-learning-model is required for PPO/MAPPO")
        if not args.red_learning_train and not args.red_learning_model.is_file():
            raise ValueError(f"Learning checkpoint does not exist: {args.red_learning_model}")
    elif args.red_learning_model is not None:
        raise ValueError("--red-learning-model requires --red-motion-policy ppo or mappo")
