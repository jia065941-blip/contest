"""Run one deterministic demonstration without modifying the existing entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from red_strategy_lab import B0RandomPolicy, B1PriorityPolicy, B2StaticAssignmentPolicy, B3RollingRulePolicy, GlobalRules, Observation, Platform, Position, Target


def demo_observation() -> Observation:
    return Observation(
        step=0,
        platforms=(
            Platform(1, "H", Position(118.0, 22.0)),
            Platform(2, "M", Position(118.2, 22.1)),
            Platform(3, "L", Position(118.4, 22.2)),
        ),
        targets=(
            Target(101, Position(120.0, 23.0), value=10),
            Target(102, Position(119.0, 24.0), value=5),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=("b0", "b1", "b2", "b3"), default="b1")
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    rules = GlobalRules(target_capacity=2, min_launch_interval=1)
    policies = {
        "b0": B0RandomPolicy(rules, args.seed),
        "b1": B1PriorityPolicy(rules),
        "b2": B2StaticAssignmentPolicy(rules),
        "b3": B3RollingRulePolicy(rules),
    }
    for assignment in policies[args.baseline].decide(demo_observation()).assignments:
        print(f"platform={assignment.platform_id} target={assignment.target_id} launch_step={assignment.launch_step} score={assignment.score:.3f}")


if __name__ == "__main__":
    main()

