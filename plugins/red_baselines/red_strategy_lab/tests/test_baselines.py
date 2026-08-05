from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from red_strategy_lab import B0RandomPolicy, B1PriorityPolicy, B2StaticAssignmentPolicy, B3RollingRulePolicy, GlobalRules, Observation, Platform, Position, Target


def observation(step: int = 0, target_values: tuple[float, float] = (10.0, 4.0)) -> Observation:
    return Observation(
        step=step,
        platforms=(
            Platform(1, "H", Position(118.0, 22.0)),
            Platform(2, "M", Position(118.2, 22.0)),
            Platform(3, "L", Position(118.4, 22.0)),
        ),
        targets=(
            Target(101, Position(118.5, 22.0), value=target_values[0]),
            Target(102, Position(121.0, 24.0), value=target_values[1]),
        ),
    )


class BaselineTests(unittest.TestCase):
    def test_b0_is_reproducible_and_honours_capacity(self) -> None:
        rules = GlobalRules(target_capacity=1, min_launch_interval=1)
        first = B0RandomPolicy(rules, seed=88).decide(observation())
        second = B0RandomPolicy(rules, seed=88).decide(observation())
        self.assertEqual(first, second)
        self.assertLessEqual(max(sum(item.target_id == target for item in first.assignments) for target in (101, 102)), 1)

    def test_b1_uses_value_and_global_capacity(self) -> None:
        rules = GlobalRules(target_capacity=1, value_weight=5.0, distance_weight=0.01, coverage_weight=1.0)
        decision = B1PriorityPolicy(rules).decide(observation())
        targets = [item.target_id for item in decision.assignments]
        self.assertEqual(targets[0], 101)
        self.assertEqual(len(targets), len(set(targets)))
        self.assertEqual([item.launch_step for item in decision.assignments], sorted(item.launch_step for item in decision.assignments))

    def test_b2_finds_capacity_constrained_static_plan(self) -> None:
        rules = GlobalRules(target_capacity=1, value_weight=2.0, distance_weight=0.1)
        policy = B2StaticAssignmentPolicy(rules, candidate_limit=2, exact_platform_limit=8)
        decision = policy.decide(observation())
        self.assertEqual({item.target_id for item in decision.assignments}, {101, 102})
        self.assertEqual(len({item.platform_id for item in decision.assignments}), len(decision.assignments))
        self.assertEqual(policy.decide(observation(step=5)).assignments, decision.assignments)

    def test_b3_replans_on_configured_interval(self) -> None:
        rules = GlobalRules(target_capacity=2, replan_interval=3, value_weight=1.0, distance_weight=0.0)
        policy = B3RollingRulePolicy(rules)
        initial = policy.decide(observation(step=0, target_values=(10, 1)))
        cached = policy.decide(observation(step=2, target_values=(1, 10)))
        replanned = policy.decide(observation(step=3, target_values=(1, 10)))
        self.assertEqual([item.target_id for item in initial.assignments], [item.target_id for item in cached.assignments])
        self.assertEqual(replanned.assignments[0].target_id, 102)


if __name__ == "__main__":
    unittest.main()
