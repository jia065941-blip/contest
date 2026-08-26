"""Checks for R0--R3 using the complete core-provided target catalogue."""

from __future__ import annotations

import unittest

from policies.red.baselines import (
    BaselineObservation,
    BaselineRules,
    PlatformState,
    R0RandomPolicy,
    R1PriorityPolicy,
    R2StaticAssignmentPolicy,
    R3RollingRulePolicy,
    TargetPrior,
)
from policies.red.contracts import Position
from policies.red.priors import initial_targets_from_observation


def observation(step: int = 0) -> BaselineObservation:
    return BaselineObservation(
        step=step,
        platforms=(
            PlatformState(1, "H", Position(118.0, 22.0)),
            PlatformState(2, "M", Position(118.2, 22.0)),
            PlatformState(3, "L", Position(118.4, 22.0)),
        ),
        targets=(
            TargetPrior(101, 9400, Position(118.5, 22.0), value=10.0),
            TargetPrior(102, 9600, Position(119.0, 22.0), value=6.0),
            TargetPrior(103, 9500, Position(121.0, 24.0), value=3.0),
        ),
    )


class RedBaselineTests(unittest.TestCase):
    def test_initial_prior_uses_every_core_provided_target(self) -> None:
        raw = {
            "entities": {
                "1": {"side": 1, "type": 9500, "health": 1, "position": {"lon": 1, "lat": 2}},
                "2": {"side": 1, "type": 9400, "health": 1, "position": {"lon": 3, "lat": 4}},
                "3": {"side": 1, "type": 9600, "health": 1, "position": {"lon": 5, "lat": 6}},
            }
        }
        self.assertEqual([item.entity_id for item in initial_targets_from_observation(raw)], [1, 2, 3])

    def test_all_baselines_assign_only_core_provided_targets(self) -> None:
        rules = BaselineRules(target_capacity=2, replan_interval=3)
        policies = (
            R0RandomPolicy(rules, seed=7),
            R1PriorityPolicy(rules),
            R2StaticAssignmentPolicy(rules),
            R3RollingRulePolicy(rules),
        )
        for policy in policies:
            with self.subTest(policy=type(policy).__name__):
                assignments = policy.decide(observation())
                self.assertTrue(assignments)
                self.assertTrue({item.target_id for item in assignments} <= {101, 102, 103})
                self.assertEqual(len({item.platform_id for item in assignments}), len(assignments))

    def test_optimizing_policy_uses_platform_target_capability(self) -> None:
        assignments = {item.platform_id: item.target_id for item in R1PriorityPolicy(BaselineRules()).decide(observation())}
        self.assertEqual(assignments[1], 101)
        self.assertEqual(assignments[2], 101)
        self.assertEqual(assignments[3], 103)

    def test_static_plan_is_retained_and_rolling_plan_replans(self) -> None:
        rules = BaselineRules(target_capacity=2, replan_interval=3)
        static = R2StaticAssignmentPolicy(rules)
        self.assertEqual(static.decide(observation()), static.decide(observation(step=5)))
        rolling = R3RollingRulePolicy(rules)
        self.assertEqual(rolling.decide(observation()), rolling.decide(observation(step=2)))


if __name__ == "__main__":
    unittest.main()
