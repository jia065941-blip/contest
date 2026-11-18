"""Configuration and assignment contracts for bundled blue baselines."""

from __future__ import annotations

import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blue_defense import (
    BLUE_POLICY_CHOICES,
    BlueDefenseConfig,
    BlueDefenseController,
    BlueObservation,
    OwnEntityState,
    InterceptorState,
    ThreatTrack,
    build_blue_policy,
)
from envengine.common.BaseStruct import Vector3d


def observation() -> BlueObservation:
    vector = Vector3d()
    return BlueObservation(
        step=0,
        targets=[
            ThreatTrack(101, "threat-a", 21000, Vector3d(1000, 0, 0), Vector3d(-300, 0, 0), 0, source_id=301),
            ThreatTrack(102, "threat-b", 21001, Vector3d(0, 2000, 0), Vector3d(0, -200, 0), 0, source_id=301),
        ],
        interceptors=[
            InterceptorState(1, Vector3d(900, 0, 0), vector, True),
            InterceptorState(2, Vector3d(0, 1900, 0), vector, True),
            InterceptorState(3, Vector3d(5000, 0, 0), vector, False),
        ],
        own_entities=[OwnEntityState(201, "asset", 9400, vector, vector, 100.0)],
    )


class BlueBaselineSmokeTests(unittest.TestCase):
    def test_configuration_registry_and_policy_constructors_share_default_shot_limit(self) -> None:
        expected = BlueDefenseConfig().max_shots_per_target
        self.assertEqual(expected, 2)
        for name in BLUE_POLICY_CHOICES:
            with self.subTest(policy=name):
                policy = build_blue_policy(name)
                self.assertEqual(policy.max_shots_per_target, expected)
                self.assertEqual(type(policy)().max_shots_per_target, expected)

    def test_every_registered_baseline_returns_valid_assignments(self) -> None:
        valid_interceptors = {1, 2}
        valid_targets = {101, 102}
        for name in ("b0_fixed_ratio_random", "b1_nearest_interceptor", "b2_threat_priority", "b3_min_cost_assignment"):
            with self.subTest(policy=name):
                controller = BlueDefenseController(BlueDefenseConfig(
                    policy_name=name, max_shots_per_target=1, seed=7,
                ))
                assignments = controller.decide(observation()).assignments
                self.assertTrue(assignments)
                self.assertLessEqual(len(assignments), 2)
                self.assertTrue({item.interceptor_id for item in assignments} <= valid_interceptors)
                self.assertTrue({item.target_id for item in assignments} <= valid_targets)


if __name__ == "__main__":
    unittest.main()
