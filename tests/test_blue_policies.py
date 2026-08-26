"""Dependency-free contract checks for every bundled blue baseline."""

from __future__ import annotations

import unittest

from policies.blue import (
    BlueObservation,
    DefendedAsset,
    InterceptorState,
    ThreatTrack,
    Vector3,
    build_blue_policy,
)


def observation() -> BlueObservation:
    vector = Vector3()
    return BlueObservation(
        sim_time=0,
        targets=[
            ThreatTrack(101, "threat-a", 1, vector, Vector3(1000, 0, 0), Vector3(-300, 0, 0), 0),
            ThreatTrack(102, "threat-b", 2, vector, Vector3(0, 2000, 0), Vector3(0, -200, 0), 0),
        ],
        interceptors=[
            InterceptorState(1, vector, Vector3(900, 0, 0), vector, True),
            InterceptorState(2, vector, Vector3(0, 1900, 0), vector, True),
            InterceptorState(3, vector, Vector3(5000, 0, 0), vector, False),
        ],
        assets=[DefendedAsset(201, "asset", vector, vector, 100.0, 1.0)],
    )


class BlueBaselineSmokeTests(unittest.TestCase):
    def test_every_registered_baseline_returns_valid_assignments(self) -> None:
        valid_interceptors = {1, 2}
        valid_targets = {101, 102}
        for name in ("b0_fixed_ratio_random", "b1_nearest_interceptor", "b2_threat_priority", "b3_min_cost_assignment"):
            with self.subTest(policy=name):
                assignments = build_blue_policy(name, max_shots_per_target=1, seed=7).decide(observation())
                self.assertTrue(assignments)
                self.assertLessEqual(len(assignments), 2)
                self.assertTrue({item.interceptor_id for item in assignments} <= valid_interceptors)
                self.assertTrue({item.target_id for item in assignments} <= valid_targets)


if __name__ == "__main__":
    unittest.main()
