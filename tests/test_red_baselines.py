"""Assignment rules and received-track fusion inside one communication domain."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from user_agents.example.common.baselines import (
    BaselineObservation,
    BaselineRules,
    PlatformState,
    R0RandomPolicy,
    R1PriorityPolicy,
    R2StaticAssignmentPolicy,
    R3StaticPriorityPolicy,
    R4RollingRulePolicy,
    R5EventRollingPolicy,
    R6PressureAssignmentPolicy,
    R7StrikePackagePolicy,
    R8SatellitePackagePolicy,
    R9HierarchicalLearningPolicy,
    TacticalMetrics,
    TargetPrior,
)
from user_agents.example.common.contracts import Position
from user_agents.example.common.commander import RedBaselineCommander
from user_agents.example.common.targets import initial_targets_from_observation, ReceivedTrackFusion


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
    def test_initial_prior_excludes_hidden_targets(self) -> None:
        raw = {
            "entities": {
                "1": {"side": 1, "type": 9500, "health": 1, "pos": {"x": 1, "y": 2}},
                "2": {"side": 1, "type": 9400, "health": 1, "pos": {"x": 3, "y": 4}},
                "3": {"side": 1, "type": 9600, "health": 1, "pos": {"x": 5, "y": 6}},
            }
        }
        self.assertEqual([item.entity_id for item in initial_targets_from_observation(raw)], [2, 3])

    def test_all_baselines_assign_only_core_provided_targets(self) -> None:
        rules = BaselineRules(target_capacity=2, replan_interval=3)
        policies = (
            R0RandomPolicy(rules, seed=7),
            R1PriorityPolicy(rules),
            R2StaticAssignmentPolicy(rules),
            R3StaticPriorityPolicy(rules),
            R4RollingRulePolicy(rules),
            R5EventRollingPolicy(rules),
            R6PressureAssignmentPolicy(rules),
            R7StrikePackagePolicy(rules),
            R8SatellitePackagePolicy(rules),
            R9HierarchicalLearningPolicy(rules),
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
        rolling = R4RollingRulePolicy(rules)
        self.assertEqual(rolling.decide(observation()), rolling.decide(observation(step=2)))

    def test_static_priority_caches_one_greedy_plan(self) -> None:
        rules = BaselineRules()
        platforms = tuple(
            PlatformState(entity_id=index + 1, kind="H", position=Position(118.0 + index / 10, 22.0))
            for index in range(6)
        )
        fixed = R3StaticPriorityPolicy(rules)
        first = fixed.decide(BaselineObservation(0, platforms, observation().targets))
        self.assertEqual(first, fixed.decide(BaselineObservation(50, platforms, observation().targets)))
        self.assertEqual(first, R1PriorityPolicy(rules).decide(BaselineObservation(0, platforms, observation().targets)))
        self.assertFalse(hasattr(first[0], "launch_step"))

    def test_track_fusion_only_updates_initial_catalogue_target_positions(self) -> None:
        fusion = ReceivedTrackFusion(observation().targets)
        changed = fusion.ingest({
            "self": {
                "detectInfo": {
                    103: {"entity_id": 103, "entity_type": 9500, "pos": {"x": 122.0, "y": 25.0, "z": 0.0}},
                    999: {"entity_id": 999, "entity_type": 9400, "pos": {"x": 1.0, "y": 1.0, "z": 0.0}},
                }
            }
        })
        positions = {item.entity_id: item.position for item in fusion.targets}
        self.assertTrue(changed)
        self.assertEqual(positions[103], Position(122.0, 25.0, 0.0))
        self.assertNotIn(999, positions)

    def test_r5_merges_events_until_its_replan_cooldown_expires(self) -> None:
        commander = RedBaselineCommander(
            targets=(TargetPrior(101, 9500, Position(120.0, 20.0)),),
            policy_name="r5_event_rolling",
        )
        commander.register_platform(1)
        first = {
            "step": 0,
            "entity_id": 1,
            "self": {
                "type": 21000,
                "health": 1,
                "isVisible": True,
                "pos": {"x": 118.0, "y": 20.0, "z": 0.0},
                "detectInfo": {},
            },
        }
        commander.begin_step((first,))
        commander.action_for(1, 0)
        self.assertEqual(commander.last_plan_step, 0)

        changed = {
            **first,
            "step": 1,
            "self": {
                **first["self"],
                "detectInfo": {
                    101: {
                        "entity_id": 101,
                        "entity_type": 9500,
                        "pos": {"x": 121.0, "y": 20.0, "z": 0.0},
                    }
                },
            },
        }
        commander.begin_step((changed,))
        commander.action_for(1, 1)
        self.assertEqual(commander.last_plan_step, 0)
        self.assertGreater(commander.event_revision, commander.planned_revision)

        commander.action_for(1, commander.rules.event_replan_cooldown)
        self.assertEqual(commander.last_plan_step, commander.rules.event_replan_cooldown)
        self.assertEqual(commander.event_revision, commander.planned_revision)

    def test_commander_builds_adaptive_metrics_from_isolated_reports(self) -> None:
        commander = RedBaselineCommander(observation().targets, "r6_pressure_assignment")
        commander.register_platform(1)
        commander.register_platform(2)
        commander.begin_step(({
            "step": 1,
            "entity_id": 1,
            "self": {
                "type": 21000,
                "health": 1,
                "isVisible": True,
                "pos": {"x": 118.0, "y": 22.0, "z": 0.0},
                "detectInfo": {
                    500: {"entity_id": 500, "entity_type": 24000},
                },
            },
        },))
        self.assertEqual(commander.tactical_metrics.observed_interceptor_count, 1)
        self.assertEqual(commander.tactical_metrics.alive_count, 1)
        self.assertEqual(commander.tactical_metrics.pressure, 0.65)

    def test_r6_rolls_from_a_bounded_low_probe_to_a_main_strike(self) -> None:
        policy = R6PressureAssignmentPolicy(BaselineRules())
        probe = policy.decide(observation())
        self.assertEqual([item.platform_id for item in probe], [3])

        commit = policy.decide(BaselineObservation(
            step=10,
            platforms=(
                PlatformState(1, "H", Position(118.0, 22.0)),
                PlatformState(2, "M", Position(118.2, 22.0)),
            ),
            targets=observation().targets,
            metrics=TacticalMetrics(alive_count=3),
        ))
        self.assertEqual([item.platform_id for item in commit], [1])

    def test_r6_prioritizes_low_platform_reassignment_under_pressure(self) -> None:
        policy = R6PressureAssignmentPolicy(BaselineRules())
        pressured = BaselineObservation(
            step=10,
            platforms=observation().platforms,
            targets=observation().targets,
            metrics=TacticalMetrics(
                observed_interceptor_count=2,
                alive_count=3,
            ),
        )
        self.assertEqual([item.platform_id for item in policy.decide(pressured)], [3])

    def test_r7_pairs_h_and_m_on_the_same_strike_target(self) -> None:
        package_observation = BaselineObservation(
            step=10,
            platforms=(
                PlatformState(1, "H", Position(118.0, 22.0)),
                PlatformState(2, "H", Position(118.1, 22.0)),
                PlatformState(3, "M", Position(118.2, 22.0)),
                PlatformState(4, "M", Position(118.3, 22.0)),
                PlatformState(5, "L", Position(118.4, 22.0)),
            ),
            targets=observation().targets,
            metrics=TacticalMetrics(alive_count=5),
        )
        assignments = {item.platform_id: item for item in R7StrikePackagePolicy(BaselineRules()).decide(package_observation)}
        self.assertEqual(assignments[1].target_id, assignments[3].target_id)
        self.assertEqual(set(assignments), {1, 3})

    def test_r8_selects_one_h_satellite_leader_per_strike_target(self) -> None:
        package_observation = BaselineObservation(
            step=10,
            platforms=(
                PlatformState(1, "H", Position(118.0, 22.0)),
                PlatformState(2, "H", Position(118.1, 22.0)),
                PlatformState(3, "M", Position(118.2, 22.0)),
                PlatformState(4, "M", Position(118.3, 22.0)),
            ),
            targets=observation().targets,
            metrics=TacticalMetrics(alive_count=4),
        )
        policy = R8SatellitePackagePolicy(BaselineRules())
        assignments = {item.platform_id: item for item in policy.decide(package_observation)}
        self.assertTrue(policy.satellite_platform_ids)
        for platform_id in policy.satellite_platform_ids:
            self.assertEqual(package_observation.platforms[platform_id - 1].kind, "H")
            self.assertIn(assignments[platform_id].target_id, {101, 102})

    def test_fusion_rejects_older_steps_and_accepts_same_step_authoritative_updates(self) -> None:
        fusion = ReceivedTrackFusion(())

        def received(detected_step, x):
            return {"self": {"detectInfo": {103: {
                "entity_id": 103, "entity_type": 9500, "detected_step": detected_step,
                "pos": {"x": x, "y": 22.0, "z": 0.0},
            }}}}

        self.assertTrue(fusion.ingest(received(10, 120.0)))
        self.assertFalse(fusion.ingest(received(9, 119.0)))
        self.assertEqual(fusion.targets[0].position.x, 120.0)
        self.assertFalse(fusion.ingest(received(10, 120.0)))
        self.assertTrue(fusion.ingest(received(10, 120.5)))
        self.assertEqual(fusion.targets[0].position.x, 120.5)
        self.assertFalse(fusion.ingest(received(9, 119.0)))
        self.assertEqual(fusion.targets[0].position.x, 120.5)
        self.assertTrue(fusion.ingest(received(11, 121.0)))

    def test_random_policy_never_assigns_an_incompatible_target(self) -> None:
        policy = R0RandomPolicy(BaselineRules(), seed=7)
        assignments = {item.platform_id: item.target_id for item in policy.decide(observation())}
        self.assertIn(assignments[1], {101, 102})
        self.assertIn(assignments[2], {101, 102})
        self.assertEqual(assignments[3], 103)

    def test_static_search_uses_public_routes_without_launch_or_follow_on_delay(self):
        from user_agents.example.coordination.search import R7StaticSearchPolicy

        rules = BaselineRules(probe_reassignment_fraction=1.0)
        polygon = (Position(0, 0), Position(500000, 0), Position(500000, 400000), Position(0, 400000))
        policy = R7StaticSearchPolicy(rules, polygon)
        platforms = tuple(PlatformState(i, "L", Position(100000, 100000)) for i in (1, 2, 3))
        observed = BaselineObservation(0, platforms, observation().targets[:1])
        first = policy.decide(observed)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(item.target_id is None and item.destination is not None for item in first))
        self.assertEqual(policy.decide(BaselineObservation(1000, platforms, observed.targets)), ())
        detected = TargetPrior(103, 9500, Position(101000, 100000))
        self.assertEqual(policy.claim_retarget(platforms[0], (detected,)), detected)
        self.assertIsNone(policy.claim_retarget(platforms[1], (detected,)))


    def test_metric_distance_is_horizontal_and_translation_invariant(self):
        from user_agents.example.common.baselines import distance_km

        self.assertEqual(distance_km(Position(0, 0, 100), Position(3000, 4000, 9000)), 5.0)
        self.assertEqual(distance_km(Position(90000, -80000), Position(93000, -76000)), 5.0)

    def test_coverage_geometry_preserves_metric_ranges_and_heights(self):
        from user_agents.example.coordination.coverage import (
            SearchPlatformStart, plan_static_coverage, radial_fan_endpoints,
        )
        from user_agents.example.common.baselines import distance_km

        bounds = (Position(90000, 90000, 800), Position(110000, 90000, 800),
                  Position(110000, 110000, 800), Position(90000, 110000, 800))
        origin = Position(100000, 100000, 800)
        endpoints = radial_fan_endpoints(origin, bounds, count=8, max_range_km=4.0)
        self.assertEqual(len(endpoints), 8)
        for endpoint in endpoints:
            self.assertAlmostEqual(distance_km(origin, endpoint), 4.0)
            self.assertEqual(endpoint.z, 800)
        plan = plan_static_coverage(bounds, 2.0, (SearchPlatformStart(1, origin),))
        self.assertLessEqual(plan.lane_spacing_km, 4.0)
        self.assertGreater(plan.coverage_distance_km, 20.0)
        for point in plan.coverage_waypoints:
            self.assertTrue(90000 - 1e-6 <= point.x <= 110000 + 1e-6)
            self.assertTrue(90000 - 1e-6 <= point.y <= 110000 + 1e-6)

    def test_search_formation_spacing_uses_metres(self):
        from itertools import combinations
        from user_agents.example.coordination.search import _equilateral_destinations
        from user_agents.example.common.baselines import distance_km

        points = _equilateral_destinations(Position(100000, -30000, 600),
                                           Position(120000, -10000, 600), 6.0)
        for left, right in combinations(points, 2):
            self.assertAlmostEqual(distance_km(left, right), 6.0)
        self.assertTrue(all(point.z == 600 for point in points))


if __name__ == "__main__":
    unittest.main()
