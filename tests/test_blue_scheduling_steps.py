from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import envengine  # Initialize the SDK before importing blue strategy modules.
from blue_defense.base import (
    BlueClock, DefendedAsset, InterceptorState, ScheduledInterceptorAssignment,
    ThreatTrack,
)
from blue_defense.coordination.b4_joint_timing_assignment import (
    B4JointTimingAssignmentPolicy, _Candidate, _ThreatAssessment,
)
from blue_defense.coordination.b5_successive_depth_coordination import (
    B5SuccessiveDepthCoordinationPolicy, _Engagement,
)
from blue_defense.coordination.b6_mixed_fire_coordination import (
    B6MixedFireCoordinationPolicy, _TargetPlan,
)
from blue_defense.scheduled.b7_planned_nearest import B7PlannedNearestPolicy
from blue_defense.scheduled.b8_threat_multi_wave import B8ThreatMultiWavePolicy
from blue_defense.scheduled.b9_min_cost_planned_multi_wave import B9MinCostPlannedMultiWavePolicy
from blue_defense.shared.information import PolicyObservation
from envengine.common.BaseStruct import Vector3d


def observation(duration_ms=1000, step=0, *, launched_map=None):
    return PolicyObservation(
        step=step,
        clock=BlueClock(step_duration_ms=duration_ms),
        targets=(ThreatTrack(1, 'target', 21000, Vector3d(0, 0, 0),
                             Vector3d(100, 0, 0), 0, 55),),
        interceptors=tuple(
            InterceptorState(entity_id, Vector3d(5000, 0, 0), Vector3d(0, 0, 0),
                             True, max_launch_angle_deg=30.0)
            for entity_id in (11, 12)
        ),
        own_entities=(),
        assets=(DefendedAsset(52, 'mother', Vector3d(50000, 0, 0), 100, 10, 9400),),
        launched_map={} if launched_map is None else launched_map,
    )


def assignment(interceptor_id=11, *, intercept_step=0.0, intercept_x=0.0):
    return ScheduledInterceptorAssignment(
        interceptor_id=interceptor_id, target_id=1, decision_step=0,
        planned_launch_step=0.0, estimated_intercept_step=intercept_step,
        estimated_intercept_pos=Vector3d(intercept_x, 0, 0),
    )


class BlueSchedulingStepTests(unittest.TestCase):
    def test_b4_grid_and_cost_preserve_physical_time_across_clocks(self):
        policy = B4JointTimingAssignmentPolicy()
        reference = None
        for duration_ms in (500, 1000, 3000):
            obs = observation(duration_ms, step=12)
            candidates = policy._candidates(obs, obs.targets[0], obs.interceptors[0])
            self.assertEqual(len(candidates), 16)
            physical = []
            for item in candidates:
                plan = item.assignment
                self.assertEqual(plan.decision_step, 12)
                physical.append((
                    obs.clock.steps_to_seconds(plan.planned_launch_step - obs.step),
                    obs.clock.steps_to_seconds(plan.estimated_intercept_step - obs.step),
                    plan.estimated_intercept_pos.x,
                    item.cost,
                ))
            self.assertAlmostEqual(physical[-1][0], 30.0)
            self.assertNotEqual(candidates[0].assignment.estimated_intercept_step,
                                int(candidates[0].assignment.estimated_intercept_step))
            if reference is None:
                reference = physical
            else:
                for actual, expected in zip(physical, reference):
                    for value, baseline in zip(actual, expected):
                        self.assertAlmostEqual(value, baseline)

    def test_b4_solver_returns_step_plans_with_unchanged_quota(self):
        obs = observation(3000, step=4)
        policy = B4JointTimingAssignmentPolicy(max_shots_per_target=2)
        plans = policy.decide(obs)
        self.assertEqual(len(plans), 2)
        self.assertEqual({plan.interceptor_id for plan in plans}, {11, 12})
        for plan in plans:
            self.assertEqual(plan.decision_step, 4)
            self.assertGreater(plan.estimated_intercept_step, plan.planned_launch_step)
            self.assertFalse(hasattr(plan, 'planned_launch_time'))

    def test_b5_minimum_gap_uses_same_seconds_for_each_clock(self):
        for duration_ms in (500, 1000, 3000):
            for gap_seconds, expected_count in ((4.75, 0), (5.0, 1)):
                with self.subTest(duration_ms=duration_ms, gap_seconds=gap_seconds):
                    obs = observation(duration_ms, step=1, launched_map={1: (11,)})
                    policy = B5SuccessiveDepthCoordinationPolicy()
                    policy._engagements[1] = _Engagement('pair', assignment())
                    candidate = _Candidate(assignment(
                        12, intercept_step=obs.clock.seconds_to_steps(gap_seconds),
                        intercept_x=gap_seconds * 100,
                    ), 0.01)
                    policy._target_candidates = lambda *_: [candidate]
                    policy._assess_threat = lambda *_: _ThreatAssessment(1000, 10)
                    plans = policy.decide(obs)
                    self.assertEqual(len(plans), expected_count)
                    if plans:
                        self.assertAlmostEqual(
                            obs.clock.steps_to_seconds(policy.last_solve_stats['intercept_gap_step_total']),
                            gap_seconds,
                        )

    def test_b6_gap_window_preserves_five_to_fifteen_seconds(self):
        for duration_ms in (500, 1000, 3000):
            obs = observation(duration_ms)
            for gap_seconds, expected_count in ((4.75, 0), (5.0, 1), (15.0, 1), (15.25, 0)):
                with self.subTest(duration_ms=duration_ms, gap_seconds=gap_seconds):
                    policy = B6MixedFireCoordinationPolicy()
                    candidate = _Candidate(assignment(
                        12, intercept_step=obs.clock.seconds_to_steps(gap_seconds),
                        intercept_x=gap_seconds * 100,
                    ), 0.01)
                    policy._target_candidates = lambda *_: [candidate]
                    candidates = policy._second_candidates(
                        obs, obs.targets[0], obs.interceptors,
                        _TargetPlan('successive', assignment()),
                        _ThreatAssessment(1000, 10),
                    )
                    self.assertEqual(len(candidates), expected_count)

    def test_b7_fractional_deadline_never_releases_early(self):
        policy = B7PlannedNearestPolicy(max_shots_per_target=1, planning_delay=1.25)
        obs = observation(3000)
        self.assertEqual(policy.decide(obs), [])
        self.assertEqual(policy._pending[0].due_step, 1.25)
        self.assertEqual(policy.decide(replace(obs, step=1)), [])
        self.assertEqual(len(policy.decide(replace(obs, step=2))), 1)

    def test_b8_fractional_wave_gap_never_releases_early(self):
        policy = B8ThreatMultiWavePolicy(wave_gap=2.5, wave_size=1)
        obs = observation(3000)
        first = policy.decide(obs)
        self.assertEqual(len(first), 1)
        self.assertEqual(policy._pending[0].due_step, 2.5)
        obs = replace(obs, launched_map={1: (first[0].interceptor_id,)})
        self.assertEqual(policy.decide(replace(obs, step=2)), [])
        self.assertEqual(len(policy.decide(replace(obs, step=3))), 1)

    def test_b9_fractional_delay_and_gap_remain_fractional(self):
        policy = B9MinCostPlannedMultiWavePolicy(planning_delay=0.25, wave_gap=1.25)
        obs = observation(3000)
        self.assertEqual(policy.decide(obs), [])
        self.assertEqual([plan.due_step for plan in policy._pending], [0.25, 1.5])
        first = policy.decide(replace(obs, step=1))
        self.assertEqual(len(first), 1)
        obs = replace(obs, launched_map={1: (first[0].interceptor_id,)})
        self.assertEqual(len(policy.decide(replace(obs, step=2))), 1)

    def test_direct_defaults_preserve_seconds_without_caching_clock(self):
        policies = (B7PlannedNearestPolicy(), B8ThreatMultiWavePolicy(),
                    B9MinCostPlannedMultiWavePolicy())
        for duration_ms in (500, 3000):
            obs = observation(duration_ms)
            for policy in policies:
                policy.reset()
                policy.decide(obs)
                due_seconds = [obs.clock.steps_to_seconds(plan.due_step)
                               for plan in policy._pending]
                if isinstance(policy, B7PlannedNearestPolicy):
                    self.assertEqual(due_seconds, [1.0, 1.0])
                    self.assertIsNone(policy.planning_delay)
                elif isinstance(policy, B8ThreatMultiWavePolicy):
                    self.assertEqual(due_seconds, [10.0])
                    self.assertIsNone(policy.wave_gap)
                else:
                    self.assertEqual(due_seconds, [1.0, 6.0])
                    self.assertIsNone(policy.planning_delay)
                    self.assertIsNone(policy.wave_gap)


if __name__ == '__main__':
    unittest.main()
