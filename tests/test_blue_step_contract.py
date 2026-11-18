"""Blue decisions use steps; physical prediction uses the scenario clock."""

from dataclasses import FrozenInstanceError, asdict, fields
import unittest
from unittest.mock import patch

from envengine.common.BaseStruct import Vector3d
from blue_defense import (
    BLUE_POLICY_CHOICES, BlueClock, BlueDefenseConfig, BlueDefenseController,
    BlueInformationStore, BlueObservation, InterceptorState, OwnEntityState,
    ScheduledInterceptorAssignment, ThreatTrack,
)


def observation(clock, step=0):
    return BlueObservation(
        step=step, clock=clock,
        targets=(
            ThreatTrack(101, "threat-a", 21000, Vector3d(10000, 0, 0),
                        Vector3d(-300, 0, 0), step, source_id=301),
            ThreatTrack(102, "threat-b", 21001, Vector3d(0, 20000, 0),
                        Vector3d(0, -200, 0), step, source_id=301),
        ),
        interceptors=tuple(InterceptorState(i, Vector3d(), Vector3d(), True)
                           for i in range(1, 7)),
        own_entities=(OwnEntityState(201, "asset", 9400, Vector3d(), Vector3d(), 100),),
    )


class BlueStepClockTests(unittest.TestCase):
    def test_clock_converts_fractional_durations_without_origin_or_rounding(self):
        for duration in (1, 50, 250, 1000):
            clock = BlueClock(start_time_ms=1783391450000, step_duration_ms=duration)
            for step in (0, 1, 6):
                self.assertEqual(clock.step_at(clock.time_at(step)), step)
            for count in (0, 1.25, 5.5):
                seconds = count * duration / 1000
                self.assertEqual(clock.steps_to_seconds(count), seconds)
                self.assertAlmostEqual(clock.seconds_to_steps(seconds), count)
            self.assertEqual(clock.continuous_step_at(clock.start_time_ms + 2.5 * duration), 2.5)
        with self.assertRaises(ValueError):
            BlueClock(step_duration_ms=1000).step_at(5500)
        for invalid in (True, float("nan"), float("inf")):
            with self.assertRaises((TypeError, ValueError)):
                clock.steps_to_seconds(invalid)

    def test_observation_and_scheduled_actions_store_only_steps(self):
        obs = observation(BlueClock(step_duration_ms=1000), 6)
        self.assertEqual(obs.sim_step, 6)
        self.assertFalse(hasattr(obs, "sim_time"))
        self.assertNotIn("sim_time", asdict(obs))
        with self.assertRaises(FrozenInstanceError):
            obs.step = 7
        action = ScheduledInterceptorAssignment(1, 101, 6, 7.25, 8.5)
        self.assertTrue({"decision_step", "planned_launch_step", "estimated_intercept_step"}
                        <= {item.name for item in fields(action)})
        self.assertFalse(any(name.endswith("_time") for name in asdict(action)))

    def test_prediction_uses_global_step_duration_and_memory_stays_600_steps(self):
        for duration in (50, 1000):
            clock = BlueClock(start_time_ms=1783391450000, step_duration_ms=duration)
            store = BlueInformationStore(clock)
            track = ThreatTrack(101, "target", 21000, Vector3d(100, 200, 300),
                                Vector3d(20, -40, 60), 10, source_id=301)
            store.receive_tracks((track,), 10)
            seconds = 20 * duration / 1000
            self.assertEqual(store.get_track(101, 30).pos,
                             Vector3d(100 + 20 * seconds, 200 - 40 * seconds, 300 + 60 * seconds))
            self.assertIsNotNone(store.get_track(101, 610))
            self.assertIsNone(store.get_track(101, 611))

    def test_prediction_requires_an_explicit_clock_before_first_decision(self):
        controller = BlueDefenseController()
        track = observation(BlueClock()).targets[0]
        controller.receive_tracks((track,), 0)
        with self.assertRaises(RuntimeError):
            controller.get_track(101, 1)
        controller.bind_clock(BlueClock(step_duration_ms=1000))
        self.assertEqual(controller.get_track(101, 1).pos.x, 9700)

    def test_all_policies_ignore_the_absolute_start_time(self):
        # Same steps and physical durations must not depend on a calendar epoch.
        for duration in (50, 1000):
            for name in BLUE_POLICY_CHOICES:
                with self.subTest(duration=duration, policy=name), patch("time.perf_counter", return_value=0):
                    config = BlueDefenseConfig(policy_name=name, seed=17)
                    left, right = BlueDefenseController(config), BlueDefenseController(config)
                    clocks = (BlueClock(step_duration_ms=duration),
                              BlueClock(start_time_ms=1783391450000, step_duration_ms=duration))
                    for step in (0, 1, 5, 20, 100):
                        self.assertEqual(left.decide(observation(clocks[0], step)),
                                         right.decide(observation(clocks[1], step)))
                    left.reset()
                    right.reset()
                    self.assertEqual(left.decide(observation(clocks[0])),
                                     right.decide(observation(clocks[1])))

    def test_binding_clock_preserves_b0_random_stream_across_resets(self):
        config = BlueDefenseConfig(seed=17)
        left, right = BlueDefenseController(config), BlueDefenseController(config)
        clock = BlueClock(step_duration_ms=1000)
        self.assertEqual(left.decide(observation(clock)), right.decide(observation(clock)))
        original_policy = left._policy
        left.bind_clock(clock)
        left.reset()
        left.bind_clock(clock)
        self.assertIs(left._policy, original_policy)
        self.assertEqual(left.decide(observation(clock)), right.decide(observation(clock)))

    def test_default_wave_durations_and_explicit_step_intervals(self):
        for name, seconds in (("b8_threat_multi_wave", 10), ("b9_min_cost_planned_multi_wave", 5)):
            for duration in (50, 1000):
                clock = BlueClock(step_duration_ms=duration)
                default = BlueDefenseController(BlueDefenseConfig(policy_name=name))
                default.bind_clock(clock)
                self.assertEqual(default._policy.wave_gap, clock.seconds_to_steps(seconds))
                explicit = BlueDefenseController(BlueDefenseConfig(policy_name=name, wave_gap_step=3.25))
                explicit.bind_clock(clock)
                explicit.bind_clock(clock)
                self.assertEqual(explicit._policy.wave_gap, 3.25)

    def test_first_decision_is_not_delayed_by_the_interval(self):
        controller = BlueDefenseController(BlueDefenseConfig(
            policy_name="b4_joint_timing_assignment", joint_decision_interval_step=100,
        ))
        clock = BlueClock(step_duration_ms=1000)
        self.assertIsNotNone(controller.decide(observation(clock)))
        self.assertIsNone(controller.decide(observation(clock, 1)))
        self.assertIsNotNone(controller.decide(observation(clock, 100)))
        controller.reset()
        self.assertIsNotNone(controller.decide(observation(clock)))


if __name__ == "__main__":
    unittest.main()
