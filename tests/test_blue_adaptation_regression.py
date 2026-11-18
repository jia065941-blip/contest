import unittest

from blue_defense import BlueClock, BlueDefenseConfig, BlueDefenseController, BlueObservation, OwnEntityState
from blue_defense.coordination.b4_joint_timing_assignment import B4JointTimingAssignmentPolicy
from blue_defense.coordination.b5_successive_depth_coordination import B5SuccessiveDepthCoordinationPolicy
from blue_defense.coordination.b6_mixed_fire_coordination import B6MixedFireCoordinationPolicy
from envengine.common.BaseStruct import Vector3d


class BlueAdaptationRegressionTests(unittest.TestCase):
    def test_b6_replans_within_five_seconds_for_each_clock(self):
        for duration in (500, 1000, 2500):
            controller = BlueDefenseController(BlueDefenseConfig(
                policy_name='b6_mixed_fire_coordination', joint_decision_interval_step=100))
            clock = BlueClock(step_duration_ms=duration)
            interval = int(clock.seconds_to_steps(5))
            def observation(step):
                return BlueObservation(step=step, clock=clock, targets=(), interceptors=(), own_entities=())
            self.assertIsNotNone(controller.decide(observation(0)))
            self.assertIsNone(controller.decide(observation(interval - 1)))
            self.assertIsNotNone(controller.decide(observation(interval)))
            controller.close()

    def test_dark_nest_damage_depends_on_entity_type(self):
        for policy_type in (B4JointTimingAssignmentPolicy, B5SuccessiveDepthCoordinationPolicy,
                            B6MixedFireCoordinationPolicy):
            controller = BlueDefenseController(BlueDefenseConfig(policy_name=policy_type.name))
            position = Vector3d()
            observation = BlueObservation(step=0, targets=(), interceptors=(), own_entities=(
                OwnEntityState(1, '移动保护对象', 9500, position, position, 2),
                OwnEntityState(2, '暗巢名称但实际是母巢', 9400, position, position, 24),
            ))
            controller.decide(observation)
            assets = {row.asset_id: row for row in controller._information.snapshot(0).assets}
            self.assertEqual(policy_type._expected_damage(21002, assets[1].entity_type), 1)
            self.assertEqual(policy_type._expected_damage(21002, assets[2].entity_type), 0)
            controller.close()


if __name__ == '__main__':
    unittest.main()
