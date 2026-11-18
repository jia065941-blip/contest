"""Policy-owned target memory and protected-object selection at the engine boundary."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

from envengine.common.BaseStruct import Vector3d
from blue_defense import (
    BaseBlueDefense, BlueClock, BlueDecision, BlueDefenseConfig, BlueDefenseController,
    BlueObservation, OwnEntityState, ThreatTrack
)
from envengine.simulator.simulator_factory import SimulatorFactory


def detected_target(identifier=101, *, time=0, x=10):
    return ThreatTrack(
        target_id=identifier, name="target", type=21000,
        pos=Vector3d(x, 20, 30),
        vel=Vector3d(2, 0, 0), detect_step=time, source_id=44001,
    )


class CustomDefense(BaseBlueDefense):
    name = "custom-memory"

    def decide(self, observation):
        return BlueDecision()


class SixtyStepDefense(CustomDefense):
    def get_tracks(self, sim_step):
        return tuple(
            replace(track, pos=Vector3d(
                track.pos.x + 2 * track.vel.x * self._clock.steps_to_seconds(sim_step - track.detect_step),
                track.pos.y, track.pos.z,
            ))
            for track in super().get_tracks(sim_step)
            if sim_step - track.detect_step <= 60
        )


class BlueTargetMemoryTests(unittest.TestCase):
    def test_default_custom_memory_has_no_builtin_expiry_or_prediction(self):
        target = detected_target()
        custom = CustomDefense()
        builtin = BlueDefenseController()
        for controller in (custom, builtin):
            controller.receive_tracks((target,), 0)
        self.assertEqual(custom.get_track(101, 30_001), target)
        self.assertIsNone(builtin.get_track(101, 30_001))
        self.assertEqual(custom.get_track(101, 10_000).pos, target.pos)
        custom.reset()
        self.assertEqual(custom.get_tracks(30_001), ())

    def test_engine_target_and_nearest_queries_follow_custom_memory(self):
        controller = SixtyStepDefense()
        controller.bind_clock(BlueClock(step_duration_ms=1000))
        controller.receive_tracks((detected_target(),), 0)
        factory = object.__new__(SimulatorFactory)
        commander = SimpleNamespace(defense_controller=controller, _observed_target_ids={101})
        factory._simulators_by_type = {35000: [commander]}
        # No world entities are available to resolve the target's state.
        factory._simulators = {}
        predicted = factory.get_blue_track(101, 40)
        self.assertEqual(predicted.pos, Vector3d(170, 20, 30))
        self.assertEqual(factory.nearest_blue_track(Vector3d(170, 20, 30), 1, 40), predicted)
        self.assertIsNone(factory.nearest_blue_track(Vector3d(), 1, 40))
        self.assertIsNone(factory.get_blue_track(101, 61))
        self.assertIsNone(factory.nearest_blue_track(Vector3d(170, 20, 30), 1, 61))
        controller.receive_tracks((detected_target(102),), 0)
        self.assertIsNone(factory.get_blue_track(102, 40))

    def test_throttled_builtin_decision_still_receives_new_detections(self):
        controller = BlueDefenseController(BlueDefenseConfig(
            policy_name="b3_min_cost_assignment", min_cost_decision_interval_step=100,
        ))
        observation = BlueObservation(
            step=0, clock=BlueClock(step_duration_ms=50),
            targets=(detected_target(),), interceptors=(), own_entities=(),
        )
        self.assertIsNotNone(controller.decide(observation))
        latest = detected_target(time=20, x=90)
        self.assertIsNone(controller.decide(replace(observation, step=20, targets=(latest,))))
        self.assertEqual(controller.get_track(101, 20), latest)
        self.assertEqual(controller.get_track(101, 40).pos.x, 92)
        controller.reset()
        self.assertEqual(controller.get_tracks(2000), ())

    def test_builtin_assets_are_derived_from_current_own_entities(self):
        controller = BlueDefenseController(BlueDefenseConfig(asset_values={5: 25.0}))

        def own(identifier, entity_type, *, health=100, visible=True):
            return OwnEntityState(
                identifier, str(identifier), entity_type, Vector3d(identifier, 0, 0), Vector3d(), health, visible,
            )

        observation = BlueObservation(
            step=0, targets=(), interceptors=(), own_entities=(
                own(1, 24000), own(2, 9400, health=0), own(3, 9500, visible=False),
                own(4, 9600), own(5, 9400),
            ),
        )
        self.assertFalse(hasattr(observation, "assets"))
        controller.decide(observation)
        assets = controller._information.snapshot(0).assets
        self.assertEqual([(item.asset_id, item.value) for item in assets], [(5, 25.0), (4, 6.0)])
        moved = replace(observation.own_entities[-1], pos=Vector3d(500, 0, 0), health=12)
        controller.decide(replace(observation, step=1000, own_entities=(moved,)))
        assets = controller._information.snapshot(1000).assets
        self.assertEqual(len(assets), 1)
        self.assertEqual((assets[0].pos, assets[0].health), (Vector3d(500, 0, 0), 12))


if __name__ == "__main__":
    unittest.main()
