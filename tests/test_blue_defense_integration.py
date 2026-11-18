from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envengine.common.BaseStruct import Vector3d
from blue_defense import BlueClock, BlueDefenseConfig, BlueDefenseController, InterceptorAssignment
from envengine.common import SimmerCommandType
from envengine.sdk.base_struct.Basic import DetectInfo
from envengine.sdk.base_struct.Message import Command
from envengine.sdk.writer import WriteConfig, init_writer
from test_blue_information import create_engine, place_sensor_targets, advance_until_detected


class BlueDefenseControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.writer = init_writer(WriteConfig(
            enable_config=False, enable_state=False, enable_event=False, enable_ai_action=False,
            output_dir=str(ROOT / ".blue_validation/controller"), verbose=False,
        ))
        cls.engine = create_engine()
        place_sensor_targets(cls.engine)
        advance_until_detected(cls.engine)
        factory = cls.engine.simulator_factory
        commander = factory.get_simulators_by_type(35000)[0]
        commander.sim_time = cls.engine.sim_time
        commander.current_step = cls.engine.current_step
        cls.observation = replace(commander._build_blue_observation(), targets=tuple(
            commander._detect_info_to_threat_track(record)
            for radar in factory.get_simulators_by_type(44000)
            for record in radar.entity_ext.entity.detectInfo.values()
        ))

    @classmethod
    def tearDownClass(cls):
        cls.writer.close()

    def test_b3_preserves_configured_decision_interval(self):
        controller = BlueDefenseController(BlueDefenseConfig(
            policy_name="b3_min_cost_assignment", min_cost_decision_interval_step=5, seed=42,
        ))
        first = controller.decide(self.observation)
        self.assertIsNotNone(first)
        self.assertTrue(first.assignments)
        self.assertIsNone(controller.decide(replace(
            self.observation, step=self.observation.step + 1,
        )))
        self.assertIsNotNone(controller.decide(replace(
            self.observation, step=self.observation.step + 5,
        )))

    def test_b7_releases_on_timed_decision_and_reset_discards_old_plan(self):
        controller = BlueDefenseController(BlueDefenseConfig(
            policy_name="b7_planned_nearest", planning_delay_step=5, seed=42,
        ))
        observation = replace(
            self.observation, step=0,
            targets=tuple(replace(track, detect_step=0) for track in self.observation.targets),
            clock=BlueClock(start_time_ms=self.engine.sim_time, step_duration_ms=1000),
        )
        self.assertEqual(controller.decide(observation).assignments, ())
        self.assertEqual(controller.decide(replace(observation, step=4)).assignments, ())
        released = controller.decide(replace(observation, step=5))
        self.assertTrue(released.assignments)
        self.assertTrue(all(isinstance(item, InterceptorAssignment) for item in released.assignments))
        controller.reset()
        self.assertEqual(controller.decide(replace(observation, step=5)).assignments, ())

    def test_b0_reset_preserves_random_stream_by_default(self):
        config = BlueDefenseConfig(policy_name="b0_fixed_ratio_random", seed=17)
        continuous = BlueDefenseController(config)
        reset = BlueDefenseController(config)
        self.assertEqual(continuous.decide(self.observation), reset.decide(self.observation))
        reset.reset()
        self.assertEqual(continuous.decide(self.observation), reset.decide(self.observation))

    def test_b0_explicit_reseed_reproduces_decision(self):
        controller = BlueDefenseController(BlueDefenseConfig(
            policy_name="b0_fixed_ratio_random", seed=17, reset_rng_on_episode=True,
        ))
        first = controller.decide(self.observation)
        controller.reset()
        self.assertEqual(first, controller.decide(self.observation))

    def test_real_commander_uses_injected_controller_on_update(self):
        engine = create_engine("b7_planned_nearest")
        factory = engine.simulator_factory
        commander = factory.get_simulators_by_type(35000)[0]
        self.assertEqual(commander.defense_controller.name, "b7_planned_nearest")
        place_sensor_targets(engine)
        advance_until_detected(engine)
        controller = commander.defense_controller
        commander._decide_from_information()
        controller.reset()
        commander.sim_time = engine.sim_time
        commander.update()
        self.assertIs(commander.defense_controller, controller)
        self.assertEqual(commander._build_blue_observation().targets, ())
        self.assertEqual(controller.get_tracks(commander.current_step), ())

    def test_real_launch_boundary_rejects_target_without_record(self):
        engine = create_engine()
        factory = engine.simulator_factory
        commander = factory.get_simulators_by_type(35000)[0]
        interceptor = factory.get_simulators_by_type(24000)[0]
        track = self.observation.targets[0]
        assignment = InterceptorAssignment(
            interceptor_id=interceptor.entity_ext.entity.id, target_id=track.target_id,
        )
        commander._execute_immediate_assignments([assignment])
        factory.process_commands()
        self.assertEqual(interceptor.launched, -1)
        self.assertEqual(commander.launched_list, {})

    def native_launch_fixture(self):
        engine = create_engine()
        factory = engine.simulator_factory
        commander = factory.get_simulators_by_type(35000)[0]
        interceptor = next(
            unit for unit in factory.get_simulators_by_type(24000)
            if factory.get_simulator_by_id(unit.entity_ext.entity.parentId).entity_ext.entity.entityType == 9600
        )
        parent = factory.get_simulator_by_id(interceptor.entity_ext.entity.parentId).entity_ext.entity
        radar = next(
            unit.entity_ext.entity for unit in factory.get_simulators_by_type(44000)
            if unit.entity_ext.entity.parentId == parent.id
        )
        target = factory.get_simulators_by_type(21001)[0].entity_ext.entity
        target.posEcf = Vector3d(parent.posEcf.x + 1000, parent.posEcf.y, parent.posEcf.z)
        target.velEcf = Vector3d(-100, 0, 0)
        target.stage = 3
        engine.current_step = 20
        engine.sim_time = commander._blue_clock.time_at(engine.current_step)
        factory.update_sim_time(engine.sim_time, engine.current_step)
        record = DetectInfo(
            detect_from=radar.id, step=engine.current_step,
            entity_id=target.id, entity_type=target.entityType, nameChn=target.nameChn,
            pos=deepcopy(target.posEcf), vel=deepcopy(target.velEcf),
        )
        return engine, commander, interceptor, record

    def test_native_launch_survives_radar_update_in_both_queue_orders(self):
        for radar_first in (True, False):
            with self.subTest(radar_first=radar_first):
                engine, commander, interceptor, record = self.native_launch_fixture()
                factory = engine.simulator_factory
                track = commander._detect_info_to_threat_track(record)
                commander.defense_controller.receive_tracks((track,), commander.current_step)
                commander._observed_target_ids.add(track.target_id)
                radar_update = Command(
                    executorId=commander.entity_ext.entity.id,
                    commandTypeId=SimmerCommandType.RADAR_DETECT_STATUS_UPDATE,
                    commandAttributes={record.entity_id: record},
                )
                if radar_first:
                    factory._send_command_callback([radar_update])
                commander._execute_immediate_assignments([
                    InterceptorAssignment(interceptor.entity_ext.entity.id, record.entity_id),
                ])
                if not radar_first:
                    factory._send_command_callback([radar_update])
                factory.process_commands()
                self.assertEqual(interceptor.launched, 1)
                self.assertEqual(commander.launched_list, {record.entity_id: [interceptor.entity_ext.entity.id]})
                self.assertEqual(commander.coordination_metrics["actual_launches"], 1)

    def test_mother_and_sentinel_both_evade_in_either_update_order(self):
        for sentinel_first in (True, False):
            with self.subTest(sentinel_first=sentinel_first):
                engine, commander, _, record = self.native_launch_fixture()
                factory = engine.simulator_factory
                mother = factory.get_simulators_by_type(9400)[0]
                sentinel = factory.get_simulators_by_type(9600)[0]
                record.vel = Vector3d(100, 0, 0)
                before = {}
                for index, ship in enumerate((mother, sentinel)):
                    position = Vector3d(record.pos.x - 1000, record.pos.y + index * 1000, record.pos.z)
                    ship.entity_ext.entity.posEcf = deepcopy(position)
                    ship._initial_entity_ext.entity.posEcf = deepcopy(position)
                    ship._attacker = None
                    before[ship.entity_ext.entity.id] = position
                commander.handel_detect_info({record.entity_id: record})
                order = (sentinel, commander, mother) if sentinel_first else (mother, commander, sentinel)
                real_id = id
                priorities = {real_id(unit.entity_ext.entity): index for index, unit in enumerate(order)}
                with patch("envengine.engine.engine.id", create=True,
                           side_effect=lambda value: priorities.get(real_id(value), real_id(value))):
                    engine.step([])
                elapsed_seconds = engine.sim_step / 1000.0
                for ship in (mother, sentinel):
                    self.assertIsNotNone(ship._attacker)
                    self.assertAlmostEqual(ship.entity_ext.entity.velEcf.length, 12.0)
                    initial = before[ship.entity_ext.entity.id]
                    self.assertAlmostEqual(ship.entity_ext.entity.posEcf.x, initial.x)
                    self.assertAlmostEqual(ship.entity_ext.entity.posEcf.y, initial.y + 12.0 * elapsed_seconds)


if __name__ == "__main__":
    unittest.main()
