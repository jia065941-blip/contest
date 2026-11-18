from dataclasses import FrozenInstanceError, replace
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envengine.common.BaseStruct import Vector3d
from blue_defense import (
    BLUE_POLICY_CHOICES, BaseBlueDefense, BlueDefenseConfig, BlueDefenseController, BlueInformationStore,
    ScheduledInterceptorAssignment, build_blue_policy,
)
from envengine import Engine
from scenarios import load_scenario
from envengine.common import DamageC, DamageCmd, SimmerCommandType
from envengine.sdk.Util import UtilsPy
from envengine.sdk.base_struct.Message import Command
from envengine.sdk.writer import WriteConfig, init_writer


def create_engine(policy_name=BLUE_POLICY_CHOICES[0], **config_overrides):
    scenario = ROOT / "scenarios/E01.json"
    profile = load_scenario(scenario).profile
    engine = Engine(profile, speed_multiplier=0)
    engine.reset()
    factory = engine.simulator_factory
    if factory.get_simulator_count() != len(profile.imagineProfile.entityList):
        raise AssertionError("真实场景的实体没有完整初始化")
    factory.get_simulators_by_type(35000)[0].set_defense_controller(
        BlueDefenseController(BlueDefenseConfig(policy_name=policy_name, seed=42, **config_overrides)),
    )
    return engine


def place_sensor_targets(engine):
    factory = engine.simulator_factory
    radars = factory.get_simulators_by_type(44000)
    units = [factory.get_simulators_by_type(entity_type)[0]
             for entity_type in (21000, 21001, 21002)]
    # 固定红方飞行状态，仅隔离原生飞行模型的起飞阶段；蓝方雷达、信息库与释放链均真实执行。
    for index, unit in enumerate(units):
        origin = radars[index].entity_ext.entity.posEcf
        unit.set_position(Vector3d(origin.x + 2000 + index * 300, origin.y, 100.0))
        unit.entity_ext.entity.velEcf = Vector3d(-100, 0, 0)
        unit.entity_ext.entity.stage = 3
        unit.update = lambda: None
    factory.start_red_missions()
    return units


def advance_until_detected(engine):
    commander = engine.simulator_factory.get_simulators_by_type(35000)[0]
    for _ in range(200):
        engine.step([])
        targets = commander.defense_controller.get_tracks(commander.current_step)
        if len(targets) >= 3 and len({track.source_id for track in targets}) >= 2:
            return
    raise AssertionError("真实视野塔没有形成多个来源的三类人族记录")


class BlueInformationNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.writer = init_writer(WriteConfig(
            enable_config=False, enable_state=False, enable_event=False, enable_ai_action=False,
            output_dir=str(ROOT / ".blue_validation/writer"), verbose=False,
        ))
        cls.engine = create_engine()
        cls.red_units = place_sensor_targets(cls.engine)
        advance_until_detected(cls.engine)
        cls.factory = cls.engine.simulator_factory
        cls.commander = cls.factory.get_simulators_by_type(35000)[0]
        cls.commander.sim_time = cls.engine.sim_time
        cls.commander.current_step = cls.engine.current_step
        cls.observation = cls.commander._build_blue_observation()
        cls.records = tuple(record for radar in cls.factory.get_simulators_by_type(44000)
                            for record in radar.entity_ext.entity.detectInfo.values())
        cls.tracks = tuple(cls.commander._detect_info_to_threat_track(record)
                           for record in cls.records)
        cls.observation = replace(cls.observation, targets=cls.tracks)

    @classmethod
    def tearDownClass(cls):
        cls.writer.close()

    def make_store(self):
        controller = BlueDefenseController(BlueDefenseConfig(seed=42))
        controller.decide(self.observation)
        return controller._information

    def install_store(self, factory):
        commander = factory.get_simulators_by_type(35000)[0]
        information = self.make_store()
        commander.defense_controller._information = information
        commander._observed_target_ids = set(information._tracks)
        return commander

    def test_empty_library_cannot_observe_existing_red_entities(self):
        self.assertEqual(BlueInformationStore(self.observation.clock).snapshot(self.observation.step).targets, ())
        self.assertGreater(len(self.factory.get_simulators_by_type(21000)), 0)

    def test_custom_controller_receives_raw_updates_before_decision_and_transport_is_drained(self):
        class RecordingDefense(BaseBlueDefense):
            name = "recording"

            def __init__(self):
                super().__init__()
                self.observations = []

            def decide(self, observation):
                self.observations.append(observation)
                return None

        engine = create_engine()
        factory = engine.simulator_factory
        commander = factory.get_simulators_by_type(35000)[0]
        controller = RecordingDefense()
        commander.set_defense_controller(controller)
        commander.sim_time = self.engine.sim_time
        commander.current_step = self.observation.step
        record = max(self.records, key=lambda item: item.step)
        expected = commander._detect_info_to_threat_track(record)
        commander.handel_detect_info({record.entity_id: record})
        self.assertEqual(factory.get_blue_track(record.entity_id, commander.current_step), expected)
        self.assertEqual(controller.observations, [])
        commander._decide_from_information()
        self.assertEqual(controller.observations[-1].targets, (expected,))
        commander._decide_from_information()
        self.assertEqual(controller.observations[-1].targets, ())
        self.assertEqual(factory.get_blue_track(record.entity_id, commander.current_step + 601), expected)

    def test_prediction_and_expiry_use_recorded_step(self):
        information = self.make_store()
        track = max(self.tracks, key=lambda item: (item.detect_step, -item.source_id))
        elapsed_steps = 5
        elapsed_seconds = self.observation.clock.steps_to_seconds(elapsed_steps)
        predicted = information.get_track(track.target_id, track.detect_step + elapsed_steps)
        self.assertEqual(predicted.detect_step, track.detect_step)
        self.assertEqual(predicted.pos.x, track.pos.x + track.vel.x * elapsed_seconds)
        self.assertIsNotNone(information.get_track(track.target_id, track.detect_step + 600))
        self.assertIsNone(information.get_track(track.target_id, track.detect_step + 601))

    def test_all_sources_merge_and_old_records_do_not_overwrite(self):
        information = BlueInformationStore(self.observation.clock)
        self.assertGreater(len({track.source_id for track in self.tracks}), 1)
        for track in sorted(self.tracks, key=lambda item: item.detect_step, reverse=True):
            information.receive_tracks([track], self.observation.step)
        current = information.snapshot(self.observation.step)
        self.assertEqual({track.target_id for track in current.targets},
                         {track.target_id for track in self.tracks})
        for track in current.targets:
            expected = max((item for item in self.tracks if item.target_id == track.target_id),
                           key=lambda item: (item.detect_step, -item.source_id))
            self.assertEqual((track.detect_step, track.source_id),
                             (expected.detect_step, expected.source_id))

    def test_radar_records_have_independent_coordinates(self):
        for record in self.records:
            entity = self.factory.get_simulator_by_id(record.entity_id).entity_ext.entity
            for record_field, entity_field in (("pos", "posEcf"), ("vel", "velEcf")):
                self.assertIsNot(getattr(record, record_field), getattr(entity, entity_field))

    def test_snapshot_is_read_only(self):
        observation = self.make_store().snapshot(self.observation.step)
        with self.assertRaises(FrozenInstanceError):
            observation.step = 0
        with self.assertRaises(FrozenInstanceError):
            observation.clock = self.observation.clock
        with self.assertRaises(TypeError):
            observation.launched_map[1] = ()
        self.assertIsInstance(observation.targets, tuple)

    def test_invalid_source_and_future_step_are_rejected(self):
        record = self.records[0]
        original_source = record.detect_from
        record.detect_from = self.red_units[0].entity_ext.entity.id
        try:
            with self.assertRaises(ValueError):
                self.commander._detect_info_to_threat_track(record)
        finally:
            record.detect_from = original_source
        with self.assertRaises(ValueError):
            self.make_store().receive_tracks(
                [replace(self.tracks[0], detect_step=self.observation.step + 1)], self.observation.step,
            )

    def test_hidden_destruction_does_not_remove_known_track(self):
        engine = create_engine()
        red_units = place_sensor_targets(engine)
        advance_until_detected(engine)
        factory = engine.simulator_factory
        commander = factory.get_simulators_by_type(35000)[0]
        commander.sim_time = engine.sim_time
        before = commander.defense_controller.get_tracks(commander.current_step)
        track = next(item for item in before if item.target_id == red_units[0].entity_ext.entity.id)
        factory._send_command_callback([Command(
            prevTriggerId=track.target_id, executorId=track.target_id,
            commandTypeId=SimmerCommandType.DAMAGE,
            commandAttributes=DamageC(DamageCmd(damagePoint=5000)).to_json(),
        )])
        factory.process_commands()
        self.assertEqual(red_units[0].entity_ext.entity.survivePoints, 0)
        self.assertEqual(before, commander.defense_controller.get_tracks(commander.current_step))

    def test_due_assignment_uses_prediction_and_expires(self):
        engine = create_engine()
        factory = engine.simulator_factory
        commander = self.install_store(factory)
        track = max(self.tracks, key=lambda item: item.detect_step)
        interceptor = factory.get_simulators_by_type(24000)[0]
        assignment = ScheduledInterceptorAssignment(
            interceptor_id=interceptor.entity_ext.entity.id, target_id=track.target_id,
            decision_step=self.observation.step, planned_launch_step=self.observation.step + 60,
            estimated_intercept_step=self.observation.step + 70,
        )
        commander.current_step = self.observation.step + 1
        commander.sim_time = self.observation.clock.time_at(commander.current_step)
        current = commander._current_assignment(assignment)
        predicted = factory.get_blue_track(track.target_id, commander.current_step)
        command = commander._assignment_to_command(current)
        target = command.commandAttributes.targetPosInfo
        for axis in ("x", "y", "z"):
            self.assertEqual(getattr(target.cPos, axis), getattr(predicted.pos, axis))
            self.assertEqual(getattr(target.cVel, axis), getattr(predicted.vel, axis))
        commander.pending_launches = {assignment.interceptor_id: assignment}
        commander.current_step = track.detect_step + 601
        commander.sim_time = self.observation.clock.time_at(commander.current_step)
        commander._execute_due_launches()
        self.assertIsNone(commander._current_assignment(assignment))
        self.assertEqual(commander.pending_launches, {})

    def test_all_ten_policies_receive_identical_information(self):
        observation = self.observation
        target_ids = {track.target_id for track in observation.targets}
        interceptor_ids = {item.interceptor_id for item in observation.interceptors if item.available}
        for name in BLUE_POLICY_CHOICES:
            with self.subTest(policy=name):
                controller = BlueDefenseController(BlueDefenseConfig(policy_name=name, seed=42))
                decision = controller.decide(observation)
                self.assertIsNotNone(decision)
                for assignment in decision.assignments:
                    self.assertIn(assignment.target_id, target_ids)
                    self.assertIn(assignment.interceptor_id, interceptor_ids)
                controller.reset()

    def test_interceptor_guidance_ignores_hidden_position_changes(self):
        engine = create_engine()
        factory = engine.simulator_factory
        self.install_store(factory)
        track = self.tracks[0]
        first, second = factory.get_simulators_by_type(24000)[:2]
        position = first.entity_ext.entity.posEcf
        second.set_position(Vector3d(position.x, position.y, position.z))
        for interceptor in (first, second):
            interceptor.sim_time = self.engine.sim_time
            interceptor.current_step = self.observation.step
            interceptor.launch(track.target_id,
                               UtilsPy.Vector3D(track.pos.x, track.pos.y, track.pos.z),
                               UtilsPy.Vector3D(track.vel.x, track.vel.y, track.vel.z))
        first.set_target_ecf()
        entity = factory.get_simulator_by_id(track.target_id).entity_ext.entity
        factory.modify_simulator_position(track.target_id, Vector3d(
            entity.posEcf.x + 1000, entity.posEcf.y + 1000, entity.posEcf.z,
        ))
        second.set_target_ecf()
        for _ in range(20):
            first.model.Update()
            second.model.Update()
        self.assert_same_native_state(first, second)
        expiry_step = max(item.detect_step for item in self.tracks) + 601
        first.current_step = second.current_step = expiry_step
        first.sim_time = second.sim_time = self.observation.clock.time_at(expiry_step)
        first.set_target_ecf()
        second.set_target_ecf()
        for _ in range(20):
            first.model.Update()
            second.model.Update()
        self.assert_same_native_state(first, second)

    def assert_same_native_state(self, first, second):
        for vector_name in ("posEcf", "velEcf"):
            left = getattr(first.model.getState(), vector_name)()
            right = getattr(second.model.getState(), vector_name)()
            for axis in ("x", "y", "z"):
                self.assertAlmostEqual(getattr(left, axis)(), getattr(right, axis)(), places=7)

    def test_b7_to_b9_release_without_new_record_and_cancel_expired_targets(self):
        for name in BLUE_POLICY_CHOICES[7:]:
            with self.subTest(policy=name):
                information = self.make_store()
                config = BlueDefenseConfig(policy_name=name, seed=42)
                policy = build_blue_policy(config)
                policy.decide(information.snapshot(self.observation.step))
                release_step = self.observation.step + math.ceil(max(
                    config.planning_delay_step, self.observation.clock.seconds_to_steps(10),
                ))
                released = policy.decide(information.snapshot(release_step))
                self.assertTrue(released)
                for assignment in released:
                    expected = information.get_track(assignment.target_id, release_step)
                    self.assertIsNotNone(expected)
                information = self.make_store()
                policy.reset()
                policy.decide(information.snapshot(self.observation.step))
                expiry = max(track.detect_step for track in self.tracks) + 601
                self.assertEqual(policy.decide(information.snapshot(expiry)), [])

    def test_engine_reset_clears_database_and_controller_plans(self):
        engine = create_engine(BLUE_POLICY_CHOICES[7])
        place_sensor_targets(engine)
        advance_until_detected(engine)
        commander = engine.simulator_factory.get_simulators_by_type(35000)[0]
        self.assertTrue(commander.defense_controller.get_tracks(commander.current_step))
        engine.reset()
        factory = engine.simulator_factory
        commander = factory.get_simulators_by_type(35000)[0]
        self.assertEqual(commander.defense_controller.get_tracks(0), ())
        self.assertEqual(commander._build_blue_observation().targets, ())
        self.assertEqual(commander.pending_launches, {})
        self.assertEqual(commander.launched_list, {})

    def test_mother_and_sentinel_evasion_use_database(self):
        engine = create_engine()
        factory = engine.simulator_factory
        track = self.make_store().snapshot(self.observation.step).targets[0]
        for entity_type in (9400, 9600):
            with self.subTest(entity_type=entity_type):
                ship = factory.get_simulators_by_type(entity_type)[0]
                ship.sim_time = self.engine.sim_time
                ship.current_step = self.observation.step
                ship.entity_ext.entity.posEcf = Vector3d(track.pos.x + 1000, track.pos.y, track.pos.z)
                controller = factory.get_simulators_by_type(35000)[0].defense_controller
                controller._information = BlueInformationStore(self.observation.clock)
                ship.detect_attacker()
                self.assertIsNone(ship._attacker)
                self.install_store(factory)
                ship.detect_attacker()
                self.assertIsNotNone(ship._attacker)
                self.assertIsNotNone(ship._perpendicular_velocity())
                ship.current_step = max(item.detect_step for item in self.tracks) + 601
                ship.sim_time = self.observation.clock.time_at(ship.current_step)
                ship.detect_attacker()
                self.assertIsNone(ship._attacker)

    def test_each_policy_launches_through_native_engine(self):
        results = []
        for name in BLUE_POLICY_CHOICES:
            with self.subTest(policy=name):
                # Re-evaluate joint policies after the first radar delivery instead of
                # waiting their configurable 100-step production cadence.
                engine = create_engine(name, joint_decision_interval_step=1)
                place_sensor_targets(engine)
                advance_until_detected(engine)
                factory = engine.simulator_factory
                for _ in range(22):
                    engine.step([])
                commander = factory.get_simulators_by_type(35000)[0]
                commander.sim_time = engine.sim_time
                observation = commander._build_blue_observation()
                known_targets = commander.defense_controller.get_tracks(commander.current_step)
                self.assertTrue(known_targets)
                self.assertFalse(hasattr(observation, "assets"))
                self.assertTrue(observation.own_entities)
                self.assertTrue({track.target_id for track in observation.targets}.issubset(
                    {track.target_id for track in known_targets},
                ))
                launched_ids = {entity_id for ids in commander.launched_list.values() for entity_id in ids}
                self.assertEqual(len(launched_ids), sum(map(len, commander.launched_list.values())))
                native_launches = sum(unit.launched == 1 for unit in factory.get_simulators_by_type(24000))
                self.assertGreater(native_launches, 0)
                self.assertEqual(native_launches, len(launched_ids))
                self.assertEqual(native_launches, commander.coordination_metrics["actual_launches"])
                results.append({"policy": name, "steps_executed": engine.current_step,
                                "simulation_time_ms": engine.sim_time, "known_targets": len(known_targets),
                                "native_launches": native_launches, "pending_launches": len(commander.pending_launches)})
        output = ROOT / ".blue_validation/native_policies.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
