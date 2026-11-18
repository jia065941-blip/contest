"""Environment-owned communication snapshots and red command legality."""

import math
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

from envengine.action_validation import IllegalActionError
from envengine.common import EntityInfo, SimmerCommandType
from envengine.engine.engine import Engine
from envengine.common.BaseStruct import Vector3d
from envengine.sdk.base_struct.Basic import DetectInfo
from envengine.sdk.base_struct.Message import Command
from envengine.simulator.interfaces import ISimulator
from envengine.simulator.simulator_factory import SimulatorFactory
from envengine.simulator.simlulator_impl.CompCruiseMissileHSimulator import CompCruiseMissileHSimulator
from envengine.simulator.simlulator_impl.CompCruiseMissileMSimulator import CompCruiseMissileMSimulator
from envengine.simulator.simlulator_impl.CompCruiseMissileLSimulator import CompCruiseMissileLSimulator


class FakeSimulator:
    handel_detect_info = ISimulator.handel_detect_info
    _consume_event = ISimulator._consume_event
    _entity_info = ISimulator._entity_info

    def __init__(self, entity_id, entity_type=21002, x=0, *, side=0, visible=True, health=1):
        self._entity_ext = SimpleNamespace(entity=SimpleNamespace(
            id=entity_id, entityType=entity_type, typeId=0, sideId=side,
            parentId=-1, childrenId=[],
            isVisible=visible, survivePoints=health, nameChn=str(entity_id),
            posEcf=Vector3d(x, 0, 0), velEcf=Vector3d(0, 0, 0),
            detectInfo={}, commRangeInfo={},
        ))
        self._delivered_to = set()
        self.received = []
        self.events = []
        self._send_events = self.events.append
        self._sim_time = 0
        self.current_step = 0

    @property
    def entity_ext(self):
        return self._entity_ext

    def command_received(self, command):
        self.received.append(command)

    def _target_entity_info_or_self(self, identifier=None):
        if identifier is None or identifier == self.entity_ext.entity.id:
            return self._entity_info(self.entity_ext.entity)
        # Communication fixtures have observations without live target models.
        observed = self.entity_ext.entity.detectInfo[identifier]
        return EntityInfo(id=observed.entity_id, type=observed.entity_type,
                          side=1-self.entity_ext.entity.sideId)


def track(entity_id=900, entity_type=9500, x=3, step=10):
    return DetectInfo(
        detect_from=1, step=step, entity_id=entity_id,
        entity_type=entity_type, pos=Vector3d(x, 4, 0), vel=Vector3d(1, 2, 3),
    )


def factory_with(*simulators):
    factory = object.__new__(SimulatorFactory)
    factory._simulators = {sim.entity_ext.entity.id: sim for sim in simulators}
    factory._simulators_by_type = {}
    factory._simulators_by_side = {}
    for simulator in simulators:
        entity = simulator.entity_ext.entity
        factory._simulators_by_type.setdefault(entity.entityType, []).append(simulator)
        factory._simulators_by_side.setdefault(entity.sideId, []).append(simulator)
    factory._command_queue = []
    factory._profile = SimpleNamespace(imagineProfile=SimpleNamespace(
        mapArea=SimpleNamespace(xMin=0, xMax=10, yMin=0, yMax=10),
    ))
    factory._capture_public_red_targets()
    return factory


def engine_with(*simulators):
    engine = object.__new__(Engine)
    engine.simulator_factory = factory_with(*simulators)
    return engine


def target_command(executor_id, **attributes):
    return Command(
        executorId=executor_id,
        commandTypeId=SimmerCommandType.CHANGE_MISSILE_TARGET,
        commandAttributes=attributes,
    )


class RedCommunicationSnapshotsTest(unittest.TestCase):
    def test_deployment_updates_communication_position_before_first_motion(self):
        for simulator_class in (CompCruiseMissileHSimulator, CompCruiseMissileMSimulator, CompCruiseMissileLSimulator):
            with self.subTest(simulator=simulator_class.__name__):
                simulator = object.__new__(simulator_class)
                simulator._entity_ext = FakeSimulator(1).entity_ext
                simulator.model = Mock()
                destination = Vector3d(5, 6, 300)

                simulator.set_position(destination)

                self.assertEqual(destination, simulator.entity_ext.entity.posEcf)
                simulator.model.Init.assert_called_once()
                simulator.model.Update.assert_not_called()

    def test_domains_use_bidirectional_links_with_multihop_relays(self):
        high = FakeSimulator(1, 21000, 0)
        medium = FakeSimulator(2, 21001, 18000)
        relay = FakeSimulator(3, 21002, 27000)
        distant = FakeSimulator(4, 21002, 40000)
        engine = engine_with(high, medium, relay, distant)
        high.handel_detect_info({900: track()})

        # H-M can exchange within 20 km, M-L within 10 km. H-L at 27 km
        # cannot communicate directly, so L receives H's track via M.
        self.assertEqual(((1,), (3,)), engine_with(high, relay).refresh_red_communications())

        domains = engine.refresh_red_communications()

        self.assertEqual(((1, 2, 3), (4,)), domains)
        peers = medium.entity_ext.entity.commRangeInfo
        self.assertEqual({1, 2, 3}, set(peers))
        self.assertEqual({"x": 27000, "y": 0, "z": 0}, peers[3]["pos"])
        self.assertEqual({"x": 0, "y": 0, "z": 0}, peers[3]["vel"])
        self.assertIn(900, relay.entity_ext.entity.detectInfo)
        self.assertNotIn(900, distant.entity_ext.entity.detectInfo)
        self.assertEqual([], engine.simulator_factory.command_queue)

    def test_bidirectional_link_accepts_the_shorter_radius_boundary(self):
        for first_type, second_type, limit in (
            (21000, 21000, 50000),
            (21001, 21001, 20000),
            (21002, 21002, 10000),
            (21000, 21001, 20000),
            (21000, 21002, 10000),
            (21001, 21002, 10000),
        ):
            with self.subTest(first=first_type, second=second_type):
                first = FakeSimulator(1, first_type, 0)
                second = FakeSimulator(2, second_type, limit)
                first.handel_detect_info({900: track()})
                self.assertEqual(((1, 2),), engine_with(first, second).refresh_red_communications())
                self.assertIn(900, second.entity_ext.entity.detectInfo)

    def test_one_way_reach_cannot_share_tracks_or_form_a_domain(self):
        for first_type, second_type, separation in (
            (21000, 21001, 30000), (21001, 21000, 30000),
            (21000, 21002, 20000), (21002, 21000, 20000),
            (21001, 21002, 15000), (21002, 21001, 15000),
        ):
            with self.subTest(first=first_type, second=second_type):
                first = FakeSimulator(1, first_type, 0)
                second = FakeSimulator(2, second_type, separation)
                first.handel_detect_info({900: track()})
                self.assertEqual(((1,), (2,)), engine_with(first, second).refresh_red_communications())
                self.assertEqual({1}, set(first.entity_ext.entity.commRangeInfo))
                self.assertEqual({2}, set(second.entity_ext.entity.commRangeInfo))
                self.assertNotIn(900, second.entity_ext.entity.detectInfo)

    def test_only_active_red_platforms_can_relay(self):
        first = FakeSimulator(1, x=0)
        last = FakeSimulator(2, x=18000)
        dead = FakeSimulator(3, 21000, 9000, health=0)
        hidden = FakeSimulator(4, 21000, 9000, visible=False)
        blue = FakeSimulator(5, 21000, 9000, side=1)
        self.assertEqual(((1,), (2,)), engine_with(first, last, dead, hidden, blue).refresh_red_communications())

    def test_detection_records_are_immutable_snapshots_of_live_vectors(self):
        red = FakeSimulator(1)
        detection = track()
        red.handel_detect_info({900: detection})
        detection.pos.x = 999
        detection.vel.y = 999
        stored = red.entity_ext.entity.detectInfo[900]
        self.assertEqual(3, stored.pos.x)
        self.assertEqual(10, stored.step)
        self.assertEqual(2, stored.vel.y)

    def test_disconnected_unit_keeps_old_track_without_new_positions(self):
        first = FakeSimulator(1)
        second = FakeSimulator(2, x=1000)
        engine = engine_with(first, second)
        first.handel_detect_info({900: track(x=3)})
        engine.refresh_red_communications()
        second.entity_ext.entity.posEcf.x = 30000
        first.handel_detect_info({900: track(x=8, step=20)})
        engine.refresh_red_communications()
        self.assertEqual(3, second.entity_ext.entity.detectInfo[900].pos.x)
        self.assertEqual(10, second.entity_ext.entity.detectInfo[900].step)
        self.assertEqual(8, first.entity_ext.entity.detectInfo[900].pos.x)
        self.assertEqual(20, first.entity_ext.entity.detectInfo[900].step)
        second.entity_ext.entity.posEcf.x = 1000
        engine.refresh_red_communications()
        self.assertEqual(8, second.entity_ext.entity.detectInfo[900].pos.x)
        self.assertEqual(20, second.entity_ext.entity.detectInfo[900].step)
        first.entity_ext.entity.detectInfo[900].pos.x = 7
        self.assertEqual(8, second.entity_ext.entity.detectInfo[900].pos.x)

    def test_satellite_is_global_but_cannot_overwrite_newer_local_track(self):
        first = FakeSimulator(1)
        remote = FakeSimulator(2, x=100000)
        satellite = FakeSimulator(3, 9202)
        first.handel_detect_info({900: track(x=6, step=30)})
        satellite.handel_detect_info({900: track(x=2, step=10)})
        engine = engine_with(first, remote, satellite)
        original = deepcopy(first.entity_ext.entity.detectInfo)

        fused = engine.fuse_cluster_detection(((1,), (2,)))

        self.assertEqual(original, first.entity_ext.entity.detectInfo)
        self.assertEqual(6, fused[0][900].pos.x)
        self.assertEqual(30, fused[0][900].step)
        self.assertEqual(2, fused[1][900].pos.x)
        self.assertEqual(10, fused[1][900].step)
        engine.refresh_red_communications()
        self.assertEqual(2, remote.entity_ext.entity.detectInfo[900].pos.x)


class RedActionLegalityTest(unittest.TestCase):
    def setUp(self):
        self.high = FakeSimulator(1, 21000)
        self.low = FakeSimulator(2, 21002)
        self.public = FakeSimulator(800, 9400, side=1)
        self.hidden = FakeSimulator(900, 9500, side=1)
        self.factory = factory_with(self.high, self.low, self.public, self.hidden)

    def test_public_initial_position_does_not_follow_live_entity(self):
        self.public.entity_ext.entity.posEcf.x = 8
        command = target_command(1, intent="track", target_id=800)
        resolved = self.factory.validate_red_command(command)
        self.assertEqual(0, resolved.commandAttributes["target"]["x"])
        self.assertNotIn("target", command.commandAttributes)

    def test_track_resolves_to_last_legally_observed_position(self):
        self.low.handel_detect_info({900: track(x=3)})
        self.hidden.entity_ext.entity.posEcf.x = 9
        command = target_command(2, intent="track", target_id=900)
        resolved = self.factory.validate_red_command(command)
        self.assertEqual({"x": 3, "y": 4, "z": 0}, resolved.commandAttributes["target"])

    def test_reject_unknown_targets_even_if_present_in_another_unit_tracks(self):
        self.high.handel_detect_info({900: track()})
        with self.assertRaises(IllegalActionError):
            self.factory.validate_red_command(target_command(2, intent="track", target_id=900))

    def test_reject_mismatched_attack_roles_and_double_coordinate_source(self):
        self.high.handel_detect_info({900: track()})
        invalid = (
            target_command(1, intent="track", target_id=900),
            target_command(2, intent="track", target_id=800),
            target_command(1, intent="track", target_id=800, target={"x": 1, "y": 2, "z": 0}),
            target_command(1, target={"x": 1, "y": 2, "z": 0}),
            target_command(1, intent="track", target_id=True),
        )
        for command in invalid:
            with self.subTest(attributes=command.commandAttributes), self.assertRaises(IllegalActionError):
                self.factory.validate_red_command(command)

    def test_waypoint_is_bounded_and_does_not_query_hidden_entities(self):
        # Searching the same coordinate as an undiscovered nest is legal. Its
        # acceptance is independent of the presence or absence of that nest.
        command = target_command(2, intent="waypoint", target={"x": 1, "y": 2, "z": 0})
        first = self.factory.validate_red_command(command)
        self.factory._simulators.pop(900)
        second = self.factory.validate_red_command(command)
        self.assertEqual(first, second)
        for value in (-1, 11, math.inf, math.nan, True):
            with self.subTest(value=value), self.assertRaises(IllegalActionError):
                self.factory.validate_red_command(target_command(2, intent="waypoint", target={"x": value, "y": 2, "z": 0}))

    def test_reject_blue_dead_unknown_executors_and_old_launch_command(self):
        self.low.entity_ext.entity.survivePoints = 0
        for executor in (2, 800, 999):
            with self.subTest(executor=executor), self.assertRaises(IllegalActionError):
                self.factory.validate_red_command(target_command(executor, intent="track", target_id=800))
        for command_type in (SimmerCommandType.MISSILE_LAUNCH, SimmerCommandType.DETECT_STATUS_UPDATE):
            with self.subTest(command_type=command_type), self.assertRaises(IllegalActionError):
                self.factory.validate_red_command(Command(executorId=1, commandTypeId=command_type, commandAttributes={}))

    def test_invalid_batch_does_not_partially_execute(self):
        valid = target_command(1, intent="track", target_id=800)
        invalid = target_command(2, intent="track", target_id=900)
        with self.assertRaises(IllegalActionError):
            self.factory.process_ai_commands([valid, invalid])
        self.assertEqual([], self.high.received)
        self.factory.process_ai_commands([valid])
        self.assertEqual(1, len(self.high.received))


if __name__ == "__main__":
    unittest.main()
