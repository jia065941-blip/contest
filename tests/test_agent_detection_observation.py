"""Detection observations preserve acquisition steps and Cartesian coordinates."""

from copy import deepcopy
from dataclasses import asdict, fields
from types import SimpleNamespace
import unittest

from envengine.agent_manager import AgentManager
from envengine.sdk.base_struct.Basic import DetectInfo
from envengine.simulator.interfaces import ISimulator
from user_agents.base import BaseAgent
from envengine.common.BaseStruct import Vector3d


class MinimalAgent(BaseAgent):
    def get_action(self):
        return []


class AgentDetectionObservationTests(unittest.TestCase):
    def manager(self, *, register=True, tracks=None):
        entity = SimpleNamespace(
            id=11, entityType=21000, sideId=0, nameChn="unit", survivePoints=100,
            isVisible=True, posEcf=Vector3d(1.0, 1.0, 3000.0), velEcf=Vector3d(),
            detectInfo=tracks or {},
        )
        simulator = SimpleNamespace(entity_ext=SimpleNamespace(entity=entity))
        profile = SimpleNamespace(imagineProfile=SimpleNamespace(
            redArea=SimpleNamespace(coordinates=[[[0, 0], [3, 0], [3, 3], [0, 3]]]),
            mapArea=SimpleNamespace(xMin=0, xMax=3, yMin=0, yMax=3),
        ))
        engine = SimpleNamespace(
            profile=profile, current_step=0, total_steps=100,
            simulator_factory=SimpleNamespace(get_all_simulators=lambda: [simulator]),
        )
        manager = AgentManager(engine)
        if register:
            manager.register_agent(MinimalAgent(11, {}))
        return manager

    @staticmethod
    def full_observation(tracks, step=0):
        return {"step": step, "entities": {11: {
            "health": 100, "isVisible": True,
            "nameChn": "unit", "type": 21000, "side": 0, "stage": 3,
            "pos": {"x": 1.0, "y": 1.0, "z": 3000.0},
            "vel": {"x": 0.0, "y": 0.0, "z": 0.0}, "commRangeInfo": {},
            "detectInfo": tracks,
        }}}

    def test_sdk_records_have_step_and_no_time_field(self):
        track = DetectInfo(step=2)
        self.assertNotIn("time", {item.name for item in fields(DetectInfo)})
        self.assertFalse(hasattr(track, "time"))
        self.assertNotIn("time", asdict(track))
        self.assertNotIn("time", track.to_dict())
        with self.assertRaises(TypeError):
            DetectInfo(time=123)

    def test_raw_records_expose_their_acquisition_step_without_a_clock(self):
        for detected_step, current_step in ((2, 20), (7, 120)):
            with self.subTest(detected_step=detected_step):
                manager = self.manager()
                track = DetectInfo(step=detected_step, entity_id=99,
                                   pos=Vector3d(2.0, 1.0, 0.0))
                observation = manager.extract_observation_for_agent(
                    self.full_observation({99: track}, step=current_step), 11,
                )
                public = observation["self"]["detectInfo"][99]
                self.assertEqual(public["detected_step"], detected_step)
                self.assertIs(type(public["detected_step"]), int)
                self.assertEqual(observation["step"] - public["detected_step"],
                                 current_step - detected_step)
                self.assertEqual(set(public), {
                    "detect_from", "entity_id", "entity_type", "nameChn",
                    "pos", "vel", "detected_step",
                })
                self.assertEqual(public["pos"], {"x": 2.0, "y": 1.0, "z": 0.0})
                public["pos"]["x"] = 999
                self.assertEqual(track.pos.x, 2.0)
                self.assertEqual(track.step, detected_step)

    def test_cached_record_keeps_its_acquisition_step_when_forwarded_later(self):
        manager = self.manager()
        track = {"entity_id": 99, "step": 2, "pos": Vector3d(2.0, 1.0, 0.0)}
        full = self.full_observation({99: track}, step=3)
        first = manager.extract_observation_for_agent(full, 11)
        full["step"] = 120
        forwarded = manager.extract_observation_for_agent(full, 11)
        self.assertEqual(first["self"]["detectInfo"], forwarded["self"]["detectInfo"])
        public = forwarded["self"]["detectInfo"][99]
        self.assertEqual(public["detected_step"], 2)
        self.assertEqual(forwarded["step"] - public["detected_step"], 118)
        self.assertEqual(track["step"], 2)
        public["pos"]["x"] = 999
        self.assertEqual(track["pos"].x, 2.0)

    def test_initialization_and_regular_observations_have_the_same_public_shape(self):
        track = DetectInfo(step=0, entity_id=99, pos=Vector3d(2.0, 1.0, 0.0))
        manager = self.manager(register=False, tracks={99: track})
        manager._episode_started = True
        manager._deployment_applied = True
        manager._run_id = "test-run"
        group = manager.register_user_agent(MinimalAgent)
        self.addCleanup(group.close)
        initial = group.agents[0].init_observation["self"]["detectInfo"]
        full = self.full_observation({99: track})
        regular = manager.extract_observation_for_agent(full, 11)["self"]["detectInfo"]
        self.assertEqual(initial, regular)
        self.assertEqual(initial[99]["detected_step"], 0)
        self.assertNotIn("time", initial[99])
        initial[99]["pos"]["x"] = 999
        self.assertEqual(track.pos.x, 2.0)
        self.assertEqual(full["entities"][11]["detectInfo"][99].step, 0)

    def test_public_records_need_no_engine_and_missing_step_is_not_invented(self):
        manager = AgentManager()
        manager.register_agent(MinimalAgent(11, {}))
        tracks = {"99": {"entity_id": 99, "detected_step": 4},
                  100: {"entity_id": 100},
                  101: {"entity_id": 101, "step": 7},
                  102: {"entity_id": 102, "time": 123}}
        original = deepcopy(tracks)
        public = manager.extract_observation_for_agent(
            self.full_observation(tracks, step=120), 11,
        )["self"]["detectInfo"]
        self.assertEqual(public[99]["detected_step"], 4)
        self.assertEqual(public[101]["detected_step"], 7)
        self.assertNotIn("detected_step", public[100])
        self.assertNotIn("detected_step", public[102])
        self.assertNotIn("time", public[102])
        self.assertEqual(tracks, original)

    def test_initial_records_start_at_step_zero(self):
        manager = self.manager()
        for track in (DetectInfo(), DetectInfo(step=0)):
            public = manager.extract_observation_for_agent(
                self.full_observation({99: track}), 11,
            )["self"]["detectInfo"][99]
            self.assertEqual(public["detected_step"], 0)

    def test_public_detection_uses_a_field_allowlist(self):
        manager = self.manager()
        source = {
            "entity_id": 99, "step": 4,
            "pos": {"x": 2.0, "y": 1.0, "z": 0.0},
            "vel": {"x": 3.0, "y": 0.0, "z": 0.0},
            "time": 123, "private_state": {"secret": True},
            "model_handle": "private",
        }
        public = manager.extract_observation_for_agent(
            self.full_observation({99: source}, step=20), 11,
        )["self"]["detectInfo"][99]
        self.assertEqual(set(public), {"entity_id", "detected_step", "pos", "vel"})
        self.assertEqual(public["detected_step"], 4)
        self.assertEqual(public["vel"], {"x": 3.0, "y": 0.0, "z": 0.0})
        public["pos"]["x"] = 999
        self.assertEqual(source["pos"]["x"], 2.0)

    def test_fusion_ignores_the_same_step_and_accepts_a_higher_step(self):
        older = DetectInfo(step=2, entity_id=99, pos=Vector3d(1.0, 1.0, 0.0))
        same_step = DetectInfo(step=2, entity_id=99, pos=Vector3d(2.0, 1.0, 0.0))
        newer = DetectInfo(step=3, entity_id=99, pos=Vector3d(3.0, 1.0, 0.0))
        stored = {99: deepcopy(older)}
        simulator = SimpleNamespace(
            _entity_ext=SimpleNamespace(entity=SimpleNamespace(detectInfo=stored)),
            _delivered_to={11},
        )
        ISimulator.handel_detect_info(simulator, {99: same_step})
        self.assertEqual(stored[99].step, 2)
        self.assertEqual(stored[99].pos.x, 1.0)
        self.assertEqual(simulator._delivered_to, {11})
        ISimulator.handel_detect_info(simulator, {99: newer})
        self.assertEqual(stored[99].step, 3)
        self.assertEqual(stored[99].pos.x, 3.0)
        self.assertEqual(simulator._delivered_to, set())
        simulator._delivered_to.add(11)
        ISimulator.handel_detect_info(simulator, {99: older})
        self.assertEqual(stored[99].step, 3)
        self.assertEqual(stored[99].pos.x, 3.0)
        self.assertEqual(simulator._delivered_to, {11})
        public = self.manager().extract_observation_for_agent(
            self.full_observation(stored, step=10), 11,
        )["self"]["detectInfo"][99]
        self.assertEqual(public["detected_step"], 3)
        self.assertEqual(public["pos"], {"x": 3.0, "y": 1.0, "z": 0.0})


if __name__ == "__main__":
    unittest.main()
