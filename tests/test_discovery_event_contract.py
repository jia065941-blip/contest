"""First-discovery events preserve the public position and acquisition step."""

from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

from envengine.common import LifePointChangeT, LifePointChangeTrigger, SimmerTriggerType
from envengine.common.BaseStruct import Vector3d
from envengine.sdk.base_struct.Basic import DetectInfo
from envengine.sdk.base_struct.Entity import Entity, EntityExt
from envengine.simulator.interfaces.isimulator import ISimulator


class Observer(ISimulator):
    simulator_sim_step = 0

    def update(self):
        pass

    def set_position(self, pos):
        pass

    def set_speed(self, speed):
        pass

    def init_model(self):
        pass


class DiscoveryEventContractTests(unittest.TestCase):
    def observer(self, observer_type=21000, side=0, target_type=9500):
        target = EntityExt(Entity(id=51, entityType=target_type, typeId=7, sideId=1-side))
        factory = SimpleNamespace(
            profile=SimpleNamespace(imagineProfile=SimpleNamespace(simTime=1783391450000)),
            get_simulator_by_id=lambda identifier: SimpleNamespace(entity_ext=target),
        )
        sent = []
        observer = Observer(EntityExt(Entity(id=10, entityType=observer_type, sideId=side,
                                             nameChn="observer", parentId=-1)),
                            lambda commands: None, sent.append, factory)
        observer.current_step = 27
        observer.sim_time = 1783391477000
        track = DetectInfo(detect_from=10, step=26, entity_id=51,
                           entity_type=target_type, nameChn="target",
                           pos=Vector3d(80000, -75000, 1500), vel=Vector3d(120, -20, 3))
        return observer, sent, track

    def test_red_and_radar_first_discoveries_use_public_vectors_and_steps(self):
        for observer_type, side, target_type in ((21000, 0, 9500), (44000, 1, 21000)):
            with self.subTest(observer_type=observer_type):
                observer, sent, track = self.observer(observer_type, side, target_type)
                observer.handel_detect_info({51: track})
                self.assertEqual(len(sent), 1)
                event = sent[0]
                self.assertEqual(event["type"], SimmerTriggerType.SENSOR_DETECT_INCREASE)
                self.assertEqual(event["logicTime"], observer.sim_time)
                payload = event["content"]["event"]
                self.assertEqual(payload["sensor"]["id"], 10)
                self.assertEqual(payload["step"], 27)
                target = payload["detectTarget"][0]
                self.assertEqual(target["entityInfo"], {"id":51, "type":target_type, "typeId":7, "side":1-side})
                self.assertEqual(target["pos"], {"x":80000, "y":-75000, "z":1500})
                self.assertEqual(target["vel"], {"x":120, "y":-20, "z":3})
                self.assertEqual(target["step"], 26)
                self.assertEqual(target["radarId"], 10)
                self.assertTrue(target["visible"])
                self.assertEqual(set(target), {"entityInfo", "pos", "vel", "step", "visible", "radarId"})
                self.assertIsInstance(json.loads(json.dumps(event)), dict)

    def test_event_and_fused_memory_are_isolated_from_mutable_source(self):
        observer, sent, track = self.observer()
        observer.handel_detect_info({51: track})
        track.pos.x = -999
        track.vel.y = 999
        self.assertEqual(sent[0]["content"]["event"]["detectTarget"][0]["pos"]["x"], 80000)
        self.assertEqual(sent[0]["content"]["event"]["detectTarget"][0]["vel"]["y"], -20)
        self.assertEqual(observer.entity_ext.entity.detectInfo[51].pos.x, 80000)

    def test_later_and_repeated_tracks_do_not_emit_another_first_discovery(self):
        observer, sent, track = self.observer()
        observer.handel_detect_info({51: track})
        later = replace(track, step=27, pos=Vector3d(81000, -75000, 1500))
        observer.handel_detect_info({51: later})
        observer.handel_detect_info({51: track})
        self.assertEqual(len(sent), 1)
        self.assertEqual(observer.entity_ext.entity.detectInfo[51].step, 27)
        self.assertEqual(observer.entity_ext.entity.detectInfo[51].pos.x, 81000)

    def test_other_observer_target_combinations_keep_existing_event_filter(self):
        for observer_type, side, target_type in ((21000, 0, 9400), (44000, 1, 21001), (9500, 1, 21000)):
            with self.subTest(observer_type=observer_type, target_type=target_type):
                observer, sent, track = self.observer(observer_type, side, target_type)
                observer.handel_detect_info({51: track})
                self.assertEqual(sent, [])
                self.assertIn(51, observer.entity_ext.entity.detectInfo)

    def test_existing_sdk_event_serialization_stays_compatible(self):
        observer, sent, _ = self.observer()
        payload = LifePointChangeT(LifePointChangeTrigger(time=123, destroyPoint=2.5))
        observer._consume_event(SimmerTriggerType.LIFE_POINT_CHANGE, payload)
        self.assertEqual(sent[0]["content"], payload.to_dict())

    def test_dictionary_event_payload_is_detached_before_delivery(self):
        observer, sent, _ = self.observer()
        payload = {"event":{"step":1, "pos":{"x":1, "y":2, "z":3}}}
        observer._consume_event(SimmerTriggerType.SENSOR_DETECT_INCREASE, payload)
        payload["event"]["pos"]["x"] = 999
        self.assertEqual(sent[0]["content"]["event"]["pos"]["x"], 1)


if __name__ == "__main__":
    unittest.main()
