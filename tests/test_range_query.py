import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "core"))

from envengine.sdk.base_struct.Basic import Vector3d  # noqa: E402
from envengine.simulator.interfaces import ISimulator  # noqa: E402
from envengine.simulator.simlulator_impl.RadarModelSimulator import (  # noqa: E402
    RadarModelSimulator,
)


def make_simulator(entity_id: int, stage: int, position: Vector3d, side_id: int = 0):
    entity = SimpleNamespace(
        id=entity_id,
        sideId=side_id,
        isVisible=True,
        survivePoints=1.0,
        stage=stage,
        posEcf=position,
        velEcf=Vector3d(0.0, 0.0, 0.0),
        lla=Vector3d(0.0, 0.0, 0.0),
        entityType=21000,
        nameChn=f"entity-{entity_id}",
    )
    return SimpleNamespace(entity_ext=SimpleNamespace(entity=entity))


class RangeQueryHarness:
    _get_targets_within_self_range = ISimulator._get_targets_within_self_range
    _is_geometrically_visible = staticmethod(ISimulator._is_geometrically_visible)

    def __init__(self):
        source = make_simulator(1, stage=3, position=Vector3d(0.0, 0.0, 0.0))
        self._entity_ext = source.entity_ext


class RadarFactory:
    def __init__(self, candidates):
        self.candidates = candidates

    def get_simulators_by_side(self, _side_id):
        return self.candidates


class RadarDetectionHarness(RangeQueryHarness):
    execute_detection = RadarModelSimulator.execute_detection

    def __init__(self, candidates):
        source = make_simulator(1, stage=1, position=Vector3d(0.0, 0.0, 0.0), side_id=1)
        self._entity_ext = source.entity_ext
        self._simulator_factory = RadarFactory(candidates)
        self._sim_time = 0
        self.detected = {}

    @property
    def entity_ext(self):
        return self._entity_ext

    @property
    def sim_time(self):
        return self._sim_time

    def handel_detect_info(self, detect_info):
        self.detected = detect_info


class TargetRangeQueryTest(unittest.TestCase):
    def test_static_target_is_included_for_damage_queries(self):
        query = RangeQueryHarness()
        static_target = make_simulator(2, stage=0, position=Vector3d(100.0, 0.0, 0.0))

        actual = query._get_targets_within_self_range([static_target], max_range=1000.0)

        self.assertEqual([static_target], actual)

    def test_radar_can_request_flight_stage_filter(self):
        boost_target = make_simulator(2, stage=2, position=Vector3d(100.0, 0.0, 0.0))
        glide_target = make_simulator(3, stage=3, position=Vector3d(200.0, 0.0, 0.0))
        radar = RadarDetectionHarness([boost_target, glide_target])

        radar.execute_detection()

        self.assertEqual([3], list(radar.detected))


if __name__ == "__main__":
    unittest.main()
