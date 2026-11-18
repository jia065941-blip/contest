"""Verify the public scene and runtime coordinate boundary end to end."""

import json
import re
import unittest

from build_tools.build_layout import ROOT, collect_sources, retained_engine_file
from envengine.common.BaseStruct import Vector3d
from envengine.environment.training_env import TrainingEnv
from envengine.render import Renderer
from envengine.sdk.base_struct.Basic import DetectInfo
from scenarios import load_scenario


class CartesianPublicContractTests(unittest.TestCase):
    def test_public_sources_and_docs_use_only_the_cartesian_contract(self):
        obsolete = re.compile(
            r"\b(?:lla|lon|lat|longitude|latitude|ecef|wgs84|pos_ecf|vel_ecf|"
            r"posEcf|velEcf|velNue|lonMin|lonMax|latMin|latMax)\b|经纬|经度|纬度",
            re.IGNORECASE,
        )
        paths = [ROOT / "README.md"]
        for package in ("user_agents", "blue_defense", "scenarios"):
            paths.extend(path for path in (ROOT / package).rglob("*")
                         if path.suffix in {".py", ".md", ".txt", ".json"})
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertIsNone(obsolete.search(path.read_text(encoding="utf-8")))

    def test_every_bundled_scene_has_one_coordinate_source(self):
        paths = [path for path in (ROOT / "scenarios").glob("*.json") if path.name != "manifest.json"]
        self.assertEqual(len(paths), 22)
        for path in paths:
            with self.subTest(scene=path.stem):
                data = json.loads(path.read_text(encoding="utf-8"))
                profile = data["imagineProfile"]
                self.assertEqual(set(profile["mapArea"]), {"xMin", "xMax", "yMin", "yMax"})
                for row in profile["entityList"]:
                    entity = row["entity"]
                    self.assertEqual(set(entity["pos"]), {"x", "y", "z"})
                    self.assertEqual(set(entity["vel"]), {"x", "y", "z"})
                    self.assertTrue({"lla", "posEcf", "velEcf", "velNue"}.isdisjoint(entity))

    def test_real_runtime_exports_nested_tracks_without_private_fields(self):
        scene = load_scenario("Rtest")
        env = TrainingEnv(scene, render_mode="none", record_logs=False)
        self.addCleanup(env.close)
        env.reset()
        units = env.engine.simulator_factory.get_simulators_by_type(21000)
        unit = units[0].entity_ext.entity
        unit.detectInfo[999] = DetectInfo(entity_id=999, entity_type=9500,
            pos=Vector3d(1000, 2000, 30), vel=Vector3d(3, 4, 0), step=0)
        snapshot = env.get_observation()
        record = snapshot["entities"][unit.id]["detectInfo"][999]
        self.assertEqual(record["pos"], {"x": 1000, "y": 2000, "z": 30})
        self.assertEqual(record["vel"], {"x": 3, "y": 4, "z": 0})
        self.assertEqual(record["detected_step"], 0)
        self.assertTrue({"time", "step", "lla", "pos_ecf", "vel_ecf"}.isdisjoint(record))
        record["pos"]["x"] = -1
        self.assertEqual(unit.detectInfo[999].pos.x, 1000)

    def test_planar_view_round_trips_away_from_zero(self):
        renderer = Renderer(10000, 110000, -50000, 30000)
        for point in ((10000, -50000), (60000, -10000), (110000, 30000)):
            pixel = renderer._pos_to_screen(*point)
            restored = renderer._screen_to_pos(*pixel)
            self.assertAlmostEqual(restored[0], point[0], delta=renderer.meter_per_pixel)
            self.assertAlmostEqual(restored[1], point[1], delta=renderer.meter_per_pixel)
        self.assertEqual(renderer.km_to_pixel_radius(10000, 0, 20),
                         renderer.km_to_pixel_radius(110000, 30000, 20))

    def test_private_wrappers_are_compiled_and_sources_are_not_distributed(self):
        sources = dict(collect_sources())
        wrapper = "envengine/sdk/Util/UtilsPy.py"
        self.assertEqual(sources["envengine.sdk.Util.UtilsPy"], wrapper)
        for relative in (wrapper, "envengine/sdk/Util/UtilsPy.pyi",
                         "envengine/simulator/models/ACMMissileModel/ACMMissilePy.py"):
            self.assertFalse(retained_engine_file(ROOT / relative, relative))
        dependency = "envengine/sdk/Util/_UtilsPy.pyd"
        self.assertTrue(retained_engine_file(ROOT / dependency, dependency))


if __name__ == "__main__":
    unittest.main()
