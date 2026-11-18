"""Scene validation shared by user files, official maps, and environment loading."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scenarios.scenario_checker import SceneValidationError, check_scenario, validate_scene


ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "scenarios"


def scene():
    coordinates = [[[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]]
    return {
        "environmentProfile": {
            "dynamicLibraryConfigs": [{"entityType": 9400, "dynamicLibraryPath": "libExample.so"}],
        },
        "imagineProfile": {
            "simStep": 1000, "simTime": 1783391450000, "simEndLogicTime": 3000000,
            "mapArea": {"xMin": -1, "xMax": 1, "yMin": -1, "yMax": 1},
            "redArea": {"coordinates": coordinates},
            "entityList": [{"entity": {
                "id": 1, "entityType": 9400, "sideId": 1, "survivePoints": 100, "parentId": -1,
                "pos": {"x": 0, "y": 0, "z": 0},
            }, "external": ""}],
        },
    }


def entity(data, index=0):
    return data["imagineProfile"]["entityList"][index]["entity"]


def with_child():
    data = scene()
    child = deepcopy(data["imagineProfile"]["entityList"][0])
    child["entity"].update(id=2, parentId=1)
    entity(data)["childrenId"] = [2]
    data["imagineProfile"]["entityList"].append(child)
    return data


class SceneValidationTests(unittest.TestCase):
    def assert_invalid(self, data, path):
        with self.assertRaises(SceneValidationError) as failure:
            validate_scene(data)
        self.assertIn(path, failure.exception.path)
        self.assertIn(path, str(failure.exception))

    def test_official_manifest_twenty_files_satisfy_the_common_format(self):
        manifest = json.loads((CATALOGUE / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["cases"]), 20)
        for entry in manifest["cases"]:
            with self.subTest(index=entry["id"]):
                data = check_scenario(CATALOGUE / entry["scenario"])
                self.assertGreater(len(data["imagineProfile"]["entityList"]), 0)

    def test_user_scene_needs_no_official_metadata_and_validation_does_not_mutate(self):
        data = scene()
        original = deepcopy(data)
        self.assertIsNone(validate_scene(data))
        self.assertEqual(data, original)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "custom.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(check_scenario(path), data)

    def test_checker_reads_only_the_selected_file_and_ignores_obsolete_sidecars(self):
        data = scene()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scene.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            path.with_name("case_info.json").write_text('{"scenario_sha256": "obsolete"}', encoding="utf-8")
            path.with_name("deployment_area.json").write_text("not json", encoding="utf-8")
            reads = []
            read_bytes, read_text = Path.read_bytes, Path.read_text

            def track(method):
                def read(source, *args, **kwargs):
                    self.assertEqual(source, path)
                    reads.append(source)
                    return method(source, *args, **kwargs)
                return read

            with patch.object(Path, "read_bytes", track(read_bytes)), \
                 patch.object(Path, "read_text", track(read_text)):
                self.assertEqual(check_scenario(path), data)
            self.assertEqual(reads, [path])
            entity(data)["survivePoints"] = 99
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(check_scenario(path), data)

    def test_loading_fills_only_missing_parent_without_changing_source_file(self):
        data = scene()
        del entity(data)["parentId"]
        validate_scene(data)
        self.assertNotIn("parentId", entity(data))
        raw = json.dumps(data).encode("utf-8")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scene.json"
            path.write_bytes(raw)
            loaded = check_scenario(path)
            self.assertEqual(entity(loaded)["parentId"], -1)
            expected = deepcopy(data)
            entity(expected)["parentId"] = -1
            self.assertEqual(loaded, expected)
            self.assertEqual(path.read_bytes(), raw)

    def test_optional_metadata_is_descriptive_and_provided_values_are_nonempty_strings(self):
        data = scene()
        for metadata in ({}, {"id": "custom", "title": "Custom map", "difficulty": "practice",
                              "description": "A self-contained scenario."}):
            data["metadata"] = metadata
            validate_scene(data)
        for value in (None, [], "custom"):
            data["metadata"] = value
            with self.subTest(metadata=value):
                self.assert_invalid(data, "$.metadata")
        for field in ("id", "title", "difficulty", "description"):
            for value in ("", " \t", None, 1, True):
                data["metadata"] = {field: value}
                with self.subTest(field=field, value=value):
                    self.assert_invalid(data, f"$.metadata.{field}")

    def test_required_fields_and_nonempty_entities(self):
        for field in ("simStep", "simTime", "simEndLogicTime", "mapArea", "redArea", "entityList"):
            data = scene()
            del data["imagineProfile"][field]
            with self.subTest(field=field):
                self.assert_invalid(data, f"$.imagineProfile.{field}")
        data = scene()
        data["imagineProfile"]["entityList"] = []
        self.assert_invalid(data, "$.imagineProfile.entityList")
        data = scene()
        del entity(data)["id"]
        self.assert_invalid(data, ".entity.id")

    def test_entity_integer_fields_exclude_bools_and_fractional_values(self):
        for field in ("id", "entityType", "sideId"):
            for value in (True, 1.5, "1"):
                data = scene()
                entity(data)[field] = value
                with self.subTest(field=field, value=value):
                    self.assert_invalid(data, f".entity.{field}")

    def test_health_must_be_nonnegative_and_finite(self):
        for value in (-1, True, float("nan"), float("inf")):
            data = scene()
            entity(data)["survivePoints"] = value
            with self.subTest(value=value):
                self.assert_invalid(data, ".entity.survivePoints")

    def test_optional_physical_fields_have_explicit_types_and_finite_vectors(self):
        valid = scene()
        entity(valid).update(
            isVisible=True, vel={"x": -10, "y": 0, "z": 5},
            stage=-1,
            step=0, time=1783391450000, nameChn="目标", nameEn="target",
        )
        validate_scene(valid)
        invalid = (
            ("isVisible", 1), ("isVisible", "false"),
            ("vel", {"x": 1, "y": 2}),
            ("vel", {"x": 0, "y": float("inf"), "z": 0}),
            ("stage", True), ("stage", "1"),
            ("step", 1.5), ("step", True), ("step", -1),
            ("time", -1), ("time", False), ("time", float("nan")),
            ("nameChn", None), ("nameEn", 123),
        )
        for field, value in invalid:
            data = deepcopy(valid)
            entity(data)[field] = value
            with self.subTest(field=field, value=value):
                self.assert_invalid(data, f".entity.{field}")

    def test_event_and_task_lists_cannot_be_silently_ignored(self):
        for field in ("eventList", "taskList"):
            data = scene()
            data["imagineProfile"][field] = []
            validate_scene(data)
            for value in ({}, "[]", [{"id": 1}]):
                data["imagineProfile"][field] = value
                with self.subTest(field=field, value=value):
                    self.assert_invalid(data, f"$.imagineProfile.{field}")

    def test_absolute_and_logic_clocks_are_distinct(self):
        data = scene()
        validate_scene(data)
        for field, value in (("simStep", 0), ("simStep", 0.5), ("simStep", True), ("simTime", -1),
                              ("simEndLogicTime", 500), ("logicTime", 3000000)):
            data = scene()
            data["imagineProfile"][field] = value
            with self.subTest(field=field, value=value):
                self.assert_invalid(data, f"$.imagineProfile.{field}")

    def test_map_bounds_and_polygon_validity(self):
        data = scene()
        data["imagineProfile"]["mapArea"]["xMin"] = 2
        self.assert_invalid(data, ".mapArea")
        for ring in ([], [[0, 0], [1, 1]], [[0, 0], [1, 0], [2, 0]],
                     [[0, 0], [1, 0], [0, 0], [0, 1]],
                     [[0, 0], [2, 2], [0, 2], [2, 0]],
                     [[0, 0], [True, 0], [0, 1]]):
            data = scene()
            data["imagineProfile"]["redArea"]["coordinates"] = [ring]
            with self.subTest(ring=ring):
                self.assert_invalid(data, ".redArea.coordinates")

    def test_implicit_and_explicit_polygon_closure_are_supported(self):
        data = scene()
        ring = data["imagineProfile"]["redArea"]["coordinates"][0]
        ring.append(deepcopy(ring[0]))
        validate_scene(data)

    def test_model_mappings_need_unique_types_and_nonempty_paths(self):
        data = scene()
        configs = data["environmentProfile"]["dynamicLibraryConfigs"]
        configs.append(deepcopy(configs[0]))
        self.assert_invalid(data, ".dynamicLibraryConfigs[1].entityType")
        data = scene()
        data["environmentProfile"]["dynamicLibraryConfigs"][0]["dynamicLibraryPath"] = " "
        self.assert_invalid(data, ".dynamicLibraryPath")
        data = scene()
        entity(data)["entityType"] = 12345
        self.assert_invalid(data, ".entity.entityType")

    def test_checking_model_registry_is_left_to_engine_creation(self):
        data = scene()
        entity(data)["entityType"] = 12345
        data["environmentProfile"]["dynamicLibraryConfigs"] = [
            {"entityType": 12345, "dynamicLibraryPath": "custom-model-binding"},
        ]
        validate_scene(data)

    def test_entity_ids_are_unique(self):
        data = scene()
        data["imagineProfile"]["entityList"].append(deepcopy(data["imagineProfile"]["entityList"][0]))
        self.assert_invalid(data, ".entityList[1].entity.id")

    def test_parent_and_child_references_must_exist_and_match(self):
        validate_scene(with_child())
        data = scene()
        entity(data)["parentId"] = 99
        self.assert_invalid(data, ".entity.parentId")
        data = scene()
        entity(data)["childrenId"] = [99]
        self.assert_invalid(data, ".entity.childrenId[0]")
        data = with_child()
        entity(data)["childrenId"] = []
        self.assert_invalid(data, ".entityList[1].entity.parentId")
        data = with_child()
        entity(data, 1)["parentId"] = -1
        self.assert_invalid(data, ".entityList[0].entity.childrenId")

    def test_parent_cycles_are_rejected(self):
        data = with_child()
        entity(data)["parentId"] = 2
        entity(data, 1)["childrenId"] = [1]
        with self.assertRaisesRegex(SceneValidationError, "cycle"):
            validate_scene(data)

    def test_positions_are_three_dimensional_finite_vectors(self):
        data = scene()
        entity(data)["pos"]["x"] = 25_000
        validate_scene(data)
        for invalid, path in (({"x": 0, "y": 0}, ".entity.pos.z"),
                              ({"x": float("inf"), "y": 0, "z": 0}, ".entity.pos.x"),
                              ({"x": True, "y": 0, "z": 0}, ".entity.pos.x")):
            data = scene()
            entity(data)["pos"] = invalid
            self.assert_invalid(data, path)

    def test_support_entities_are_not_forced_inside_the_map_rectangle(self):
        data = scene()
        data["imagineProfile"]["mapArea"].update(xMin=10, xMax=11)
        validate_scene(data)

    def test_external_is_optional_or_an_encoded_json_object(self):
        data = scene()
        row = data["imagineProfile"]["entityList"][0]
        del row["external"]
        validate_scene(data)
        row["external"] = '{"custom": {"enabled": true}}'
        validate_scene(data)
        for value in (None, {}, "[]", "not json", '{"x": 1, "x": 2}', '{"x": NaN}'):
            row["external"] = value
            with self.subTest(value=value):
                self.assert_invalid(data, ".external")

    def test_load_rejects_duplicate_keys_and_nonstandard_numbers_with_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scene.json"
            base = json.dumps(scene())
            for text, expected in (
                (base.replace('"id": 1', '"id": 1, "id": 2'), ".entity.id"),
                (base.replace('"survivePoints": 100', '"survivePoints": NaN'), ".survivePoints"),
                (base.replace('"survivePoints": 100', '"survivePoints": Infinity'), ".survivePoints"),
                (base.replace('"survivePoints": 100', '"survivePoints": 1e999'), ".survivePoints"),
                ("{", "$"),
            ):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text[:70]), self.assertRaises(SceneValidationError) as error:
                    check_scenario(path)
                self.assertIn(expected, error.exception.path)



if __name__ == "__main__":
    unittest.main()
