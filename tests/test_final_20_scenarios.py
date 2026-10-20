from __future__ import annotations

import hashlib
import json
import statistics
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "scenarios" / "cases"
FINAL = CASES / "final20"
RED_TYPES = {"high": 21000, "medium": 21001, "low": 21002}
BLUE_TYPES = {"target": 9400, "site": 9600, "ship": 9500}
INTERCEPTOR_TYPE = 24000
RADAR_TYPE = 44000
EXPECTED_CASES = {"easy": 6, "medium": 6, "hard": 8}
EXPECTED_ROOTS = {
    "easy": {"target": 5, "site": 10, "ship": 9},
    "medium": {"target": 7, "site": 15, "ship": 14},
    "hard": {"target": 10, "site": 20, "ship": 18},
}
EXPECTED_RED = {
    "easy": {"high": 35, "medium": 49, "low": 126},
    "medium": {"high": 50, "medium": 70, "low": 180},
    "hard": {"high": 70, "medium": 98, "low": 252},
}
EXPECTED_INTERCEPTORS = {"easy": 189, "medium": 270, "hard": 378}
EXPECTED_HP = {"target": 16.0, "site": 12.0, "ship": 0.8}
SOURCE_CASE = {"easy": "E01", "medium": "M01", "hard": "H01"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def entities(scenario: dict) -> list[dict]:
    return [item["entity"] for item in scenario["imagineProfile"]["entityList"]]


class Final20ScenarioTests(unittest.TestCase):
    def test_case_distribution_is_six_six_eight(self) -> None:
        for difficulty, count in EXPECTED_CASES.items():
            prefix = {"easy": "E", "medium": "M", "hard": "H"}[difficulty]
            self.assertEqual(
                [f"{prefix}{index:02d}" for index in range(1, count + 1)],
                sorted(path.name for path in (FINAL / difficulty).iterdir() if path.is_dir()),
            )

    def test_inventory_health_ratio_and_timing_contracts(self) -> None:
        for scenario_path in sorted(FINAL.glob("*/*/scenario.json")):
            with self.subTest(case=scenario_path.parent.name):
                scenario = load(scenario_path)
                info = load(scenario_path.with_name("case_info.json"))
                difficulty = info["difficulty"]
                rows = entities(scenario)
                counts = Counter(row["entityType"] for row in rows)

                self.assertEqual(EXPECTED_ROOTS[difficulty], info["root_counts"])
                self.assertGreaterEqual(info["root_counts"]["target"], 4)
                self.assertIn(info["root_counts"]["site"] - info["root_counts"]["ship"], (1, 2))
                self.assertGreaterEqual(info["root_counts"]["site"] / info["root_counts"]["target"], 2.0)
                self.assertLessEqual(info["root_counts"]["site"] / info["root_counts"]["target"], 2.15)
                self.assertEqual({"easy": 24, "medium": 36, "hard": 48}[difficulty], info["blue_root_total"])
                self.assertEqual(EXPECTED_RED[difficulty], info["red_missile_counts"])
                self.assertEqual(EXPECTED_INTERCEPTORS[difficulty], info["interceptor_total"])
                self.assertEqual(EXPECTED_HP, info["blue_hp_by_kind"])
                for kind, entity_type in BLUE_TYPES.items():
                    self.assertEqual(info["root_counts"][kind], counts[entity_type])
                    self.assertTrue(all(row["survivePoints"] == EXPECTED_HP[kind] for row in rows if row["entityType"] == entity_type))
                for kind, entity_type in RED_TYPES.items():
                    self.assertEqual(EXPECTED_RED[difficulty][kind], counts[entity_type])
                self.assertEqual(EXPECTED_INTERCEPTORS[difficulty], counts[INTERCEPTOR_TYPE])
                self.assertGreaterEqual(info["blue_health_to_red_nominal_damage_ratio"], 0.19)
                self.assertLessEqual(info["blue_health_to_red_nominal_damage_ratio"], 0.20)
                self.assertAlmostEqual(2.25, info["interceptor_to_high_medium_ratio"])
                self.assertAlmostEqual(1.5, info["interceptor_to_low_ratio"])
                self.assertEqual(3000, info["max_steps"])
                profile = scenario["imagineProfile"]
                self.assertEqual(info["max_steps"] * profile["simStep"], profile["simEndLogicTime"])
                self.assertEqual(info["map_area"], profile["mapArea"])
                self.assertEqual(hashlib.sha256(scenario_path.read_bytes()).hexdigest(), info["scenario_sha256"])

    def test_entity_graph_locations_and_radar_configuration_are_valid(self) -> None:
        for scenario_path in sorted(FINAL.glob("*/*/scenario.json")):
            with self.subTest(case=scenario_path.parent.name):
                scenario = load(scenario_path)
                info = load(scenario_path.with_name("case_info.json"))
                rows = entities(scenario)
                by_id = {row["id"]: row for row in rows}
                self.assertEqual(len(rows), len(by_id))
                area = scenario["imagineProfile"]["mapArea"]
                for row in rows:
                    parent_id = row.get("parentId", -1)
                    if parent_id != -1:
                        self.assertIn(parent_id, by_id)
                        self.assertIn(row["id"], by_id[parent_id].get("childrenId", ()))
                    for child_id in row.get("childrenId", ()):
                        self.assertIn(child_id, by_id)
                        self.assertEqual(row["id"], by_id[child_id].get("parentId"))
                    if row["entityType"] in BLUE_TYPES.values():
                        self.assertLessEqual(area["lonMin"], row["lla"]["x"])
                        self.assertGreaterEqual(area["lonMax"], row["lla"]["x"])
                        self.assertLessEqual(area["latMin"], row["lla"]["y"])
                        self.assertGreaterEqual(area["latMax"], row["lla"]["y"])
                for item in scenario["imagineProfile"]["entityList"]:
                    if item["entity"]["entityType"] == RADAR_TYPE:
                        self.assertNotIn("distance", json.loads(item["external"])["raderExternal"])
                defense_ids = {
                    row["id"] for row in rows if row["entityType"] in (BLUE_TYPES["site"], BLUE_TYPES["ship"])
                }
                target_ids = {row["id"] for row in rows if row["entityType"] == BLUE_TYPES["target"]}
                site_ids = {row["id"] for row in rows if row["entityType"] == BLUE_TYPES["site"]}
                ship_ids = {row["id"] for row in rows if row["entityType"] == BLUE_TYPES["ship"]}
                self.assertEqual(target_ids | site_ids | ship_ids, set(info["objective_ids"]))
                self.assertEqual(target_ids | site_ids, set(info["public_objective_ids"]))
                self.assertEqual(ship_ids, set(info["hidden_objective_ids"]))
                self.assertEqual(len(site_ids), info["radar_total"])
                for root_id in site_ids:
                    radar_ids = [child for child in by_id[root_id].get("childrenId", ()) if by_id[child]["entityType"] == RADAR_TYPE]
                    self.assertEqual(1, len(radar_ids))
                    self.assertEqual(by_id[root_id]["lla"], by_id[radar_ids[0]]["lla"])
                for root_id in ship_ids:
                    self.assertFalse(any(by_id[child]["entityType"] == RADAR_TYPE for child in by_id[root_id].get("childrenId", ())))
                assigned = {int(key): value for key, value in info["interceptor_counts"].items()}
                self.assertEqual(defense_ids, set(assigned))
                self.assertLessEqual(max(assigned.values()) - min(assigned.values()), 1)
                for root_id, amount in assigned.items():
                    root = by_id[root_id]
                    interceptor_ids = [child for child in root.get("childrenId", ()) if by_id[child]["entityType"] == INTERCEPTOR_TYPE]
                    self.assertEqual(amount, len(interceptor_ids))
                    for child_id in interceptor_ids:
                        self.assertEqual(root["lla"], by_id[child_id]["lla"])

    def test_red_launch_positions_match_source_and_outputs_exist(self) -> None:
        checksums: set[str] = set()
        for scenario_path in sorted(FINAL.glob("*/*/scenario.json")):
            difficulty = scenario_path.parents[1].name
            scenario = load(scenario_path)
            info = load(scenario_path.with_name("case_info.json"))
            source = load(CASES / difficulty / SOURCE_CASE[difficulty] / "scenario.json")
            for entity_type in RED_TYPES.values():
                expected = {
                    (row["lla"]["x"], row["lla"]["y"], row["lla"].get("z", 0.0))
                    for row in entities(source) if row["entityType"] == entity_type
                }
                actual = {
                    (row["lla"]["x"], row["lla"]["y"], row["lla"].get("z", 0.0))
                    for row in entities(scenario) if row["entityType"] == entity_type
                }
                self.assertEqual(expected, actual)
            checksum = info["scenario_sha256"]
            self.assertNotIn(checksum, checksums)
            checksums.add(checksum)
            self.assertGreaterEqual(len(info["expected_route_options"]), 3)
            self.assertEqual("role_layered_paths_v2", info["topology_model"])
            markers = info["red_launch_markers"]
            self.assertEqual({"高/中", "低"}, {marker["label"] for marker in markers})
            self.assertEqual(
                {frozenset(("high", "medium")), frozenset(("low",))},
                {frozenset(marker["roles"]) for marker in markers},
            )
            self.assertTrue(scenario_path.with_name("map_info.png").is_file())
        self.assertEqual(20, len(checksums))
        self.assertTrue((FINAL / "final_20_overview.png").is_file())
        self.assertTrue((FINAL / "overview_easy.png").is_file())
        self.assertTrue((FINAL / "overview_medium.png").is_file())
        self.assertTrue((FINAL / "overview_hard.png").is_file())
        self.assertTrue((FINAL / "README.md").is_file())

    def test_role_layers_and_ship_dispersion_are_physical(self) -> None:
        topology_signatures: set[str] = set()
        for scenario_path in sorted(FINAL.glob("*/*/scenario.json")):
            with self.subTest(case=scenario_path.parent.name):
                scenario = load(scenario_path)
                info = load(scenario_path.with_name("case_info.json"))
                rows = entities(scenario)
                sites = [(row["lla"]["x"], row["lla"]["y"]) for row in rows if row["entityType"] == BLUE_TYPES["site"]]
                ships = [(row["lla"]["x"], row["lla"]["y"]) for row in rows if row["entityType"] == BLUE_TYPES["ship"]]
                targets = [(row["lla"]["x"], row["lla"]["y"]) for row in rows if row["entityType"] == BLUE_TYPES["target"]]

                self.assertGreaterEqual(sum(lon >= 124.5 for lon, _lat in targets) / len(targets), 0.70)
                self.assertGreaterEqual(sum(lon <= 123.3 for lon, _lat in sites) / len(sites), 0.75)
                self.assertGreater(statistics.median(lon for lon, _lat in targets) - statistics.median(lon for lon, _lat in sites), 2.0)
                self.assertGreater(max(lon for lon, _lat in ships) - min(lon for lon, _lat in ships), 4.0)
                self.assertGreater(max(lat for _lon, lat in ships) - min(lat for _lon, lat in ships), 4.0)
                self.assertEqual(0.15, info["placement_variation"]["cross_layer_fraction"]["site"])
                self.assertEqual(0.20, info["placement_variation"]["cross_layer_fraction"]["target"])
                self.assertEqual("fixed_offset_table", info["placement_variation"]["model"])
                self.assertFalse(info["placement_variation"]["experiment_seed_dependent"])
                self.assertNotIn("seed", info["placement_variation"])

                signature = json.dumps(info["placement_paths"], sort_keys=True)
                self.assertNotIn(signature, topology_signatures)
                topology_signatures.add(signature)
        self.assertEqual(20, len(topology_signatures))


if __name__ == "__main__":
    unittest.main()
