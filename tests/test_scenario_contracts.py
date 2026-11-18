from __future__ import annotations

import hashlib
import json
import math
import unittest
from collections import Counter
from pathlib import Path

from scenarios import load_scenario
from scenarios.grid_maps.deployment import balanced_plan, compile_actions, compile_positions, validate_plan


CATALOGUE = Path(__file__).resolve().parents[1] / "scenarios"
RED_TYPES = {"high": 21000, "medium": 21001, "low": 21002}
BLUE_TYPES = {"target": 9400, "site": 9600, "ship": 9500}
EXPECTED_CASES = {"easy": 6, "medium": 6, "hard": 8}
TIER_INVENTORIES = {
    "easy": {9400: 4, 9600: 7, 9500: 10, 44000: 7, 24000: 78,
             21000: 15, 21001: 24, 21002: 85, 9202: 2, 35000: 1},
    "medium": {9400: 6, 9600: 12, 9500: 15, 44000: 12, 24000: 146,
               21000: 24, 21001: 37, 21002: 128, 9202: 3, 35000: 1},
    "hard": {9400: 8, 9600: 20, 9500: 20, 44000: 20, 24000: 348,
             21000: 40, 21001: 66, 21002: 214, 9202: 4, 35000: 1},
}
RADAR_TYPE = 44000
INTERCEPTOR_TYPE = 24000


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def checksum(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(scenario):
    return [item["entity"] for item in scenario["imagineProfile"]["entityList"]]


def rectangle_contains(coordinates, pos):
    points = coordinates[0]
    return (min(point[0] for point in points) <= pos["x"] <= max(point[0] for point in points)
            and min(point[1] for point in points) <= pos["y"] <= max(point[1] for point in points))


class ScenarioContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load(CATALOGUE / "manifest.json")
        cls.cases = [(CATALOGUE / entry["scenario"], load(CATALOGUE / entry["scenario"]), entry)
                     for entry in cls.manifest["cases"]]

    def test_case_distribution_and_manifest_completeness(self):
        expected_ids = {f"{prefix}{index:02d}"
                        for prefix, count in (("E", 6), ("M", 6), ("H", 8))
                        for index in range(1, count + 1)}
        self.assertEqual({path.name for path in CATALOGUE.glob("*.json")},
                         {f"{identifier}.json" for identifier in expected_ids} | {"manifest.json", "R01.json", "Rtest.json"})
        self.assertEqual({path.name for path in (CATALOGUE / "images").glob("*.png")},
                         {f"{identifier}.png" for identifier in expected_ids})
        self.assertEqual(len(self.cases), 20)
        declared = set()
        for item in self.manifest["cases"]:
            self.assertNotIn(item["id"], declared)
            declared.add(item["id"])
            self.assertEqual(item["scenario"], f"{item['id']}.json")
        self.assertEqual(declared, expected_ids)
        self.assertEqual(Counter(entry["difficulty"] for entry in self.manifest["cases"]), EXPECTED_CASES)
        self.assertEqual(Counter(scenario["metadata"]["difficulty"] for _, scenario, _ in self.cases),
                         EXPECTED_CASES)

    def test_manifest_hashes_and_embedded_metadata_identity(self):
        signatures = set()
        for path, scenario, entry in self.cases:
            with self.subTest(case=entry["id"]):
                metadata = scenario["metadata"]
                self.assertEqual(path.stem, metadata["id"])
                for field in ("id", "difficulty", "title"):
                    self.assertEqual(metadata[field], entry[field])
                self.assertTrue(metadata["description"].strip())
                actual_hash = checksum(path)
                self.assertNotIn(actual_hash, signatures)
                signatures.add(actual_hash)
                self.assertEqual({f"{entry['id']}.json", f"images/{entry['id']}.png"},
                                 set(entry["files_sha256"]))
                self.assertEqual(path.parent, CATALOGUE)
                for filename, expected_hash in entry["files_sha256"].items():
                    self.assertEqual(expected_hash, checksum(CATALOGUE / filename), filename)
        self.assertEqual(len(signatures), 20)

    def test_runtime_inventory_health_and_timing_preserve_tier_balance(self):
        for path, scenario, entry in self.cases:
            with self.subTest(case=entry["id"]):
                profile = scenario["imagineProfile"]
                entities = rows(scenario)
                self.assertEqual(len(entities), len({row["id"] for row in entities}))
                self.assertEqual(Counter(row["entityType"] for row in entities),
                                 TIER_INVENTORIES[entry["difficulty"]])
                self.assertEqual(self.manifest["max_steps"], 3000)
                self.assertEqual(self.manifest["max_steps"] * profile["simStep"], profile["simEndLogicTime"])
                self.assertEqual(profile["satelliteUseStep"], 80)
                initial_health = {9400: 24, 9600: 12, 9500: 2}
                for entity in entities:
                    if entity["entityType"] in initial_health:
                        self.assertAlmostEqual(entity["survivePoints"], initial_health[entity["entityType"]])

    def test_entity_graph_objectives_and_radar_contract(self):
        for path, scenario, entry in self.cases:
            with self.subTest(case=entry["id"]):
                profile = scenario["imagineProfile"]
                entities = rows(scenario)
                by_id = {row["id"]: row for row in entities}
                area = profile["mapArea"]
                for entity in entities:
                    parent = entity.get("parentId", -1)
                    if parent != -1:
                        self.assertIn(parent, by_id)
                        self.assertIn(entity["id"], by_id[parent].get("childrenId", ()))
                    children = entity.get("childrenId", ())
                    self.assertEqual(len(children), len(set(children)))
                    for child in children:
                        self.assertIn(child, by_id)
                        self.assertEqual(entity["id"], by_id[child].get("parentId"))
                    if entity["entityType"] in BLUE_TYPES.values():
                        self.assertLessEqual(area["xMin"], entity["pos"]["x"])
                        self.assertGreaterEqual(area["xMax"], entity["pos"]["x"])
                        self.assertLessEqual(area["yMin"], entity["pos"]["y"])
                        self.assertGreaterEqual(area["yMax"], entity["pos"]["y"])
                for item in profile["entityList"]:
                    if item["entity"]["entityType"] == RADAR_TYPE:
                        self.assertNotIn("distance", json.loads(item["external"])["raderExternal"])
                objective_ids = {row["id"] for row in entities if row["entityType"] in BLUE_TYPES.values()}
                self.assertEqual(objective_ids, set(load_scenario(path).reward_policy.objective_ids))
                site_ids = {row["id"] for row in entities if row["entityType"] == BLUE_TYPES["site"]}
                ship_ids = {row["id"] for row in entities if row["entityType"] == BLUE_TYPES["ship"]}
                self.assertEqual(len(site_ids), sum(row["entityType"] == RADAR_TYPE for row in entities))
                interceptor_counts = []
                for root_id in site_ids | ship_ids:
                    root = by_id[root_id]
                    children = [by_id[child] for child in root.get("childrenId", ())]
                    self.assertEqual(1 if root_id in site_ids else 0,
                                     sum(child["entityType"] == RADAR_TYPE for child in children))
                    interceptors = [child for child in children if child["entityType"] == INTERCEPTOR_TYPE]
                    self.assertTrue(interceptors)
                    interceptor_counts.append(len(interceptors))
                    for child in children:
                        self.assertEqual(root["pos"], child["pos"])
                self.assertEqual(sum(interceptor_counts), sum(row["entityType"] == INTERCEPTOR_TYPE for row in entities))
                self.assertLessEqual(max(interceptor_counts) - min(interceptor_counts), 1)

    def test_launch20_free_deployment_contract(self):
        self.assertTrue(self.manifest["all_units_depart_at_start"])
        self.assertFalse(self.manifest["grid_required"])
        for path, scenario, entry in self.cases:
            with self.subTest(case=entry["id"]):
                profile = scenario["imagineProfile"]
                deployment = load_scenario(path).deployment
                self.assertEqual(deployment.launch_altitude, 3000)
                self.assertEqual(deployment.coordinates,
                                 tuple(tuple(map(tuple, polygon)) for polygon in profile["redArea"]["coordinates"]))
                self.assertEqual(len(deployment.coordinates), 1)
                points = deployment.coordinates[0]
                self.assertEqual(len(points), 4)
                self.assertTrue(all(math.isfinite(axis) for point in points for axis in point))
                west, east = min(point[0] for point in points), max(point[0] for point in points)
                south, north = min(point[1] for point in points), max(point[1] for point in points)
                self.assertLess(west, east)
                self.assertLess(south, north)
                self.assertEqual({(west, south), (west, north), (east, south), (east, north)}, set(points))
                for entity in rows(scenario):
                    if entity["entityType"] in RED_TYPES.values():
                        self.assertTrue(rectangle_contains(deployment.coordinates, entity["pos"]))
                        self.assertEqual(deployment.launch_altitude, entity["pos"]["z"])

    def test_grid_plan_uses_current_inventory_and_legal_deployment_area(self):
        for path, scenario, entry in self.cases:
            with self.subTest(case=entry["id"]):
                profile = scenario["imagineProfile"]
                counts = Counter(row["entityType"] for row in rows(scenario))
                expected = {kind_id: counts[kind_id] for kind_id in RED_TYPES.values()}
                plan = balanced_plan(expected)
                validate_plan(plan, expected)
                observed = {row["id"]: {"type": row["entityType"], "pos": {"z": row["pos"]["z"]}}
                            for row in rows(scenario) if row["entityType"] in RED_TYPES.values()}
                deployment_entities = {entity_id: {"type": entity["type"]}
                                       for entity_id, entity in observed.items()}
                positions = compile_positions(plan, deployment_entities, profile["redArea"]["coordinates"][0])
                self.assertEqual(set(positions), set(observed))
                actions = compile_actions(plan, observed, profile["redArea"]["coordinates"])
                self.assertEqual(positions, {entity_id: (x, y)
                                            for _, entity_id, x, y, _ in actions})
                self.assertEqual(set(observed), {action[1] for action in actions})
                self.assertEqual(len(actions), len({tuple(action[2:]) for action in actions}))
                for _, entity_id, x, y, altitude in actions:
                    self.assertTrue(rectangle_contains(profile["redArea"]["coordinates"], {"x": x, "y": y}))
                    self.assertEqual(altitude, observed[entity_id]["pos"]["z"])


if __name__ == "__main__":
    unittest.main()
