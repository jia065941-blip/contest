import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envengine.environment.scene_adapter import entity_position, profile_from_scene
from scenarios import load_scenario


class Launch20ScenarioLoadingTests(unittest.TestCase):
    def test_blue_coordinates_and_attached_positions_are_consistent(self):
        case_root = ROOT / 'scenarios'
        manifest = json.loads((case_root / 'manifest.json').read_text(encoding='utf-8'))
        for case in manifest['cases']:
            path = case_root / case['scenario']
            raw = json.loads(path.read_text(encoding='utf-8'))
            entities = {row['entity']['id']: row['entity'] for row in raw['imagineProfile']['entityList']}
            for entity in entities.values():
                if entity['entityType'] not in (9400, 9500, 9600, 44000, 24000):
                    continue
                with self.subTest(map=path.stem, entity=entity['id']):
                    self.assertTrue(all(math.isfinite(entity['pos'][axis]) for axis in 'xyz'))
                    self.assertTrue(all(math.isfinite(entity['vel'][axis]) for axis in 'xyz'))
                    if entity['parentId'] != -1:
                        parent = entities[entity['parentId']]
                        self.assertEqual(entity['pos'], parent['pos'])


    def test_tier_inventory_and_attack_agent_counts(self):
        case_root = ROOT / 'scenarios'
        manifest = json.loads((case_root / 'manifest.json').read_text(encoding='utf-8'))
        expected = {'easy': (78, 41), 'medium': (146, 64), 'hard': (348, 114)}
        for case in manifest['cases']:
            with self.subTest(map=case['id']):
                path = case_root / case['scenario']
                raw = json.loads(path.read_text(encoding='utf-8'))
                counts = Counter(item['entity']['entityType'] for item in raw['imagineProfile']['entityList'])
                interceptors, attackers = expected[case['difficulty']]
                self.assertEqual(counts[24000], interceptors)
                self.assertEqual(counts[21000] + counts[21001], attackers)

    def test_all_scenarios_load_with_empty_communication_dictionaries(self):
        case_root = ROOT / 'scenarios'
        manifest = json.loads((case_root / 'manifest.json').read_text(encoding='utf-8'))
        expected_ids = {f'E{i:02}' for i in range(1, 7)} | {f'M{i:02}' for i in range(1, 7)} | {
            f'H{i:02}' for i in range(1, 9)}
        self.assertEqual({case['id'] for case in manifest['cases']}, expected_ids)
        self.assertEqual(len(manifest['cases']), 20)
        for case in manifest['cases']:
            with self.subTest(map=case['id']):
                path = case_root / case['scenario']
                bundle = load_scenario(path)
                self.assertEqual(bundle.reward_policy.scenario_id, case['id'])
                raw = json.loads(path.read_text(encoding='utf-8'))
                profile = profile_from_scene(raw)
                self.assertEqual(len(profile.imagineProfile.entityList), len(raw['imagineProfile']['entityList']))
                for item in profile.imagineProfile.entityList:
                    expected_pos = next(row['entity']['pos'] for row in raw['imagineProfile']['entityList']
                                        if row['entity']['id'] == item.entity.id)
                    actual_pos = entity_position(item.entity)
                    for axis in 'xyz':
                        self.assertAlmostEqual(getattr(actual_pos, axis), expected_pos[axis], places=6)
                    self.assertIsInstance(item.entity.commRangeInfo, dict)
                    self.assertEqual(item.entity.commRangeInfo, {})
                for name, expected_hash in case['files_sha256'].items():
                    content = (case_root / name).read_bytes()
                    hashes = {hashlib.sha256(content).hexdigest()}
                    if name.endswith('.json'):
                        normalized = content.replace(b'\r\n', b'\n')
                        hashes.add(hashlib.sha256(normalized).hexdigest())
                        hashes.add(hashlib.sha256(normalized.replace(b'\n', b'\r\n')).hexdigest())
                    self.assertIn(expected_hash, hashes, name)


    def test_every_bundled_scene_including_practice_maps_loads(self):
        paths = sorted(path for path in (ROOT / 'scenarios').glob('*.json')
                       if path.name != 'manifest.json')
        self.assertEqual(len(paths), 22)
        for path in paths:
            with self.subTest(scene=path.stem):
                bundle = load_scenario(path)
                self.assertGreater(len(bundle.profile.imagineProfile.entityList), 0)


if __name__ == '__main__':
    unittest.main()
