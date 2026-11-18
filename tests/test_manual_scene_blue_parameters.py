from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

from scenarios import load_scenario
from blue_defense.blue_checker import check_blue_policy

ROOT = Path(__file__).resolve().parents[1]


class ManualSceneBlueTests(unittest.TestCase):
    def test_twenty_scenes_load_with_manual_parameters(self):
        manifest = json.loads((ROOT / 'scenarios/manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(manifest['cases']), 20)
        for case in manifest['cases']:
            with self.subTest(map=case['id']):
                path = ROOT / 'scenarios' / case['scenario']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), case['files_sha256'][path.name])
                scene = load_scenario(path)
                profile = scene.profile.imagineProfile
                counts = Counter(row.entity.entityType for row in profile.entityList)
                self.assertEqual(counts[21000], {'E': 15, 'M': 24, 'H': 40}[case['id'][0]])
                self.assertEqual(counts[24000], {'E': 78, 'M': 146, 'H': 348}[case['id'][0]])
                self.assertEqual(counts[9202], {'E': 2, 'M': 3, 'H': 4}[case['id'][0]])
                self.assertEqual(profile.satelliteUseStep, 80)
                self.assertEqual(profile.missilePincerAttackIntervalStep, 20)
                self.assertEqual(profile.missilePincerAttackMinAngle, 30)
                for row in profile.entityList:
                    entity = row.entity
                    if entity.entityType in (9400, 9600, 9500):
                        self.assertEqual(entity.survivePoints, {9400: 24, 9600: 12, 9500: 2}[entity.entityType])
                        self.assertEqual(scene.asset_values[entity.id], {9400: 5, 9600: 2, 9500: 1}[entity.entityType])

    def test_ten_blue_entries_load_and_use_current_values(self):
        entries = sorted((ROOT / 'blue_defense').glob('*/b[0-9]_*.py'))
        self.assertEqual(len(entries), 10)
        for path in entries:
            with self.subTest(blue=path.stem):
                policy = check_blue_policy(path)(seed=7, asset_values={})
                try:
                    for kind, expected in ((9400, 5), (9600, 2), (9500, 1)):
                        self.assertEqual(policy.asset_value(-1, kind), expected)
                finally:
                    policy.close()


if __name__ == '__main__':
    unittest.main()
