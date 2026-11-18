"""A parsed scene is reusable input, independent of environment state."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from envengine.environment.training_env import TrainingEnv
from scenarios import ScenarioConfig, load_scenario
from scenarios.cases import calculate_reward, load_reward_policy
from scenarios.scenario_checker import check_scenario


ROOT = Path(__file__).resolve().parents[1]
E01 = ROOT / 'scenarios/E01.json'


class ScenarioConfigTests(unittest.TestCase):
    def test_each_load_reads_scene_once_and_returns_independent_configuration(self):
        reads = []
        read_bytes, read_text = Path.read_bytes, Path.read_text

        def tracked_bytes(path, *args, **kwargs):
            if path.resolve() == E01:
                reads.append(path)
            return read_bytes(path, *args, **kwargs)

        def tracked_text(path, *args, **kwargs):
            if path.resolve() == E01:
                reads.append(path)
            return read_text(path, *args, **kwargs)

        with patch.object(Path, 'read_bytes', tracked_bytes), patch.object(Path, 'read_text', tracked_text), \
             patch('scenarios.loader.check_scenario', wraps=check_scenario) as checker:
            first = load_scenario(E01)
            self.assertEqual(len(reads), 1)
            checker.assert_called_once_with(E01)
            second = load_scenario(E01)
            self.assertEqual(len(reads), 2)
            self.assertEqual(checker.call_count, 2)
        self.assertIsInstance(first, ScenarioConfig)
        self.assertEqual(first.path, E01)
        self.assertIsNot(first, second)
        self.assertIsNot(first.profile, second.profile)
        original = second.profile.imagineProfile.entityList[0].entity.survivePoints
        first.profile.imagineProfile.entityList[0].entity.survivePoints += 1
        self.assertEqual(second.profile.imagineProfile.entityList[0].entity.survivePoints, original)
        deployment = deepcopy(first.deployment.coordinates)
        first.profile.imagineProfile.redArea.coordinates[0][0] += 1
        self.assertEqual(first.deployment.coordinates, deployment)

    def test_asset_values_are_independent_copies_of_scoring_weights(self):
        scene = load_scenario(E01)
        values = scene.asset_values
        self.assertTrue(values)
        self.assertEqual(values, dict(scene.reward_policy.objective_weights))
        identifier = next(iter(values))
        expected = values[identifier]
        values[identifier] = 999
        values[-1] = 100
        self.assertEqual(scene.asset_values[identifier], expected)
        self.assertNotIn(-1, scene.asset_values)

    def test_preview_configuration_skips_scoring_but_keeps_deployment_rules(self):
        with patch('scenarios.loader._reward_policy_from_scene',
                   side_effect=AssertionError('preview must not construct scoring rules')):
            scene = load_scenario(E01, include_rules=False)
        self.assertIsNone(scene.reward_policy)
        self.assertEqual(scene.asset_values, {})
        self.assertEqual(scene.deployment.launch_altitude, 3000)
        self.assertTrue(scene.deployment.coordinates)
        self.assertFalse(hasattr(scene.deployment, 'coordinates_hm'))

    def test_copied_custom_scene_uses_edited_entities_for_score_without_sibling_files(self):
        original = load_scenario(E01).reward_policy
        data = json.loads(E01.read_text(encoding='utf-8'))
        data['metadata'] = {'id': 'custom-score', 'title': 'Custom score'}
        rows = data['imagineProfile']['entityList']
        targets = [row for row in rows if row['entity']['entityType'] == 9400]
        changed, removed = targets[:2]
        changed_id, removed_id = changed['entity']['id'], removed['entity']['id']
        changed['entity'].update(entityType=9500, survivePoints=73)
        rows.remove(removed)
        added = deepcopy(targets[2])
        added_id = max(row['entity']['id'] for row in rows) + 1
        added['entity'].update(id=added_id, survivePoints=31)
        rows.append(added)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'custom.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            self.assertEqual(list(path.parent.iterdir()), [path])
            scene = load_scenario(path)
            policy = scene.reward_policy
            self.assertEqual(policy, load_reward_policy(path))
            self.assertEqual(policy.scenario_id, 'custom-score')
            self.assertNotIn(removed_id, policy.objective_ids)
            self.assertIn(added_id, policy.objective_ids)
            self.assertEqual(dict(policy.objective_weights)[changed_id], 1)
            self.assertEqual(dict(policy.objective_weights)[added_id], 5)
            self.assertEqual(dict(policy.objective_initial_health)[changed_id], 73)
            self.assertEqual(dict(policy.objective_initial_health)[added_id], 31)
            self.assertEqual(scene.deployment.launch_altitude, 3000)
            result = calculate_reward(policy=policy, target_health={changed_id: 0, added_id: 0})
            self.assertAlmostEqual(result.score, 100 * 6 / (sum(dict(original.objective_weights).values()) - 4))
            path.with_name('case_info.json').write_text('invalid obsolete scoring data', encoding='utf-8')
            path.with_name('deployment_area.json').write_text('{"coordinates": []}', encoding='utf-8')
            reloaded = load_scenario(path)
            self.assertEqual(reloaded.reward_policy, policy)
            self.assertEqual(reloaded.deployment, scene.deployment)
            self.assertEqual(load_reward_policy(path), policy)
            data.pop('metadata')
            path.write_text(json.dumps(data), encoding='utf-8')
            without_metadata = load_scenario(path).reward_policy
            self.assertEqual(without_metadata.objective_weights, policy.objective_weights)
            self.assertEqual(without_metadata.objective_initial_health, policy.objective_initial_health)

    def test_no_countable_targets_has_no_scoring_policy(self):
        data = json.loads(E01.read_text(encoding='utf-8'))
        data.pop('metadata', None)
        data['imagineProfile']['entityList'] = [
            row for row in data['imagineProfile']['entityList']
            if row['entity']['entityType'] not in (9400, 9500, 9600, 44000, 24000)
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'practice.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            scene = load_scenario(path)
            self.assertIsNone(scene.reward_policy)
            self.assertIsNone(load_reward_policy(path))
            self.assertEqual(scene.asset_values, {})

    def test_deployment_altitude_comes_from_initial_red_units(self):
        data = json.loads(E01.read_text(encoding='utf-8'))
        red = [row['entity'] for row in data['imagineProfile']['entityList']
               if row['entity']['entityType'] in (21000, 21001, 21002)]
        for entity in red:
            entity['pos']['z'] = 1700
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'height.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            scene = load_scenario(path)
            self.assertEqual(scene.deployment.launch_altitude, 1700)
            self.assertEqual(scene.deployment.coordinates, load_scenario(E01).deployment.coordinates)
            data['imagineProfile']['entityList'] = [
                row for row in data['imagineProfile']['entityList']
                if row['entity']['entityType'] not in (21000, 21001, 21002)
            ]
            path.write_text(json.dumps(data), encoding='utf-8')
            self.assertIsNone(load_scenario(path).deployment.launch_altitude)

    def test_environment_consumes_copies_without_reading_files_for_config_or_legacy_profile(self):
        scene = load_scenario(E01)
        before = scene.profile.to_dict()

        def build_engine(profile, speed):
            return Mock(profile=profile, total_steps=3000)

        for source in (scene, scene.profile):
            with self.subTest(config=isinstance(source, ScenarioConfig)), \
                 patch('envengine.environment.training_env.Engine', side_effect=build_engine), \
                 patch.object(Path, 'read_bytes', side_effect=AssertionError('environment must not read files')), \
                 patch.object(Path, 'read_text', side_effect=AssertionError('environment must not read files')), \
                 patch('builtins.open', side_effect=AssertionError('environment must not read files')):
                env = TrainingEnv(source, record_logs=False)
                self.assertIs(env.config, scene if isinstance(source, ScenarioConfig) else None)
                self.assertIsNot(env.engine.profile, scene.profile)
                env.engine.profile.imagineProfile.entityList[0].entity.survivePoints += 1
                env.engine.profile.imagineProfile.redArea.coordinates[0][0] += 1
            self.assertEqual(scene.profile.to_dict(), before)

    def test_manager_enforces_layout_while_environment_exports_static_metadata(self):
        scene = load_scenario(E01)
        before = scene.profile.to_dict()
        env = TrainingEnv(scene, record_logs=False)
        self.addCleanup(env.close)
        state = env.reset()
        self.assertIs(env.config, scene)
        self.assertEqual(state['deployment_area']['launch_altitude_m'], 3000)
        for name in ('get_deployment_observation', 'apply_deployment', 'register_user_agent',
                     'set_deployment_constraints', '_validate_deployment_position', '_point_in_polygon',
                     '_episode_prepared', 'speed_distributor'):
            self.assertFalse(hasattr(env, name), name)
        self.assertFalse(any(name.startswith('_deployment') for name in vars(env)))
        manager = env.agent_manager
        observation = manager.get_deployment_observation(seed=7)
        self.assertEqual(set(observation), {'step', 'entities', 'public_targets', 'coordinates', 'seed'})
        self.assertEqual(observation['coordinates'],
                         [list(point) for point in state['deployment_area']['coordinates'][0]])
        plan = {identifier: tuple(observation['coordinates'][0])
                for identifier in observation['entities']}
        unit = env.engine.simulator_factory.get_simulators_by_type(21000)[0]
        identifier = unit.entity_ext.entity.id
        with self.assertRaises(ValueError):
            manager.apply_deployment({**plan, identifier: (*plan[identifier], scene.deployment.launch_altitude - 1)})
        self.assertIsNone(manager.apply_deployment(plan))
        unit.entity_ext.entity.survivePoints -= 1
        unit.entity_ext.entity.survivePoints += 0.01
        env.engine.profile.imagineProfile.redArea.coordinates[0][0] += 0.01
        self.assertEqual(scene.profile.to_dict(), before)
        self.assertEqual(env.get_observation()['deployment_area'], state['deployment_area'])
        env.reset()
        self.assertEqual(scene.profile.to_dict(), before)


if __name__ == '__main__':
    unittest.main()
