"""Independent deployment entry points and atomic environment application."""

from copy import deepcopy
from dataclasses import replace
import inspect
import json
from pathlib import Path
from types import MappingProxyType
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from envengine.environment.training_env import TrainingEnv
from scenarios import load_scenario
from scenarios.grid_maps.deployment import balanced_plan
from user_agents.base import BaseDeployment, BaseAgent
from user_agents.red_checker import check_deployment, check_red_agent


ROOT = Path(__file__).resolve().parents[1]
E01 = ROOT / 'scenarios/E01.json'
CUSTOM = """
from user_agents.base import BaseDeployment
class Deployment(BaseDeployment):
    def deploy(self, observation):
        return {}
"""


class DeploymentCheckerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def write(self, source, name='entry.py'):
        path = self.directory / name
        path.write_text(textwrap.dedent(source), encoding='utf-8')
        return path

    def test_deployment_is_a_separate_zero_argument_abstract_role(self):
        self.assertEqual(list(inspect.signature(BaseDeployment).parameters), [])
        self.assertEqual(BaseDeployment.__abstractmethods__, {'deploy'})
        self.assertFalse(issubclass(BaseDeployment, BaseAgent))
        with self.assertRaises(TypeError):
            BaseDeployment()

        class Minimal(BaseDeployment):
            def deploy(self, observation):
                return {}

        implementation = Minimal()
        self.assertEqual(implementation.deploy({}), {})
        self.assertIsNone(implementation.close())

    def test_both_checkers_share_one_load_and_never_construct_or_execute(self):
        path = self.write(r"""
            from pathlib import Path
            from user_agents.base import BaseDeployment, BaseAgent
            marker = Path(__file__).with_suffix('.loads')
            marker.write_text(marker.read_text() + 'load\n' if marker.exists() else 'load\n')
            class Deployment(BaseDeployment):
                def __init__(self):
                    raise AssertionError('checker must not construct deployment')
                def deploy(self, observation):
                    raise AssertionError('checker must not deploy')
            class RedAttack(BaseAgent):
                def __init__(self, entity_id, init_observation):
                    raise AssertionError('checker must not construct actor')
                def get_action(self):
                    raise AssertionError('checker must not decide')
        """)
        actor = check_red_agent(path)
        deployment = check_deployment(path)
        self.assertIs(check_red_agent(path), actor)
        self.assertIs(check_deployment(path), deployment)
        self.assertEqual(path.with_suffix('.loads').read_text(), 'load\n')

    def test_invalid_deployment_exports_and_synchronous_signatures_are_rejected(self):
        sources = {
            'missing': 'value = 1',
            'unrelated': 'class Deployment: pass',
            'abstract': 'from user_agents.base import BaseDeployment\nclass Deployment(BaseDeployment): pass',
            'instance': CUSTOM + '\nDeployment = Deployment()',
            'constructor': CUSTOM.replace('class Deployment(BaseDeployment):',
                                          'class Deployment(BaseDeployment):\n    def __init__(self, required): pass'),
            'no_observation': CUSTOM.replace('def deploy(self, observation):', 'def deploy(self):'),
            'async': CUSTOM.replace('def deploy(', 'async def deploy('),
            'generator': CUSTOM.replace('return {}', 'yield {}'),
            'static': CUSTOM.replace('    def deploy(', '    @staticmethod\n    def deploy('),
        }
        for name, source in sources.items():
            with self.subTest(invalid=name), self.assertRaises(ValueError):
                check_deployment(self.write(source, name + '.py'))

    def test_example_actor_and_deployment_exports_are_independent(self):
        path = ROOT / 'user_agents/example/assignment/r0_random.py'
        self.assertTrue(issubclass(check_red_agent(path), BaseAgent))
        self.assertTrue(issubclass(check_deployment(path), BaseDeployment))
        self.assertFalse(hasattr(check_red_agent(path), 'deploy'))


    def test_example_grid_plan_exposes_only_two_dimensional_positions(self):
        entities = {identifier: {'type': kind}
                    for identifier, kind in ((11, 21000), (22, 21001), (33, 21002))}
        plan = self.directory / 'grid.json'
        plan.write_text(json.dumps(balanced_plan({21000: 1, 21001: 1, 21002: 1})))
        observation = {'step': 0, 'entities': entities,
                       'public_targets': {}, 'seed': 7,
                       'coordinates': [[0, 0], [1, 0], [1, 1], [0, 1]]}
        original = deepcopy(observation)
        deployment = check_deployment(ROOT / 'user_agents/example/assignment/r0_random.py')()
        self.addCleanup(deployment.close)
        deployment.plan_path = plan
        positions = deployment.deploy(observation)
        self.assertEqual(set(positions), set(entities))
        self.assertEqual(len(set(positions.values())), len(entities))
        for position in positions.values():
            self.assertEqual(len(position), 2)
            self.assertTrue(all(0 < coordinate < 1 for coordinate in position))
        self.assertEqual(observation, original)


class EnvironmentDeploymentTests(unittest.TestCase):
    def environment(self):
        env = TrainingEnv(load_scenario(E01), record_logs=False)
        self.addCleanup(env.close)
        return env

    def plan(self, observation):
        return {identifier: tuple(observation['coordinates'][0])
                for identifier in observation['entities']}

    def test_observation_contains_only_deployment_inputs_and_uses_independent_copies(self):
        env = self.environment()
        with self.assertRaises(RuntimeError):
            env.agent_manager.get_deployment_observation()
        with self.assertRaises(RuntimeError):
            env.agent_manager.apply_deployment({})
        env.reset()
        observation = env.agent_manager.get_deployment_observation(seed=7)
        self.assertEqual(set(observation), {
            'step', 'entities', 'public_targets', 'coordinates', 'seed',
        })
        self.assertEqual(observation['step'], 0)
        self.assertEqual(observation['seed'], 7)
        self.assertEqual(observation['coordinates'], env.engine.profile.imagineProfile.redArea.coordinates)
        self.assertEqual(len(observation['entities']), 126)
        for identifier, entity in observation['entities'].items():
            self.assertEqual(set(entity), {'id', 'type'})
            self.assertIn(entity['type'], (21000, 21001, 21002))
        identifier = next(iter(observation['entities']))
        original = observation['entities'][identifier]['type']
        observation['entities'][identifier]['type'] = -1
        observation['coordinates'][0][0] += 1
        again = env.agent_manager.get_deployment_observation(seed=7)
        self.assertEqual(again['entities'][identifier]['type'], original)
        self.assertNotEqual(observation['coordinates'], again['coordinates'])

    def test_deployment_observation_rejects_zero_or_multiple_regions(self):
        scene = load_scenario(E01)
        polygon = scene.deployment.coordinates[0]
        for regions in ((), (polygon, polygon)):
            with self.subTest(region_count=len(regions)):
                configuration = replace(scene, deployment=replace(scene.deployment, coordinates=regions))
                env = TrainingEnv(configuration, record_logs=False)
                self.addCleanup(env.close)
                env.reset()
                with self.assertRaises(ValueError):
                    env.agent_manager.get_deployment_observation(seed=7)

    def test_invalid_complete_plans_cannot_partially_move_or_start_units(self):
        env = self.environment()
        before = env.reset()
        valid = self.plan(env.agent_manager.get_deployment_observation(seed=7))
        first, second = list(valid)[:2]
        missing = dict(valid)
        del missing[first]
        cases = {
            'not_mapping': [],
            'missing': missing,
            'foreign': {**valid, -1: valid[first]},
            'float_keys': {float(key): value for key, value in valid.items()},
            'bool_key': {**missing, True: valid[first]},
            'nan': {**valid, second: (float('nan'), valid[second][1])},
            'infinite': {**valid, second: (valid[second][0], float('inf'))},
            'short': {**valid, second: valid[second][:1]},
            'nested': {**valid, second: [valid[second]]},
            'bounds': {**valid, second: (valid[second][0] + 1e9, valid[second][1])},
            'extra_altitude': {**valid, second: (*valid[second], 3000)},
        }
        factory = env.engine.simulator_factory
        for name, plan in cases.items():
            with self.subTest(invalid=name), \
                 patch.object(factory, 'modify_simulator_position') as move, \
                 patch.object(factory, 'modify_simulator_speed') as speed, \
                 patch.object(factory, 'start_red_missions') as start:
                with self.assertRaises(ValueError):
                    env.agent_manager.apply_deployment(plan)
                move.assert_not_called()
                speed.assert_not_called()
                start.assert_not_called()
                after = env._get_observation()
                for identifier, entity in before['entities'].items():
                    for field in ('pos', 'vel', 'stage', 'health', 'isVisible'):
                        self.assertEqual(after['entities'][identifier][field], entity[field],
                                         (name, identifier, field))

    def test_valid_mapping_starts_once_without_agents_and_returns_ready_state(self):
        env = self.environment()
        env.reset()
        plan = self.plan(env.agent_manager.get_deployment_observation(seed=7))
        original = deepcopy(plan)
        integral_plan = MappingProxyType({np.int64(key): np.array(value) for key, value in plan.items()})
        factory = env.engine.simulator_factory
        with patch.object(factory, 'start_red_missions', wraps=factory.start_red_missions) as start, \
             patch.object(factory, 'modify_simulator_speed') as speed:
            self.assertIsNone(env.agent_manager.apply_deployment(integral_plan))
            ready = env.get_observation()
            start.assert_called_once_with()
            speed.assert_not_called()
            self.assertEqual(ready['step'], 0)
            self.assertEqual(env.agent_manager.get_agent_count(), 0)
            for identifier, position in plan.items():
                for axis, expected in zip(('x', 'y'), position):
                    self.assertAlmostEqual(ready['entities'][identifier]['pos'][axis], expected, delta=1e-6)
            with self.assertRaises(RuntimeError):
                env.agent_manager.apply_deployment(plan)
            with self.assertRaises(RuntimeError):
                env.agent_manager.get_deployment_observation()
            start.assert_called_once_with()
        self.assertEqual(plan, original)

    def test_two_dimensional_plan_preserves_each_units_existing_altitude(self):
        data = json.loads(E01.read_text(encoding='utf-8'))
        heights = {21000: 1200, 21001: 2300, 21002: 3400}
        for row in data['imagineProfile']['entityList']:
            entity = row['entity']
            if entity['entityType'] in heights:
                entity['pos']['z'] = heights[entity['entityType']]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mixed-heights.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            scene = load_scenario(path)
        self.assertIsNone(scene.deployment.launch_altitude)
        env = TrainingEnv(scene, record_logs=False)
        self.addCleanup(env.close)
        env.reset()
        factory = env.engine.simulator_factory
        observation = env.agent_manager.get_deployment_observation(seed=7)
        original_heights = {identifier: factory.get_simulator_by_id(identifier).entity_ext.entity.posEcf.z
                            for identifier in observation['entities']}
        plan = self.plan(observation)
        with patch.object(factory, 'modify_simulator_position', wraps=factory.modify_simulator_position) as move, \
             patch.object(factory, 'start_red_missions', wraps=factory.start_red_missions) as start:
            env.agent_manager.apply_deployment(plan)
        self.assertEqual(move.call_count, len(plan))
        for arguments in move.call_args_list:
            identifier, position = arguments.args
            self.assertEqual(position.z, original_heights[identifier])
        ready = env.get_observation()
        self.assertEqual({identifier: ready['entities'][identifier]['pos']['z']
                          for identifier in plan}, original_heights)
        start.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
