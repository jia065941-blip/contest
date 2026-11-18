"""Red file loading, per-entity ownership and run-local resource lifetimes."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from uuid import uuid4
from unittest.mock import Mock, patch

import yaml

from envengine.action_validation import IllegalActionError
from envengine.agent_manager import AgentManager
from envengine.environment.training_env import TrainingEnv
from scenarios import load_scenario
from scenarios.cases import RewardTracker
from simulator import Simulator
from user_agents.base import BaseAgent
from user_agents.red_checker import check_deployment, check_red_agent


ROOT = Path(__file__).resolve().parents[1]
E01 = ROOT / 'scenarios/E01.json'
DEFAULT_RED = ROOT / 'user_agents/example/assignment/r0_random.py'
CUSTOM = """
from user_agents.base import BaseAgent

class RedAttack(BaseAgent):
    def get_obs(self, observation):
        super().get_obs(observation)

    def get_action(self):
        return []
"""


class MinimalRed(BaseAgent):
    def __init__(self, entity_id, init_observation):
        super().__init__(entity_id, init_observation)
        self.closed = 0
        self.resets = 0

    def get_obs(self, observation):
        super().get_obs(observation)

    def get_action(self):
        return []

    def on_reset(self):
        self.resets += 1

    def close(self):
        self.closed += 1


class MisleadingHookRed(MinimalRed):
    @classmethod
    def create_shared(cls, *args):
        raise AssertionError("The framework must ignore this unrelated user method")


class RedPluginTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def write_policy(self, source=CUSTOM, name='policy.py'):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding='utf-8')
        return path

    def test_checker_loads_once_without_constructing_agents_or_calling_user_hooks(self):
        path = self.write_policy(r"""
            from pathlib import Path
            from user_agents.base import BaseAgent
            marker = Path(__file__).with_suffix('.loads')
            marker.write_text(marker.read_text() + 'load\n' if marker.exists() else 'load\n')

            class RedAttack(BaseAgent):
                def __init__(self, entity_id, init_observation):
                    raise AssertionError('checker must not construct an entity agent')
                @classmethod
                def create_shared(cls, initial):
                    raise AssertionError('checker must not construct shared resources')
                def get_obs(self, observation):
                    raise AssertionError('checker must not deliver observations')
                def get_action(self):
                    raise AssertionError('checker must not decide')
        """)
        implementation = check_red_agent(path)
        self.assertIs(check_red_agent(path), implementation)
        self.assertTrue(issubclass(implementation, BaseAgent))
        self.assertEqual(path.with_suffix('.loads').read_text(), 'load\n')

    def test_invalid_export_roles_and_signatures_are_rejected(self):
        sources = {
            'missing': 'value = 1',
            'unrelated': 'class RedAttack: pass',
            'abstract': 'from user_agents.base import BaseAgent\nclass RedAttack(BaseAgent): pass',
            'instance': CUSTOM + '\nRedAttack = RedAttack(11, {})',
            'async_action': CUSTOM.replace('def get_action(', 'async def get_action('),
            'bad_observation_signature': CUSTOM.replace('def get_obs(self, observation):', 'def get_obs(self):'),
            'generator': CUSTOM.replace('return []', 'yield []'),
            'constructor': CUSTOM.replace('class RedAttack(BaseAgent):',
                                          'class RedAttack(BaseAgent):\n    def __init__(self, required): pass'),
        }
        for name, source in sources.items():
            with self.subTest(invalid=name), self.assertRaises(ValueError):
                check_red_agent(self.write_policy(source, name + '.py'))

    def test_same_named_files_are_isolated_and_failed_imports_can_be_retried(self):
        first = check_red_agent(self.write_policy(CUSTOM, 'first/policy.py'))
        second = check_red_agent(self.write_policy(CUSTOM, 'second/policy.py'))
        self.assertIsNot(first, second)
        self.assertNotEqual(first.__module__, second.__module__)
        path = self.write_policy("raise RuntimeError('broken red import')\n", 'retry.py')
        with self.assertRaisesRegex(ValueError, 'broken red import'):
            check_red_agent(path)
        self.assertFalse(any(getattr(module, '__file__', None) == str(path)
                             for module in tuple(sys.modules.values()) if module is not None))
        self.write_policy(CUSTOM, 'retry.py')
        self.assertTrue(issubclass(check_red_agent(path), BaseAgent))
        for invalid in (self.directory / 'missing.py', self.write_policy(CUSTOM, 'policy.txt')):
            with self.subTest(path=invalid), self.assertRaises(ValueError):
                check_red_agent(invalid)

    def test_builtin_entry_keeps_package_imports_and_public_base_identity(self):
        implementation = check_red_agent(DEFAULT_RED)
        self.assertTrue(issubclass(implementation, BaseAgent))
        self.assertEqual(implementation.__module__, 'user_agents.example.assignment.r0_random')

    def test_custom_file_check_does_not_import_any_default_baseline_or_native_engine(self):
        path = self.write_policy()
        script = """
            import importlib.abc
            import sys
            class NoExamples(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if any(fullname == name or fullname.startswith(name + '.') for name in (
                        'user_agents.example',
                    )):
                        raise AssertionError('Custom plugin imported an example: ' + fullname)
            sys.meta_path.insert(0, NoExamples())
            from user_agents.red_checker import check_red_agent
            from user_agents.base import BaseAgent
            assert issubclass(check_red_agent(sys.argv[1]), BaseAgent)
            for name in ('numpy', 'torch', 'scipy', 'pygame', 'envengine'):
                assert name not in sys.modules, name
        """
        result = subprocess.run([sys.executable, '-c', textwrap.dedent(script), str(path)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class RedGroupTests(unittest.TestCase):
    def initial(self):
        run_id = uuid4().hex
        return {entity_id: {
            'run_id': run_id, 'entity_id': entity_id,
            'self': {'id': entity_id, 'type': 21000, 'health': 100},
            'seed': 7, 'entities': {101: {'type': 9400, 'health': 100}},
        } for entity_id in (11, 22)}

    def test_framework_owns_only_agent_instances_and_ignores_user_shared_hook(self):
        managers = [AgentManager(), AgentManager()]
        initials = [self.initial(), self.initial()]
        groups = [manager.register_red_agents(MisleadingHookRed, (11, 22), initial)
                  for manager, initial in zip(managers, initials)]
        for group in groups:
            self.addCleanup(group.close)
            self.assertFalse(hasattr(group, 'shared'))
        first, second = groups
        self.assertIsNot(first.agents[0], second.agents[0])
        first.agents[0].init_observation['entities'][101]['health'] = 0
        self.assertEqual(first.agents[1].init_observation['entities'][101]['health'], 100)
        self.assertEqual(initials[0][11]['entities'][101]['health'], 100)
        managers[0].reset_all()
        self.assertEqual([agent.resets for agent in first.agents], [1, 1])
        self.assertEqual([agent.resets for agent in second.agents], [0, 0])
        first.close()
        first.close()
        self.assertEqual([agent.closed for agent in first.agents], [1, 1])
        self.assertEqual([agent.closed for agent in second.agents], [0, 0])

    def test_partial_registration_failure_closes_agents_and_keeps_existing_registration(self):
        manager = AgentManager()
        existing = MinimalRed(999, {})
        manager.register_agent(existing)
        created = []

        class Failing(MisleadingHookRed):
            def __init__(self, entity_id, initial):
                if created:
                    raise RuntimeError('second constructor failed')
                super().__init__(entity_id, initial)
                created.append(self)

        with self.assertRaisesRegex(RuntimeError, 'second constructor failed'):
            manager.register_red_agents(Failing, (11, 22), self.initial())
        self.assertEqual(manager.get_all_agents(), [existing])
        self.assertIs(manager.get_agent_by_entity(999), existing)
        self.assertIsNone(manager.get_agent_by_entity(11))
        self.assertEqual(created[0].closed, 1)
        group = manager.register_red_agents(MisleadingHookRed, (11, 22), self.initial())
        self.addCleanup(group.close)
        self.assertEqual(len(group.agents), 2)

    def test_manager_delivers_every_local_observation_before_deciding_and_enforces_ownership(self):
        events = []

        class Ordered(MinimalRed):
            def get_obs(self, observation):
                super().get_obs(observation)
                events.append(('observe', self.entity_id))
                observation['self']['health'] = -1

            def get_action(self):
                events.append(('decide', self.entity_id))
                return [[3, self.entity_id, 0, 0]]

        manager = AgentManager()
        agents = [Ordered(11, {}), Ordered(22, {})]
        for agent in agents:
            manager.register_agent(agent)
        observations = {11: {'step': 1, 'self': {'health': 100}},
                        22: {'step': 1, 'self': {'health': 100}}}
        manager.collect_actions_from_agents(observations)
        self.assertEqual(events, [('observe', 11), ('observe', 22), ('decide', 11), ('decide', 22)])
        self.assertEqual([value['self']['health'] for value in observations.values()], [100, 100])
        agents[0].entity_id = 22
        self.assertIs(dict(manager.get_registered_agents())[11], agents[0])
        isolated = manager.extract_observation_for_agent({
            'step': 1,
            'entities': {11: {
                'nameChn': 'unit-11', 'type': 21000, 'side': 0, 'stage': 3,
                'health': 100, 'isVisible': True, 'pos': {'x': 1000, 'y': 2000, 'z': 100},
                'vel': {'x': 100, 'y': 0, 'z': 0}, 'detectInfo': {}, 'commRangeInfo': {},
            }},
        }, 11)
        self.assertEqual(isolated['entity_id'], 11)
        self.assertNotIn('agent_id', isolated)
        agents[0].get_action = lambda: [[3, 22, 0, 0]]
        with self.assertRaises(IllegalActionError):
            manager.collect_actions_from_agents(observations)


class RedEnvironmentTests(unittest.TestCase):
    def test_actor_registration_follows_deployment_and_receives_no_team_plan(self):
        env = TrainingEnv(load_scenario(E01), record_logs=False)
        self.addCleanup(env.close)
        with self.assertRaises(RuntimeError):
            env.agent_manager.register_user_agent(MinimalRed, seed=7)
        env.reset()
        with self.assertRaises(RuntimeError):
            env.agent_manager.register_user_agent(MinimalRed, seed=7)
        deployment = check_deployment(DEFAULT_RED)()
        self.addCleanup(deployment.close)
        with patch.object(deployment, 'deploy', wraps=deployment.deploy) as plan:
            self.assertIsNone(env.agent_manager.apply_deployment(deployment.deploy(env.agent_manager.get_deployment_observation(seed=7))))
            ready = env.get_observation()
            group = env.agent_manager.register_user_agent(MinimalRed, seed=7)
            self.addCleanup(group.close)
            run_ids = {agent.init_observation['run_id'] for agent in group.agents}
            self.assertEqual(len(run_ids), 1)
            self.assertTrue(all(isinstance(value, str) and value for value in run_ids))
            for agent in group.agents:
                initial = agent.init_observation
                self.assertFalse({'deployment', 'plan', 'agent_setup', 'platforms',
                                  'controlled_entities', 'shared', 'agents'} & initial.keys())
                self.assertNotIn('agent_id', initial)
                self.assertFalse(hasattr(agent, 'agent_id'))
                self.assertEqual(initial['entity_id'], agent.entity_id)
                self.assertEqual(initial['self']['pos'], ready['entities'][agent.entity_id]['pos'])
                self.assertEqual(initial['self']['vel'], ready['entities'][agent.entity_id]['vel'])
                self.assertEqual(set(initial['self']['pos']), {'x', 'y', 'z'})
                self.assertEqual(set(initial['map_area']), {'xMin', 'xMax', 'yMin', 'yMax'})
                self.assertTrue(all(value['type'] in (9400, 9600) for value in initial['entities'].values()))
                self.assertFalse(hasattr(agent, 'deploy'))
            group.reset()
            env._generate_actions_from_agents()
            plan.assert_called_once()
        env.close()
        self.assertTrue(all(agent.closed == 0 for agent in group.agents))
        group.close()
        self.assertTrue(all(agent.closed == 1 for agent in group.agents))


    def test_cli_runs_a_short_scene_with_default_process_logging_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            scene = json.loads(E01.read_text(encoding='utf-8'))
            scene['imagineProfile']['simEndLogicTime'] = 2 * scene['imagineProfile']['simStep']
            scene_path = directory / 'scenario.json'
            scene_path.write_text(json.dumps(scene), encoding='utf-8')
            config = yaml.safe_load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
            config.update(agent=str(DEFAULT_RED), scenario=str(scene_path), render_mode='none',
                          blue_mode=str(ROOT / config['blue_mode']), output_dir=str(directory / 'output'))
            path = directory / 'run.yaml'
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            result = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--config', str(path)],
                                    cwd=directory, text=True, encoding='utf-8', capture_output=True, timeout=30,
                                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('共推进 2 步', result.stdout)
            self.assertNotIn('entities', result.stdout)
            self.assertNotIn('[训练环境]', result.stdout)
            self.assertEqual(result.stderr, '')
            self.assertFalse(Path(config['output_dir']).exists())


if __name__ == '__main__':
    unittest.main()
