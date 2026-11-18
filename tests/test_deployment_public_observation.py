from copy import deepcopy
from pathlib import Path
import unittest

from envengine.environment.training_env import TrainingEnv
from scenarios import load_scenario
from user_agents.red_checker import check_deployment, check_red_agent


ROOT = Path(__file__).resolve().parents[1]
E01 = ROOT / 'scenarios/E01.json'
ENTRY = ROOT / 'user_agents/example/assignment/r0_random.py'


class DeploymentPublicObservationTests(unittest.TestCase):
    def environment(self):
        env = TrainingEnv(load_scenario(E01), record_logs=False)
        self.addCleanup(env.close)
        env.reset()
        return env

    def test_real_scene_exposes_public_targets_and_deployment_region_without_hidden_entities(self):
        env = self.environment()
        observation = env.agent_manager.get_deployment_observation(seed=42)
        public = observation['public_targets']
        factory = env.engine.simulator_factory
        expected = {
            simulator.entity_ext.entity.id: simulator.entity_ext.entity
            for kind in (9400, 9600)
            for simulator in factory.get_simulators_by_type(kind)
        }
        hidden_ids = {simulator.entity_ext.entity.id
                      for simulator in factory.get_simulators_by_type(9500)}
        self.assertEqual(len(expected), 11)
        self.assertEqual(len(observation['entities']), 126)
        self.assertEqual(set(public), set(expected))
        self.assertTrue(hidden_ids.isdisjoint(public))
        for identifier, target in public.items():
            source = expected[identifier]
            self.assertEqual(set(target), {'nameChn', 'pos', 'health', 'type', 'side'})
            self.assertEqual(target['pos'], {
                'x': source.posEcf.x, 'y': source.posEcf.y, 'z': source.posEcf.z,
            })
            self.assertEqual(target['health'], source.survivePoints)
            self.assertEqual(target['type'], source.entityType)
        self.assertEqual(observation['coordinates'], env.engine.profile.imagineProfile.redArea.coordinates)

    def test_deployment_observation_mutation_cannot_change_engine_or_next_observation(self):
        env = self.environment()
        manager = env.agent_manager
        observation = manager.get_deployment_observation(seed=42)
        expected = deepcopy(observation)
        identifier = next(iter(observation['public_targets']))
        observation['public_targets'][identifier]['pos']['x'] += 100
        observation['public_targets'][identifier]['health'] = 0
        observation['coordinates'][0][0] += 100
        again = manager.get_deployment_observation(seed=42)
        self.assertEqual(again, expected)
        entity = env.engine.simulator_factory.get_simulator_by_id(identifier).entity_ext.entity
        self.assertEqual(entity.posEcf.x, expected['public_targets'][identifier]['pos']['x'])
        self.assertEqual(entity.survivePoints, expected['public_targets'][identifier]['health'])

    def test_same_public_directory_reaches_real_combat_agent_initialization(self):
        env = self.environment()
        manager = env.agent_manager
        observation = manager.get_deployment_observation(seed=42)
        deployment = check_deployment(ENTRY)()
        self.addCleanup(deployment.close)
        manager.apply_deployment(deployment.deploy(observation))
        group = manager.register_user_agent(check_red_agent(ENTRY), seed=42)
        self.addCleanup(group.close)
        self.assertNotIn('run_id', observation)
        run_ids = {agent.init_observation['run_id'] for agent in group.agents}
        self.assertEqual(len(run_ids), 1)
        run_id = next(iter(run_ids))
        self.assertIsInstance(run_id, str)
        self.assertTrue(run_id)
        area = env.engine.profile.imagineProfile.mapArea
        for agent in group.agents:
            self.assertEqual(agent.init_observation['entities'], observation['public_targets'])
            self.assertEqual(agent.init_observation['map_area'], {
                key: getattr(area, key) for key in ('xMin', 'xMax', 'yMin', 'yMax')
            })

    def test_reset_before_registration_changes_run_id_while_observation_reads_keep_it(self):
        env = self.environment()
        manager = env.agent_manager
        first = manager._run_id
        self.assertIsInstance(first, str)
        self.assertTrue(first)
        manager.get_deployment_observation(seed=42)
        self.assertEqual(manager._run_id, first)
        env.reset()
        second = manager._run_id
        self.assertIsInstance(second, str)
        self.assertTrue(second)
        self.assertNotEqual(second, first)
        manager.get_deployment_observation(seed=42)
        self.assertEqual(manager._run_id, second)

    def test_training_reset_preserves_the_registered_groups_run_id(self):
        env = self.environment()
        manager = env.agent_manager
        observation = manager.get_deployment_observation(seed=42)
        deployment = check_deployment(ENTRY)()
        self.addCleanup(deployment.close)
        manager.apply_deployment(deployment.deploy(observation))
        group = manager.register_user_agent(check_red_agent(ENTRY), seed=42)
        self.addCleanup(group.close)
        run_ids = {agent.init_observation['run_id'] for agent in group.agents}
        env.reset()
        self.assertEqual({manager._run_id}, run_ids)
        next_observation = manager.get_deployment_observation(seed=42)
        manager.apply_deployment(deployment.deploy(next_observation))
        self.assertEqual({agent.init_observation['run_id'] for agent in group.agents}, run_ids)

    def test_two_real_environments_have_independent_run_ids_and_public_directories(self):
        first, second = self.environment(), self.environment()
        left = first.agent_manager.get_deployment_observation(seed=42)
        right = second.agent_manager.get_deployment_observation(seed=42)
        identifier = next(iter(left['public_targets']))
        expected = deepcopy(right)
        left['public_targets'][identifier]['pos']['x'] += 100
        left['public_targets'][identifier]['health'] = 0
        self.assertEqual(second.agent_manager.get_deployment_observation(seed=42), expected)

        groups = []
        for env in (first, second):
            deployment = check_deployment(ENTRY)()
            self.addCleanup(deployment.close)
            observation = env.agent_manager.get_deployment_observation(seed=42)
            env.agent_manager.apply_deployment(deployment.deploy(observation))
            group = env.agent_manager.register_user_agent(check_red_agent(ENTRY), seed=42)
            self.addCleanup(group.close)
            groups.append(group)
        first_ids, second_ids = ({agent.init_observation['run_id'] for agent in group.agents}
                                 for group in groups)
        self.assertTrue(first_ids.isdisjoint(second_ids))
        self.assertIsNot(groups[0].agents[0]._resource, groups[1].agents[0]._resource)


if __name__ == '__main__':
    unittest.main()
