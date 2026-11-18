import json
from pathlib import Path
import unittest

from envengine import TrainingEnv
from envengine.environment.scene_adapter import profile_from_scene
from envengine.common import DamageC, DamageCmd, SimmerCommandType
from envengine.sdk.base_struct.Message import Command
from envengine.sdk.writer import WriteConfig, init_writer
from example_evaluation import RunSummary
from scenarios import load_scenario
from scenarios.cases import RewardTracker, calculate_reward


ROOT = Path(__file__).resolve().parents[1]


class DestroyedObjectiveRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_scenario('E01')
        cls.raw = json.loads(cls.bundle.path.read_text(encoding='utf-8'))
        cls.writer = init_writer(WriteConfig(
            enable_config=False, enable_state=False, enable_event=False, enable_ai_action=False,
            output_dir=str(ROOT / '.blue_validation/reward'), verbose=False,
        ))

    @classmethod
    def tearDownClass(cls):
        cls.writer.close()

    def setUp(self):
        self.env = TrainingEnv(profile_from_scene(self.raw), render_mode='none')
        self.addCleanup(self.env.close)
        self.initial = self.env.reset()
        self.tracker = RewardTracker(self.bundle.reward_policy)

    def damage(self, entity_id, amount):
        factory = self.env.engine.simulator_factory
        factory._send_command_callback([Command(
            prevTriggerId=entity_id, executorId=entity_id, commandTypeId=SimmerCommandType.DAMAGE,
            commandAttributes=DamageC(DamageCmd(damagePoint=amount)).to_json(),
        )])
        factory.process_commands()

    def observation(self):
        return self.env._get_observation()

    def test_partial_damage_including_near_zero_health_scores_zero(self):
        for entity_id in self.bundle.reward_policy.objective_ids:
            health = self.initial['entities'][entity_id]['health']
            self.damage(entity_id, health - .001)
        observation = self.observation()
        self.assertTrue(all(0 < observation['entities'][key]['health'] < .002
                            for key in self.bundle.reward_policy.objective_ids))
        self.tracker.check_completion(10, observation)
        result = self.tracker.finish(observation)
        self.assertEqual(result.score, 0)
        self.assertEqual(result.destroyed_ids, ())
        self.assertFalse(result.completed)

    def test_each_kind_gets_its_full_weight_only_after_destruction(self):
        expected_weight = 0
        for entity_type, weight in ((9400, 5), (9600, 2), (9500, 1)):
            entity_id = next(key for key in self.bundle.reward_policy.objective_ids
                             if self.initial['entities'][key]['type'] == entity_type)
            health = self.initial['entities'][entity_id]['health']
            self.damage(entity_id, health / 2)
            before = self.tracker.finish(self.observation())
            self.assertEqual(before.destroyed_weight, expected_weight)
            self.damage(entity_id, health)
            expected_weight += weight
            self.tracker.check_completion(20, self.observation())
            result = self.tracker.finish(self.observation())
            self.assertEqual(result.destroyed_weight, expected_weight)
            self.assertAlmostEqual(result.score, 100 * expected_weight / result.total_objective_weight)
            self.assertEqual(result.K, result.raw_score)
            self.assertEqual(result.T, 0)

    def test_all_destroyed_scores_100_and_preserves_first_destruction_step(self):
        for entity_id in self.bundle.reward_policy.objective_ids:
            self.damage(entity_id, self.initial['entities'][entity_id]['health'] + 1)
        observation = self.observation()
        self.tracker.check_completion(30, observation)
        self.tracker.check_completion(40, observation)
        result = self.tracker.finish(observation)
        self.assertEqual(result.score, 100)
        self.assertTrue(result.completed)
        self.assertEqual(dict(result.destruction_steps), {key: 30 for key in self.bundle.reward_policy.objective_ids})
        health = {str(key): observation['entities'][key]['health'] for key in self.bundle.reward_policy.objective_ids}
        late = calculate_reward(policy=self.bundle.reward_policy, target_health=health,
                                destruction_steps={key: 3000 for key in health})
        self.assertEqual(late.score, result.score)

    def test_nonobjective_destruction_adds_no_score(self):
        unit = self.env.engine.get_simulators_by_type(24000)[0].entity_ext.entity
        self.damage(unit.id, unit.survivePoints + 1)
        self.assertEqual(unit.survivePoints, 0)
        self.assertEqual(self.tracker.finish(self.observation()).score, 0)

    def test_all_maps_destroy_carriers_and_their_remaining_inventory(self):
        case_root = ROOT / 'scenarios'
        manifest = json.loads((case_root / 'manifest.json').read_text(encoding='utf-8'))
        for case in manifest['cases']:
            path = case_root / case['scenario']
            with self.subTest(map=path.stem):
                bundle = load_scenario(path)
                env = TrainingEnv(profile_from_scene(json.loads(path.read_text(encoding='utf-8'))),
                                  render_mode='none')
                try:
                    initial = env.reset()
                    factory = env.engine.simulator_factory
                    factory._send_command_callback([
                        Command(prevTriggerId=key, executorId=key, commandTypeId=SimmerCommandType.DAMAGE,
                                commandAttributes=DamageC(DamageCmd(
                                    damagePoint=initial['entities'][key]['health'] + 1)).to_json())
                        for key in bundle.reward_policy.objective_ids
                    ])
                    factory.process_commands()
                    env.engine.step([])
                    observation = env._get_observation()
                    tracker = RewardTracker(bundle.reward_policy)
                    tracker.check_completion(1, observation)
                    result = tracker.finish(observation)
                    self.assertEqual(result.score, 100)
                    self.assertTrue(result.completed)
                    for unit in factory.get_simulators_by_type(24000):
                        self.assertEqual(unit.entity_ext.entity.survivePoints, 0)
                finally:
                    env.close()

    def test_summary_score_matches_destroyed_objectives_with_partial_survivors(self):
        collector = RunSummary(scenario='E01', policies={}, reward_tracker=self.tracker)
        collector.start(self.initial)
        first, second = self.bundle.reward_policy.objective_ids[:2]
        self.damage(first, self.initial['entities'][first]['health'])
        self.damage(second, self.initial['entities'][second]['health'] / 2)
        observation = self.observation()
        collector.update(15, observation)
        summary = collector.build(observation, termination_reason='test', red_launched=0)
        expected = 100 * sum(row['weight'] for row in summary['objectives'] if row['final_health'] <= 0) / sum(
            row['weight'] for row in summary['objectives'])
        self.assertAlmostEqual(summary['score']['score'], expected)
        self.assertEqual(summary['score']['destroyed_ids'], [first])
        self.assertEqual(self.tracker.finish(self.env.reset()).score, 0)


if __name__ == '__main__':
    unittest.main()
