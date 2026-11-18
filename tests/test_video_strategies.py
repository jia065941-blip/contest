from collections import Counter
import json
from pathlib import Path
import unittest

from envengine.environment.training_env import TrainingEnv
from blue_defense.blue_checker import check_blue_policy
from envengine.sdk.writer import WriteConfig, init_writer
from scenarios import load_scenario
from user_agents.red_checker import check_deployment, check_red_agent
from user_agents.example.spatial.geometry import public_geometry
from user_agents.example.spatial import runtime


ROOT = Path(__file__).resolve().parents[1]
CASES = (('M01', 'r7_v3_0_spatial_packages', 'b9_min_cost_planned_multi_wave'),
         ('E06', 'r7_v4_0_search_recovery', 'b8_threat_multi_wave'))


class VideoStrategyTests(unittest.TestCase):
    def test_public_geometry_matches_all_twenty_scenes(self):
        rows = json.loads((ROOT / 'user_agents/example/spatial/public_geometry.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(len(rows), 20)
        for row in rows:
            with self.subTest(map=row['id']):
                scene = load_scenario(ROOT / 'scenarios' / (row['id'] + '.json'))
                bounds, duration = public_geometry([list(point) for point in scene.deployment.coordinates[0]])
                self.assertEqual(bounds, {key: getattr(scene.profile.imagineProfile.mapArea, key)
                                         for key in ('xMin', 'xMax', 'yMin', 'yMax')})
                self.assertEqual(duration, scene.profile.imagineProfile.simStep)

    def test_real_deployment_and_twenty_steps_for_both_video_strategies(self):
        writer = init_writer(WriteConfig(enable_config=False, enable_state=False,
                                        enable_event=False, enable_ai_action=False, verbose=False))
        self.addCleanup(writer.close)
        searches = {}
        for map_id, red_name, blue_name in CASES:
            with self.subTest(map=map_id, strategy=red_name):
                scene = load_scenario(ROOT / 'scenarios' / (map_id + '.json'))
                blue = check_blue_policy(ROOT / 'blue_defense/scheduled' / (blue_name + '.py'))(
                    seed=42, asset_values=scene.asset_values)
                env = TrainingEnv(scene, record_logs=False)
                try:
                    env.register_blue_policy(blue)
                    env.reset()
                    manager = env.agent_manager
                    red_path = ROOT / 'user_agents/example/coordination' / (red_name + '.py')
                    deployment = check_deployment(red_path)()
                    observation = manager.get_deployment_observation(seed=42)
                    positions = deployment.deploy(observation)
                    self.assertEqual(set(positions), set(observation['entities']))
                    self.assertEqual(len(set(positions.values())), len(positions))
                    public_ids = set(observation['public_targets'])
                    self.assertTrue({entry['target_id'] for entry in deployment.deployment.assignments.values()
                                     if entry['target_id'] is not None} <= public_ids)
                    roles = Counter(entry['role'] for entry in deployment.deployment.assignments.values())
                    searches[red_name] = roles['search']
                    manager.apply_deployment(positions)
                    # 正式入口先关闭部署对象，再创建作战实例。
                    deployment.close()
                    group = manager.register_user_agent(check_red_agent(red_path), seed=42)
                    group.reset()
                    for _ in range(20):
                        _, _, done, _ = env.step()
                        self.assertFalse(done)
                    for agent in group.agents:
                        self.assertEqual(len(agent.observation_history), 0)
                        self.assertIsNotNone(agent._commander)
                        self.assertTrue(agent._commander.current_snapshot_ids <= agent._commander.expected_platform_ids)
                        self.assertTrue({target.entity_id for target in agent._commander.targets
                                         if target.entity_type == 9500} <= {
                                             identifier for identifiers in agent._commander.observed_target_ids.values()
                                             for identifier in identifiers})
                    # 同一真实观测重复决策时，目标指令不重复发送。
                    for agent in group.agents:
                        actions = agent.get_action()
                        self.assertTrue(all(int(row[0]) in (1, 2, 3) and int(row[1]) == agent.entity_id
                                            for row in actions))
                    group.close()
                    self.assertFalse(runtime._SESSIONS)
                    self.assertFalse(runtime._PLANS)
                finally:
                    env.close()
                    blue.close()
        self.assertGreater(searches['r7_v4_0_search_recovery'], searches['r7_v3_0_spatial_packages'])


if __name__ == '__main__':
    unittest.main()
