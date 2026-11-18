from pathlib import Path
import unittest
from uuid import uuid4

from simulator import Simulator


class R7InitialPositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.simulator = Simulator({
            'agent': root / 'user_agents/example/coordination/r7_v4_0_search_recovery.py',
            'scenario': root / 'scenarios/E06.json', 'seed': 7,
            'blue_mode': root / 'blue_defense/immediate/b0_fixed_ratio_random.py',
            'render_mode': 'none', 'logging': {'enabled': False, 'example_evaluation': False}})
        try:
            cls.simulator._initialize()
        except BaseException:
            cls.simulator._close(None)
            raise
        cls.agents = cls.simulator._env.agent_manager.get_registered_agents()
        cls.entity_id, cls.agent = cls.agents[0]

    @classmethod
    def tearDownClass(cls):
        cls.simulator._close(None)

    def test_actual_native_positions_register(self):
        plan = self.agent.deployment
        submitted = {int(row[1]): (row[2], row[3]) for row in plan.actions}
        self.assertEqual(set(submitted), {identifier for identifier, _ in self.agents})
        differences = []
        for identifier, agent in self.agents:
            position = agent.init_observation['self']['pos']
            x, y = submitted[identifier]
            differences.append(((position['x'] - x)**2 + (position['y'] - y)**2)**0.5)
        self.assertGreater(max(differences), 100)

    def test_nonfinite_native_position_is_rejected(self):
        for axis in ('x', 'y', 'z'):
            with self.subTest(axis=axis):
                observation = self.invalid_observation()
                observation['self']['pos'][axis] = float('nan')
                with self.assertRaisesRegex(ValueError, '本机初始位置必须是有限数值'):
                    type(self.agent)(self.entity_id, observation)

    def test_position_outside_deployment_is_rejected(self):
        observation = self.invalid_observation()
        observation['self']['pos']['x'] = observation['map_area']['xMax'] + 1
        with self.assertRaisesRegex(ValueError, '超出合法部署区域'):
            type(self.agent)(self.entity_id, observation)

    def invalid_observation(self):
        original = self.agent.init_observation
        return dict(original, run_id=str(uuid4()),
                    self=dict(original['self'], pos=dict(original['self']['pos'])))


if __name__ == '__main__':
    unittest.main()
