import json
from pathlib import Path
import unittest

from simulator import Simulator
from envengine.action_validation import IllegalActionError
from envengine.common import SimmerCommandType
from envengine.sdk.base_struct.Message import Command
from user_agents.example.common.waypoints import bounded_waypoint

ROOT = Path(__file__).resolve().parents[1]


class R7WaypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((ROOT / 'tests/fixtures/h03_waypoint_outside.json').read_text(encoding='utf-8'))
        cls.simulator = Simulator({
            'agent': ROOT / 'user_agents/example/coordination/r7_v4_0_search_recovery.py',
            'scenario': ROOT / 'scenarios/H03.json', 'seed': 7,
            'blue_mode': ROOT / 'blue_defense/immediate/b0_fixed_ratio_random.py',
            'render_mode': 'none',
            'logging': {'enabled': False, 'state_interval_steps': 100, 'example_evaluation': False}})
        try:
            cls.simulator._initialize()
        except BaseException:
            cls.simulator._close(None)
            raise
        cls.factory = cls.simulator._env.engine.simulator_factory

    @classmethod
    def tearDownClass(cls):
        cls.simulator._close(None)

    def command(self, x, y):
        return Command(executorId=self.record['executor_id'],
                       commandTypeId=SimmerCommandType.CHANGE_MISSILE_TARGET,
                       commandAttributes={'intent': 'waypoint', 'target_id': None,
                                          'target': {'x': x, 'y': y, 'z': 0.0}})

    def test_recorded_failure_is_rejected_and_bounded_command_is_accepted(self):
        target, area = self.record['target'], self.record['map_area']
        with self.assertRaisesRegex(IllegalActionError, 'Waypoint lies outside the map'):
            self.factory.validate_red_command(self.command(target['x'], target['y']))
        x, y = bounded_waypoint(target['x'], target['y'], area)
        self.assertEqual(x, target['x'])
        self.assertEqual(y, area['yMax'])
        self.factory.validate_red_command(self.command(x, y))
        self.assertGreater(target['y'], area['yMax'])
        self.assertEqual(bounded_waypoint(x, y, area), (x, y))

    def test_recorded_in_map_position_is_unchanged(self):
        position = self.record['own_position']
        self.assertEqual(bounded_waypoint(position['x'], position['y'], self.record['map_area']),
                         (position['x'], position['y']))

    def test_nonfinite_waypoint_is_rejected(self):
        position = self.record['target']
        for invalid in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaisesRegex(ValueError, '有限数值'):
                bounded_waypoint(invalid, position['y'], self.record['map_area'])
            with self.assertRaisesRegex(ValueError, '有限数值'):
                bounded_waypoint(position['x'], invalid, self.record['map_area'])


if __name__ == '__main__':
    unittest.main()
