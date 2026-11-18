from pathlib import Path
import unittest

from simulator import Simulator
from user_agents.example.spatial.strike_coordination import StrikeCoordination


class R7CommitmentNativeTests(unittest.TestCase):
    def test_real_reports_merge_by_latest_step_and_startup_does_not_consume_satellite(self):
        root = Path(__file__).resolve().parents[1]
        sim = Simulator({
            'agent': root / 'user_agents/example/coordination/r7_v3_0_spatial_packages.py',
            'scenario': root / 'scenarios/E06.json', 'seed': 7,
            'blue_mode': root / 'blue_defense/immediate/b0_fixed_ratio_random.py',
            'render_mode': 'none', 'logging': {'enabled': False, 'example_evaluation': False},
        })
        snapshots = []
        try:
            sim._initialize()
            for _ in range(20):
                sim._step()
                agent = next(iter(dict(sim._env.agent_manager.get_registered_agents()).values()))
                commander = agent._commander
                task = commander.task_for(agent.entity_id)
                self.assertTrue(task.strike_reports)
                self.assertFalse(agent.sat_used)
                snapshots.append(task.strike_reports)
            deployment = dict(agent.deployment.report, assignments=agent.deployment.assignments)
            forward = StrikeCoordination(deployment, commander.initial_targets)
            backward = StrikeCoordination(deployment, commander.initial_targets)
            for reports in snapshots:
                forward.restore_reports(reports)
            for reports in reversed(snapshots):
                backward.restore_reports(reports)
            backward.restore_reports(snapshots[0])
            self.assertEqual(forward.strike_reports, backward.strike_reports)
            self.assertFalse(forward.accounted_losses)
            self.assertFalse(backward.accounted_losses)
        finally:
            sim._close(None)


if __name__ == '__main__':
    unittest.main()
