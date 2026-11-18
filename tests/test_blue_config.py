"""Blue-owned defaults and explicit algorithm configuration."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

from envengine.common.BaseStruct import Vector3d
from blue_defense import BlueDefenseConfig, build_blue_policy, create_blue_config


ROOT = Path(__file__).resolve().parents[1]


class BlueConfigTests(unittest.TestCase):
    def test_runtime_factory_keeps_algorithm_defaults_and_copies_scoring_weights(self):
        weights = {101: 5.0, 102: 2.0}
        config = create_blue_config(seed=17, asset_values=weights)
        self.assertEqual(config, BlueDefenseConfig(seed=17, asset_values=weights))
        self.assertEqual(create_blue_config(), BlueDefenseConfig())
        self.assertIsNot(config.asset_values, weights)
        weights[101] = 999
        self.assertEqual(config.asset_values[101], 5.0)
        self.assertEqual(config.seed, 17)

    def test_policy_wave_defaults_are_resolved_against_the_scenario_clock(self):
        for policy in ("b8_threat_multi_wave", "b9_min_cost_planned_multi_wave"):
            with self.subTest(policy=policy):
                config = create_blue_config(policy_name=policy, seed=7)
                self.assertIsNone(config.wave_gap_step)
                self.assertIsNone(build_blue_policy(config).wave_gap)

    def test_explicit_algorithm_intervals_use_steps(self):
        config = BlueDefenseConfig(
            policy_name="b9_min_cost_planned_multi_wave", seed=11,
            max_shots_per_target=4, wave_size=3, reset_rng_on_episode=True,
            min_cost_decision_interval_step=250, joint_decision_interval_step=7500,
            planning_delay_step=2000, wave_gap_step=3250,
        )
        policy = build_blue_policy(config)
        self.assertEqual((policy.planning_delay, policy.wave_gap, policy.wave_size),
                         (2000, 3250, 3))
        self.assertEqual(config.min_cost_decision_interval_step, 250)
        self.assertEqual(config.joint_decision_interval_step, 7500)
        self.assertTrue(config.reset_rng_on_episode)

    def test_zero_delay_preserves_other_defaults(self):
        config = BlueDefenseConfig(max_shots_per_target=3, planning_delay_step=0)
        self.assertEqual(config.planning_delay_step, 0)
        self.assertEqual(config.joint_decision_interval_step, 100)
        self.assertFalse(config.reset_rng_on_episode)

    def test_main_help_remains_dependency_light(self):
        script = """
            import sys
            import main
            for name in ('numpy', 'torch', 'scipy', 'pygame', 'envengine'):
                assert name not in sys.modules, name
            main.main(['--help'])
        """
        result = subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('--config', result.stdout)


if __name__ == "__main__":
    unittest.main()
