from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from red_strategy_lab import GlobalRules


def load_runner_module():
    path = LAB_ROOT / "scripts" / "run_scenario.py"
    specification = importlib.util.spec_from_file_location("run_scenario", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load scenario runner")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class ScenarioContractTests(unittest.TestCase):
    def test_all_baselines_accept_platform_scenario_initial_state(self) -> None:
        runner = load_runner_module()
        scenario = runner.load_scenario(LAB_ROOT.parent / "scenarios" / "platform.json")
        targets = runner.targets_from_scenario(scenario)
        observation = runner.initial_observation_from_scenario(scenario, targets)
        self.assertGreater(len(observation.platforms), 0)
        self.assertGreater(len(targets), 0)
        rules = GlobalRules(target_capacity=2, min_launch_interval=1)
        for name in ("b0", "b1", "b2", "b3"):
            decision = runner.choose_policy(name, rules, seed=20260730).decide(observation)
            self.assertGreater(len(decision.assignments), 0, name)
            counts: dict[int, int] = {}
            for assignment in decision.assignments:
                counts[assignment.target_id] = counts.get(assignment.target_id, 0) + 1
            self.assertTrue(all(count <= rules.target_capacity for count in counts.values()), name)


if __name__ == "__main__":
    unittest.main()

