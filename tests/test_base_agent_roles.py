"""Entity-bound red roles with local observations and optional lifecycle hooks."""

import inspect
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

from user_agents.base import BaseAgent


ROOT = Path(__file__).resolve().parents[1]


class BaseAgentRoleTests(unittest.TestCase):
    def test_only_action_is_abstract_and_construction_uses_entity_identity(self):
        self.assertEqual(list(inspect.signature(BaseAgent).parameters),
                         ["entity_id", "init_observation", "history_length"])
        history = inspect.signature(BaseAgent).parameters["history_length"]
        self.assertEqual(history.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(history.default, 10)
        self.assertEqual(BaseAgent.__abstractmethods__, {"get_action"})
        import user_agents.base as public_base
        self.assertFalse(hasattr(public_base, 'BaseRedAttack'))
        self.assertFalse(hasattr(public_base, 'AgentType'))
        self.assertFalse(hasattr(BaseAgent, "create_shared"))
        self.assertFalse(hasattr(BaseAgent, "deploy"))
        with self.assertRaises(TypeError):
            BaseAgent(11, {})

    def test_default_observation_and_optional_feedback(self):
        class Minimal(BaseAgent):
            def get_action(self):
                return []

        agent = Minimal(11, {})
        self.assertEqual(agent.entity_id, 11)
        self.assertFalse(hasattr(agent, "agent_id"))
        self.assertFalse(hasattr(agent, "agent_type"))
        self.assertIsNone(agent.latest_observation)
        local = {"step": 1, "self": {"pos": {"x": 118.0, "y": 22.0, "z": 3000.0}}}
        agent.get_obs(local)
        self.assertIs(agent.latest_observation, local)
        self.assertEqual(agent.get_action(), [])
        agent.record_step({"step": 2}, [], 1.5)
        self.assertEqual(agent.get_recent_rewards(), [1.5])
        agent.reset()
        self.assertEqual(agent.get_recent_rewards(), [])
        self.assertIsNone(agent.latest_observation)

    def test_custom_observation_method_can_extend_default_lifecycle(self):
        events = []

        class Entrant(BaseAgent):
            def get_obs(self, observation):
                super().get_obs(observation)
                events.append(("observation", self.latest_observation["step"]))

            def get_action(self):
                events.append(("decide", self.latest_observation["step"]))
                return [[1, self.entity_id, 118.0, 22.0]]

            def on_feedback(self, observation, action, reward, info=None):
                events.append(("feedback", self.get_recent_rewards(), self.get_recent_observations()))

            def on_reset(self):
                events.append(("reset", self.latest_observation, self.get_recent_rewards()))

        agent = Entrant(11, {})
        agent.get_obs({"step": 3})
        actions = agent.get_action()
        agent.record_step({"step": 4}, actions, 2.5, {"done": False})
        agent.reset()
        self.assertEqual(events, [
            ("observation", 3), ("decide", 3),
            ("feedback", [2.5], [{"step": 4}]), ("reset", None, []),
        ])
        self.assertEqual(agent.get_recent_actions(), [])
        self.assertEqual(list(agent.info_history), [])

    def test_history_length_limits_local_records(self):
        class Legacy(BaseAgent):
            def get_obs(self, observation):
                self.observation = observation

            def get_action(self):
                return []

            def reset(self):
                super().reset()

        agent = Legacy(11, {}, history_length=2)
        for step in (1, 2, 3):
            agent.record_step({'step': step}, [], float(step))
        self.assertEqual(agent.get_recent_rewards(), [2.0, 3.0])
        self.assertEqual(agent.get_recent_observations(), [{'step': 2}, {'step': 3}])
        agent.reset()
        self.assertEqual(agent.get_recent_rewards(), [])

    def test_base_and_yaml_help_do_not_load_examples_or_numerical_dependencies(self):
        script = """
            import importlib.abc
            from contextlib import redirect_stdout
            from io import StringIO
            import sys

            class NoExamples(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'user_agents.example' or fullname.startswith('user_agents.example.'):
                        raise AssertionError('Base attempted an example import: ' + fullname)

            sys.meta_path.insert(0, NoExamples())
            from user_agents.base import BaseAgent
            import main
            help_text = StringIO()
            with redirect_stdout(help_text):
                try:
                    main.main(['--help'])
                except SystemExit as error:
                    assert error.code == 0
            assert '--config' in help_text.getvalue()
            class Custom(BaseAgent):
                def get_action(self):
                    return []
            assert not Custom.__abstractmethods__
            for name in ('numpy', 'torch', 'scipy', 'pygame', 'envengine'):
                assert name not in sys.modules, name
        """
        result = subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
