from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


CORE_ROOT = Path(__file__).resolve().parents[1] / "core"


from policies.red.learning.unified_mappo_policy import UnifiedMAPPOSharedPolicy

def _load_deploy_agent():
    class _BaseAgent:
        def __init__(self, agent_id, entity_id, agent_type, init_observation):
            self.agent_id = agent_id
            self.entity_id = entity_id
            self.agent_type = agent_type
            self.init_observation = init_observation

        def reset(self):
            return None

    class _AgentType:
        DEPLOY = "deploy"

    base_agent = types.ModuleType("user_agents.base_agent")
    base_agent.BaseAgent = _BaseAgent
    base_agent.AgentType = _AgentType
    modules = {
        "user_agents": types.ModuleType("user_agents"),
        "user_agents.base_agent": base_agent,
    }
    module_path = CORE_ROOT / "user_agents" / "deploy_agent.py"
    spec = importlib.util.spec_from_file_location(
        "_deploy_agent_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class _DeferredPolicy:
    def __init__(self) -> None:
        self.entities: list[int] = []

    def select_deployment(self, entity_id, observation, deploy_bounds):
        del observation, deploy_bounds
        self.entities.append(int(entity_id))
        return {
            "presence": 1,
            "lon": 0.0,
            "lat": 0.0,
            "defer_position": True,
        }


class DeferredDeploymentTests(unittest.TestCase):
    def test_deferred_entities_are_registered_without_prelaunch_setlla(self):
        module = _load_deploy_agent()
        policy = _DeferredPolicy()
        entities = {
            entity_id: {
                "type": 21002,
                "position": {"lon": 100.0, "lat": 10.0, "alt": 10_000.0},
            }
            for entity_id in range(1, 12)
        }
        agent = module.DeployAgent(
            -1,
            -1,
            {"entities": entities},
            [[[115.0, 15.0], [117.0, 15.0], [117.0, 17.0], [115.0, 17.0]]],
            [],
            unified_policy=policy,
        )
        agent.set_observation({"entities": entities})

        first = agent.get_action()
        self.assertEqual(first.shape, (0, 5))
        self.assertEqual(agent.index, 10)
        second = agent.get_action()
        self.assertEqual(second.shape, (1, 5))
        self.assertEqual(int(second[0, 0]), module.ACTION_COMPLETE)
        self.assertEqual(agent.index, 11)
        self.assertEqual(policy.entities, list(range(1, 12)))
        self.assertFalse(np.any(first[:, 0] == module.ACTION_DEPLOY))

    def test_trajectory_policy_only_registers_bounds_during_deployment(self):
        policy = UnifiedMAPPOSharedPolicy.__new__(UnifiedMAPPOSharedPolicy)
        policy.trajectory_counterfactual = True
        policy.dynamic_lifecycle = True
        policy._current_global_state = np.zeros(1, dtype=np.float32)
        policy._current_team_context = None
        policy._encode_team_context = lambda observation: np.zeros(
            1, dtype=np.float32
        )
        policy._encoders = {11: SimpleNamespace(agent_id=3)}
        policy._deployment = {}
        policy._active_deployment_count = 0
        policy._deployment_legal_count = 0
        observation = {
            "entities": {
                11: {"position": {"lon": 123.4, "lat": 45.6}}
            }
        }

        decision = policy.select_deployment(
            11, observation, (115.0, 117.0, 15.0, 17.0)
        )
        self.assertTrue(decision["defer_position"])
        self.assertEqual((decision["lon"], decision["lat"]), (0.0, 0.0))
        self.assertEqual(
            decision["deploy_bounds"], (115.0, 117.0, 15.0, 17.0)
        )
        self.assertEqual(policy._deployment[11], decision)

    def test_non_deferred_policy_still_emits_deploy(self):
        module = _load_deploy_agent()
        entities = {1: {"type": 21002, "position": {}}}
        agent = module.DeployAgent(
            -1, -1, {"entities": entities},
            [[[115.0, 15.0], [117.0, 15.0], [117.0, 17.0], [115.0, 17.0]]],
            [],
        )
        agent.set_observation({"entities": entities})
        actions = agent.get_action()
        self.assertEqual(actions.shape, (1, 5))
        self.assertEqual(int(actions[0, 0]), module.ACTION_DEPLOY)


if __name__ == "__main__":
    unittest.main()


