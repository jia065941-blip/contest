"""Keep decisions in per-entity agents and learning orchestration in examples."""

import ast
from copy import deepcopy
from dataclasses import asdict
import inspect
from pathlib import Path
import unittest
from uuid import uuid4
from unittest.mock import Mock, patch

import numpy as np

from envengine.agent_manager import AgentManager
from envengine.environment.training_env import TrainingEnv
from user_agents.example.assignment.r0_random import RedAttack
from user_agents.example.common.config import RedAgentConfig
from user_agents.example.common.commander import RedBaselineCommander


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = {"entities": {
    101: {"type": 9400, "pos": {"x": 119.0, "y": 22.0}},
}}


def observation(entity_id=1, *, step=1, peers=()):
    return {
        "step": step, "entity_id": entity_id,
        "self": {
            "type": 21000, "health": 100, "isVisible": True,
            "pos": {"x": 118.0, "y": 22.0, "z": 3000},
            "detectInfo": {}, "commRangeInfo": {peer: {} for peer in peers},
        },
    }


def register_example(config, entity_ids=(1, 2)):
    polygon = ((117, 21), (122, 21), (122, 25), (117, 25))
    run_id = uuid4().hex
    initials = {entity_id: {
        'run_id': run_id, 'entity_id': entity_id,
        'self': observation(entity_id)['self'],
        'entities': deepcopy(PUBLIC['entities']),
        'search_polygon': polygon,
        'map_area': {'xMin': 117, 'xMax': 122, 'yMin': 21, 'yMax': 25},
        'seed': config.seed, 'max_steps': config.max_steps,
        'options': {'config': asdict(config)},
    } for entity_id in entity_ids}
    manager = AgentManager()
    group = manager.register_red_agents(RedAttack, entity_ids, initials)
    return manager, group


class UserAgentBoundaryTests(unittest.TestCase):
    def test_public_constructor_accepts_only_local_initial_data(self):
        self.assertEqual(list(inspect.signature(RedAttack).parameters),
                         ["entity_id", "init_observation"])
        manager, group = register_example(RedAgentConfig(motion_policy="straight"))
        self.addCleanup(group.close)
        first, second = group.agents
        self.assertFalse(hasattr(first, "set_commander"))
        self.assertFalse(hasattr(group, 'shared'))
        self.assertFalse(hasattr(first, 'shared'))
        self.assertFalse(hasattr(first, 'agent_id'))
        self.assertFalse(hasattr(RedAttack, 'create_shared'))
        self.assertIs(first._resource, second._resource)
        first.init_observation["entities"][101]["pos"]["x"] = 999
        self.assertEqual(second.init_observation["entities"][101]["pos"]["x"], 119.0)
        self.assertEqual(PUBLIC["entities"][101]["pos"]["x"], 119.0)

    def test_framework_has_no_example_planning_or_learning_hooks(self):
        for name in ("set_team_controller_factory", "set_shared_learning_policy",
                     "set_agent_runtime", "_prepare_team_controllers", "get_deployment_observation",
                     "apply_deployment", "register_user_agent", "set_deployment_constraints",
                     "_validate_deployment_position", "_point_in_polygon"):
            self.assertFalse(hasattr(TrainingEnv, name), name)
        tree = ast.parse((ROOT / "envengine/environment/training_env.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                self.assertNotEqual(node.id, 'SpeedDistributor')
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith(("user_agents.red", "user_agents.example")))
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, {
                    "_team_controllers", "_team_controller_factory", "_shared_learning_policy",
                    "_agent_runtime", "prepare_observation", "begin_environment_step",
                    "end_environment_step", "finish_environment_step", "create_shared", "_resource",
                    "_deployment_coordinates", "_deployment_coordinates_hm", "_deployment_launch_altitude",
                    "_episode_prepared", "speed_distributor",
                })
        for file in ("main.py", "user_agents/example/common/runtime.py"):
            tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if file == "main.py":
                        self.assertNotIn(node.func.id, {
                            "RedAttack", "RedBaselineCommander", "build_learning_motion_policy",
                        })
                if file.endswith("runtime.py") and isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("envengine"))

    def test_get_obs_only_reports_and_get_action_performs_planning(self):
        manager, group = register_example(RedAgentConfig(motion_policy="straight", seed=7))
        self.addCleanup(group.close)
        first, second = group.agents
        with patch.object(RedBaselineCommander, "begin_step", autospec=True,
                          side_effect=RedBaselineCommander.begin_step) as plan:
            first.get_obs(observation(1, peers=(2,)))
            second.get_obs(observation(2, peers=(1,)))
            self.assertEqual(plan.call_count, 0)
            self.assertEqual((first._execution.launch_step, second._execution.launch_step), (-1, -1))
            actions = first.get_action()
            self.assertEqual(plan.call_count, 1)
            self.assertEqual({item["entity_id"] for item in plan.call_args.args[1]}, {1, 2})
            second.get_action()
            self.assertEqual(plan.call_count, 1)
        self.assertEqual(list(actions[0]), [1, 1, 119.0, 22.0])
        self.assertEqual(first._execution._commander.target_id_for(1), 101)
        self.assertIs(first._execution._commander, second._execution._commander)
        self.assertEqual((first._execution.launch_step, second._execution.launch_step), (1, 1))

    def test_precomputed_action_payload_does_not_replace_agent_decision(self):
        manager, group = register_example(RedAgentConfig(motion_policy="straight"), (1,))
        self.addCleanup(group.close)
        agent = group.agents[0]
        local = observation()
        local["example_decision"] = {
            "actions": [[1, 1, 121.0, 24.0]],
            "state": {"launch_step": 999, "sat_used": True},
        }
        agent.get_obs(local)
        actions = agent.get_action()
        self.assertEqual(list(actions[0]), [1, 1, 119.0, 22.0])
        self.assertEqual(agent._execution._commander.target_id_for(1), 101)
        self.assertEqual(agent._execution.launch_step, 1)

    def test_local_mailbox_cannot_submit_another_platforms_report(self):
        manager, group = register_example(RedAgentConfig(motion_policy="straight"))
        self.addCleanup(group.close)
        first, _ = group.agents
        with self.assertRaisesRegex(ValueError, "own observation"):
            first.get_obs(observation(2))
        first.get_obs(observation(1))
        self.assertEqual(int(first.get_action()[0, 1]), 1)

    def test_reset_all_discards_group_mailboxes_and_previous_target_memory(self):
        manager, group = register_example(RedAgentConfig(motion_policy="straight"))
        self.addCleanup(group.close)
        agents = group.agents
        observations = {i: observation(i, peers=(3 - i,)) for i in (1, 2)}
        for local in observations.values():
            local["self"]["detectInfo"][103] = {
                "entity_id": 103, "entity_type": 9500, "detected_step": 1,
                "pos": {"x": 119.5, "y": 22, "z": 0},
            }
        manager.collect_actions_from_agents(observations)
        previous = agents[0]._execution._commander
        self.assertIn(103, {item.entity_id for item in previous.targets})
        manager.reset_all()
        self.assertEqual([agent._execution.launch_step for agent in agents], [-1, -1])
        manager.collect_actions_from_agents({i: observation(i, step=0) for i in (1, 2)})
        self.assertIsNot(agents[0]._execution._commander, previous)
        self.assertNotIn(103, {item.entity_id for item in agents[0]._execution._commander.targets})
        self.assertIsNot(agents[0]._execution._commander, agents[1]._execution._commander)

    def test_actor_and_record_step_expose_only_local_training_calls(self):
        policy = Mock(training=True)
        policy.select_action.return_value = 1
        with patch("user_agents.example.common.runtime.build_learning_motion_policy", return_value=policy):
            manager, group = register_example(RedAgentConfig(motion_policy="ppo", training=True), (1,))
        self.addCleanup(group.close)
        runtime = group.agents[0]._resource
        agent = group.agents[0]
        for name in ("get_global_state_context", "begin_environment_step", "end_environment_step",
                     "finish_environment_step", "observe"):
            self.assertFalse(hasattr(agent._execution._actor, name), name)
        agent.get_obs(observation())
        action = agent.get_action()
        agent.record_step(observation(step=2), action, 1.0, {"done": False})
        policy.select_action.assert_called_once()
        self.assertEqual(policy.select_action.call_args.args[0].shape, (85,))
        self.assertEqual(len(runtime._transitions), 1)
        for name in ("get_global_state_context", "begin_environment_step", "end_environment_step",
                     "finish_environment_step", "observe"):
            getattr(policy, name).assert_not_called()
        transition = runtime._transitions[0]
        self.assertIsNone(transition.global_state)
        self.assertIsNone(transition.next_global_state)
        self.assertTrue(np.isfinite(transition.observation).all())

    def test_training_runner_pairs_each_frames_local_reports_with_its_global_states(self):
        policy = Mock(training=True)
        policy.select_action.return_value = 1
        states = [(np.full(70, step), np.full(70, step + 1)) for step in (1, 2)]
        policy.get_global_state_context.side_effect = states
        with patch("user_agents.example.common.runtime.build_learning_motion_policy", return_value=policy):
            manager, group = register_example(RedAgentConfig(motion_policy="ppo", training=True))
        self.addCleanup(group.close)
        runtime, agents = group.agents[0]._resource, group.agents

        for step in (1, 2):
            runtime.begin_training_step({"step": step, "entities": {}})
            actions = manager.collect_actions_from_agents({
                i: observation(i, step=step, peers=(3 - i,)) for i in (1, 2)
            })
            for agent, action in zip(agents, actions, strict=True):
                agent.record_step(
                    observation(agent.entity_id, step=step + 1, peers=(3 - agent.entity_id,)),
                    action, 1.0, {"done": False},
                )
            self.assertEqual(policy.observe.call_count, (step - 1) * 2)
            runtime.finish_training_step({"step": step + 1, "entities": {}})

        transitions = [call.args[0] for call in policy.observe.call_args_list]
        self.assertEqual([item.entity_id for item in transitions], [1, 2, 1, 2])
        for index, transition in enumerate(transitions):
            global_state, next_global_state = states[index // 2]
            np.testing.assert_array_equal(transition.global_state, global_state)
            np.testing.assert_array_equal(transition.next_global_state, next_global_state)
        self.assertEqual(policy.finish_environment_step.call_count, 2)

    def test_sparse_entity_ids_use_private_dense_learning_indices(self):
        entity_ids = (2576, 880000, 900001)
        policy = Mock(training=True)
        policy.select_action.return_value = 1
        with patch("user_agents.example.common.runtime.build_learning_motion_policy", return_value=policy):
            manager, group = register_example(RedAgentConfig(motion_policy="ppo", training=True), entity_ids)
        self.addCleanup(group.close)

        def observations(step):
            return {
                entity_id: observation(entity_id, step=step,
                                       peers=tuple(peer for peer in entity_ids if peer != entity_id))
                for entity_id in entity_ids
            }

        manager.collect_actions_from_agents(observations(1))
        expected = [[0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]]
        encoded = [call.args[0] for call in policy.select_action.call_args_list]
        np.testing.assert_allclose(np.stack(encoded)[:, 17:21], expected)
        self.assertEqual([agent.entity_id for agent in group.agents], list(entity_ids))
        for agent in group.agents:
            self.assertFalse({'agent_id', 'agent_index', 'team_size'} & agent.init_observation.keys())

        manager.reset_all()
        manager.collect_actions_from_agents(observations(0))
        encoded = [call.args[0] for call in policy.select_action.call_args_list[-3:]]
        np.testing.assert_allclose(np.stack(encoded)[:, 17:21], expected)

    def test_private_example_models_reset_and_close_once_without_cross_run_sharing(self):
        policies = [Mock(training=True), Mock(training=True)]
        with patch('user_agents.example.common.runtime.build_learning_motion_policy', side_effect=policies) as build:
            runs = [register_example(RedAgentConfig(motion_policy='ppo', training=True)) for _ in range(2)]
        for manager, group in runs:
            self.addCleanup(group.close)
        first, second = (group for manager, group in runs)
        self.assertEqual(build.call_count, 2)
        self.assertIs(first.agents[0]._resource, first.agents[1]._resource)
        self.assertIsNot(first.agents[0]._resource, second.agents[0]._resource)
        runs[0][0].reset_all()
        policies[0].reset_episode.assert_called_once_with()
        policies[1].reset_episode.assert_not_called()
        first.close()
        first.close()
        policies[0].close.assert_called_once_with()
        policies[1].close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
