"""Information-flow regressions between the engine and red planners."""

from copy import deepcopy
from dataclasses import asdict
from uuid import uuid4
from types import SimpleNamespace
import unittest

import numpy as np

from envengine.action_validation import IllegalActionError
from envengine.common import SimmerCommandType
from envengine.agent_manager import AgentManager
from envengine.environment.command_converter import CommandConverter
from envengine.environment.training_env import TrainingEnv
from user_agents.example.assignment.r0_random import RedAttack
from user_agents.example.common.config import RedAgentConfig


def track(entity_id=9501, x=1.5, detected_step=1, entity_type=9500):
    return {"entity_id": entity_id, "entity_type": entity_type, "detected_step": detected_step,
            "pos": {"x": x, "y": 1.5, "z": 0.0}}


def own(x=1.0, tracks=None):
    return {"nameChn": "unit", "stage": 3, "vel": {"x": 0.0, "y": 0.0, "z": 0.0},
            "type": 21002, "side": 0, "health": 1.0, "isVisible": True,
            "pos": {"x": x, "y": 1.0, "z": 3000.0},
            "detectInfo": deepcopy(tracks or {}), "commRangeInfo": {}}


class CommunicationScopedPlanningTests(unittest.TestCase):
    def make_env(self, groups=((1,), (2,)), left_tracks=None, right_tracks=None):
        env = object.__new__(TrainingEnv)
        env.agent_manager = AgentManager()
        polygon = ((0, 0), (3, 0), (3, 3), (0, 3))
        run_id = uuid4().hex
        for identifier in (1, 2):
            initial = {
                'run_id': run_id, 'entity_id': identifier,
                'self': own(float(identifier)),
                'entities': {9401: {'type': 9400, 'pos': {'x': 2.5, 'y': 2.5}}},
                'seed': 7, 'max_steps': 1000, 'search_polygon': polygon,
                'map_area': {'xMin': 0, 'xMax': 3, 'yMin': 0, 'yMax': 3},
                'options': {'config': asdict(RedAgentConfig(
                    policy_name='r1_priority', motion_policy='straight', seed=7,
                ))},
            }
            agent = RedAttack(identifier, initial)
            env.agent_manager.register_agent(agent)
            self.addCleanup(agent.close)
        self.groups = groups
        self.full = {"step": 0, "entities": {1: own(tracks=left_tracks), 2: own(2.0, right_tracks)}}

        def refresh_communications():
            for entity_id, entity in self.full["entities"].items():
                group = next(item for item in self.groups if entity_id in item)
                entity["commRangeInfo"] = {
                    member: {"pos": deepcopy(self.full["entities"][member]["pos"])}
                    for member in group if member != entity_id
                }
            return self.groups

        env.engine = SimpleNamespace(
            refresh_red_communications=refresh_communications,
            simulator_factory=SimpleNamespace(is_using_satellite=lambda: False),
        )
        env._get_observation = lambda: deepcopy(self.full)
        return env

    def test_disconnected_planner_cannot_assign_another_groups_hidden_target(self):
        env = self.make_env(left_tracks={9501: track()})
        commands = env._generate_actions_from_agents()
        by_id = {item["executor_id"]: item for item in commands}
        self.assertEqual(by_id[1]["target"], {"x": 1.5, "y": 1.5, "z": 0.0})
        self.assertEqual(by_id[2]["target"], {"x": 2.25, "y": 2.25, "z": 0.0})
        self.assertEqual(by_id[1]["intent"], "waypoint")
        self.assertIsNone(by_id[1]["target_id"])
        self.assertEqual(by_id[2]["intent"], "waypoint")
        first = env.agent_manager.get_agent(1)._execution._commander
        second = env.agent_manager.get_agent(2)._execution._commander
        self.assertIsNot(first, second)
        self.assertEqual(first.target_id_for(1), 9501)
        self.assertIsNone(second.target_id_for(2))
        self.assertNotIn(9501, {item.entity_id for item in second.targets})
        self.assertEqual(set(second.reports), {2})

    def test_connected_members_share_one_planner_and_stable_groups_keep_it(self):
        received = {9501: track()}
        env = self.make_env(((1, 2),), received, received)
        env._generate_actions_from_agents()
        controller = env.agent_manager.get_agent(1)._execution._commander
        self.assertIs(controller, env.agent_manager.get_agent(2)._execution._commander)
        self.full["step"] = 1
        env._generate_actions_from_agents()
        self.assertIs(controller, env.agent_manager.get_agent(1)._execution._commander)

    def test_split_preserves_received_memory_without_new_cross_group_updates(self):
        received = {9501: track(x=1.5)}
        env = self.make_env(((1, 2),), received, received)
        env._generate_actions_from_agents()
        previous = env.agent_manager.get_agent(1)._execution._commander
        self.groups = ((1,), (2,))
        self.full["step"] = 20
        self.full["entities"][1]["detectInfo"][9501] = track(x=2.7, detected_step=20)
        env._generate_actions_from_agents()
        first = env.agent_manager.get_agent(1)._execution._commander
        second = env.agent_manager.get_agent(2)._execution._commander
        self.assertIsNot(first, previous)
        self.assertIsNot(second, previous)
        self.assertEqual(next(t.position.x for t in first.targets if t.entity_id == 9501), 2.7)
        self.assertEqual(next(t.position.x for t in second.targets if t.entity_id == 9501), 1.5)

    def test_merge_replans_from_newly_authorized_received_tracks(self):
        env = self.make_env(left_tracks={9501: track()})
        env._generate_actions_from_agents()
        previous = env.agent_manager.get_agent(2)._execution._commander
        self.groups = ((1, 2),)
        self.full["step"] = 20
        self.full["entities"][2]["detectInfo"] = {9501: track()}
        commands = env._generate_actions_from_agents()
        controller = env.agent_manager.get_agent(2)._execution._commander
        self.assertIsNot(controller, previous)
        self.assertIs(controller, env.agent_manager.get_agent(1)._execution._commander)
        self.assertEqual(len(commands), 2)
        self.assertTrue(all(item["target"] == {"x": 1.5, "y": 1.5, "z": 0.0}
                            for item in commands))
        self.assertEqual([controller.target_id_for(entity_id) for entity_id in (1, 2)],
                         [9501, 9501])

    def test_disconnected_r9_context_does_not_include_other_groups_threat(self):
        env = self.make_env(left_tracks={2401: track(2401, entity_type=24000)})
        env._generate_actions_from_agents()
        first = env.agent_manager.get_agent(1)._execution._commander
        second = env.agent_manager.get_agent(2)._execution._commander
        self.assertGreater(first.learning_task_context(1)[3], 0.0)
        self.assertEqual(second.learning_task_context(2)[3], 0.0)

    def test_independent_runtimes_do_not_share_same_id_group_state(self):
        env = self.make_env()
        env._generate_actions_from_agents()
        first = env.agent_manager.get_agent(1)._execution._commander
        other = self.make_env(left_tracks={9501: track()})
        other._generate_actions_from_agents()
        second = other.agent_manager.get_agent(1)._execution._commander
        self.assertIsNot(first, second)
        self.assertNotIn(9501, {item.entity_id for item in first.targets})
        self.assertIn(9501, {item.entity_id for item in second.targets})


    def test_report_copy_cannot_be_changed_after_get_obs(self):
        env = self.make_env(left_tracks={9501: track()})
        agent = env.agent_manager.get_agent(1)
        observation = env.agent_manager.extract_observation_for_agent(self.full, 1)
        agent.get_obs(observation)
        observation["self"]["detectInfo"][9501]["pos"]["x"] = 999
        actions = agent.get_action()
        np.testing.assert_array_equal(actions[0], [1, 1, 1.5, 1.5])
        self.assertEqual(next(item.position.x for item in agent._execution._commander.targets
                              if item.entity_id == 9501), 1.5)

    def test_agent_cannot_issue_command_for_another_executor(self):
        env = self.make_env()
        agent = env.agent_manager.get_agent(1)
        agent.get_action = lambda: np.array([[1, 2, 1.5, 1.5]])
        with self.assertRaisesRegex(IllegalActionError, "只能控制自身"):
            env._generate_actions_from_agents()


class RedActionProtocolTests(unittest.TestCase):
    def test_target_action_uses_coordinates_and_preserves_internal_command(self):
        result = CommandConverter.common_converter([[[1, 17, 1.5, 2.5]]])[0]
        self.assertEqual(result["executor_id"], 17)
        self.assertEqual(result["commandType_id"], SimmerCommandType.CHANGE_MISSILE_TARGET)
        self.assertEqual(result["intent"], "waypoint")
        self.assertEqual(result["target"], {"x": 1.5, "y": 2.5, "z": 0.0})
        self.assertIsNone(result["target_id"])

    def test_satellite_and_maneuver_keep_native_types_and_values(self):
        satellite, maneuver = CommandConverter.common_converter([
            [[2, 17, 0, 0], [3, 17, -0.5, 0]],
        ])
        self.assertEqual(satellite["executor_id"], 17)
        self.assertEqual(satellite["commandType_id"], SimmerCommandType.EXECUTE_SATELLITE_DETECTION)
        self.assertEqual(maneuver["executor_id"], 17)
        self.assertEqual(maneuver["commandType_id"], SimmerCommandType.SET_DESIRED_ACC_Z)
        self.assertEqual(maneuver["acc_z"], -0.5)

    def test_unsupported_and_malformed_actions_fail(self):
        for row in ([0, 1, 1, 0], [4, 1, 9501, 0], [5, 1, 1, 0], [999, 1, 0, 0],
                    [1, 1, float("nan"), 0], [1, 1.5, 1.5, 2.5], [1, 1, 2],
                    [2, 1, 1, 0], [2, 1, 0, 1], [3, 1, 0, 1]):
            with self.subTest(row=row), self.assertRaises(IllegalActionError):
                CommandConverter.common_converter([[row]])


if __name__ == "__main__":
    unittest.main()
