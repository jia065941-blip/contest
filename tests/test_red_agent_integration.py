"""Red execution and assignment constrained to one communication domain."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from uuid import uuid4
import unittest
from unittest.mock import patch

from user_agents.example.assignment.r0_random import RedAttack
from user_agents.example.common.contracts import Position
from user_agents.example.common.config import RedAgentConfig
from user_agents.example.common.commander import RedBaselineCommander
from user_agents.example.common.baselines import RED_POLICY_CHOICES, TargetPrior


POLYGON = (Position(117.0, 21.0), Position(122.0, 21.0), Position(122.0, 25.0), Position(117.0, 25.0))


def initial_observations(entity_ids=(1,), public=None, *, motion_policy='reactive_evasion'):
    polygon = tuple((point.x, point.y) for point in POLYGON)
    run_id = uuid4().hex
    return {entity_id: {
        'run_id': run_id, 'entity_id': entity_id,
        'self': platform_observation(entity_id, 21000)['self'],
        'entities': deepcopy((public or {}).get('entities', {})),
        'seed': 7, 'max_steps': 1000, 'search_polygon': polygon,
        'map_area': {'xMin': 117, 'xMax': 122, 'yMin': 21, 'yMax': 25},
        'options': {'config': asdict(RedAgentConfig(motion_policy=motion_policy, seed=7))},
    } for entity_id in entity_ids}


def platform_observation(entity_id: int, entity_type: int, *, step: int = 1, x: float = 118.0) -> dict:
    return {
        "step": step, "entity_id": entity_id,
        "self": {
            "type": entity_type, "health": 100.0, "isVisible": True,
            "pos": {"x": x, "y": 22.0, "z": 100.0},
            "vel": {"x": 100.0, "y": 0.0, "z": 0.0},
            "detectInfo": {}, "commRangeInfo": [entity_id],
        },
    }


def target(entity_id=101, entity_type=9400, x=119.0) -> TargetPrior:
    return TargetPrior(entity_id, entity_type, Position(x, 22.0), value=10.0)


def report_track(observation: dict, tracked_target: TargetPrior, x: float) -> dict:
    observation["self"]["detectInfo"][tracked_target.entity_id] = {
        "entity_id": tracked_target.entity_id, "entity_type": tracked_target.entity_type,
        "detected_step": observation["step"], "pos": {"x": x, "y": 22.0, "z": 0.0},
    }
    return observation


class CommanderStub:
    def __init__(self):
        self.commands = [[1.0, 1.0, 119.0, 22.0], [1.0, 1.0, 120.0, 22.0]]
        self.satellite_leaders = set()

    def register_platform(self, entity_id):
        pass

    def begin_step(self, observations):
        pass

    def report(self, observation):
        pass

    def action_for(self, entity_id, step):
        return self.commands.pop(0) if self.commands else None

    def should_use_satellite(self, entity_id):
        return entity_id in self.satellite_leaders

    def target_id_for(self, entity_id):
        return None

    def reset(self):
        self.commands.clear()
        self.satellite_leaders.clear()


class RedDomainIntegrationTests(unittest.TestCase):
    def agents(self, entity_ids=(1,), *, motion_policy='reactive_evasion', public=None):
        initials = initial_observations(entity_ids, public, motion_policy=motion_policy)
        agents = tuple(RedAttack(entity_id, initials[entity_id]) for entity_id in entity_ids)
        for agent in agents:
            self.addCleanup(agent.close)
        return agents

    def agent(self, commander=None, *, motion_policy="straight"):
        if commander is not None:
            self.enterContext(patch("user_agents.example.common.coordination.RedBaselineCommander", return_value=commander))
        return self.agents(motion_policy=motion_policy)[0]

    def commander(self, policy="r1_priority", ids=(1,), targets=None):
        commander = RedBaselineCommander(
            (target(),) if targets is None else targets, policy, search_polygon=POLYGON,
        )
        for entity_id in ids:
            commander.register_platform(entity_id)
        return commander

    def test_first_step_guides_to_public_targets_and_searches_for_hidden_nests(self):
        commander = self.commander("r3_static_priority", (1, 2, 3), (target(), target(102, 9600, 120.0)))
        commander.begin_step(tuple(platform_observation(i, kind) for i, kind in ((1, 21000), (2, 21001), (3, 21002))))
        actions = [commander.action_for(i, 1) for i in (1, 2, 3)]
        for entity_id, action in zip((1, 2), actions[:2]):
            target_id = commander.target_id_for(entity_id)
            self.assertIn(target_id, {101, 102})
            self.assertEqual(action, [1, entity_id, {101: 119.0, 102: 120.0}[target_id], 22.0])
        self.assertEqual(actions[2], [1, 3, 118.25, 24.0])
        self.assertIsNone(commander.target_id_for(3))

    def test_initial_catalogue_cannot_make_hidden_nests_public(self):
        hidden = target(103, 9500, 121.0)
        commander = self.commander(ids=(3,), targets=(target(), hidden))
        commander.begin_step((platform_observation(3, 21002),))
        self.assertEqual(commander.action_for(3, 1), [1, 3, 118.25, 24.0])
        self.assertIsNone(commander.target_id_for(3))
        self.assertNotIn(103, {item.entity_id for item in commander.targets})

    def test_received_hidden_nest_is_attacked_using_received_coordinates(self):
        hidden = target(103, 9500, 121.0)
        commander = self.commander(ids=(3,))
        obs = report_track(platform_observation(3, 21002), hidden, 121.0)
        commander.begin_step((obs,))
        self.assertEqual(commander.action_for(3, 1), [1, 3, 121.0, 22.0])
        self.assertEqual(commander.target_id_for(3), 103)

    def test_hidden_track_does_not_cross_disconnected_domains(self):
        hidden = target(103, 9500, 121.0)
        informed = self.commander(ids=(1,))
        isolated = self.commander(ids=(2,))
        informed.begin_step((report_track(platform_observation(1, 21002), hidden, 121.0),))
        isolated.begin_step((platform_observation(2, 21002),))
        self.assertEqual(informed.action_for(1, 1), [1, 1, 121.0, 22.0])
        self.assertEqual(informed.target_id_for(1), 103)
        self.assertEqual(isolated.action_for(2, 1), [1, 2, 120.75, 24.0])
        self.assertIsNone(isolated.target_id_for(2))
        self.assertNotIn(103, {item.entity_id for item in isolated.targets})

    def test_each_platform_must_have_received_the_hidden_target(self):
        # The engine normally fuses all domain tracks before this call. This
        # extra guard fails closed if a caller passes an incomplete snapshot.
        commander = self.commander(ids=(1, 2))
        commander.begin_step((
            report_track(platform_observation(1, 21002), target(103, 9500, 121.0), 121.0),
            platform_observation(2, 21002),
        ))
        self.assertEqual(commander.action_for(1, 1), [1, 1, 121.0, 22.0])
        self.assertEqual(commander.target_id_for(1), 103)
        self.assertEqual(commander.action_for(2, 1), [1, 2, 120.75, 24.0])
        self.assertIsNone(commander.target_id_for(2))

    def test_commander_rejects_out_of_domain_reports(self):
        commander = self.commander(ids=(1,))
        with self.assertRaisesRegex(ValueError, "domain"):
            commander.begin_step((platform_observation(2, 21000),))
        with self.assertRaisesRegex(ValueError, "domain"):
            commander.report(platform_observation(2, 21000))

    def test_current_domain_snapshot_is_required(self):
        commander = self.commander()
        with self.assertRaisesRegex(RuntimeError, "begin_step"):
            commander.action_for(1, 1)

    def test_agent_decides_without_external_binding(self):
        initial = initial_observations(public={"entities": {
            101: {"type": 9400, "pos": {"x": 119.0, "y": 22.0}},
        }})
        agent = RedAttack(1, initial[1])
        self.addCleanup(agent.close)
        agent.get_obs(platform_observation(1, 21000))
        actions = agent.get_action()
        self.assertEqual(list(actions[0]), [1, 1, 119.0, 22.0])
        self.assertEqual(agent._execution.launch_step, 1)
        self.assertFalse(hasattr(agent, "set_commander"))

    def test_communication_changes_preserve_local_state(self):
        agent, other = self.agents((1, 2))
        observations = [platform_observation(i, 21000) for i in (1, 2)]
        for i, observation in enumerate(observations, 1):
            observation["self"]["commRangeInfo"] = {3 - i: {}}
        observations[0]["self"]["detectInfo"][500] = {
            "entity_id": 500, "entity_type": 24000,
            "pos": {"x": 117.0, "y": 22.0, "z": 100.0}, "vel": {"x": 1.0, "y": 0.0, "z": 0.0},
        }
        agent.get_obs(observations[0])
        other.get_obs(observations[1])
        self.assertIn(2, agent.get_action()[:, 0])
        other.get_action()
        previous = agent._execution._commander
        self.assertTrue(agent._execution.sat_used)
        agent.get_obs(platform_observation(1, 21000, step=7))
        self.assertNotIn(2, agent.get_action()[:, 0])
        self.assertIsNot(agent._execution._commander, previous)
        self.assertEqual(agent._execution.launch_step, 1)
        agent.reset()
        other.reset()
        self.assertIsNone(agent.latest_observation)
        self.assertEqual(agent._execution.launch_step, -1)
        self.assertFalse(agent._execution.sat_used)

    def test_maneuver_starts_immediately_and_target_changes_preserve_start_time(self):
        agent = self.agent(CommanderStub(), motion_policy="random_masked")
        agent.get_obs(platform_observation(1, 21000))
        self.assertEqual(set(agent.get_action()[:, 0]), {3, 1})
        agent.get_obs(platform_observation(1, 21000, step=5))
        self.assertEqual(set(agent.get_action()[:, 0]), {3, 1})
        self.assertEqual(agent._execution.launch_step, 1)

    def test_later_satellite_selection_is_requested_once_per_platform(self):
        commander = CommanderStub()
        agent = self.agent(commander)
        agent.get_obs(platform_observation(1, 21000))
        self.assertNotIn(2, agent.get_action()[:, 0])
        commander.satellite_leaders.add(1)
        agent.get_obs(platform_observation(1, 21000, step=5))
        self.assertEqual(list(agent.get_action()[:, 0]).count(2), 1)
        agent.get_obs(platform_observation(1, 21000, step=6))
        self.assertEqual(agent.get_action().shape, (0, 4))

    def test_missing_member_does_not_block_live_member_and_reset_clears_tracks(self):
        commander = self.commander("r4_rolling_rules", (1, 2))
        commander.begin_step((platform_observation(1, 21000),))
        self.assertEqual(commander.action_for(1, 1), [1, 1, 119.0, 22.0])
        self.assertEqual(commander.absent_platform_ids, frozenset({2}))
        self.assertEqual(commander.tactical_metrics.pressure, 0.0)
        commander.reset()
        self.assertIsNone(commander.current_snapshot_ids)
        self.assertFalse(commander.guidance)
        self.assertFalse(commander.reports)

    def test_r4_replans_only_a_bounded_subset(self):
        commander = self.commander("r4_rolling_rules", range(1, 5))
        commander.begin_step(tuple(platform_observation(i, 21000) for i in range(1, 5)))
        self.assertTrue(all(commander.action_for(i, 1) == [1, i, 119.0, 22.0] for i in range(1, 5)))
        commander.begin_step(tuple(report_track(platform_observation(i, 21000, step=21), target(), 120.0) for i in range(1, 5)))
        actions = [commander.action_for(i, 21) for i in range(1, 5)]
        self.assertEqual(sum(action is not None for action in actions), 1)
        updated = next(action for action in actions if action is not None)
        self.assertEqual(updated[2:], [120.0, 22.0])
        self.assertEqual(commander.target_id_for(int(updated[1])), 101)

    def test_r5_replans_after_event_cooldown_using_received_coordinates(self):
        commander = self.commander("r5_event_rolling")
        commander.begin_step((platform_observation(1, 21000),))
        self.assertEqual(commander.action_for(1, 1), [1, 1, 119.0, 22.0])
        commander.begin_step((report_track(platform_observation(1, 21000, step=2), target(), 120.0),))
        self.assertIsNone(commander.action_for(1, 2))
        self.assertEqual(commander.action_for(1, 6), [1, 1, 120.0, 22.0])
        self.assertEqual(commander.target_id_for(1), 101)

    def test_r9_after_domain_split_does_not_learn_the_other_domains_threat(self):
        joined = self.commander("r9_hierarchical_learning", (1, 2))
        joined.begin_step((platform_observation(1, 21000), platform_observation(2, 21000)))
        joined.action_for(1, 1)
        joined.action_for(2, 1)
        # A split constructs independent planners from each member's own
        # received observations. The new threat is observed only after split.
        informed = self.commander("r9_hierarchical_learning", (1,))
        isolated = self.commander("r9_hierarchical_learning", (2,))
        observed = platform_observation(1, 21000)
        observed["self"]["detectInfo"][500] = {"entity_id": 500, "entity_type": 24000}
        informed.begin_step((observed,))
        isolated.begin_step((platform_observation(2, 21000),))
        informed.action_for(1, 1)
        isolated.action_for(2, 1)
        self.assertGreater(informed.learning_task_context(1)[3], isolated.learning_task_context(2)[3])
        self.assertEqual(isolated.learning_task_context(2)[3], 0.0)

    def test_absent_members_are_removed_without_creating_threat_pressure(self):
        commander = self.commander("r9_hierarchical_learning", (1, 2))
        commander.begin_step((platform_observation(1, 21000), platform_observation(2, 21000)))
        commander.action_for(1, 1)
        commander.action_for(2, 1)
        commander.begin_step((platform_observation(1, 21000, step=2),))
        self.assertEqual(set(commander.reports), {1})
        self.assertNotIn(2, commander.target_by_platform)
        self.assertEqual(commander.tactical_metrics.alive_count, 1)
        self.assertEqual(commander.tactical_metrics.observed_interceptor_count, 0)
        self.assertEqual(commander.learning_task_context(1)[3], 0.0)
        self.assertIsNone(commander.action_for(2, 2))

    def test_every_current_policy_guides_all_units_without_explicit_launch(self):
        for name in RED_POLICY_CHOICES:
            with self.subTest(policy=name):
                commander = self.commander(name, (1, 2, 3))
                commander.begin_step(tuple(platform_observation(i, kind) for i, kind in ((1, 21000), (2, 21001), (3, 21002))))
                actions = [commander.action_for(i, 1) for i in (1, 2, 3)]
                self.assertTrue(all(action is not None for action in actions))
                self.assertTrue(all(action[0] == 1 for action in actions))
                self.assertTrue(all(117 <= action[2] <= 122 and 21 <= action[3] <= 25
                                    for action in actions))

    def test_reactive_motion_accepts_received_dictionary_tracks(self):
        agent = self.agent(CommanderStub(), motion_policy="reactive_evasion")
        observation = platform_observation(1, 21000)
        observation["self"]["detectInfo"][500] = {
            "entity_id": 500, "entity_type": 24000,
            "pos": {"x": 117.0, "y": 22.0, "z": 100.0},
            "vel": {"x": 1.0, "y": 0.0, "z": 0.0},
        }
        agent.get_obs(observation)
        self.assertEqual(set(agent.get_action()[:, 0]), {3, 2, 1})


if __name__ == "__main__":
    unittest.main()
