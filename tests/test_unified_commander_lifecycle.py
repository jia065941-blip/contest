from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import types
import unittest
from unittest.mock import patch

import numpy as np

CORE_ROOT = Path(__file__).resolve().parents[1] / "core"
from policies.red.learning.red_policy import (
    ObservationEncoder,
    UNIFIED_LOCAL_OBSERVATION_DIM,
    UNIFIED_TARGET_SLOTS,
)

from policies.red.baselines import TargetPrior
from policies.red.contracts import Position
from policies.red.tracks import InitialCatalogueTrackFusion
from policies.red.unified_mappo_commander import UnifiedMAPPOCommander


def _load_attack_agent():
    """Load the real agent module without importing the native simulator SDK."""

    class _BaseAgent:
        def __init__(self, agent_id, entity_id, agent_type, init_observation):
            self.agent_id = agent_id
            self.entity_id = entity_id
            self.agent_type = agent_type
            self.init_observation = init_observation

        def record_step(self, observation, action, reward, info=None):
            del observation, action, reward, info

        def reset(self):
            return None

    class _AgentType:
        AIRCRAFT = "aircraft"

    basic = types.ModuleType("envengine.sdk.base_struct.Basic")
    basic.Vector3d = object
    base_agent = types.ModuleType("user_agents.base_agent")
    base_agent.BaseAgent = _BaseAgent
    base_agent.AgentType = _AgentType
    modules = {
        "envengine": types.ModuleType("envengine"),
        "envengine.sdk": types.ModuleType("envengine.sdk"),
        "envengine.sdk.base_struct": types.ModuleType("envengine.sdk.base_struct"),
        "envengine.sdk.base_struct.Basic": basic,
        "user_agents": types.ModuleType("user_agents"),
        "user_agents.base_agent": base_agent,
    }
    module_path = CORE_ROOT / "user_agents" / "attack_missile_agent.py"
    spec = importlib.util.spec_from_file_location(
        "_attack_missile_agent_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module.AttackMissileAgent


class _LifecyclePolicy:
    def __init__(self, active_entity_ids: set[int]) -> None:
        self.active_entity_ids = set(active_entity_ids)
        self.retarget_interval = 200
        self.targets: tuple[TargetPrior, ...] = ()
        self.reserved_targets: tuple[TargetPrior, ...] = ()
        self.configure_history: list[tuple[TargetPrior, ...]] = []
        self.reserved_history: list[tuple[TargetPrior, ...]] = []
        self.actor_visibility: dict[int, frozenset[int]] = {}
        self.decisions: dict[int, list[dict]] = {}

    def configure_targets(self, targets, *, reserved_targets=()) -> None:
        self.targets = tuple(targets)
        self.reserved_targets = tuple(reserved_targets)
        self.configure_history.append(self.targets)
        self.reserved_history.append(self.reserved_targets)

    def set_actor_target_visibility(self, visibility) -> None:
        self.actor_visibility = {
            int(entity_id): frozenset(int(target_id) for target_id in target_ids)
            for entity_id, target_ids in visibility.items()
        }

    def queue(self, entity_id: int, **decision) -> None:
        self.decisions.setdefault(entity_id, []).append(decision)

    def consume_lifecycle_decision(self, entity_id: int) -> dict:
        return self.decisions[entity_id].pop(0)

    def diagnostics(self) -> dict:
        return {}


def _targets() -> tuple[TargetPrior, ...]:
    return (
        TargetPrior(1, 9400, Position(120.0, 20.0), value=5.0),
        TargetPrior(2, 9600, Position(121.0, 21.0), value=2.0),
        TargetPrior(50, 9500, Position(130.0, 30.0), value=1.0),
    )


def _observation(
    *, entity_id: int = 11, agent_id: int = 3, step: int = 7,
    entity_type: int = 21002, detect_info: dict | None = None,
) -> dict:
    return {
        "step": step,
        "entity_id": entity_id,
        "agent_id": agent_id,
        "self": {
            "type": entity_type,
            "health": 100.0,
            "isVisible": True,
            "position": {"lon": 110.0, "lat": 10.0, "alt": 10_000.0},
            "velocity": {"speed": 300.0, "up": 0.0, "heading": 0.0},
            "detectInfo": detect_info or {},
            "commRangeInfo": [],
        },
    }


def _detected_ship(*, detect_from: int | None = 77, time: int | None = 6):
    return SimpleNamespace(
        entity_id=50,
        entity_type=9500,
        detect_from=detect_from,
        time=time,
        lla=SimpleNamespace(x=122.5, y=22.5, z=0.0),
    )


class UnifiedCommanderLifecycleTests(unittest.TestCase):
    def _make_commander(self, policy: _LifecyclePolicy) -> UnifiedMAPPOCommander:
        commander = UnifiedMAPPOCommander(
            _targets(),
            max_steps=100,
            search_polygon=(
                Position(115.0, 15.0),
                Position(117.0, 15.0),
                Position(117.0, 17.0),
                Position(115.0, 17.0),
            ),
        )
        commander.attach_policy(policy)
        commander.register_agent_identity(11, 3)
        return commander

    def test_provenance_same_frame_launch_and_consuming_ledger(self) -> None:
        with patch.dict(
            os.environ,
            {"RED_UNIFIED_DYNAMIC_LIFECYCLE": "1"},
            clear=False,
        ):
            policy = _LifecyclePolicy({11})
            commander = self._make_commander(policy)

            self.assertEqual(policy.retarget_interval, 1)
            self.assertEqual(
                {target.entity_id for target in policy.configure_history[0]},
                {1, 2, -100},
            )
            self.assertEqual(
                {target.entity_id for target in policy.reserved_history[0]},
                {50},
            )
            commander.begin_step((
                _observation(detect_info={50: _detected_ship()}),
            ))
            self.assertEqual(commander.track_fusion.source_for(50), 77)
            self.assertEqual(commander.first_discovery_step[50], 6)
            self.assertIn(50, {target.entity_id for target in policy.targets})

            policy.queue(
                11,
                kind="launch",
                target_id=50,
                initial_lon=116.25,
                initial_lat=16.25,
                initial_alt=9_500.0,
            )
            commands = commander.action_for(11, 7)
            self.assertIsNotNone(commands)
            assert commands is not None
            self.assertEqual([int(row[0]) for row in commands], [4, 1])
            self.assertEqual(commands[0][2:], [116.25, 16.25, 9_500.0])
            commander.record_maneuver(11, 7, -1)

            events = commander.consume_decision_events()
            self.assertEqual([event["event_type"] for event in events], [
                "LAUNCH", "MANEUVER",
            ])
            self.assertEqual(set(events[0]), {
                "event_id", "event_type", "agent_id", "entity_id", "step",
                "target_id", "search_source", "satellite_source", "action",
            })
            self.assertEqual(events[0]["agent_id"], 3)
            self.assertEqual(events[0]["entity_id"], 11)
            self.assertEqual(events[0]["target_id"], 50)
            self.assertIsNone(events[0]["search_source"])
            self.assertEqual(commander.consume_decision_events(), ())
            self.assertEqual(len(commander.decision_ledger), 2)

            # RETARGET is legal on the immediately following timestep.
            policy.queue(11, kind="retarget", target_id=-100)
            retarget = commander.action_for(11, 8)
            self.assertEqual(int(retarget[0][0]), 2)
            new_events = commander.consume_decision_events()
            self.assertEqual(
                [event["event_type"] for event in new_events],
                ["RETARGET", "SEARCH"],
            )
            self.assertEqual(new_events[1]["search_source"], 11)

            commander.reset()
            self.assertEqual(commander.decision_ledger, ())
            self.assertEqual(commander.consume_decision_events(), ())
            self.assertNotIn(50, {target.entity_id for target in commander.targets})

    def test_low_platform_rejects_9400_9600_and_undiscovered_9500(self) -> None:
        for target_id in (1, 2, 50):
            with self.subTest(target_id=target_id), patch.dict(
                os.environ,
                {"RED_UNIFIED_DYNAMIC_LIFECYCLE": "1"},
                clear=False,
            ):
                policy = _LifecyclePolicy({11})
                commander = self._make_commander(policy)
                commander.begin_step((_observation(),))
                policy.queue(11, kind="launch", target_id=target_id)
                with self.assertRaises(RuntimeError):
                    commander.action_for(11, 1)
                self.assertNotIn(11, commander.launched_ids)
                self.assertNotIn(11, commander.target_by_platform)

    def test_high_and_medium_platforms_reject_detected_9500(self) -> None:
        for entity_type in (21000, 21001):
            with self.subTest(entity_type=entity_type), patch.dict(
                os.environ,
                {"RED_UNIFIED_DYNAMIC_LIFECYCLE": "1"},
                clear=False,
            ):
                policy = _LifecyclePolicy({11})
                commander = self._make_commander(policy)
                commander.begin_step((
                    _observation(
                        entity_type=entity_type,
                        detect_info={50: _detected_ship(detect_from=11)},
                    ),
                ))
                policy.queue(11, kind="launch", target_id=50)
                with self.assertRaises(RuntimeError):
                    commander.action_for(11, 1)
                self.assertNotIn(11, commander.launched_ids)
                self.assertNotIn(11, commander.target_by_platform)

    def test_provenance_falls_back_to_receiving_entity_and_step(self) -> None:
        fusion = InitialCatalogueTrackFusion(())
        observation = _observation(
            entity_id=77,
            step=12,
            detect_info={50: _detected_ship(detect_from=None, time=None)},
        )
        self.assertTrue(fusion.ingest(observation))
        self.assertEqual(fusion.source_for(50), 77)
        record = fusion.provenance_for(50)[0]
        self.assertEqual((record.detect_from, record.step), (77, 12))


    def test_9500_visibility_follows_engine_fused_actor_detect_info(self) -> None:
        with patch.dict(
            os.environ,
            {"RED_UNIFIED_DYNAMIC_LIFECYCLE": "1"},
            clear=False,
        ):
            policy = _LifecyclePolicy({11, 12})
            commander = self._make_commander(policy)
            commander.register_agent_identity(12, 4)
            commander.begin_step((
                _observation(
                    entity_id=11,
                    agent_id=3,
                    detect_info={50: _detected_ship(detect_from=11)},
                ),
                _observation(entity_id=12, agent_id=4),
            ))
            self.assertIn(50, {target.entity_id for target in policy.targets})
            self.assertEqual(commander.visible_9500_by_platform[11], {50})
            self.assertEqual(commander.visible_9500_by_platform[12], set())

            self.assertEqual(policy.actor_visibility[11], {50})
            self.assertEqual(policy.actor_visibility[12], set())
            policy.queue(12, kind="launch", target_id=50)
            with self.assertRaises(RuntimeError):
                commander.action_for(12, 7)

            # The engine has now fused A's track into B's detectInfo because
            # both entities belong to the same communication component.
            commander.begin_step((
                _observation(
                    entity_id=11,
                    agent_id=3,
                    step=8,
                    detect_info={50: _detected_ship(detect_from=11)},
                ),
                _observation(
                    entity_id=12,
                    agent_id=4,
                    step=8,
                    detect_info={50: _detected_ship(detect_from=11)},
                ),
            ))
            self.assertEqual(policy.actor_visibility[12], {50})
            policy.queue(12, kind="launch", target_id=50)
            actions = commander.action_for(12, 8)
            self.assertEqual(int(actions[-1][0]), 1)

    def test_search_credit_requires_search_before_current_source_detection(self) -> None:
        with patch.dict(
            os.environ,
            {"RED_UNIFIED_DYNAMIC_LIFECYCLE": "1"},
            clear=False,
        ):
            policy = _LifecyclePolicy({11})
            commander = self._make_commander(policy)
            commander.begin_step((
                _observation(
                    entity_id=11,
                    step=6,
                    detect_info={50: _detected_ship(detect_from=11, time=6)},
                ),
            ))
            commander._record_decision(
                "SEARCH",
                entity_id=11,
                step=7,
                target_id=-100,
                search_source=11,
                action=(116.0, 16.0),
            )
            self.assertIsNone(commander._search_source(11, 50))

            commander._record_decision(
                "SEARCH",
                entity_id=11,
                step=5,
                target_id=-100,
                search_source=11,
                action=(116.0, 16.0),
            )
            self.assertEqual(commander._search_source(11, 50), 11)

    def test_private_target_encoder_isolated_and_retains_only_current_target(self) -> None:
        targets = [
            {
                "entity_id": 1,
                "type": 9400,
                "nameChn": "目标1",
                "position": {"lon": 120.0, "lat": 20.0, "alt": 0.0},
            },
            {
                "entity_id": 50,
                "type": 9500,
                "nameChn": "无人船50",
                "position": {"lon": 130.0, "lat": 30.0, "alt": 0.0},
            },
        ]
        first = ObservationEncoder(
            {"entities": {}},
            target_slots=UNIFIED_TARGET_SLOTS,
            agent_id=3,
            locally_observed_target_types=(9500,),
        )
        second = ObservationEncoder(
            {"entities": {}},
            target_slots=UNIFIED_TARGET_SLOTS,
            agent_id=4,
            locally_observed_target_types=(9500,),
        )
        first.set_targets(targets)
        second.set_targets(targets)

        hidden_observation = _observation(entity_id=11, agent_id=3)
        first_encoded = first.encode(
            hidden_observation,
            launched=False,
            launch_step=-1,
            satellite_used=False,
            maneuver_state=0,
        )
        _, first_valid = first.target_view(hidden_observation)
        self.assertTrue(first_valid[0])
        self.assertFalse(first_valid[1])
        self.assertTrue(np.all(first.last_target_coordinates[1] == 0.0))
        hidden_slice = slice(
            ObservationEncoder.SELF_FEATURES + ObservationEncoder.TARGET_FEATURES,
            ObservationEncoder.SELF_FEATURES + 2 * ObservationEncoder.TARGET_FEATURES,
        )
        self.assertTrue(np.all(first_encoded[hidden_slice] == 0.0))

        detected_observation = _observation(
            entity_id=11,
            agent_id=3,
            step=6,
            detect_info={50: _detected_ship(detect_from=11, time=6)},
        )
        first.encode(
            detected_observation,
            launched=False,
            launch_step=-1,
            satellite_used=False,
            maneuver_state=0,
            current_target_index=1,
        )
        self.assertTrue(first.last_target_valid_mask[1])
        self.assertTrue(
            np.allclose(first.last_target_coordinates[1], [122.5, 22.5])
        )

        second.encode(
            _observation(entity_id=12, agent_id=4, step=6),
            launched=False,
            launch_step=-1,
            satellite_used=False,
            maneuver_state=0,
        )
        self.assertFalse(second.last_target_valid_mask[1])
        self.assertTrue(np.all(second.last_target_coordinates[1] == 0.0))

        first.encode(
            _observation(entity_id=11, agent_id=3, step=7),
            launched=True,
            launch_step=6,
            satellite_used=False,
            maneuver_state=0,
            current_target_index=1,
        )
        self.assertTrue(first.last_target_valid_mask[1])
        self.assertTrue(
            np.allclose(first.last_target_coordinates[1], [122.5, 22.5])
        )
        first.encode(
            _observation(entity_id=11, agent_id=3, step=8),
            launched=True,
            launch_step=6,
            satellite_used=False,
            maneuver_state=0,
            current_target_index=None,
        )
        self.assertFalse(first.last_target_valid_mask[1])

class _AgentPolicy:
    training = False

    def __init__(self) -> None:
        self.encoder = None

    def register_encoder(self, entity_id: int, encoder) -> None:
        del entity_id
        self.encoder = encoder

    def target_index_for(self, target_id: int | None) -> int | None:
        return 0 if target_id == 1 else None

    def select_maneuver(self, entity_id: int, encoded: np.ndarray) -> int:
        del entity_id, encoded
        return 2


class _AgentCommander:
    def __init__(self) -> None:
        self.identity = None
        self.maneuvers: list[tuple[int, int, int]] = []

    def register_agent_identity(self, entity_id: int, agent_id: int) -> None:
        self.identity = (entity_id, agent_id)

    def report(self, observation: dict) -> None:
        del observation

    def action_for(self, entity_id: int, step: int):
        del step
        return [[4.0, entity_id, 116.0, 16.0, 10_000.0],
                [1.0, entity_id, 120.0, 20.0]]

    def should_use_satellite(self, entity_id: int) -> bool:
        del entity_id
        return False

    def target_id_for(self, entity_id: int) -> int:
        del entity_id
        return 1

    def learning_task_context(self, entity_id: int):
        del entity_id
        return (1.0, 0.0, 0.0, 0.0, 0.0)

    def record_maneuver(self, entity_id: int, step: int, maneuver: int) -> None:
        self.maneuvers.append((entity_id, step, maneuver))


class AttackMissileLifecycleTests(unittest.TestCase):
    def test_unified_actor_uses_full_local_target_capacity_and_launch_maneuvers_same_frame(self) -> None:
        AttackMissileAgent = _load_attack_agent()
        policy = _AgentPolicy()
        commander = _AgentCommander()
        init_observation = {
            "entities": {
                1: {
                    "type": 9400,
                    "nameChn": "目标1",
                    "health": 100.0,
                    "position": {"lon": 120.0, "lat": 20.0, "alt": 0.0},
                }
            }
        }
        agent = AttackMissileAgent(
            3,
            11,
            init_observation,
            commander,
            motion_policy="unified_mappo",
            learning_policy=policy,
            hierarchical_learning=True,
        )
        self.assertIs(policy.encoder, agent.learning_encoder)
        self.assertEqual(policy.encoder.observation_dim, UNIFIED_LOCAL_OBSERVATION_DIM)
        self.assertFalse(policy.encoder.hierarchical_task_context)
        self.assertEqual(commander.identity, (11, 3))

        agent.set_observation(_observation(step=9))
        actions = agent.get_action()
        self.assertEqual(actions.shape, (3, 5))
        self.assertEqual(actions[:, 0].astype(int).tolist(), [4, 1, 0])
        self.assertEqual(commander.maneuvers, [(11, 9, 1)])


if __name__ == "__main__":
    unittest.main()
