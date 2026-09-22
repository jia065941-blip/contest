from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "core"))

from envengine.common import SimmerCommandType  # noqa: E402
from envengine.engine.engine import Engine  # noqa: E402
from envengine.environment.individual_reward import (  # noqa: E402
    target_reward_vector,
    target_reward_vector_fixed_21,
)
from envengine.sdk.base_struct.Message import Command  # noqa: E402
from envengine.simulator.simulator_factory import SimulatorFactory  # noqa: E402


def make_entity(
    entity_id: int,
    entity_type: int,
    *,
    health: float = 100.0,
    parent_id: int = -1,
):
    return SimpleNamespace(
        id=entity_id,
        entityType=entity_type,
        sideId=0,
        survivePoints=health,
        parentId=parent_id,
        posEcf=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        velEcf=SimpleNamespace(x=4.0, y=5.0, z=6.0),
    )


class FakeSimulator:
    def __init__(
        self,
        entity,
        *,
        update_log: list[tuple[float, int]] | None = None,
        sim_step: float = 50.0,
        damage_on_hit: float = 0.0,
    ):
        self.entity_ext = SimpleNamespace(entity=entity)
        self.simulator_sim_step = sim_step
        self.sim_time = 0.0
        self.update_log = update_log
        self.damage_on_hit = damage_on_hit
        self.launched = -1
        self.target_id = None

    def update(self) -> None:
        if self.update_log is not None:
            self.update_log.append((self.sim_time, self.entity_ext.entity.id))

    def command_received(self, command: Command) -> None:
        if command.commandTypeId == SimmerCommandType.DAMAGE:
            entity = self.entity_ext.entity
            entity.survivePoints = max(
                0.0,
                float(entity.survivePoints) - self.damage_on_hit,
            )
        elif (
            command.commandTypeId == SimmerCommandType.INTERCEPTOR_LAUNCH
            and self.launched == -1
        ):
            self.target_id = command.commandAttributes.targetId
            self.entity_ext.entity.parentId = -1
            self.launched = 1

    def reset(self) -> None:
        return None

    def init_model(self) -> None:
        return None


class FakeEngineFactory:
    def __init__(self, simulators):
        self.simulators = simulators
        self.contexts = []

    def set_event_context(self, *, step: int, sim_time: float) -> None:
        self.contexts.append((step, sim_time))

    def process_ai_commands(self, _commands) -> None:
        return None

    def update_sim_time(self, sim_time: float) -> None:
        self.sim_time = float(sim_time)
        for simulator in self.simulators:
            simulator.sim_time = float(sim_time)

    def get_all_lived_simulators(self):
        return self.simulators

    def process_commands(self) -> None:
        return None


def make_factory(simulators: dict[int, FakeSimulator]) -> SimulatorFactory:
    factory = SimulatorFactory.__new__(SimulatorFactory)
    factory._profile = SimpleNamespace(
        imagineProfile=SimpleNamespace(simTime=0.0)
    )
    factory._simulators = simulators
    factory._simulators_by_type = {}
    factory._simulators_by_side = {}
    factory._command_queue = []
    factory.current_round = 3
    factory.target_hit_relation = {}
    factory._causal_event_ledger = []
    factory._causal_event_sequence = 0
    factory._causal_event_consume_cursor = 0
    factory._event_step = 7
    factory._event_sim_time = 700.0
    factory.process_hit = lambda command: command
    return factory


class EnvironmentCausalPrimitiveTests(unittest.TestCase):
    def test_fixed_21_reward_does_not_change_legacy_normalization(self) -> None:
        previous = {54: {"health": 100.0}, 55: {"health": 100.0}}
        current = {54: {"health": 70.0}, 55: {"health": 100.0}}
        kwargs = {
            "current_entities": current,
            "previous_entities": previous,
            "initial_health": {54: 100.0, 55: 100.0},
            "objective_weights": {54: 7.0, 55: 7.0},
        }

        fixed = target_reward_vector_fixed_21(**kwargs)
        legacy = target_reward_vector(**kwargs)

        self.assertAlmostEqual(fixed[54], (7.0 / 21.0) * 0.3)
        self.assertAlmostEqual(legacy[54], (7.0 / 14.0) * 0.3)
        self.assertEqual(fixed[55], 0.0)

    def test_same_time_updates_follow_stable_entity_ids(self) -> None:
        update_log = []
        later_id = FakeSimulator(make_entity(20, 21000), update_log=update_log)
        earlier_id = FakeSimulator(make_entity(10, 21000), update_log=update_log)
        factory = FakeEngineFactory([later_id, earlier_id])

        engine = Engine.__new__(Engine)
        engine.simulator_factory = factory
        engine.current_step = 0
        engine.sim_time = 0.0
        engine.sim_step = 100.0
        engine.speed_multiplier = 0.0
        engine.share_detect_between_missiles = lambda: None

        engine.step([])

        self.assertEqual(
            update_log,
            [(0.0, 10), (0.0, 20), (50.0, 10), (50.0, 20)],
        )
        self.assertEqual(factory.contexts, [(1, 0.0), (1, 100.0)])

    def test_hit_and_launch_ledger_is_incremental_and_resettable(self) -> None:
        attacker = FakeSimulator(make_entity(101, 21001))
        target = FakeSimulator(
            make_entity(54, 9400, health=10.0),
            damage_on_hit=4.0,
        )
        interceptor = FakeSimulator(
            make_entity(202, 24000, parent_id=960),
        )
        factory = make_factory({101: attacker, 54: target, 202: interceptor})

        factory.command_queue.append(
            Command(
                prevTriggerId=101,
                executorId=54,
                commandTypeId=SimmerCommandType.DAMAGE,
            )
        )
        factory.process_commands()

        first_batch = factory.consume_causal_events()
        self.assertEqual(len(first_batch), 1)
        hit = first_batch[0]
        self.assertEqual(hit["event_type"], "direct_hit")
        self.assertEqual(hit["event_id"], "3:1")
        self.assertEqual(hit["attacking_entity_id"], 101)
        self.assertEqual(hit["target_entity_id"], 54)
        self.assertEqual(hit["actual_damage"], 4.0)
        self.assertEqual(
            factory.target_hit_relation[54][0]["event_id"],
            hit["event_id"],
        )
        self.assertEqual(factory.consume_causal_events(), ())

        launch = Command(
            executorId=202,
            commandTypeId=SimmerCommandType.INTERCEPTOR_LAUNCH,
            commandAttributes=SimpleNamespace(targetId=101),
        )
        factory.command_queue.append(launch)
        factory.process_commands()

        second_batch = factory.consume_causal_events()
        self.assertEqual(len(second_batch), 1)
        interception = second_batch[0]
        self.assertEqual(interception["event_type"], "interceptor_launch")
        self.assertEqual(interception["event_id"], "3:2")
        self.assertEqual(interception["interceptor_entity_id"], 202)
        self.assertEqual(interception["intercepted_red_entity_id"], 101)
        self.assertEqual(interception["defending_parent_entity_id"], 960)

        factory.command_queue.append(launch)
        factory.process_commands()
        self.assertEqual(factory.consume_causal_events(), ())
        self.assertEqual(len(factory.causal_event_ledger), 2)

        factory.reset_all()
        self.assertEqual(factory.causal_event_ledger, ())
        self.assertEqual(factory.consume_causal_events(), ())
        self.assertEqual(factory.current_round, 4)


if __name__ == "__main__":
    unittest.main()

