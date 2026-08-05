from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from red_strategy_lab.adapter import launch_rows, observation_from_engine
from red_strategy_lab.deployment import assign_slots, make_slots
from red_strategy_lab.domain import Assignment, Decision, Platform, Position, Target
from red_strategy_lab.evaluation import EpisodeMetrics, split_seeds
from red_strategy_lab.geometry import distance_km, point_in_polygon
from red_strategy_lab.optimization import ParameterSearch


class ContractTests(unittest.TestCase):
    def test_deployment_slots_are_inside_and_separated(self) -> None:
        polygon = [Position(0, 0), Position(0, 1), Position(1, 1), Position(1, 0)]
        slots = make_slots(polygon, count=3, min_distance_km=20)
        self.assertTrue(all(point_in_polygon(slot.position, polygon) for slot in slots))
        self.assertTrue(all(distance_km(a.position, b.position) >= 20 for index, a in enumerate(slots) for b in slots[index + 1:]))
        positions = assign_slots((Platform(1, "H", Position(0, 0)), Platform(2, "L", Position(0, 0))), slots)
        self.assertEqual(set(positions), {1, 2})

    def test_adapter_contract_and_launch_schedule(self) -> None:
        raw = {"step": 4, "entities": {"8": {"side": 0, "type": 21000, "position": {"lon": 118, "lat": 22}, "health": 1}}}
        target = Target(99, Position(120, 23), value=5)
        converted = observation_from_engine(raw, [target], launched_ids={8})
        self.assertEqual(converted.platforms[0].kind, "H")
        self.assertTrue(converted.platforms[0].launched)
        decision = Decision(assignments=(Assignment(8, 99, 4, 1.0), Assignment(9, 99, 3, 1.0), Assignment(10, 99, 5, 1.0)))
        self.assertEqual(launch_rows(decision, [target], 4), [[1.0, 8.0, 120, 23]])

    def test_metrics_and_seed_split(self) -> None:
        metrics = EpisodeMetrics(initial_target_value=10)
        metrics.record_decision(Decision(assignments=(Assignment(1, 9, 0, 1), Assignment(2, 9, 0, 1))), target_capacity=1)
        metrics.destroyed_target_value = 5
        summary = metrics.summary()
        self.assertEqual(summary["completion_rate"], 0.5)
        self.assertEqual(summary["constraint_violations"], 1)
        self.assertEqual(split_seeds([3, 1, 2, 4]), ((1, 2), (3, 4)))

    def test_parameter_search_uses_tail_risk(self) -> None:
        search = ParameterSearch(risk_weight=1.0, tail_fraction=0.5)
        result = search.select(
            [{"name": 1.0}, {"name": 2.0}],
            lambda candidate: [8, 8] if candidate["name"] == 1.0 else [20, -10],
        )
        self.assertEqual(result.parameters["name"], 1.0)


if __name__ == "__main__":
    unittest.main()
