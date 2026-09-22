from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from envengine.environment.training_env import _learning_runtime_fields
from envengine.simulator.simulator_factory import SimulatorFactory
from experiments.unified_mappo.model import (
    HybridActionMask,
    HybridMAPPOConfig,
    HybridMAPPOTrainer,
)
from policies.red.learning.red_policy import UNIFIED_LOCAL_OBSERVATION_DIM
from policies.red.learning.unified_mappo_policy import (
    TARGET_SET_FEATURE_DIM, UnifiedMAPPOSharedPolicy,
)
from tools.train_start_state_option_curriculum import (
    c0_team_search_partition,
    search_segment_credits,
    team_search_boundary_credits,
    units_for_stage,
)


class SearchSegmentCreditTests(unittest.TestCase):
    def test_each_boundary_receives_only_its_own_leg_outcome(self):
        samples = [
            {
                "step": 11,
                "targets": [{"alive": True, "distance_m": 200_000.0}],
            },
            {
                "step": 19,
                "targets": [{"alive": True, "distance_m": 100_000.0}],
            },
            {
                "step": 21,
                "targets": [{"alive": True, "distance_m": 100_000.0}],
            },
            {
                "step": 30,
                "targets": [{"alive": True, "distance_m": 150_000.0}],
            },
        ]
        credits = search_segment_credits(
            [10, 20], samples, {2573: 25}
        )

        self.assertEqual([row["step"] for row in credits], [10, 20])
        self.assertGreater(credits[0]["proximity_gain"], 0.0)
        self.assertEqual(credits[0]["direct_return"], 0.0)
        self.assertEqual(credits[1]["proximity_gain"], 0.0)
        self.assertAlmostEqual(credits[1]["direct_return"], 1.0 / 9.0)

    def test_empirical_advantages_are_normalized_only_on_action_boundaries(self):
        advantages = torch.tensor([1.0, 3.0, 100.0])
        mask = torch.tensor([True, True, False])

        normalized = HybridMAPPOTrainer._normalize_empirical_advantages(
            advantages, mask
        )

        torch.testing.assert_close(normalized, torch.tensor([-1.0, 1.0, 0.0]))

    def test_team_credit_matches_agent_and_latest_pre_detection_boundary(self):
        credits = team_search_boundary_credits(
            boundary_steps=[10, 20, 10, 30],
            boundary_agent_ids=[1, 1, 2, 2],
            direct_first_steps_by_entity={
                "101": {"168": 15, "2569": 25},
                "102": {"169": 29, "2569": 24},
            },
            agent_id_by_entity={"101": 1, "102": 2},
            objective_9500_count=9,
        )
        by_key = {
            (row["agent_id"], row["step"]): row for row in credits
        }

        self.assertAlmostEqual(by_key[(1, 10)]["return"], 1.0 / 9.0)
        self.assertEqual(by_key[(1, 20)]["return"], 0.0)
        self.assertAlmostEqual(by_key[(2, 10)]["return"], 2.0 / 9.0)
        self.assertEqual(by_key[(2, 30)]["return"], 0.0)

    def test_team_grid_novelty_is_shared_and_revisits_are_zero(self):
        credits = team_search_boundary_credits(
            boundary_steps=[10, 10, 20, 20],
            boundary_agent_ids=[1, 2, 1, 2],
            direct_first_steps_by_entity={},
            agent_id_by_entity={},
            objective_9500_count=9,
            boundary_search_indices=[3, 3, 4, 3],
            search_cell_count=192,
        )
        by_key = {
            (row["agent_id"], row["step"]): row for row in credits
        }
        self.assertAlmostEqual(
            by_key[(1, 10)]["grid_novelty_return"], 1.0 / 384.0
        )
        self.assertAlmostEqual(
            by_key[(2, 10)]["grid_novelty_return"], 1.0 / 384.0
        )
        self.assertAlmostEqual(
            by_key[(1, 20)]["grid_novelty_return"], 1.0 / 192.0
        )
        self.assertEqual(by_key[(2, 20)]["grid_novelty_return"], 0.0)

    def test_team_partition_is_bounded_rotating_and_ignores_damage_anchors(self):
        trace = ROOT / "tests" / "fixtures" / "team_partition_trace.json"
        self.addCleanup(lambda: trace.unlink(missing_ok=True))
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_text(
            '{"steps": [{"step": 2, "actions": ['
            + ",".join(
                '{"executor_id": %d, "commandType_id": 200}' % entity_id
                for entity_id in range(100, 120)
            )
            + "]}]}",
            encoding="utf-8",
        )
        row = {
            "seed": 7,
            "trace": str(trace),
            "damage_anchors": {
                "900": {"attacking_entity_id": 107},
            },
        }
        entity_types = {entity_id: 21002 for entity_id in range(100, 120)}

        controlled, reserved = c0_team_search_partition(
            row,
            entity_types,
            required_attack_reserve=9,
            max_controlled_l=8,
            selector_offset=0,
        )

        self.assertEqual(len(controlled), 8)
        self.assertEqual(len(reserved), 9)
        self.assertTrue(
            {unit["executor_id"] for unit in controlled}.isdisjoint(reserved)
        )
        self.assertIn(107, {unit["executor_id"] for unit in controlled})


def _satellite_factory() -> SimulatorFactory:
    factory = object.__new__(SimulatorFactory)
    factory.sim_time = 1_000.0
    factory.sat_use_minutes = 3.0
    factory.red_sat_max_use_count = 2
    factory.red_sat_use_count = 0
    factory._satellite_use_end_time = 0.0
    factory._satellite_use_count_by_entity = {}
    factory._satellite_use_end_time_by_entity = {}
    factory.record_satellite_request_accepted = lambda simulator: "accepted"
    return factory


def _simulator(entity_id: int):
    return SimpleNamespace(
        entity_ext=SimpleNamespace(
            entity=SimpleNamespace(
                id=entity_id,
                nameChn=f"red-{entity_id}",
                velEcf=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            )
        )
    )


class SatelliteActionContractTests(unittest.TestCase):
    def test_requests_have_independent_counts_and_overlapping_three_minute_windows(self):
        factory = _satellite_factory()
        first = _simulator(11)
        second = _simulator(12)

        factory._process_missile_use_satellite(None, first)
        first_end = factory.satellite_use_end_time(11)
        self.assertEqual(first_end, 181_000.0)
        factory.sim_time = 2_000.0
        factory._process_missile_use_satellite(None, second)

        self.assertEqual(factory.satellite_use_count(11), 1)
        self.assertEqual(factory.satellite_use_count(12), 1)
        self.assertTrue(factory.is_using_satellite(11))
        self.assertTrue(factory.is_using_satellite(12))
        factory._process_missile_use_satellite(None, first)
        self.assertEqual(factory.satellite_use_count(11), 1)

        factory.sim_time = first_end
        self.assertFalse(factory.is_using_satellite(11))
        self.assertTrue(factory.is_using_satellite(12))

    def test_actor_runtime_fields_use_requesting_entity_state(self):
        factory = _satellite_factory()
        first = _simulator(11)
        second = _simulator(12)
        factory._process_missile_use_satellite(None, first)

        first_fields = _learning_runtime_fields(first, factory)
        second_fields = _learning_runtime_fields(second, factory)
        self.assertTrue(first_fields["is_using_satellite"])
        self.assertEqual(first_fields["satellite_remaining_uses"], 1)
        self.assertFalse(second_fields["is_using_satellite"])
        self.assertEqual(second_fields["satellite_remaining_uses"], 2)


class TemporalAttackOptionContractTests(unittest.TestCase):
    @staticmethod
    def _policy() -> UnifiedMAPPOSharedPolicy:
        policy = object.__new__(UnifiedMAPPOSharedPolicy)
        policy.attack_option_min_dwell_steps = 60
        policy._attack_option_by_entity = {
            11: {
                "target_index": 0,
                "target_id": 51,
                "target_lon": 120.0,
                "target_lat": 20.0,
                "last_decision_step": 10,
                "controller": "student",
                "completed": False,
            }
        }
        policy._attack_option_termination_count = 0
        policy._attack_option_termination_reasons = {}
        policy._search_option_distance_sample_count = 0
        policy._search_option_min_distance_km = float("inf")
        policy._search_option_last_distance_km = None
        policy._search_option_emergency_boundary_count = 0
        return policy

    def test_goal_boundary_is_closed_during_hmin_then_opens_for_legal_retarget(self):
        policy = self._policy()
        valid = torch.tensor([True, True, False])
        self.assertFalse(policy._attack_option_boundary(11, valid, 69))
        self.assertTrue(policy._attack_option_boundary(11, valid, 70))
        self.assertIn(11, policy.active_student_option_entity_ids)

    def test_destroyed_or_illegal_target_terminates_immediately(self):
        policy = self._policy()
        valid = torch.tensor([False, True, False])
        self.assertTrue(policy._attack_option_boundary(11, valid, 11))
        self.assertNotIn(11, policy.active_student_option_entity_ids)
        self.assertEqual(policy._attack_option_termination_count, 1)
        self.assertEqual(
            policy._attack_option_termination_reasons,
            {"target_illegal_or_destroyed": 1},
        )

    def test_search_option_reopens_near_waypoint_after_minimum_dwell(self):
        policy = self._policy()
        policy.search_option_chaining = True
        policy.search_option_dwell_steps = 120
        policy.search_option_reopen_distance_km = 15.0
        policy.search_option_emergency_reopen_distance_km = 5.0
        policy._attack_option_by_entity[11].update({
            "target_index": 1,
            "target_id": -100,
            "target_lon": 120.0,
            "target_lat": 20.0,
            "option_kind": "search",
        })
        valid = torch.tensor([False, True, False])
        near = {"self": {"position": {"lon": 120.05, "lat": 20.0}}}
        far = {"self": {"position": {"lon": 121.0, "lat": 20.0}}}
        self.assertFalse(policy._attack_option_boundary(11, valid, 129, near))
        self.assertFalse(policy._attack_option_boundary(11, valid, 130, far))
        self.assertTrue(policy._attack_option_boundary(11, valid, 130, near))
        self.assertIn(11, policy.active_student_option_entity_ids)

    def test_search_option_emergency_boundary_prevents_early_waypoint_self_destruct(self):
        policy = self._policy()
        policy.search_option_chaining = True
        policy.search_option_dwell_steps = 120
        policy.search_option_reopen_distance_km = 15.0
        policy.search_option_emergency_reopen_distance_km = 5.0
        policy._attack_option_by_entity[11].update({
            "target_index": 1,
            "target_id": -100,
            "target_lon": 120.0,
            "target_lat": 20.0,
            "option_kind": "search",
        })
        valid = torch.tensor([False, True, False])
        imminent = {
            "self": {"position": {"lon": 120.02, "lat": 20.0}}
        }
        self.assertTrue(
            policy._attack_option_boundary(11, valid, 20, imminent)
        )

    def test_search_option_reopens_immediately_when_9500_becomes_legal(self):
        policy = self._policy()
        policy.search_option_chaining = True
        policy.search_option_dwell_steps = 120
        policy.search_option_reopen_distance_km = 15.0
        policy.search_option_emergency_reopen_distance_km = 5.0
        policy._attack_option_by_entity[11].update({
            "target_index": 1,
            "target_id": -100,
            "target_lon": 120.0,
            "target_lat": 20.0,
            "option_kind": "search",
        })
        valid = torch.tensor([True, True, False])
        far = {"self": {"position": {"lon": 121.0, "lat": 20.0}}}
        self.assertTrue(policy._attack_option_boundary(11, valid, 11, far))

    def test_lightweight_search_boundary_uses_only_local_9500_track(self):
        policy = self._policy()
        policy.trajectory_counterfactual = False
        policy.search_option_dwell_steps = 120
        policy.search_option_reopen_distance_km = 15.0
        policy.search_option_emergency_reopen_distance_km = 5.0
        policy._target_slot_ids = [51, 168, -100]
        policy._attack_option_by_entity[11].update({
            "target_index": 2,
            "target_id": -100,
            "target_lon": 120.0,
            "target_lat": 20.0,
            "last_decision_step": 10,
            "option_kind": "search",
        })
        far = {
            "step": 11,
            "self": {
                "position": {"lon": 121.0, "lat": 20.0},
                "detectInfo": {},
            },
        }
        self.assertEqual(
            policy.persistent_search_boundary_entity_ids({11: far}),
            frozenset(),
        )
        far["self"]["detectInfo"] = {
            168: {"entity_id": 168, "entity_type": 9500}
        }
        self.assertEqual(
            policy.persistent_search_boundary_entity_ids({11: far}),
            frozenset({11}),
        )

    def test_lightweight_attack_boundary_skips_deterministic_keep_steps(self):
        policy = self._policy()
        policy.trajectory_counterfactual = False
        policy._target_slot_ids = [51, 168, -100]
        policy._full_target_runtime_states = {51: (1.0, 0.0, 1.0)}
        observation = {"step": 69, "self": {}}
        self.assertEqual(
            policy.persistent_search_boundary_entity_ids({11: observation}),
            frozenset(),
        )
        observation["step"] = 70
        self.assertEqual(
            policy.persistent_search_boundary_entity_ids({11: observation}),
            frozenset({11}),
        )

    @staticmethod
    def _evasion_policy() -> UnifiedMAPPOSharedPolicy:
        policy = object.__new__(UnifiedMAPPOSharedPolicy)
        policy.trajectory_counterfactual = False
        policy.deterministic_evasion_tcpa_seconds = 120.0
        policy.deterministic_evasion_dcpa_km = 20.0
        policy.deterministic_evasion_max_track_age_steps = 10
        policy.deterministic_evasion_pulse_steps = 10
        policy.deterministic_evasion_cooldown_steps = 10
        policy._deterministic_evasion_state = {}
        policy._deterministic_evasion_trigger_count = 0
        policy._deterministic_evasion_active_step_count = 0
        policy._deterministic_evasion_visible_threat_count = 0
        return policy

    @staticmethod
    def _evasion_observation(*, step: int, track_y: float, track_vx: float):
        return {
            "step": step,
            "sim_time": float(step * 1000),
            "sim_step": 1000.0,
            "self": {
                "position": {"lon": 0.0, "lat": 0.0},
                "pos_ecf": {"x": 0.0, "y": 0.0, "z": 0.0},
                "vel_ecf": {"x": 100.0, "y": 0.0, "z": 0.0},
                "detectInfo": {
                    24001: {
                        "entity_id": 24001,
                        "entity_type": 24000,
                        "time": float(step * 1000),
                        "pos_ecf": {"x": 10_000.0, "y": track_y, "z": 0.0},
                        "vel_ecf": {"x": track_vx, "y": 0.0, "z": 0.0},
                    }
                },
            },
        }

    def test_deterministic_evasion_uses_only_closing_local_track_and_retriggers(self):
        policy = self._evasion_policy()
        closing = self._evasion_observation(
            step=10, track_y=2_000.0, track_vx=-100.0
        )
        first = policy._deterministic_evasion_maneuver(11, closing)
        self.assertEqual(first, 2)
        self.assertEqual(policy._deterministic_evasion_trigger_count, 1)
        self.assertEqual(
            policy._deterministic_evasion_maneuver(11, {**closing, "step": 11}),
            first,
        )
        cooldown = {**closing, "step": 20, "sim_time": 20_000.0}
        self.assertEqual(
            policy._deterministic_evasion_maneuver(11, cooldown), 1
        )
        retrigger = self._evasion_observation(
            step=30, track_y=2_000.0, track_vx=-100.0
        )
        self.assertEqual(
            policy._deterministic_evasion_maneuver(11, retrigger), 2
        )
        self.assertEqual(policy._deterministic_evasion_trigger_count, 2)

    def test_deterministic_evasion_ignores_receding_interceptor(self):
        policy = self._evasion_policy()
        receding = self._evasion_observation(
            step=10, track_y=0.0, track_vx=500.0
        )
        self.assertEqual(
            policy._deterministic_evasion_maneuver(11, receding), 1
        )
        self.assertEqual(policy._deterministic_evasion_trigger_count, 0)

    def test_c3a_controls_one_root_attack_unit_as_a_sticky_option(self):
        groups = [
            [{"unit_id": "root", "executor_id": 11, "timestep": 7, "command_type": 200}],
            [{"unit_id": "later", "executor_id": 11, "timestep": 20, "command_type": 3014}],
        ]
        self.assertEqual(units_for_stage(groups, "C3a"), [groups[0][0]])


    def test_low_performance_target_mask_allows_only_discovered_9500_or_search(self):
        policy = object.__new__(UnifiedMAPPOSharedPolicy)
        policy.dynamic_lifecycle = True
        policy._targets = [
            {"entity_id": 51, "type": 9400},
            {"entity_id": 52, "type": 9500},
            {"entity_id": 53, "type": 9600},
            {"entity_id": -100, "type": -1},
        ]
        allowed = policy._target_valid_for_entity_type(
            21002, torch.tensor([True, True, True, True])
        )
        self.assertEqual(allowed.tolist(), [False, True, False, True])

    def test_high_and_medium_target_masks_reject_zero_hit_rate_9500(self):
        policy = object.__new__(UnifiedMAPPOSharedPolicy)
        policy.dynamic_lifecycle = True
        policy._targets = [
            {"entity_id": 51, "type": 9400},
            {"entity_id": 52, "type": 9500},
            {"entity_id": 53, "type": 9600},
            {"entity_id": -100, "type": -1},
        ]
        for entity_type in (21000, 21001):
            with self.subTest(entity_type=entity_type):
                allowed = policy._target_valid_for_entity_type(
                    entity_type, torch.tensor([True, True, True, True])
                )
                self.assertEqual(
                    allowed.tolist(), [True, False, True, False]
                )

    def test_dense_counterfactual_exports_only_current_legal_targets(self):
        policy = object.__new__(UnifiedMAPPOSharedPolicy)
        policy._target_slot_ids = [51, -100, 52]
        policy._search_bounds = (120.0, 122.0, 20.0, 22.0)
        policy._pending_motion = {
            11: SimpleNamespace(
                mask_target=True,
                target_valid_mask=torch.tensor([True, True, False]),
                target_coordinates=torch.tensor([
                    [120.5, 20.5], [0.0, 0.0], [121.5, 21.5]
                ]),
                action=SimpleNamespace(search_xy=torch.tensor([0.0, 0.0])),
            )
        }
        candidates = policy.prepared_legal_targets(11)
        self.assertEqual([row["target_id"] for row in candidates], [51, -100])
        self.assertEqual(candidates[0]["target"], {"x": 120.5, "y": 20.5, "z": 0.0})
        self.assertEqual(candidates[1]["target"], {"x": 121.0, "y": 21.0, "z": 0.0})
        attack_candidates = policy.prepared_legal_targets(
            11, include_search=False
        )
        self.assertEqual(
            [row["target_id"] for row in attack_candidates], [51]
        )


class JointProbabilityContractTests(unittest.TestCase):
    def test_only_effective_heads_enter_joint_log_probability_and_entropy(self):
        trainer = HybridMAPPOTrainer(
            HybridMAPPOConfig(
                observation_dim=8,
                target_slots=2,
                hidden_dim=16,
                device="cpu",
            )
        )
        observations = torch.zeros(1, 8)
        coordinates = torch.zeros(1, 2, 2)
        target_valid = torch.ones(1, 2, dtype=torch.bool)
        mask = HybridActionMask(
            presence=torch.zeros(1, dtype=torch.bool),
            initial_position=torch.zeros(1, dtype=torch.bool),
            search_position=torch.zeros(1, dtype=torch.bool),
            retarget=torch.zeros(1, dtype=torch.bool),
            target=torch.zeros(1, dtype=torch.bool),
            maneuver=torch.ones(1, dtype=torch.bool),
            satellite=torch.zeros(1, dtype=torch.bool),
        )
        output = trainer.model.act(
            observations,
            coordinates,
            target_valid,
            mask,
            deterministic=True,
        )
        maneuver = torch.distributions.Categorical(
            logits=trainer.model.distribution_parameters(observations)[
                "maneuver_logits"
            ]
        )
        self.assertTrue(torch.allclose(
            output.log_prob,
            maneuver.log_prob(output.action.maneuver_index),
        ))
        self.assertTrue(torch.allclose(output.entropy, maneuver.entropy()))
        self.assertFalse(output.action_mask.target.item())
        self.assertEqual(output.action.satellite.item(), 0)


    def test_satellite_head_is_active_on_same_frame_launch_but_not_wait(self):
        trainer = HybridMAPPOTrainer(
            HybridMAPPOConfig(
                observation_dim=8, target_slots=2, hidden_dim=16, device="cpu"
            )
        )
        observations = torch.zeros(1, 8)
        coordinates = torch.zeros(1, 2, 2)
        target_valid = torch.ones(1, 2, dtype=torch.bool)
        mask = HybridActionMask(
            presence=torch.ones(1, dtype=torch.bool),
            initial_position=torch.ones(1, dtype=torch.bool),
            search_position=torch.zeros(1, dtype=torch.bool),
            retarget=torch.zeros(1, dtype=torch.bool),
            target=torch.ones(1, dtype=torch.bool),
            maneuver=torch.ones(1, dtype=torch.bool),
            satellite=torch.ones(1, dtype=torch.bool),
        )
        with torch.no_grad():
            trainer.model.presence_head.bias.fill_(-100.0)
            trainer.model.satellite_head.bias.fill_(100.0)
        waiting = trainer.model.act(
            observations, coordinates, target_valid, mask, deterministic=True
        )
        self.assertFalse(waiting.action_mask.satellite.item())
        self.assertEqual(waiting.action.satellite.item(), 0)
        with torch.no_grad():
            trainer.model.presence_head.bias.fill_(100.0)
        launching = trainer.model.act(
            observations, coordinates, target_valid, mask, deterministic=True
        )
        self.assertTrue(launching.action_mask.satellite.item())
        self.assertEqual(launching.action.satellite.item(), 1)

    def test_search_cell_mask_controls_sampling_and_log_prob_recomputation(self):
        trainer = HybridMAPPOTrainer(
            HybridMAPPOConfig(
                observation_dim=8,
                target_slots=2,
                hidden_dim=16,
                search_grid_width=2,
                search_grid_height=2,
                search_target_index=0,
                device="cpu",
            )
        )
        with torch.no_grad():
            trainer.model.search_head.weight.zero_()
            trainer.model.search_head.bias.copy_(
                torch.tensor([0.0, 1.0, 2.0, 100.0])
            )
        observations = torch.zeros(1, 8)
        coordinates = torch.zeros(1, 2, 2)
        target_valid = torch.tensor([[True, False]])
        cell_valid = torch.tensor([[False, True, False, False]])
        active = torch.ones(1, dtype=torch.bool)
        inactive = torch.zeros(1, dtype=torch.bool)
        mask = HybridActionMask(
            presence=inactive,
            initial_position=inactive,
            search_position=active,
            retarget=inactive,
            target=active,
            maneuver=inactive,
            satellite=inactive,
        )
        output = trainer.model.act(
            observations,
            coordinates,
            target_valid,
            mask,
            search_valid_mask=cell_valid,
            deterministic=True,
            independent_target=True,
        )
        self.assertEqual(output.action.search_index.item(), 1)
        recomputed, _, _ = trainer.model.evaluate_actions(
            observations,
            coordinates,
            target_valid,
            output.action_mask,
            output.action,
            search_valid_mask=cell_valid,
        )
        self.assertTrue(torch.allclose(output.log_prob, recomputed, atol=1e-6))


class ReachableSearchMaskTests(unittest.TestCase):
    def test_mask_uses_only_self_position_and_public_search_bounds(self):
        policy = object.__new__(UnifiedMAPPOSharedPolicy)
        policy.trainer = SimpleNamespace(
            device=torch.device("cpu"),
            config=SimpleNamespace(
                search_grid_width=16,
                search_grid_height=12,
            ),
        )
        policy.search_reachable_mask = True
        policy.search_leg_min_distance_km = 50.0
        policy.search_leg_max_distance_km = 350.0
        policy._search_bounds = (115.8, 128.2, 18.4, 28.2)
        policy._search_reachable_mask_sample_count = 0
        policy._search_reachable_valid_count_sum = 0
        policy._search_reachable_valid_count_min = 192
        policy._search_reachable_valid_count_max = 0
        first = policy._search_grid_reachable_mask({
            "self": {"position": {"lon": 118.5, "lat": 23.0}},
            "hidden_target": {"lon": 121.0, "lat": 24.0},
        })
        second = policy._search_grid_reachable_mask({
            "self": {"position": {"lon": 118.5, "lat": 23.0}},
            "hidden_target": {"lon": 127.0, "lat": 28.0},
        })
        self.assertTrue(torch.equal(first, second))
        self.assertGreater(int(first.sum().item()), 1)
        self.assertLess(int(first.sum().item()), 192)
        self.assertEqual(policy._search_reachable_mask_sample_count, 2)


if __name__ == "__main__":
    unittest.main()


def test_target_set_features_use_only_local_bda_and_cluster_members():
    policy = object.__new__(UnifiedMAPPOSharedPolicy)
    policy.local_observation_dim = UNIFIED_LOCAL_OBSERVATION_DIM
    policy._target_slot_ids = [51, 52] + [None] * 23
    policy._attack_option_by_entity = {
        12: {"target_id": 51, "completed": False},
        13: {"target_id": 51, "completed": False},
    }
    policy._external_target_by_entity = {
        12: 52,
        13: 52,
    }
    policy._full_target_runtime_states = {52: (0.1, 0.9, 1.0)}
    valid = np.zeros(25, dtype=np.bool_)
    valid[:2] = True
    policy._encoders = {
        11: SimpleNamespace(target_feature_dim=15, last_target_valid_mask=valid, last_entity_type=21000),
        12: SimpleNamespace(target_feature_dim=15, last_target_valid_mask=valid, last_entity_type=21000),
        13: SimpleNamespace(target_feature_dim=15, last_target_valid_mask=valid, last_entity_type=21001),
    }
    track = SimpleNamespace(
        entity_id=51,
        health_remaining=40.0,
        health_max=100.0,
        health_observed=True,
    )
    observation = {
        "self": {
            "detectInfo": {51: track},
            "commRangeInfo": [11, 12],
        }
    }
    encoded = np.zeros(UNIFIED_LOCAL_OBSERVATION_DIM, dtype=np.float32)
    features = policy._target_set_features(
        encoded,
        entity_id=11,
        observation=observation,
    )
    assert features.shape == (25, TARGET_SET_FEATURE_DIM)
    np.testing.assert_allclose(features[0, 12:15], [0.4, 0.6, 1.0])
    np.testing.assert_allclose(features[1, 12:15], [1.0, 0.0, 0.0])
    assert np.isclose(features[0, 20], 1.0)
    assert np.isclose(features[0, 21], 1.0)
    assert np.isclose(features[1, 20], 0.0)
    assert np.isclose(features[1, 21], 0.0)
    assert np.isclose(features[0, 15], 1.0 / 3.0)
    assert np.isclose(features[0, 16], 1.0)
    assert np.isclose(features[0, 17], 1.0)
    assert np.isclose(features[0, 18], 0.0)
    assert np.isclose(features[0, 19], 0.0)
    # An active student option overrides the native/teacher assignment.
    assert np.isclose(features[1, 16], 0.0)

    observation["self"]["commRangeInfo"] = [11, 13]
    features = policy._target_set_features(
        encoded,
        entity_id=11,
        observation=observation,
    )
    assert np.isclose(features[0, 16], 1.0)
    assert np.isclose(features[0, 17], 0.0)
    assert np.isclose(features[0, 18], 1.0)
    assert np.isclose(features[0, 19], 0.0)

    policy._attack_option_by_entity[13]["completed"] = True
    features = policy._target_set_features(
        encoded,
        entity_id=11,
        observation=observation,
    )
    # Once the student option is complete, the external assignment supplies
    # only the new observation context without changing legacy feature
    # semantics, reopening, or taking ownership of an option.
    assert np.isclose(features[0, 16], 0.0)
    assert np.isclose(features[1, 16], 0.0)
    assert np.isclose(features[1, 18], 0.0)
    assert np.isclose(features[1, 22], 1.0 / 8.0)
    assert np.isclose(features[1, 24], 1.0 / 8.0)
