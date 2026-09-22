from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from policies.red.learning.red_policy import (
    GlobalStateEncoder,
    ObservationEncoder,
    UNIFIED_LOCAL_OBSERVATION_DIM,
    UNIFIED_TARGET_SLOTS,
    UNIFIED_TEAM_OBSERVATION_DIM,
)
from experiments.unified_mappo.run_pipeline import (
    config_from_args,
    trajectory_checkpoint_contract,
)
from policies.red.learning.unified_mappo_policy import (
    FIXED_TARGET_SLOT_IDS,
    SEARCH_TARGET_ID,
    UnifiedMAPPOSharedPolicy,
)



def _target(entity_id: int, entity_type: int, lon: float, lat: float) -> dict:
    return {
        "entity_id": entity_id,
        "type": entity_type,
        "nameChn": (
            "区域搜索" if entity_id == SEARCH_TARGET_ID
            else "目标" if entity_type == 9400
            else "拦截阵地" if entity_type == 9600
            else "无人船"
        ),
        "position": {"lon": lon, "lat": lat, "alt": 0.0},
    }


def _observation(
    *,
    entity_id: int,
    entity_type: int = 21002,
    detect_info: dict | None = None,
) -> dict:
    return {
        "step": 6,
        "entity_id": entity_id,
        "agent_id": entity_id,
        "self": {
            "type": entity_type,
            "health": 100.0,
            "isVisible": True,
            "position": {"lon": 110.0, "lat": 10.0, "alt": 1_000.0},
            "velocity": {"speed": 300.0, "up": 0.0, "heading": 0.0},
            "detectInfo": detect_info or {},
            "commRangeInfo": [],
        },
    }


class UnifiedTargetIsolationTests(unittest.TestCase):
    def test_fixed_architecture_covers_24_objectives_plus_search(self) -> None:
        self.assertEqual(UNIFIED_TARGET_SLOTS, 25)
        self.assertEqual(UNIFIED_LOCAL_OBSERVATION_DIM, 610)
        self.assertEqual(UNIFIED_TEAM_OBSERVATION_DIM, 625)
        self.assertEqual(
            GlobalStateEncoder(target_slots=UNIFIED_TARGET_SLOTS).state_dim,
            190,
        )

        policy = UnifiedMAPPOSharedPolicy.__new__(UnifiedMAPPOSharedPolicy)
        policy._encoders = {}
        policy._targets = []
        policy._target_slot_ids = list(FIXED_TARGET_SLOT_IDS)
        target_ids = [51, 52, 53, 54, 106, 168, 169, *range(1000, 1017)]
        targets = [
            _target(entity_id, 9400, 120.0 + index / 100.0, 20.0)
            for index, entity_id in enumerate(target_ids)
        ]
        targets.append(_target(SEARCH_TARGET_ID, -1, 116.0, 16.0))
        policy.configure_targets(targets)
        self.assertEqual(len(policy._targets), 25)
        with self.assertRaisesRegex(ValueError, "固定容量"):
            policy.configure_targets([
                *targets,
                _target(999, 9400, 130.0, 30.0),
            ])

    def test_policy_builds_target_coordinates_and_masks_per_actor(self) -> None:
        targets = [
            _target(51, 9400, 120.0, 20.0),
            _target(168, 9500, 122.5, 22.5),
            _target(SEARCH_TARGET_ID, -1, 116.0, 16.0),
        ]
        first = ObservationEncoder(
            {"entities": {}},
            target_slots=UNIFIED_TARGET_SLOTS,
            agent_id=11,
            include_target_runtime_state=True,
            include_agent_identity=True,
        )
        second = ObservationEncoder(
            {"entities": {}},
            target_slots=UNIFIED_TARGET_SLOTS,
            agent_id=12,
            include_target_runtime_state=True,
            include_agent_identity=True,
        )
        first.set_targets(targets)
        second.set_targets(targets)
        detection = SimpleNamespace(
            entity_id=168,
            entity_type=9500,
            time=6,
            lla=SimpleNamespace(x=122.5, y=22.5, z=0.0),
        )
        first.encode(
            _observation(entity_id=11, detect_info={168: detection}),
            launched=False,
            launch_step=-1,
            satellite_used=False,
            maneuver_state=0,
        )
        second.encode(
            _observation(entity_id=12),
            launched=False,
            launch_step=-1,
            satellite_used=False,
            maneuver_state=0,
        )

        policy = UnifiedMAPPOSharedPolicy.__new__(UnifiedMAPPOSharedPolicy)
        policy._targets = []
        policy._target_slot_ids = list(FIXED_TARGET_SLOT_IDS)
        policy._target_slot_templates = {}
        policy._visible_target_ids = frozenset()
        policy._public_target_ids = frozenset()
        policy._actor_visible_target_ids_by_entity = {}
        policy._full_target_runtime_states = {}
        policy._actor_runtime_target_ids = frozenset()
        policy._encoders = {11: first, 12: second}
        policy.dynamic_lifecycle = True
        policy.trainer = SimpleNamespace(device=torch.device("cpu"))
        policy.configure_targets(targets)
        policy.set_actor_target_visibility({11: (168,), 12: ()})

        first.encode(
            _observation(entity_id=11, detect_info={168: detection}),
            launched=False,
            launch_step=-1,
            satellite_used=False,
            maneuver_state=0,
        )
        second.encode(
            _observation(entity_id=12),
            launched=False,
            launch_step=-1,
            satellite_used=False,
            maneuver_state=0,
        )

        first_coordinates, first_valid = policy._target_tensors_for_entity(11)
        second_coordinates, second_valid = policy._target_tensors_for_entity(12)
        self.assertEqual(first_coordinates.shape, (1, 25, 2))
        self.assertEqual(first_valid.shape, (1, 25))
        self.assertEqual(first_valid[0, [0, 5, 7]].tolist(), [False, True, True])
        self.assertEqual(second_valid[0, [0, 5, 7]].tolist(), [False, False, True])
        self.assertTrue(
            np.allclose(first_coordinates[0, 5].numpy(), [122.5, 22.5])
        )
        self.assertTrue(
            np.allclose(second_coordinates[0, 5].numpy(), [0.0, 0.0])
        )


    def test_runtime_target_state_is_hidden_until_team_discovery(self) -> None:
        public_targets = [
            _target(51, 9400, 120.0, 20.0),
            _target(SEARCH_TARGET_ID, -1, 116.0, 16.0),
        ]
        hidden_target = _target(168, 9500, 122.5, 22.5)
        encoder = ObservationEncoder(
            {"entities": {}},
            target_slots=UNIFIED_TARGET_SLOTS,
            agent_id=11,
            include_target_runtime_state=True,
            include_agent_identity=True,
        )
        encoder.set_targets(public_targets)
        encoder.set_target_runtime_states({
            51: (0.8, 0.2, 1.0),
            168: (0.9, 0.1, 1.0),
        })
        before = encoder.encode(
            _observation(entity_id=11),
            launched=False,
            launch_step=-1,
            satellite_used=False,
            maneuver_state=0,
        )
        encoder.set_target_runtime_states({
            51: (0.8, 0.2, 1.0),
            168: (0.1, 0.9, 1.0),
        })
        after = encoder.encode(
            _observation(entity_id=11),
            launched=False,
            launch_step=-1,
            satellite_used=False,
            maneuver_state=0,
        )
        self.assertTrue(np.array_equal(before, after))
        self.assertNotIn(168, encoder._target_runtime_states)

        encoder.set_targets([public_targets[0], hidden_target, public_targets[1]])
        encoder.set_target_runtime_states({168: (0.25, 0.75, 1.0)})
        discovered = encoder.encode(
            _observation(entity_id=11),
            launched=False,
            launch_step=-1,
            satellite_used=False,
            maneuver_state=0,
        )
        slot_start = (
            ObservationEncoder.SELF_FEATURES
            + ObservationEncoder.IDENTITY_FEATURES
            + encoder.target_feature_dim
        )
        self.assertTrue(np.allclose(
            discovered[slot_start + 12:slot_start + 15],
            [0.25, 0.75, 1.0],
        ))


    def test_pipeline_contract_is_v7_25_610_190_and_cpu(self) -> None:
        args = SimpleNamespace(
            reward_mode="weighted_damage_trajectory_counterfactual",
            max_steps=1200,
            hidden_dim=32,
            learning_rate=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            value_coef=0.5,
            entropy_coef=0.005,
            deployment_policy_share=0.5,
            max_grad_norm=0.5,
            update_epochs=1,
            minibatch_size=8,
            initial_coordinate_log_std=-2.0,
            counterfactual_value_coef=0.1,
            parameter_seed=7,
        )
        config = config_from_args(args)
        self.assertEqual(config.observation_dim, 610)
        self.assertEqual(config.critic_state_dim, 190)
        self.assertEqual(config.critic_focal_observation_dim, 610)
        self.assertEqual(config.target_slots, 25)
        self.assertEqual(config.device, "cpu")

        contract = trajectory_checkpoint_contract(args.max_steps)
        self.assertEqual(contract["algorithm"], "target_conditioned_ctde_mappo_v7")
        self.assertEqual(contract["actor_observation_dim"], 610)
        self.assertEqual(contract["critic_state_dim"], 190)
        self.assertEqual(contract["critic_focal_observation_dim"], 610)
        self.assertEqual(contract["target_slots"], 25)
        self.assertEqual(contract["device"], "cpu")

if __name__ == "__main__":
    unittest.main()
