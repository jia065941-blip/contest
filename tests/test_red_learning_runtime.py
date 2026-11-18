"""Learning motion categories and their public command routing."""

from __future__ import annotations

import unittest
from copy import deepcopy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from envengine.common import SimmerCommandType
from envengine.environment.command_adapter import CommandAdapter
from envengine.environment.command_converter import CommandConverter

from user_agents.example.learning.models import (
    LearningActionAdapter,
    ObservationEncoder,
    build_learning_motion_policy,
)


MAP_AREA = {"xMin": 0.0, "xMax": 200000.0, "yMin": 0.0, "yMax": 200000.0}


class RedLearningRuntimeTests(unittest.TestCase):
    def test_random_masked_policy_and_action_adapter_are_runtime_safe(self) -> None:
        initial = {"map_area": MAP_AREA,
            "entities": {
                "101": {
                    "type": 9400,
                    "pos": {"x": 118.5, "y": 22.0, "z": 0.0},
                    "nameChn": "母巢_101",
                }
            }
        }
        adapter = LearningActionAdapter()
        encoder = ObservationEncoder(initial, agent_index=0)
        observation = {
            "step": 1,
            "self": {
                "pos": {"x": 118.0, "y": 22.0, "z": 100.0},
                "health": 100.0,
                "type": 21000,
                "detectInfo": {},
                "commRangeInfo": [],
            },
        }
        policy = build_learning_motion_policy("random_masked", seed=7)
        action = policy.select_action(
            encoder.encode(
                observation,
                launched=True,
                launch_step=0,
                satellite_used=False,
                maneuver_state=0,
            ),
            adapter.build_action_mask(),
        )
        engine_action = adapter.to_engine_action(action, entity_id=1)
        self.assertEqual(engine_action.shape, (1, 4))
        self.assertEqual(engine_action[0, 0], 3.0)
        self.assertEqual(engine_action[0, 1], 1.0)
        self.assertTrue(np.isin(engine_action[0, 2], [-1.0, 0.0, 1.0]))
        self.assertFalse(hasattr(adapter, "launch_target"))

    def test_model_categories_remain_zero_based_and_route_through_maneuver_code_three(self):
        adapter = LearningActionAdapter()
        for category, acceleration in ((0, -1), (1, 0), (2, 1)):
            with self.subTest(model_category=category):
                action = adapter.to_engine_action(category, entity_id=17)
                np.testing.assert_array_equal(action, [[3, 17, acceleration, 0]])
                converted = CommandConverter.common_converter([action])
                native = CommandAdapter.common_adapter(converted)
                self.assertEqual(len(native), 1)
                self.assertEqual(native[0].executorId, 17)
                self.assertEqual(native[0].commandTypeId, SimmerCommandType.SET_DESIRED_ACC_Z)
                self.assertEqual(native[0].commandAttributes['acc_z'], acceleration)

    def test_checkpointed_policy_requires_an_explicit_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_path"):
            build_learning_motion_policy("mappo")

    def test_hierarchical_encoder_appends_only_legal_task_context(self) -> None:
        initial = {"map_area": MAP_AREA, "entities": {}}
        encoder = ObservationEncoder(initial, agent_index=0, hierarchical_task_context=True)
        encoded = encoder.encode(
            {
                "step": 1,
                "self": {
                    "pos": {"x": 118.0, "y": 22.0, "z": 100.0},
                    "health": 100.0,
                    "type": 21000,
                    "detectInfo": {},
                    "commRangeInfo": [],
                },
            },
            launched=True,
            launch_step=0,
            satellite_used=False,
            maneuver_state=0,
            task_context=(1.0, 0.0, 0.0, 0.25, 0.5),
        )
        self.assertEqual(encoder.observation_dim, 90)
        np.testing.assert_allclose(encoded[-5:], [1.0, 0.0, 0.0, 0.25, 0.5])

    def test_actor_catalogue_excludes_hidden_priors_and_reads_received_track_types(self):
        initial = {"map_area": MAP_AREA, "entities": {
            "101": {"type": 9400, "pos": {"x": 118.5, "y": 22.0}},
            "103": {"type": 9500, "pos": {"x": 121.0, "y": 22.0}},
        }}
        encoder = ObservationEncoder(initial)
        self.assertEqual([item["entity_id"] for item in encoder.targets], [101])
        encoded = encoder.encode({
            "step": 1,
            "self": {
                "pos": {"x": 118.0, "y": 22.0}, "type": 21000,
                "health": 100, "commRangeInfo": [1],
                "detectInfo": {103: {"entity_id": 103, "entity_type": 9500}},
            },
        }, launched=True, launch_step=1, satellite_used=False, maneuver_state=0)
        self.assertEqual(encoder.observation_dim, 85)
        np.testing.assert_allclose(encoded[-4:], [1 / 200, 0, 1 / 10, 1 / 64])

    def test_detection_age_uses_the_original_detected_step_as_frames_advance(self):
        encoder = ObservationEncoder({"map_area": MAP_AREA, "entities": {
            101: {"type": 9400, "pos": {"x": 118.5, "y": 22.0}},
        }}, max_steps=1000)
        observation = {
            "step": 120,
            "self": {
                "pos": {"x": 118.0, "y": 22.0},
                "type": 21000,
                "detectInfo": {101: {
                    "entity_id": 101, "entity_type": 9400, "detected_step": 100,
                    "pos": {"x": 118.5, "y": 22.0, "z": 0.0},
                }},
            },
        }

        def age_feature():
            encoded = encoder.encode(
                observation, launched=True, launch_step=0, satellite_used=False, maneuver_state=0,
            )
            return encoded[encoder.SELF_FEATURES + 6]

        self.assertAlmostEqual(age_feature(), 20 / 1000)
        observation["step"] = 130
        self.assertAlmostEqual(age_feature(), 30 / 1000)
        self.assertEqual(observation["self"]["detectInfo"][101]["detected_step"], 100)


class CartesianEncodingTests(unittest.TestCase):
    def initial(self):
        return {"map_area": {"xMin": -2000, "xMax": 4000, "yMin": 7000, "yMax": 15000},
                "entities": {101: {"type": 9400, "pos": {"x": 1000, "y": 11000, "z": 0}}}}

    def local(self):
        return {"step": 1, "self": {"type": 21000, "health": 100,
                "pos": {"x": -2000, "y": 7000, "z": 100},
                "vel": {"x": 3.0, "y": 4.0, "z": 0.0}, "detectInfo": {}}}

    def encode(self, initial, local):
        return ObservationEncoder(initial).encode(
            local, launched=True, launch_step=0, satellite_used=False, maneuver_state=0,
        )

    def test_local_features_use_metric_vectors_and_map_extent(self):
        encoded = self.encode(self.initial(), self.local())
        np.testing.assert_allclose(encoded[[1, 2, 13, 14, 15, 16]], [0, 0, 5 / 1500, 0, 0.6, 0.8])
        np.testing.assert_allclose(encoded[21:26], [0.5, 0.5, 0.5, 0.6, 0.8])
        self.assertTrue(np.isfinite(encoded).all())

    def test_horizontal_features_are_invariant_to_map_translation_and_scale(self):
        initial, local = self.initial(), self.local()
        before = self.encode(initial, local)
        transformed = deepcopy(initial)
        transformed_local = deepcopy(local)
        for name in ("xMin", "xMax"):
            transformed["map_area"][name] = initial["map_area"][name] * 10 + 1234567
        for name in ("yMin", "yMax"):
            transformed["map_area"][name] = initial["map_area"][name] * 10 - 987654
        for position in (transformed["entities"][101]["pos"], transformed_local["self"]["pos"]):
            position["x"] = position["x"] * 10 + 1234567
            position["y"] = position["y"] * 10 - 987654
        after = self.encode(transformed, transformed_local)
        np.testing.assert_allclose(after, before)

    def test_global_features_use_the_same_map_scale_and_vector_speed(self):
        from user_agents.example.learning.models.red_policy import GlobalStateEncoder

        initial = self.initial()
        full = {"step": 1, "map_area": initial["map_area"], "entities": {
            1: self.local()["self"],
            101: {**initial["entities"][101], "health": 100},
        }}
        encoded = GlobalStateEncoder().encode(full)
        self.assertEqual(encoded.shape, (70,))
        np.testing.assert_allclose(encoded[[3, 4, 31, 36, 40, 41]], [0, 0, 5 / 1500, 0.5, 0.5, 0.5])

if __name__ == "__main__":
    unittest.main()
