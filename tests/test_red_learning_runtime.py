"""Dependency-free deployment checks for the red learning motion layer."""

from __future__ import annotations

import unittest

import numpy as np

from policies.red.learning import (
    LearningActionAdapter,
    ObservationEncoder,
    build_learning_motion_policy,
)


class RedLearningRuntimeTests(unittest.TestCase):
    def test_random_masked_policy_and_action_adapter_are_runtime_safe(self) -> None:
        initial = {
            "entities": {
                "101": {
                    "position": {"lon": 118.5, "lat": 22.0, "alt": 0.0},
                    "nameChn": "目标_101",
                }
            }
        }
        adapter = LearningActionAdapter([dict(initial["entities"]["101"], entity_id=101)])
        encoder = ObservationEncoder(initial, agent_id=1)
        observation = {
            "step": 1,
            "self": {
                "position": {"lon": 118.0, "lat": 22.0, "alt": 100.0},
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
            adapter.build_action_mask(launched=True, satellite_used=False),
        )
        engine_action = adapter.to_engine_action(action, entity_id=1)
        self.assertEqual(engine_action.shape, (1, 4))
        self.assertEqual(engine_action[0, 1], 1.0)
        self.assertTrue(np.isin(engine_action[0, 2], [-1.0, 0.0, 1.0]))

    def test_checkpointed_policy_requires_an_explicit_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "RED_LEARNING_MODEL"):
            build_learning_motion_policy("mappo")

    def test_hierarchical_encoder_appends_only_legal_task_context(self) -> None:
        initial = {"entities": {}}
        encoder = ObservationEncoder(initial, agent_id=1, hierarchical_task_context=True)
        encoded = encoder.encode(
            {
                "step": 1,
                "self": {
                    "position": {"lon": 118.0, "lat": 22.0, "alt": 100.0},
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


if __name__ == "__main__":
    unittest.main()
