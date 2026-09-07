"""Focused tests for the red MAPPO optimizer and reward shaping."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from policies.red.learning.mappo_policy import (
        MAPPOConfig,
        MAPPORewardShaper,
        MAPPOSharedPolicy,
    )
    from policies.red.learning.red_policy import PolicyTransition
except ModuleNotFoundError as error:  # pragma: no cover - minimal runtime
    if error.name != "torch":
        raise
    MAPPOConfig = MAPPORewardShaper = MAPPOSharedPolicy = None
    PolicyTransition = None


@unittest.skipIf(MAPPOSharedPolicy is None, "PyTorch is not installed")
class RedMAPPOTests(unittest.TestCase):
    @staticmethod
    def _full_observation(target_health: float) -> dict:
        return {
            "step": 1,
            "entities": {
                1: {
                    "type": 21000,
                    "nameChn": "高性能飞行器_1",
                    "health": 100.0,
                    "position": {"lon": 118.0, "lat": 22.0, "alt": 1000.0},
                    "velocity": {"speed": 500.0},
                },
                51: {
                    "type": 9400,
                    "nameChn": "目标_51",
                    "health": target_health,
                    "position": {"lon": 120.0, "lat": 22.0, "alt": 0.0},
                },
            },
        }

    def test_reward_uses_weighted_damage_delta_and_ignores_alive_bonus(self) -> None:
        config = MAPPOConfig(observation_dim=90, device="cpu")
        shaper = MAPPORewardShaper(config)
        shaper.begin_environment_step(self._full_observation(100.0))
        shaper.end_environment_step(self._full_observation(90.0))
        observation = np.zeros(90, dtype=np.float32)
        next_observation = np.zeros(90, dtype=np.float32)
        observation[4] = next_observation[4] = 1.0
        transition = PolicyTransition(
            agent_id=1,
            observation=observation,
            action=1,
            action_mask=np.ones(3, dtype=np.bool_),
            reward=1.0,
            next_observation=next_observation,
            done=False,
        )
        self.assertAlmostEqual(shaper.shape(transition), 10.0, places=5)

    def test_short_rollout_performs_a_clipped_ppo_update(self) -> None:
        config = MAPPOConfig(
            observation_dim=4,
            rollout_size=4,
            min_update_size=4,
            minibatch_size=4,
            update_epochs=1,
            device="cpu",
        )
        policy = MAPPOSharedPolicy(config)
        full_observation = self._full_observation(100.0)
        for step in range(4):
            full_observation["step"] = step
            policy.begin_environment_step(full_observation)
            observation = np.asarray([step / 4.0, 0.0, 0.0, 1.0], dtype=np.float32)
            action_mask = np.ones(3, dtype=np.bool_)
            action = policy.select_action(observation, action_mask)
            policy.end_environment_step(full_observation)
            global_state, next_global_state = policy.get_global_state_context()
            policy.observe(PolicyTransition(
                agent_id=1,
                observation=observation,
                action=action,
                action_mask=action_mask,
                reward=1.0,
                next_observation=observation.copy(),
                done=step == 3,
                global_state=global_state,
                next_global_state=next_global_state,
            ))
            policy.finish_environment_step()
        self.assertEqual(policy.update_count, 1)
        self.assertEqual(policy.transition_count, 4)
        self.assertEqual(policy.last_metrics["samples"], 4.0)
        self.assertIn("approx_kl", policy.last_metrics)


if __name__ == "__main__":
    unittest.main()
