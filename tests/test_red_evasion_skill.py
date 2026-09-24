"""Focused tests for the independent PPO evasion skill."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from policies.red.learning.red_policy import PolicyTransition
from policies.red.skills.evasion_ppo import (
    EvasionObservationEncoder,
    EvasionPPOConfig,
    EvasionPPOPolicy,
    EvasionPPOV3Config,
    EvasionPPOV3Policy,
    EvasionPPOV4Config,
    EvasionPPOV4Policy,
)
from policies.red.skills.evasion_ppo.reward import EvasionRewardShaper


def track(*, detect_from: int, distance: float, timestamp: int = 1) -> dict:
    return {
        "detect_from": detect_from,
        "time": timestamp,
        "entity_id": 9001,
        "entity_type": 24000,
        "pos_ecf": {"x": distance, "y": 0.0, "z": 0.0},
        "vel_ecf": {"x": -1000.0, "y": 0.0, "z": 0.0},
    }


def observation(step: int, detection: dict | None = None) -> dict:
    return {
        "step": step,
        "self": {
            "type": 21000,
            "health": 100.0,
            "stage": 3,
            "position": {"lon": 118.0, "lat": 22.0, "alt": 10000.0},
            "pos_ecf": {"x": 0.0, "y": float(step) * 500.0, "z": 0.0},
            "detectInfo": {} if detection is None else {9001: detection},
        },
    }


class EvasionEncoderTests(unittest.TestCase):
    def test_encoder_rejects_peer_fused_tracks(self) -> None:
        encoder = EvasionObservationEncoder(entity_id=77, max_steps=100)
        vector = encoder.encode(
            observation(1, track(detect_from=88, distance=10_000.0)),
            launch_step=0,
            maneuver_state=0,
        )
        self.assertEqual(vector.shape, (EvasionObservationEncoder.OBSERVATION_DIM,))
        self.assertFalse(EvasionObservationEncoder.has_threat(vector))

    def test_encoder_exposes_local_closing_track(self) -> None:
        encoder = EvasionObservationEncoder(entity_id=77, max_steps=100)
        vector = encoder.encode(
            observation(1, track(detect_from=77, distance=10_000.0)),
            launch_step=0,
            maneuver_state=-1,
        )
        self.assertTrue(EvasionObservationEncoder.has_threat(vector))
        self.assertAlmostEqual(
            EvasionObservationEncoder.nearest_distance(vector),
            np.hypot(10_000.0, 500.0) / 100_000.0,
        )
        self.assertGreater(EvasionObservationEncoder.maximum_closing_speed(vector), 0.0)
        self.assertGreaterEqual(EvasionObservationEncoder.minimum_cpa_distance(vector), 0.0)

    def test_v3_encoder_rejects_non_threatening_overflight(self) -> None:
        encoder = EvasionObservationEncoder(
            entity_id=77,
            max_steps=100,
            maximum_cpa_distance_m=15_000.0,
            maximum_time_to_go_s=120.0,
            minimum_closing_speed_mps=50.0,
        )
        crossing = track(detect_from=77, distance=10_000.0)
        crossing["pos_ecf"] = {"x": 10_000.0, "y": 50_000.0, "z": 0.0}
        vector = encoder.encode(
            observation(1, crossing), launch_step=0, maneuver_state=0
        )
        self.assertFalse(EvasionObservationEncoder.has_threat(vector))


class EvasionRewardTests(unittest.TestCase):
    def _vector(self, distance_norm: float | None) -> np.ndarray:
        value = np.zeros(EvasionObservationEncoder.OBSERVATION_DIM, dtype=np.float32)
        value[EvasionObservationEncoder.HEALTH_INDEX] = 1.0
        if distance_norm is not None:
            offset = EvasionObservationEncoder.THREAT_START
            value[offset + EvasionObservationEncoder.THREAT_VALID_OFFSET] = 1.0
            value[offset + EvasionObservationEncoder.THREAT_DISTANCE_OFFSET] = distance_norm
            value[offset + EvasionObservationEncoder.THREAT_CLOSING_OFFSET] = 0.5
        return value

    def test_close_disappearance_is_counted_as_interception(self) -> None:
        config = EvasionPPOConfig(device="cpu", clear_grace_steps=1)
        shaper = EvasionRewardShaper(config)
        outcome = shaper.shape(PolicyTransition(
            agent_id=1,
            observation=self._vector(0.01),
            action=1,
            action_mask=np.ones(3, dtype=np.bool_),
            reward=0.0,
            next_observation=np.zeros(EvasionObservationEncoder.OBSERVATION_DIM, dtype=np.float32),
            done=True,
        ))
        self.assertTrue(outcome.skill_terminal)
        self.assertLess(outcome.value, 0.0)
        self.assertEqual(shaper.finish_episode()["total"]["failures"], 1)

    def test_clear_track_is_counted_as_evasion(self) -> None:
        config = EvasionPPOConfig(device="cpu", clear_grace_steps=1)
        shaper = EvasionRewardShaper(config)
        next_vector = self._vector(None)
        outcome = shaper.shape(PolicyTransition(
            agent_id=1,
            observation=self._vector(0.2),
            action=0,
            action_mask=np.ones(3, dtype=np.bool_),
            reward=0.0,
            next_observation=next_vector,
            done=False,
        ))
        self.assertTrue(outcome.skill_terminal)
        self.assertGreater(outcome.value, 0.0)
        self.assertEqual(shaper.finish_episode()["total"]["successes"], 1)


class EvasionPolicyTests(unittest.TestCase):
    def test_no_threat_forces_stop_and_checkpoint_round_trip(self) -> None:
        config = EvasionPPOConfig(observation_dim=EvasionObservationEncoder.OBSERVATION_DIM, device="cpu")
        policy = EvasionPPOPolicy(config)
        policy.set_training(False)
        vector = np.zeros(EvasionObservationEncoder.OBSERVATION_DIM, dtype=np.float32)
        mask = policy.prepare_action_mask(vector, np.ones(3, dtype=np.bool_))
        self.assertEqual(policy.select_action(vector, mask), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skill.pt"
            policy.save(str(path))
            restored = EvasionPPOPolicy(config)
            restored.load(str(path))
        self.assertEqual(restored.update_count, policy.update_count)

    def test_v3_checkpoint_does_not_replace_v2_format(self) -> None:
        config = EvasionPPOV3Config(device="cpu")
        policy = EvasionPPOV3Policy(config)
        policy.set_training(False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skill_v3.pt"
            policy.save(str(path))
            restored = EvasionPPOV3Policy(config)
            restored.load(str(path))
            with self.assertRaises(ValueError):
                EvasionPPOPolicy(EvasionPPOConfig(device="cpu")).load(str(path))
        self.assertEqual(restored.algorithm_name, "ppo_evasion_skill_v3")

    def test_v4_mirror_is_involutive_and_swaps_actions(self) -> None:
        vector = np.arange(
            EvasionObservationEncoder.OBSERVATION_DIM, dtype=np.float32
        )
        mirrored = EvasionPPOV4Policy.mirror_observation(vector)
        np.testing.assert_array_equal(
            EvasionPPOV4Policy.mirror_observation(mirrored), vector
        )
        self.assertEqual(EvasionPPOV4Policy.mirror_action(0), 2)
        self.assertEqual(EvasionPPOV4Policy.mirror_action(1), 1)
        self.assertEqual(EvasionPPOV4Policy.mirror_action(2), 0)

    def test_v4_adds_an_independent_mirrored_transition(self) -> None:
        policy = EvasionPPOV4Policy(EvasionPPOV4Config(device="cpu"))
        vector = np.zeros(EvasionObservationEncoder.OBSERVATION_DIM, dtype=np.float32)
        offset = EvasionObservationEncoder.THREAT_START
        vector[offset + EvasionObservationEncoder.THREAT_VALID_OFFSET] = 1.0
        vector[offset + EvasionObservationEncoder.THREAT_DISTANCE_OFFSET] = 0.5
        vector[offset + 2] = 0.25
        action_mask = np.ones(3, dtype=np.bool_)
        action = policy.select_action(vector, action_mask)
        policy.observe(PolicyTransition(
            agent_id=7,
            observation=vector,
            action=action,
            action_mask=action_mask,
            reward=0.0,
            next_observation=vector.copy(),
            done=False,
        ))
        self.assertEqual(len(policy._buffer), 2)
        mirrored = policy._buffer[1][0]
        self.assertEqual(mirrored.agent_id, 7 + policy.MIRROR_AGENT_OFFSET)
        self.assertEqual(mirrored.action, policy.mirror_action(action))
        self.assertAlmostEqual(mirrored.observation[offset + 2], -0.25)


if __name__ == "__main__":
    unittest.main()
