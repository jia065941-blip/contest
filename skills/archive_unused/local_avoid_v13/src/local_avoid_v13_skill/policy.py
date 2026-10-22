"""Shared Actor with isolated encoder and gate state per simulator agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from skill_generation.artifacts import load_bundle, read_artifact
from skill_generation.monitor import Monitor

from .encoder import EnvironmentConfig, LocalAvoidRuntimeEncoder
from .gate import STOP, prepare_action_mask
from .model import LocalAvoidV13Skill, default_bundle_path


class LocalAvoidControlPolicy:
    training = False
    algorithm_name = "local_avoid_asymmetric_ppo_v1"
    is_evasion_skill = True
    requires_target_context = True
    gate_mode = "mission_v12"

    def __init__(self, bundle_dir: str | Path | None = None, *, max_steps: int = 1000) -> None:
        self.bundle_dir = Path(bundle_dir or default_bundle_path())
        checked = load_bundle(self.bundle_dir)
        config_payload = json.loads(read_artifact(checked, "environment_config").decode("utf-8"))
        self.environment_config = EnvironmentConfig.from_dict(config_payload["environment_config"])
        self.max_steps = int(max_steps)
        # Simulator control can issue hundreds of thousands of calls. Keep the
        # Runtime counters and exceptions, but do not emit one INFO record per call.
        self.skill = LocalAvoidV13Skill(self.bundle_dir, monitor=Monitor(enabled=False))
        self.skill.open()
        self._encoders: dict[int, LocalAvoidRuntimeEncoder] = {}
        self._entities: dict[int, int] = {}
        self._recovery: dict[int, bool] = {}
        self._current_agent_id: int | None = None

    def register_agent(self, agent_id: int, entity_id: int) -> None:
        self._entities[int(agent_id)] = int(entity_id)
        self._encoders[int(agent_id)] = LocalAvoidRuntimeEncoder(
            entity_id=entity_id, max_steps=self.max_steps, config=self.environment_config)

    def reset_agent(self, agent_id: int) -> None:
        encoder = self._encoders.get(int(agent_id))
        if encoder is not None:
            encoder.reset()
        self._recovery.pop(int(agent_id), None)

    def reset_episode(self) -> None:
        for encoder in self._encoders.values():
            encoder.reset()
        self._recovery.clear()
        self._current_agent_id = None

    def encode_motion_observation(self, observation: Mapping[str, Any], *,
                                  agent_id: int, entity_id: int,
                                  launch_step: int, maneuver_state: int,
                                  goal_position: Any) -> np.ndarray:
        agent_id = int(agent_id)
        if agent_id not in self._encoders:
            self.register_agent(agent_id, entity_id)
        if int(maneuver_state) != 0:
            self._recovery[agent_id] = True
        self._current_agent_id = agent_id
        return self._encoders[agent_id].encode(
            observation, launch_step=launch_step,
            maneuver_state=maneuver_state, goal_position=goal_position)

    def prepare_action_mask(self, observation: np.ndarray,
                            base_mask: np.ndarray) -> np.ndarray:
        if self._current_agent_id is None:
            raise RuntimeError("encode_motion_observation must precede action masking")
        mask, pending = prepare_action_mask(
            observation, self.environment_config,
            self._recovery.get(self._current_agent_id, False))
        self._recovery[self._current_agent_id] = pending
        return np.asarray(base_mask, dtype=np.bool_) & mask

    def action_for(self, *, agent_id: int, entity_id: int,
                   observation: Mapping[str, Any], launch_step: int,
                   maneuver_state: int, goal_position: Any) -> int:
        agent_id = int(agent_id)
        if agent_id not in self._encoders:
            self.register_agent(agent_id, entity_id)
        if int(maneuver_state) != 0:
            self._recovery[agent_id] = True
        encoded = self._encoders[agent_id].encode(
            observation, launch_step=launch_step, maneuver_state=maneuver_state,
            goal_position=goal_position)
        mask, pending = prepare_action_mask(
            encoded, self.environment_config, self._recovery.get(agent_id, False))
        self._recovery[agent_id] = pending
        return self.select_action(encoded, mask)

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        result = self.skill.infer({"observation": np.asarray(observation, dtype=np.float32)})
        selected = int(np.argmax(result.outputs["logits"]))
        if bool(action_mask[selected]):
            return selected
        valid = np.flatnonzero(action_mask)
        return int(valid[0]) if valid.size else STOP

    def set_training(self, training: bool) -> None:
        if training:
            raise ValueError("local_avoid_v13 is inference-only")

    def close(self) -> None:
        self.skill.close()
