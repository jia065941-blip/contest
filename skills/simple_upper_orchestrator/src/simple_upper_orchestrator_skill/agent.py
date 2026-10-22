"""Per-platform official agent facade."""

from __future__ import annotations

import numpy as np

from user_agents.base_agent import AgentType, BaseAgent


class UpperAgent(BaseAgent):
    def __init__(self, agent_id, entity_id, init_observation, coordinator):
        super().__init__(agent_id, entity_id, AgentType.AIRCRAFT, init_observation)
        self.coordinator = coordinator
        self.latest_observation = None
        self.launch_step = -1
        self.maneuver_state = 0

    def set_observation(self, observation: dict) -> None:
        self.latest_observation = observation
        if self.launch_step < 0 and observation.get("self"):
            self.launch_step = int(observation["step"])
        observation["_orchestrator_launch_step"] = self.launch_step
        observation["_orchestrator_maneuver_state"] = self.maneuver_state
        self.coordinator.submit(self.entity_id, observation)

    def get_action(self) -> np.ndarray:
        if self.latest_observation is None:
            raise RuntimeError("the environment must provide an isolated observation")
        rows = self.coordinator.actions_for(self.entity_id, self.launch_step, self.maneuver_state)
        for row in rows:
            if int(row[1]) != self.entity_id:
                raise RuntimeError("coordinator returned an action for another owner")
            if int(row[0]) == 0:
                self.maneuver_state = int(row[2])
        return np.asarray(rows, dtype=np.float64).reshape(-1, 4)

    def reset(self):
        super().reset()
        self.latest_observation = None
        self.launch_step = -1
        self.maneuver_state = 0
        self.coordinator.reset_member(self.entity_id)
