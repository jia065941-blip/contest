"""Adapter that exposes red_strategy_lab policies through the core agent API."""
from __future__ import annotations
import numpy as np
from user_agents.base_agent import BaseAgent, AgentType
from red_strategy_lab.domain import Target, Position
from user_agents.red_policy_commander import RedPolicyCommander


class RedBaselineAgent(BaseAgent):
    _commander = None

    def __init__(self, agent_id, entity_id, init_observation):
        super().__init__(agent_id, entity_id, AgentType.AIRCRAFT, init_observation)
        if RedBaselineAgent._commander is None:
            targets = tuple(
                Target(
                    int(key),
                    Position(float(value['position']['lon']), float(value['position']['lat'])),
                    value=(10.0 if str(value.get('nameChn', '')).startswith('目标') else 6.0 if str(value.get('nameChn', '')).startswith('拦截阵地') else 3.0),
                )
                for key, value in init_observation['entities'].items()
                if value.get('side') == 1 and value.get('health', 0) > 0
            )
            RedBaselineAgent._commander = RedPolicyCommander(targets)
        RedBaselineAgent._commander.register_platform(entity_id)
        self.lastest_step = 0

    def set_observation(self, observation: dict) -> None:
        commander = RedBaselineAgent._commander
        commander.report(observation)
        self.lastest_step = int(observation.get('step', 0))

    def get_action(self):
        commander = RedBaselineAgent._commander
        row = commander.action_for(self.entity_id, self.lastest_step)
        return np.array([row], dtype=np.float64) if row else np.empty((0, 4), dtype=np.float64)

    def reset(self):
        super().reset()
        if RedBaselineAgent._commander is not None:
            RedBaselineAgent._commander.reset()
