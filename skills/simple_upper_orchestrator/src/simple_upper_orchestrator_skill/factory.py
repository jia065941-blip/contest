"""Factory installed at the official user-agent export boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from local_avoidance_skill import load_external_policy

from .agent import UpperAgent
from .coordinator import TeamCoordinator
from .deploy_agent import StaggeredDeployAgent


class SimpleUpperFactory:
    orchestrator_config = None
    instances = []

    def __init__(self, official_config):
        del official_config
        if self.orchestrator_config is None:
            raise RuntimeError("SimpleUpperFactory has not been configured")
        self.config = self.orchestrator_config
        self.policy = load_external_policy(
            self.config.runtime_root, self.config.checkpoint, max_steps=self.config.max_steps,
        )
        self._created = False
        self._coordinator = None
        self.instances.append(self)

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()

    def create_agents(self, public_observation, platforms, *, search_polygon=None):
        del search_polygon
        if self._created:
            raise RuntimeError("create_agents may only be called once")
        self._created = True
        members = tuple((int(agent), int(entity)) for agent, entity in platforms)
        coordinator = TeamCoordinator(members, public_observation, self.policy, self.config)
        self._coordinator = coordinator
        for agent_id, entity_id in members:
            self.policy.register_agent(agent_id, entity_id, self.config.max_steps)
        return [
            UpperAgent(agent_id, entity_id, deepcopy(public_observation), coordinator)
            for agent_id, entity_id in members
        ]

    def create_deploy_agent(self, entities, coordinates, coordinates_hm, *, plan_path=None, launch_altitude=None):
        del entities
        if plan_path is not None:
            raise ValueError("simple upper orchestrator does not use preset deployment plans")
        if self._coordinator is None:
            raise RuntimeError("create_agents must run before create_deploy_agent")
        return StaggeredDeployAgent(
            coordinates, coordinates_hm, config=self.config.deployment,
            batch_size=self.config.deployment_batch_size, launch_altitude=launch_altitude,
            on_plan=self._coordinator.set_deployment_plan,
            on_mission_plan=self._coordinator.set_mission_plan,
            public_targets=self._coordinator.public_targets,
            l_first_wave_per_target=self.config.l_first_wave_per_target,
            expected_l_target_count=self.config.l_strike.expected_target_count,
            hm_config=replace(
                self.config.hm_strike,
                reserve_h=(
                    self.config.reserve_h_count
                    or self.config.hm_strike.reserve_h
                ),
                reserve_m=0,
            ),
        )
