"""Construct red examples without access to an environment or simulator."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from .config import LEARNING_MOTION_CHOICES, RedAgentConfig
from .contracts import Position
from .coordination import _CommunicationDirectory
from .learning import build_learning_motion_policy
from .priors import initial_targets_from_observation


class _LocalActor:
    """Only the local inference capability is installed in a platform agent."""

    def __init__(self, policy):
        self.select_action = policy.select_action
        self.training = bool(getattr(policy, "training", False))


class RedAgentFactory:
    """Own example configuration, per-group mailboxes and optional training.

    Construction receives configuration and public data, never ``env`` or an
    engine handle. Decisions still execute inside each Agent's ``get_action``.
    A trusted training caller separately drives the training methods below.
    """

    def __init__(self, config: RedAgentConfig):
        self.config = config
        self._learning_policy = (
            build_learning_motion_policy(
                config.motion_policy, seed=config.seed, model_path=config.model_path,
                max_steps=config.max_steps, training=config.training,
                observation_dim=90 if config.policy_name == "r9_hierarchical_learning" else 85,
            )
            if config.motion_policy in LEARNING_MOTION_CHOICES else None
        )
        self._transitions = []
        self._created = False

    def create_agents(self, public_observation: dict, platforms, *, search_polygon=None):
        from .attack_missile_agent import AttackMissileAgent

        if self._created:
            raise RuntimeError("Create a separate red example factory for each simulation")
        self._created = True
        platforms = tuple(platforms)
        polygon = tuple(
            point if isinstance(point, Position) else Position(*point)
            for point in search_polygon
        ) if search_polygon else None
        communications = _CommunicationDirectory(
            (entity_id for _, entity_id in platforms),
            initial_targets_from_observation(public_observation), self.config, polygon,
        )
        actor = _LocalActor(self._learning_policy) if self._learning_policy is not None else None
        agents = []
        for agent_id, entity_id in platforms:
            agent = AttackMissileAgent(agent_id, entity_id, deepcopy(public_observation))
            agent._configure_example(
                self.config, communications, actor,
                self._transitions.append if self.config.training else None,
            )
            agents.append(agent)
        return agents

    def create_deploy_agent(self, entities: dict, coordinates, coordinates_hm, *,
                            plan_path=None, launch_altitude=None):
        if plan_path is None:
            from .deploy_agent import DeployAgent

            return DeployAgent(-1, -1, {}, coordinates, coordinates_hm)
        from .grid_deploy_agent import GridDeployAgent

        entities = deepcopy(entities)
        if launch_altitude is not None:
            for entity in entities.values():
                entity.setdefault("position", {})["alt"] = launch_altitude
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        return GridDeployAgent(-1, -1, {"entities": entities}, coordinates, coordinates_hm, plan)

    def reset_training_episode(self) -> None:
        if self._learning_policy is not None and self.config.training:
            self._learning_policy.reset_episode()
        self._transitions.clear()

    def begin_training_step(self, observation: dict) -> None:
        if self._learning_policy is not None and self.config.training:
            self._learning_policy.begin_environment_step(observation)

    def finish_training_step(self, observation: dict) -> None:
        policy = self._learning_policy
        if policy is None or not self.config.training:
            return
        policy.end_environment_step(observation)
        global_state, next_global_state = policy.get_global_state_context()
        # Agents retain a bound append capability, so drain the original list
        # in place rather than leaving subsequent reports in an abandoned list.
        pending = tuple(self._transitions)
        self._transitions.clear()
        for transition in pending:
            policy.observe(replace(
                transition, global_state=global_state, next_global_state=next_global_state,
            ))
        policy.finish_environment_step()

    def save_model(self, path: str) -> None:
        if self._learning_policy is None:
            raise RuntimeError("This red example has no learning model to save")
        self._learning_policy.save(path)
