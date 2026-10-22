"""Legality-preserving shared upper-layer coordinator."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np

from hm_public_strike_packages_skill import (
    HMMissileSnapshot,
    HMPublicStrikePackagesSkill,
    public_targets_from_directory,
)
from l_satellite_strike_packages_skill import (
    LMissileSnapshot,
    LSatelliteStrikePackagesSkill,
    tracks_from_isolated_observations,
)


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _position(value) -> dict[str, float] | None:
    lla = _field(value, "lla")
    if lla is not None:
        return {"lon": float(_field(lla, "x", 0.0)), "lat": float(_field(lla, "y", 0.0))}
    raw = _field(value, "position")
    if raw is None:
        return None
    return {"lon": float(_field(raw, "lon", 0.0)), "lat": float(_field(raw, "lat", 0.0))}


class TeamCoordinator:
    def __init__(self, members, public_observation, policy, config):
        self.members = {int(entity): int(agent) for agent, entity in members}
        self.policy = policy
        self.config = config
        directory = {}
        for raw_id, entity in public_observation.get("entities", {}).items():
            if int(entity.get("type", -1)) in (9400, 9600):
                directory[int(raw_id)] = {
                    "entity_id": int(raw_id), "entity_type": int(entity["type"]),
                    "position": deepcopy(entity.get("position", {})),
                }
        self.public_targets = public_targets_from_directory(directory, config=config.hm_strike)
        self._public_by_id = {target.target_id: target for target in self.public_targets}
        self._hm = HMPublicStrikePackagesSkill(config.hm_strike, public_targets=self.public_targets)
        self._l = LSatelliteStrikePackagesSkill(self.config.l_strike)
        self._observations: dict[int, dict] = {}
        self._actions: dict[int, tuple[tuple[float, ...], ...]] = {}
        self._goals: dict[int, dict[str, float]] = {}
        self._assignments: dict[int, int] = {}
        self._goal_versions: dict[tuple[int, int], tuple[float, float]] = {}
        self._planned_step: int | None = None
        self._satellite_committed = False
        self._reset_members: set[int] = set()

    def submit(self, entity_id: int, observation: dict) -> None:
        if self._reset_members == set(self.members):
            self._reset_members.clear()
        self._observations[int(entity_id)] = observation

    def actions_for(self, entity_id: int, launch_step: int, maneuver_state: int):
        observation = self._observations.get(int(entity_id))
        if observation is None:
            raise RuntimeError(f"no observation submitted for entity {entity_id}")
        step = int(observation["step"])
        if self._planned_step != step:
            self._plan_assignments(step)
            self._plan_motion(step)
            self._planned_step = step
        return self._actions.get(int(entity_id), ())

    def _current_owners(self, step: int) -> tuple[int, ...]:
        return tuple(
            owner for owner in sorted(self.members)
            if owner in self._observations
            and int(self._observations[owner].get("step", -1)) == step
        )

    def _alive(self, entity_id: int) -> bool:
        own = self._observations[entity_id].get("self") or {}
        return bool(own) and float(own.get("health", 1.0)) > 0.0

    def _plan_assignments(self, step: int) -> None:
        current_owners = self._current_owners(step)
        rows: dict[int, list[tuple[float, ...]]] = {owner: [] for owner in self.members}
        hm_members = []
        for owner in current_owners:
            own = self._observations[owner].get("self") or {}
            kind = int(own.get("type", -1))
            if kind in (21000, 21001):
                hm_members.append(HMMissileSnapshot(owner, kind, alive=self._alive(owner), launched=True))
        hm_decision = self._hm.decide(step=step, members=hm_members)
        for package in hm_decision.packages:
            for owner in package.member_ids:
                self._assignments[owner] = package.target.target_id
                self._goals[owner] = {"lon": package.target.lon, "lat": package.target.lat}
        for action in hm_decision.attack_actions:
            rows[action.entity_id].append((4, action.entity_id, action.target_id, 0))

        l_observations: dict[int, dict] = {}
        for owner in current_owners:
            observation = self._observations[owner]
            own = observation.get("self") or {}
            if int(own.get("type", -1)) != 21002:
                continue
            l_observations[owner] = observation
        common_tracks = self._common_l_tracks(l_observations, step)
        l_members = [
            LMissileSnapshot(owner, alive=self._alive(owner), launched=True)
            for owner in sorted(l_observations)
        ]
        l_decision = self._l.decide(step=step, members=l_members, tracks=common_tracks)
        for package in l_decision.packages:
            for owner in package.member_ids:
                detected = (self._observations[owner].get("self") or {}).get("detectInfo") or {}
                if package.target.target_id not in {int(value) for value in detected}:
                    continue
                self._assignments[owner] = package.target.target_id
                target = detected.get(package.target.target_id, detected.get(str(package.target.target_id)))
                goal = _position(target)
                if goal is not None:
                    self._goals[owner] = goal
        for action in l_decision.attack_actions:
            detected = (self._observations[action.entity_id].get("self") or {}).get("detectInfo") or {}
            if action.target_id in {int(value) for value in detected}:
                rows[action.entity_id].append((4, action.entity_id, action.target_id, 0))
        if l_decision.satellite_request is not None and not self._satellite_committed:
            requester = l_decision.satellite_request.requester_id
            rows[requester].append((3, requester, 0, 0))
            self._satellite_committed = True

        # Owner-local coordinate refresh. No coordinate from another observation is consumed.
        current_owner_set = frozenset(current_owners)
        for owner, target_id in sorted(self._assignments.items()):
            if owner not in current_owner_set:
                continue
            detected = (self._observations[owner].get("self") or {}).get("detectInfo") or {}
            target = detected.get(target_id, detected.get(str(target_id)))
            if target is None:
                continue
            goal = _position(target)
            if goal is None:
                continue
            version = (goal["lon"], goal["lat"])
            key = (owner, target_id)
            if self._goal_versions.get(key) != version:
                self._goals[owner] = goal
                self._goal_versions[key] = version
                if not any(row[0] == 4 and int(row[2]) == target_id for row in rows[owner]):
                    rows[owner].append((4, owner, target_id, 0))
        self._actions = {owner: tuple(value) for owner, value in rows.items()}

    @staticmethod
    def _common_l_tracks(observations: dict[int, dict], step: int):
        if not observations:
            return ()
        by_owner = {
            owner: {track.target_id: track for track in tracks_from_isolated_observations({owner: observation}, step=step)}
            for owner, observation in observations.items()
        }
        common_ids = set.intersection(*(set(tracks) for tracks in by_owner.values()))
        return tuple(
            max((tracks[target_id] for tracks in by_owner.values()), key=lambda track: track.observed_step)
            for target_id in sorted(common_ids)
        )

    def _fallback_goal(self, owner: int) -> dict[str, float]:
        own = (self._observations[owner].get("self") or {})
        position = own.get("position") or {}
        lon, lat = float(position.get("lon", 0.0)), float(position.get("lat", 0.0))
        direction = own.get("velocity") or own.get("vel_lla") or {}
        dx, dy = float(_field(direction, "x", 0.0)), float(_field(direction, "y", 0.0))
        norm = math.hypot(dx, dy)
        if norm <= 1e-12:
            return {"lon": lon, "lat": lat}
        return {"lon": lon + dx / norm * 0.01, "lat": lat + dy / norm * 0.01}

    def _plan_motion(self, step: int) -> None:
        current_owners = frozenset(self._current_owners(step))
        for owner in sorted(self.members):
            if owner not in current_owners:
                self._actions[owner] = ()
                continue
            observation = self._observations[owner]
            if not self._alive(owner):
                self._actions[owner] = ()
                continue
            agent_id = self.members[owner]
            state = int(observation.get("_orchestrator_maneuver_state", 0))
            launch_step = int(observation.get("_orchestrator_launch_step", step))
            encoded = self.policy.encode_motion_observation(
                observation, agent_id=agent_id, entity_id=owner,
                launch_step=launch_step, maneuver_state=state,
                goal_position=self._goals.get(owner, self._fallback_goal(owner)),
            )
            mask = self.policy.prepare_action_mask(encoded, np.ones(3, dtype=np.bool_))
            selected = int(self.policy.select_action(encoded, mask))
            if selected not in (0, 1, 2):
                raise RuntimeError(f"external policy returned invalid action {selected}")
            maneuver = float(selected - 1)
            self._actions[owner] = ((0, owner, maneuver, 0), *self._actions.get(owner, ()))

    def reset_member(self, entity_id: int) -> None:
        entity_id = int(entity_id)
        if entity_id in self._reset_members:
            return
        if not self._reset_members:
            self._observations.clear()
            self._actions.clear()
            self._goals.clear()
            self._assignments.clear()
            self._goal_versions.clear()
            self._planned_step = None
            self._satellite_committed = False
            self._hm.reset()
            self._l.reset()
            reset = getattr(self.policy, "reset_episode", None)
            if callable(reset):
                reset()
        self._reset_members.add(entity_id)
