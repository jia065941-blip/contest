"""Legality-preserving shared upper-layer coordinator."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math

import numpy as np

from dynamic_retarget_skill import (
    DynamicRetargetSkill,
    RetargetMemberSnapshot,
    RetargetTargetSnapshot,
)
from hm_public_strike_packages_skill import (
    public_targets_from_directory,
)
from l_satellite_strike_packages_skill import (
    tracks_from_isolated_observations,
)
from target_damage_probability_skill import (
    CommitmentEvidence,
    DeliveryEvidence,
    TargetDamageProbabilitySkill,
    TargetPriorEvidence,
)
from terminal_guidance_override_skill import (
    TerminalGuidanceSkill,
    TerminalGuidanceSnapshot,
)
from reserve_echelon_skill import (
    ReserveEchelonConfig,
    ReserveEchelonSkill,
    ReserveMemberSnapshot,
    ReservePackageSnapshot,
    TargetDeficitSnapshot,
)
from reserve_holding_maneuver_skill import (
    HoldingSnapshot,
    ReserveHoldingManeuverConfig,
    ReserveHoldingManeuverSkill,
)

from .deploy_agent import PreMissionPlan


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _position(value) -> dict[str, float] | None:
    lla = _field(value, "lla")
    if lla is not None:
        lon, lat = _field(lla, "x"), _field(lla, "y")
        if lon is None or lat is None:
            return None
        return {"lon": float(lon), "lat": float(lat)}
    raw = _field(value, "position")
    if raw is None:
        return None
    lon, lat = _field(raw, "lon"), _field(raw, "lat")
    if lon is None or lat is None:
        return None
    return {"lon": float(lon), "lat": float(lat)}


def _distance_km(first: dict[str, float], second: dict[str, float]) -> float:
    first_lat = math.radians(first["lat"])
    second_lat = math.radians(second["lat"])
    longitude_delta = ((second["lon"] - first["lon"] + 180.0) % 360.0) - 180.0
    sin_lat = math.sin((second_lat - first_lat) / 2.0)
    sin_lon = math.sin(math.radians(longitude_delta) / 2.0)
    haversine = (
        sin_lat * sin_lat
        + math.cos(first_lat) * math.cos(second_lat) * sin_lon * sin_lon
    )
    return 2.0 * 6_371.0088 * math.asin(math.sqrt(min(1.0, max(0.0, haversine))))


@dataclass(frozen=True)
class _TargetPosition:
    position: dict[str, float]
    source: str
    observed_step: float | None
    age_steps: float | None
    source_time: float | None = None
    fingerprint: tuple | None = None


@dataclass(frozen=True)
class _TrackFreshness:
    fingerprint: tuple
    last_changed_step: int


@dataclass
class _OwnerPositionSample:
    step: float
    lon: float
    lat: float


@dataclass
class _OwnerVelocityState:
    positions: deque[_OwnerPositionSample]
    last_valid_east: float | None = None
    last_valid_north: float | None = None
    source: str = "none"

    @property
    def previous_step(self) -> float | None:
        return None if not self.positions else self.positions[-1].step

    @property
    def previous_lon(self) -> float | None:
        return None if not self.positions else self.positions[-1].lon

    @property
    def previous_lat(self) -> float | None:
        return None if not self.positions else self.positions[-1].lat


@dataclass(frozen=True)
class _HMProgress:
    step: int
    missile_type: int
    stage: int
    position: dict[str, float] | None
    target_id: int | None
    target_position: dict[str, float] | None
    target_position_source: str | None
    target_track_age_steps: float | None
    current_distance_km: float | None
    minimum_target_distance_metres: float | None
    communication_group: tuple[int, ...]


@dataclass(frozen=True)
class _LProgress:
    step: int
    stage: int
    position: dict[str, float] | None
    target_id: int
    target_position: dict[str, float]
    current_distance_km: float | None
    minimum_target_distance_metres: float | None
    communication_group: tuple[int, ...]


@dataclass(frozen=True)
class _EarlyLoss:
    event_id: str
    step: int
    owner: int
    member_type: int
    stage: int
    target_id: int
    target_position: dict[str, float]
    member_position: dict[str, float]
    communication_group: tuple[int, ...]
    distance_km: float | None
    minimum_distance_metres: float | None
    delivery_probability: float | None
    nominal_damage: float


@dataclass(frozen=True)
class _LedgerEvent:
    step: int
    operation: str
    event_id: str
    payload: object


class TeamCoordinator:
    def __init__(self, members, public_observation, policy, config):
        self.members = {int(entity): int(agent) for agent, entity in members}
        self.policy = policy
        self.config = config
        self._audit_path = config.audit_path
        self._audited_beliefs: dict[tuple[tuple[int, ...], int], tuple] = {}
        self._audited_retarget_status: dict[int, tuple] = {}
        self._audited_formation_steps: dict[int, int] = {}
        self._audited_missile_trace_steps: dict[int, int] = {}
        self._audited_terminal_guidance: dict[int, tuple] = {}
        self._missile_trace_history: dict[int, deque[dict]] = {}
        if self._audit_path is not None:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            self._audit_path.write_text("", encoding="utf-8")
        directory = {}
        for raw_id, entity in public_observation.get("entities", {}).items():
            if int(entity.get("type", -1)) in (9400, 9600):
                directory[int(raw_id)] = {
                    "entity_id": int(raw_id), "entity_type": int(entity["type"]),
                    "position": deepcopy(entity.get("position", {})),
                }
        self.public_targets = public_targets_from_directory(directory, config=config.hm_strike)
        self._public_by_id = {target.target_id: target for target in self.public_targets}
        self._dynamic_retarget = DynamicRetargetSkill(self.config.dynamic_retarget)
        self._reserve_echelon = ReserveEchelonSkill(ReserveEchelonConfig(
            max_dispatch_stage=self.config.reserve_max_stage,
            deadline_step=self.config.reserve_deadline_step,
            # The adapter supplies only deficits visible to the package's
            # current communication group through persistent owner ledgers.
            require_same_group_evidence=False,
            h_max_distance_km=self.config.reserve_h_max_distance_km,
            l_max_distance_km=self.config.reserve_l_max_distance_km,
            h_max_turn_angle_degrees=self.config.reserve_h_max_turn_angle_degrees,
            l_max_turn_angle_degrees=self.config.reserve_l_max_turn_angle_degrees,
            seconds_per_step=self.config.seconds_per_step,
        ))
        self._reserve_holding = ReserveHoldingManeuverSkill(
            ReserveHoldingManeuverConfig(
                corridor_half_width_km=self.config.reserve_holding_half_width_km,
                heading_offset_deg=(
                    self.config.reserve_holding_heading_offset_degrees
                ),
                max_turn_steps=self.config.reserve_holding_max_turn_steps,
                straight_transition_steps=(
                    self.config.reserve_holding_straight_steps
                ),
            )
        )
        self._terminal_guidance = TerminalGuidanceSkill(
            self.config.terminal_guidance
        )
        self._mission_plan: PreMissionPlan | None = None
        self._l_roster: tuple[int, ...] = ()
        self._l_rank: dict[int, int] = {}
        self._l_core_ids: frozenset[int] | None = None
        self._l_core_assignments: dict[int, int] = {}
        self._l_progress: dict[int, _LProgress] = {}
        self._l_early_losses: dict[int, _EarlyLoss] = {}
        self._l_catalogues: dict[int, dict[int, object]] = {
            owner: {} for owner in self.members
        }
        self._l_target_count: int | None = None
        self._observations: dict[int, dict] = {}
        self._actions: dict[int, tuple[tuple[float, ...], ...]] = {}
        self._goals: dict[int, dict[str, float]] = {}
        self._assignments: dict[int, int] = {}
        self._goal_versions: dict[tuple[int, int], tuple[float, float]] = {}
        self._hm_progress: dict[int, _HMProgress] = {}
        self._hm_types: dict[int, int] = {}
        self._core_hm_assignments: dict[int, int] = {}
        self._reserve_h_ids: frozenset[int] = frozenset()
        self._reserve_packages: dict[int, ReservePackageSnapshot] = {}
        self._hm_target_regions: dict[int, str] = {}
        self._hm_early_losses: dict[int, _EarlyLoss] = {}
        self._reserve_retarget_counts: dict[int, int] = {}
        self._reserve_member_snapshots: dict[int, ReserveMemberSnapshot] = {}
        self._reserve_seen_members: set[int] = set()
        self._reserve_package_terminations: dict[int, dict] = {}
        self._reserve_holding_anchors: dict[int, tuple[float, float]] = {}
        self._reserve_reachability_origins: dict[
            int, tuple[float, float]
        ] = {}
        self._audited_reserve_deficits: set[tuple] = set()
        self._last_groups: dict[int, tuple[int, ...]] = {}
        self._terminalized: set[int] = set()
        self._owner_ledgers: dict[int, dict[str, _LedgerEvent]] = {
            owner: {} for owner in self.members
        }
        self._damage_by_group: dict[tuple[int, ...], TargetDamageProbabilitySkill] = {}
        self._retarget_counts: dict[int, int] = {}
        self._last_retarget_steps: dict[int, int] = {}
        self._planned_step: int | None = None
        self._satellite_committed = False
        self._reset_members: set[int] = set()
        self._deployment_waves: dict[int, int] = {}
        self._track_freshness: dict[tuple[int, int], _TrackFreshness] = {}
        self._initial_attack_sent: set[int] = set()
        self._motion_guidance: dict[int, dict] = {}
        self._owner_velocity_history: dict[int, _OwnerVelocityState] = {
            owner: self._new_owner_velocity_state() for owner in self.members
        }

    def set_mission_plan(self, plan: PreMissionPlan) -> None:
        if not isinstance(plan, PreMissionPlan):
            raise ValueError("mission plan must be a PreMissionPlan")
        if set(plan.entity_types) != set(self.members):
            raise ValueError("mission plan entity roster does not match coordinator members")
        for owner, target_id in plan.hm_assignments.items():
            if owner not in self.members:
                raise ValueError(f"mission plan contains unknown owner {owner}")
            if plan.entity_types[owner] not in (21000, 21001):
                raise ValueError(f"mission plan owner {owner} is not H/M")
            if target_id not in self._public_by_id:
                raise ValueError(f"mission plan target {target_id} is not public")
        for package in plan.reserve_packages:
            if (
                package.provisional_target_id not in self._public_by_id
                or package.fallback_target_id not in self._public_by_id
            ):
                raise ValueError("reserve H package targets must be public")
        if any(target_id not in self._public_by_id for target_id in plan.reserve_target_regions):
            raise ValueError("reserve responsibility target must be public")
        expected_l = tuple(
            owner
            for owner, entity_type in sorted(plan.entity_types.items())
            if entity_type == 21002
        )
        if plan.l_roster != expected_l:
            raise ValueError("mission plan L roster does not match entity types")
        if self._mission_plan is not None:
            if self._mission_plan != plan:
                raise RuntimeError("immutable mission plan cannot change between rounds")
            return
        self._mission_plan = plan
        self._l_roster = plan.l_roster
        self._l_rank = {
            owner: rank for rank, owner in enumerate(self._l_roster)
        }
        self._restore_mission_plan_state()

    def set_pre_mission_plan(
        self,
        hm_assignments,
        entity_types,
        entity_core,
        l_roster,
    ) -> None:
        """Compatibility entry point for callers that hold the four plan parts."""
        self.set_mission_plan(PreMissionPlan(
            hm_assignments,
            entity_types,
            entity_core,
            tuple(l_roster),
        ))

    def _restore_mission_plan_state(self) -> None:
        if self._mission_plan is None:
            return
        self._reserve_echelon.reset()
        self._reserve_holding.reset()
        self._reserve_holding_anchors.clear()
        self._reserve_reachability_origins.clear()
        self._reserve_packages = {
            package.package_id: package
            for package in self._mission_plan.reserve_packages
        }
        self._reserve_h_ids = frozenset(
            member_id
            for package in self._mission_plan.reserve_packages
            for member_id in package.member_ids
        )
        self._assignments = {
            owner: target_id
            for owner, target_id in self._mission_plan.hm_assignments.items()
        }
        self._core_hm_assignments = dict(self._mission_plan.hm_assignments)
        self._hm_target_regions = dict(self._mission_plan.reserve_target_regions)
        self._hm_types = {
            owner: entity_type
            for owner, entity_type in self._mission_plan.entity_types.items()
            if entity_type in (21000, 21001)
        }
        for owner, target_id in sorted(self._assignments.items()):
            target = self._public_by_id[target_id]
            goal = {"lon": target.lon, "lat": target.lat}
            self._goals[owner] = goal
            self._goal_versions[(owner, target_id)] = (target.lon, target.lat)
            event_id = f"commit:{owner}:{target_id}"
            evidence = CommitmentEvidence(
                owner,
                self._hm_types[owner],
                target_id,
                "core" if self._hm_types[owner] == 21000 else "flexible",
                event_id,
            )
            self._add_ledger_event(
                (owner,), _LedgerEvent(0, "commit", event_id, evidence)
            )
            self._write_audit(
                "initial_commitment",
                0,
                owner=owner,
                type=evidence.missile_type,
                target=evidence.target_id,
                commitment=evidence.commitment_type,
                group=(owner,),
                event_id=event_id,
            )
        for package in self._mission_plan.reserve_packages:
            self._audit_initial_reserve(package, 0, source="pre_mission")

    def _audit_initial_reserve(
        self, package: ReservePackageSnapshot, step: int, *, source: str
    ) -> None:
        fields = {
            "package": package.package_id,
            "members": package.member_ids,
            "types": package.member_types,
            "provisional_target": package.provisional_target_id,
            "fallback_target": package.fallback_target_id,
            "responsibility_region": package.responsibility_region,
            "source": source,
        }
        self._write_audit("initial_reserve", step, **fields)
        self._write_audit(
            "reserve_staging",
            step,
            state="STAGING",
            committed=False,
            **fields,
        )

    def set_deployment_plan(self, plan) -> None:
        self._deployment_waves = {
            int(point.entity_id): (
                2
                if getattr(point, "wave_role", point.band) == "second"
                else 1
            )
            for point in plan.points
            if int(point.entity_type) == 21002
        }

    def submit(self, entity_id: int, observation: dict) -> None:
        if self._reset_members == set(self.members):
            self._reset_members.clear()
        owner = int(entity_id)
        self._observations[owner] = observation
        self._update_owner_velocity(owner, observation)

    def actions_for(self, entity_id: int, launch_step: int, maneuver_state: int):
        observation = self._observations.get(int(entity_id))
        if observation is None:
            raise RuntimeError(f"no observation submitted for entity {entity_id}")
        step = int(observation["step"])
        if self._planned_step != step:
            for owner in self._current_owners(step):
                self._update_owner_velocity(owner, self._observations[owner])
            current_hm = self._plan_assignments(step)
            self._plan_motion(step)
            self._record_missile_traces(step, current_hm)
            self._record_formation_status(step)
            self._planned_step = step
        actions = self._actions.get(int(entity_id), ())
        target_id = self._assignments.get(int(entity_id))
        if any(
            int(row[0]) == 4 and int(row[2]) == target_id
            for row in actions
        ):
            self._initial_attack_sent.add(int(entity_id))
        return actions

    def _current_owners(self, step: int) -> tuple[int, ...]:
        return tuple(
            owner for owner in sorted(self.members)
            if owner in self._observations
            and int(self._observations[owner].get("step", -1)) == step
        )

    def _alive(self, entity_id: int) -> bool:
        own = self._observations[entity_id].get("self") or {}
        return bool(own) and float(own.get("health", 1.0)) > 0.0

    def _plan_assignments(self, step: int) -> dict[int, dict]:
        current_owners = self._current_owners(step)
        rows: dict[int, list[tuple[float, ...]]] = {owner: [] for owner in self.members}
        current_hm: dict[int, dict] = {}
        current_l: dict[int, dict] = {}
        terminal_hm: dict[int, str] = {}
        terminal_l: dict[int, str] = {}
        for owner in current_owners:
            own = self._observations[owner].get("self") or {}
            if self._mission_plan is None:
                continue
            kind = self._mission_plan.entity_types.get(owner)
            try:
                observed_id = int(own.get("entity_id", owner))
                observed_kind = int(own.get("type", -1))
            except (TypeError, ValueError):
                continue
            if observed_id != owner or observed_kind != kind:
                continue
            if kind in (21000, 21001):
                self._last_groups[owner] = self._communication_group(owner, own)
                if self._alive(owner):
                    current_hm[owner] = own
                else:
                    terminal_hm[owner] = "observed_dead"
            elif kind == 21002:
                self._last_groups[owner] = self._communication_group(owner, own)
                if self._alive(owner):
                    current_l[owner] = own
                elif owner in self._l_progress or any(
                    owner in package.member_ids
                    for package in self._reserve_packages.values()
                ):
                    terminal_l[owner] = "observed_dead"
        for owner in self._hm_progress:
            if owner not in current_hm and owner not in terminal_hm:
                terminal_hm[owner] = "owner_missing"
        self._finalize_terminal_hm(step, terminal_hm)
        for owner in self._l_progress:
            if owner not in current_l and owner not in terminal_l:
                terminal_l[owner] = "owner_missing"
        for owner in self._reserve_seen_members:
            if (
                self._mission_plan.entity_types.get(owner) == 21002
                and owner not in current_l
                and owner not in terminal_l
            ):
                terminal_l[owner] = "owner_missing"
        self._finalize_terminal_l(step, terminal_l)

        # H/M ownership is fixed by deployment. The first owner action selects
        # that target; subsequent commands only refresh the same ID from that
        # owner's isolated detectInfo unless explicit dynamic retargeting runs.
        for owner in sorted(current_hm):
            target_id = self._assignments.get(owner)
            if target_id not in self._public_by_id:
                continue
            if owner not in self._initial_attack_sent:
                self._set_target_action(rows, owner, target_id)
                continue
            track = self._owner_detected_public_targets(owner, step).get(target_id)
            if track is None:
                continue
            goal = track.position
            version = (goal["lon"], goal["lat"])
            key = (owner, target_id)
            if self._goal_versions.get(key) != version:
                self._goals[owner] = goal
                self._goal_versions[key] = version
                self._set_target_action(rows, owner, target_id)

        # Satellite authority is a deployment-time roster fact. Runtime L
        # target selection remains strictly owner-local.
        if (
            self._l_roster
            and not self._satellite_committed
            and step >= self.config.l_strike.satellite_trigger_step
            and self._l_roster[0] in current_l
        ):
            requester = self._l_roster[0]
            rows[requester].append((3, requester, 0, 0))
            self._satellite_committed = True

        expected = self.config.l_strike.expected_target_count
        l_tracks_by_owner = {
            owner: self._update_l_catalogue(owner, step)
            for owner in sorted(current_l)
        }
        if (
            self.config.l_first_wave_per_target > 0
            and self._satellite_committed
            and self._l_core_ids is None
        ):
            for owner in sorted(current_l):
                target_ids = tuple(sorted(l_tracks_by_owner[owner]))
                if target_ids and (
                    expected is None or len(target_ids) >= expected
                ):
                    target_count = len(target_ids)
                    core_count = min(
                        len(self._l_roster),
                        self.config.l_first_wave_per_target * target_count,
                    )
                    self._l_target_count = target_count
                    self._l_core_ids = frozenset(self._l_roster[:core_count])
                    break
        for owner in sorted(current_l):
            owner_tracks = l_tracks_by_owner[owner]
            target_id = self._assignments.get(owner)
            if target_id is None:
                held_package = self._reserve_packages.get(owner)
                if (
                    held_package is not None
                    and held_package.state == "STAGING"
                    and held_package.member_types == (21002,)
                ):
                    continue
                target_ids = tuple(sorted(owner_tracks))
                if not target_ids or (
                    expected is not None and len(target_ids) < expected
                ):
                    continue
                if self.config.l_first_wave_per_target > 0:
                    if self._l_core_ids is None or len(target_ids) != self._l_target_count:
                        continue
                target_id = target_ids[self._l_rank[owner] % len(target_ids)]
                if self.config.l_first_wave_per_target > 0:
                    if owner in self._l_core_ids:
                        self._l_core_assignments[owner] = target_id
                    elif len(target_ids) >= 2:
                        fallback_id = target_ids[
                            (self._l_rank[owner] + 1) % len(target_ids)
                        ]
                        package = ReservePackageSnapshot(
                            package_id=owner,
                            member_ids=(owner,),
                            member_types=(21002,),
                            state="STAGING",
                            provisional_target_id=target_id,
                            fallback_target_id=fallback_id,
                            responsibility_region="l",
                            dispatch_count=0,
                        )
                        self._reserve_packages[package.package_id] = package
                        self._audit_initial_reserve(package, step, source="satellite")
                        continue
                self._assignments[owner] = target_id
            track = owner_tracks.get(target_id)
            if track is None:
                continue
            goal = {"lon": track.lon, "lat": track.lat}
            version = (track.lon, track.lat)
            key = (owner, target_id)
            if self._goal_versions.get(key) != version:
                self._goals[owner] = goal
                self._goal_versions[key] = version
                self._set_target_action(rows, owner, target_id)

        self._record_hm_progress(step, current_hm)
        self._record_l_progress(step, current_l, l_tracks_by_owner)
        self._plan_reserve_dispatches(
            step, current_hm, current_l, l_tracks_by_owner, rows,
        )
        self._plan_dynamic_retargets(step, current_hm, rows)
        self._actions = {owner: tuple(value) for owner, value in rows.items()}
        return current_hm

    @staticmethod
    def _set_target_action(
        rows: dict[int, list[tuple[float, ...]]],
        owner: int,
        target_id: int,
        *,
        priority: bool = False,
    ) -> None:
        existing = [row for row in rows[owner] if int(row[0]) == 4]
        if priority:
            rows[owner] = [row for row in rows[owner] if int(row[0]) != 4]
            rows[owner].append((4, owner, target_id, 0))
            return
        if existing:
            return
        rows[owner].append((4, owner, target_id, 0))

    def _finalize_terminal_hm(
        self, step: int, terminal_reasons: Mapping[int, str]
    ) -> None:
        for owner in sorted(terminal_reasons):
            if owner in self._terminalized:
                continue
            progress = self._hm_progress.pop(owner, None)
            target_id = self._assignments.pop(owner, None)
            is_reserve = owner in self._reserve_h_ids
            group = (
                progress.communication_group
                if progress is not None
                else self._last_groups.get(owner, (owner,))
            )
            recipients = tuple(item for item in group if item in self._owner_ledgers)
            if not recipients:
                recipients = (owner,)
            if is_reserve:
                self._terminate_reserve_member(
                    step, owner, terminal_reasons[owner], group
                )
            valid_delivery = (
                target_id is not None
                and progress is not None
                and progress.target_id == target_id
            )
            if valid_delivery:
                event_id = f"terminal:{owner}:{step}:{target_id}"
                evidence = DeliveryEvidence(
                    owner,
                    progress.missile_type,
                    target_id,
                    step,
                    progress.stage,
                    progress.minimum_target_distance_metres,
                    event_id=event_id,
                    probability=(
                        None
                        if progress.target_position_source == "detect_fresh"
                        else 0.15
                    ),
                )
                forced_low_confidence = evidence.probability is not None
                delivery_probability = self._delivery_probability(evidence)
                event = _LedgerEvent(step, "delivery", event_id, evidence)
            else:
                event_id = f"terminal-release:{owner}:{step}"
                forced_low_confidence = True
                delivery_probability = None
                event = _LedgerEvent(step, "release", event_id, owner)
            missile_type = (
                progress.missile_type
                if progress is not None
                else self._hm_types.get(owner)
            )
            core_target_id = self._core_hm_assignments.get(owner)
            if (
                not is_reserve
                and core_target_id in self._public_by_id
                and progress is not None
                and (
                    (
                        progress.stage <= 3
                        and progress.minimum_target_distance_metres is not None
                        and progress.minimum_target_distance_metres > 10_000.0
                    )
                    or (
                        delivery_probability is not None
                        and delivery_probability <= 0.15
                    )
                )
            ):
                target = self._public_by_id[core_target_id]
                target_position = {"lon": target.lon, "lat": target.lat}
                loss = _EarlyLoss(
                    event_id=f"early-loss:hm:{owner}:{step}:{core_target_id}",
                    step=step,
                    owner=owner,
                    member_type=progress.missile_type,
                    stage=progress.stage,
                    target_id=core_target_id,
                    target_position=target_position,
                    member_position=progress.position or target_position,
                    communication_group=group,
                    distance_km=progress.current_distance_km,
                    minimum_distance_metres=(
                        progress.minimum_target_distance_metres
                    ),
                    delivery_probability=delivery_probability,
                    nominal_damage=self._nominal_core_damage(
                        progress.missile_type, target.target_type
                    ),
                )
                self._record_early_loss(
                    self._hm_early_losses, loss, recipients
                )
            terminal_event_id = (
                f"missile-terminal:{owner}:{step}"
                if missile_type == 21001
                and self.config.missile_trace_interval_steps > 0
                else None
            )
            self._write_audit(
                "terminal_evidence",
                step,
                owner=owner,
                type=missile_type,
                target=target_id,
                last_stage=None if progress is None else progress.stage,
                min_distance_m=(
                    None
                    if progress is None
                    else progress.minimum_target_distance_metres
                ),
                source=(
                    None if progress is None else progress.target_position_source
                ),
                age=(
                    None if progress is None else progress.target_track_age_steps
                ),
                forced_low_confidence=forced_low_confidence,
                last_group=group,
                delivery_probability=delivery_probability,
                evidence_recorded=valid_delivery and not is_reserve,
                terminal_reason=terminal_reasons[owner],
                missile_terminal_event_id=terminal_event_id,
                event_id=event_id,
            )
            if terminal_event_id is not None:
                history = list(self._missile_trace_history.get(owner, ()))
                self._write_audit(
                    "missile_terminal",
                    step,
                    id=owner,
                    type=missile_type,
                    current_target_id=target_id,
                    terminal_reason=terminal_reasons[owner],
                    last_sample=None if not history else history[-1],
                    history=history,
                    history_length=len(history),
                    event_id=terminal_event_id,
                    terminal_evidence_event_id=event_id,
                )
            if not is_reserve:
                self._add_ledger_event(recipients, event)
            self._goals.pop(owner, None)
            self._goal_versions = {
                key: value for key, value in self._goal_versions.items() if key[0] != owner
            }
            self._terminalized.add(owner)
            self._terminal_guidance.reset(owner)
            self._owner_velocity_history[owner] = self._new_owner_velocity_state()

    def _finalize_terminal_l(
        self, step: int, terminal_reasons: Mapping[int, str]
    ) -> None:
        for owner in sorted(terminal_reasons):
            if owner in self._terminalized:
                continue
            progress = self._l_progress.pop(owner, None)
            target_id = self._assignments.pop(owner, None)
            self._goals.pop(owner, None)
            self._goal_versions = {
                key: value
                for key, value in self._goal_versions.items()
                if key[0] != owner
            }
            group = (
                progress.communication_group
                if progress is not None
                else self._last_groups.get(owner, (owner,))
            )
            is_reserve = any(
                owner in package.member_ids
                for package in self._reserve_packages.values()
            )
            self._terminate_reserve_member(
                step, owner, terminal_reasons[owner], group
            )
            self._terminalized.add(owner)
            self._write_audit(
                "terminal_evidence",
                step,
                owner=owner,
                type=21002,
                target=target_id,
                last_stage=None if progress is None else progress.stage,
                min_distance_m=(
                    None
                    if progress is None
                    else progress.minimum_target_distance_metres
                ),
                source="owner_catalogue" if progress is not None else None,
                age=None,
                forced_low_confidence=True,
                last_group=group,
                delivery_probability=None,
                evidence_recorded=progress is not None and not is_reserve,
                terminal_reason=terminal_reasons[owner],
                missile_terminal_event_id=None,
                event_id=f"terminal:l:{owner}:{step}",
            )
            if (
                self.config.l_first_wave_per_target == 0
                or progress is None
                or owner not in (self._l_core_ids or ())
                or owner in self._l_early_losses
                or progress.stage > 3
                or progress.minimum_target_distance_metres is None
                or progress.minimum_target_distance_metres <= 5_000.0
            ):
                continue
            recipients = tuple(
                item for item in group if item in self._owner_ledgers
            ) or (owner,)
            loss = _EarlyLoss(
                event_id=f"early-loss:l:{owner}:{step}:{progress.target_id}",
                step=step,
                owner=owner,
                member_type=21002,
                stage=progress.stage,
                target_id=progress.target_id,
                target_position=progress.target_position,
                member_position=progress.position or progress.target_position,
                communication_group=progress.communication_group,
                distance_km=progress.current_distance_km,
                minimum_distance_metres=progress.minimum_target_distance_metres,
                delivery_probability=None,
                nominal_damage=0.8,
            )
            self._record_early_loss(self._l_early_losses, loss, recipients)

    def _record_early_loss(
        self,
        losses: dict[int, _EarlyLoss],
        loss: _EarlyLoss,
        recipients: tuple[int, ...],
    ) -> None:
        previous = losses.get(loss.owner)
        if previous is not None:
            if previous != loss:
                raise ValueError(
                    f"early loss owner {loss.owner} was reused with new evidence"
                )
            return
        losses[loss.owner] = loss
        self._add_ledger_event(
            recipients,
            _LedgerEvent(loss.step, "early_loss", loss.event_id, loss),
        )
        self._write_audit(
            "reserve_early_loss",
            loss.step,
            owner=loss.owner,
            target=loss.target_id,
            type=loss.member_type,
            target_type=(
                self._public_by_id[loss.target_id].target_type
                if loss.target_id in self._public_by_id
                else 9500
            ),
            nominal_damage=loss.nominal_damage,
            stage=loss.stage,
            min_distance_m=loss.minimum_distance_metres,
            probability=loss.delivery_probability,
            last_group=loss.communication_group,
            recipients=recipients,
            event_id=loss.event_id,
        )

    def _terminate_reserve_member(
        self,
        step: int,
        owner: int,
        reason: str,
        group: tuple[int, ...],
    ) -> None:
        terminated = []
        for package_id, package in tuple(self._reserve_packages.items()):
            if owner not in package.member_ids:
                continue
            self._reserve_packages.pop(package_id)
            status = {
                "package": package_id,
                "members": package.member_ids,
                "owner": owner,
                "previous_state": package.state,
                "status": "TERMINATED",
                "reason": reason,
                "group": group,
            }
            self._reserve_package_terminations[package_id] = status
            self._write_audit("reserve_terminal", step, **status)
            terminated.append(package)
            for member_id in package.member_ids:
                self._reserve_holding.reset(member_id)
                self._reserve_holding_anchors.pop(member_id, None)
            self._reserve_reachability_origins.pop(package_id, None)
            if package.state == "STAGING":
                for member_id in package.member_ids:
                    self._assignments.pop(member_id, None)
                    self._goals.pop(member_id, None)
                    self._goal_versions = {
                        key: value
                        for key, value in self._goal_versions.items()
                        if key[0] != member_id
                    }
        if terminated:
            self._reserve_member_snapshots.pop(owner, None)
            self._reserve_seen_members.discard(owner)
            self._reserve_retarget_counts.pop(owner, None)
            # Remove terminated definitions from the pure skill. Surviving
            # coordinator snapshots re-register on the next decision.
            self._reserve_echelon.reset()

    def _add_ledger_event(
        self, owners: tuple[int, ...], event: _LedgerEvent
    ) -> None:
        for owner in owners:
            ledger = self._owner_ledgers.setdefault(owner, {})
            previous = ledger.get(event.event_id)
            if previous is not None and previous != event:
                raise ValueError(f"event_id {event.event_id!r} was reused with new payload")
            ledger[event.event_id] = event

    def _write_audit(self, event: str, step: int, **fields) -> None:
        if self._audit_path is None:
            return
        record = {"event": event, "step": int(step), **fields}
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._audit_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")

    @staticmethod
    def _legal_position(value) -> dict[str, float] | None:
        try:
            position = _position(value)
        except (TypeError, ValueError):
            return None
        if position is None or not all(
            math.isfinite(component) for component in position.values()
        ):
            return None
        return position

    def _new_owner_velocity_state(self) -> _OwnerVelocityState:
        return _OwnerVelocityState(
            deque(maxlen=self.config.velocity_window_steps)
        )

    def _update_owner_velocity(self, owner: int, observation: Mapping) -> None:
        """Update an owner-local velocity from a bounded position window."""
        state = self._owner_velocity_history.get(owner)
        if state is None:
            state = self._new_owner_velocity_state()
            self._owner_velocity_history[owner] = state
        raw_step = observation.get("step")
        if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
            return
        step = float(raw_step)
        position = self._legal_position(observation.get("self") or {})
        if (
            not math.isfinite(step)
            or not self._valid_lonlat(position)
        ):
            return
        assert position is not None
        lon, lat = position["lon"], position["lat"]
        if state.positions and step <= state.positions[-1].step:
            return
        current = _OwnerPositionSample(step, lon, lat)
        state.positions.append(current)
        if len(state.positions) < 2:
            return

        oldest = state.positions[0]
        delta_steps = step - oldest.step
        delta_seconds = delta_steps * float(self.config.seconds_per_step)
        longitude_delta = ((lon - oldest.lon + 180.0) % 360.0) - 180.0
        mean_latitude = math.radians((oldest.lat + lat) / 2.0)
        sin_latitude = math.sin(mean_latitude)
        wgs84_a = 6_378_137.0
        wgs84_e2 = 6.69437999014e-3
        denominator = math.sqrt(1.0 - wgs84_e2 * sin_latitude * sin_latitude)
        prime_vertical_radius = wgs84_a / denominator
        meridian_radius = (
            wgs84_a * (1.0 - wgs84_e2) / denominator**3
        )
        east = (
            math.radians(longitude_delta)
            * prime_vertical_radius
            * math.cos(mean_latitude)
            / delta_seconds
        )
        north = (
            math.radians(lat - oldest.lat)
            * meridian_radius
            / delta_seconds
        )
        speed = math.hypot(east, north)
        if not all(math.isfinite(value) for value in (east, north, speed)):
            return
        if speed < 1.0:
            state.last_valid_east = None
            state.last_valid_north = None
            state.source = "none"
            return
        if speed > 1_000.0:
            state.positions.clear()
            state.positions.append(current)
            return
        state.last_valid_east = east
        state.last_valid_north = north
        state.source = "estimated_window"

    @staticmethod
    def _ecf_velocity_to_local(
        velocity: Mapping | object,
        position: Mapping[str, float],
    ) -> tuple[float, float] | None:
        components = tuple(_field(velocity, name) for name in ("x", "y", "z"))
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in components
        ):
            return None
        x, y, z = (float(value) for value in components)
        if not all(math.isfinite(value) for value in (x, y, z)):
            return None
        longitude = math.radians(position["lon"])
        latitude = math.radians(position["lat"])
        east = -math.sin(longitude) * x + math.cos(longitude) * y
        north = (
            -math.sin(latitude) * math.cos(longitude) * x
            - math.sin(latitude) * math.sin(longitude) * y
            + math.cos(latitude) * z
        )
        return east, north

    def _owner_local_velocity(
        self,
        owner: int,
        own: Mapping,
        position: Mapping[str, float] | None,
    ) -> tuple[str, float | None, float | None]:
        observed = tuple(
            value for value in (own.get("velocity"), own.get("vel_lla"))
            if value is not None
        )
        if position is not None:
            for value in observed:
                local = self._ecf_velocity_to_local(value, position)
                if local is not None:
                    return "observed", local[0], local[1]
        for value in observed:
            speed, heading = _field(value, "speed"), _field(value, "heading")
            if (
                isinstance(speed, bool)
                or isinstance(heading, bool)
                or not isinstance(speed, (int, float))
                or not isinstance(heading, (int, float))
            ):
                continue
            speed, heading = float(speed), float(heading)
            if not math.isfinite(speed) or speed < 0 or not math.isfinite(heading):
                continue
            heading_radians = math.radians(heading)
            return (
                "observed",
                speed * math.sin(heading_radians),
                speed * math.cos(heading_radians),
            )
        state = self._owner_velocity_history.get(owner)
        if (
            state is not None
            and state.source == "estimated_window"
            and state.last_valid_east is not None
            and state.last_valid_north is not None
        ):
            return (
                "estimated_window",
                state.last_valid_east,
                state.last_valid_north,
            )
        return "none", None, None

    def _missile_trace_sample(
        self,
        step: int,
        owner: int,
        current_hm: Mapping[int, dict],
    ) -> dict:
        own = current_hm[owner]
        position = self._legal_position(own)
        target_id = self._assignments.get(owner)
        target_track = self._owner_target_tracks(owner, step).get(target_id)
        aimpoint = None if target_track is None else target_track.position
        aimpoint_distance_km = (
            None
            if position is None or aimpoint is None
            else _distance_km(position, aimpoint)
        )
        group = self._communication_group(owner, own)
        active_h_ids = tuple(
            candidate_id
            for candidate_id, candidate in sorted(current_hm.items())
            if int(candidate.get("type", -1)) == 21000
            and target_id is not None
            and self._assignments.get(candidate_id) == target_id
        )
        h_positions = {
            candidate_id: candidate_position
            for candidate_id in active_h_ids
            if (
                candidate_position := self._legal_position(
                    current_hm[candidate_id]
                )
            ) is not None
        }
        nearest_h_id = None
        nearest_h_separation_km = None
        if position is not None and h_positions:
            nearest_h_id, nearest_h_separation_km = min(
                (
                    (candidate_id, _distance_km(position, candidate_position))
                    for candidate_id, candidate_position in h_positions.items()
                ),
                key=lambda item: (item[1], item[0]),
            )
        raw_stage = own.get("stage")
        try:
            stage = (
                None
                if isinstance(raw_stage, bool) or raw_stage is None
                else max(0, int(raw_stage))
            )
        except (TypeError, ValueError, OverflowError):
            stage = None
        raw_launch_step = self._observations[owner].get(
            "_orchestrator_launch_step"
        )
        try:
            launch_step = (
                None
                if isinstance(raw_launch_step, bool) or raw_launch_step is None
                else int(raw_launch_step)
            )
        except (TypeError, ValueError, OverflowError):
            launch_step = None
        if launch_step is not None and (launch_step < 0 or launch_step > step):
            launch_step = None
        progress = self._hm_progress.get(owner)
        motion = self._motion_guidance.get(owner)
        if motion is None or motion["step"] != step:
            motion = {}
        velocity_source, velocity_east, velocity_north = (
            self._owner_local_velocity(owner, own, position)
        )
        return {
            "event": "missile_trace",
            "step": int(step),
            "id": owner,
            "type": 21001,
            "stage": stage,
            "position": position,
            "velocity": self._audit_velocity(own),
            "velocity_source": motion.get("velocity_source", velocity_source),
            "velocity_east": velocity_east,
            "velocity_north": velocity_north,
            "current_target_id": target_id,
            "aimpoint": None if aimpoint is None else dict(aimpoint),
            "track_source": (
                None if target_track is None else target_track.source
            ),
            "track_fingerprint": (
                None if target_track is None else target_track.fingerprint
            ),
            "track_age": (
                None if target_track is None else target_track.age_steps
            ),
            "aimpoint_distance_km": aimpoint_distance_km,
            "comm_group": group,
            "same_target_active_h_ids": active_h_ids,
            "nearest_h_id": nearest_h_id,
            "nearest_h_separation_km": nearest_h_separation_km,
            "same_group": (
                None if nearest_h_id is None else nearest_h_id in set(group)
            ),
            "launch_step": launch_step,
            "steps_since_launch": (
                None if launch_step is None else step - launch_step
            ),
            "maneuver_state": int(
                self._observations[owner].get(
                    "_orchestrator_maneuver_state", 0
                )
            ),
            "progress_step": None if progress is None else progress.step,
            "minimum_aimpoint_distance_m": (
                None
                if progress is None
                else progress.minimum_target_distance_metres
            ),
            "retarget_count": self._retarget_counts.get(owner, 0),
            "last_retarget_step": self._last_retarget_steps.get(owner),
            "baseline_action": motion.get("baseline_action"),
            "final_action": motion.get("final_action"),
            "override_reason": motion.get("override_reason"),
            "predicted_miss_distance": motion.get("predicted_miss_distance"),
            "control_state": motion.get("control_state"),
            "no_improvement_count": motion.get("no_improvement_count"),
            "heading_error": motion.get("heading_error"),
        }

    def _record_missile_traces(
        self, step: int, current_hm: Mapping[int, dict]
    ) -> None:
        interval = self.config.missile_trace_interval_steps
        if self._audit_path is None or interval == 0:
            return
        for owner, own in sorted(current_hm.items()):
            if int(own.get("type", -1)) != 21001:
                continue
            previous_step = self._audited_missile_trace_steps.get(owner)
            if previous_step is not None and step - previous_step < interval:
                continue
            sample = self._missile_trace_sample(step, owner, current_hm)
            history = self._missile_trace_history.setdefault(
                owner,
                deque(maxlen=self.config.missile_terminal_history),
            )
            if history and history[-1]["step"] == step:
                continue
            history.append(deepcopy(sample))
            self._write_audit(
                "missile_trace",
                step,
                **{
                    name: value
                    for name, value in sample.items()
                    if name not in ("event", "step")
                },
            )
            self._audited_missile_trace_steps[owner] = step

    @staticmethod
    def _audit_velocity(own: Mapping) -> dict[str, float] | None:
        for field_name in ("velocity", "vel_lla"):
            value = own.get(field_name)
            if value is None:
                continue
            components = {}
            for name in ("x", "y", "z", "speed", "up", "heading"):
                raw = _field(value, name)
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    continue
                component = float(raw)
                if math.isfinite(component):
                    components[name] = component
            if components:
                return components
        return None

    @staticmethod
    def _formation_offsets_km(
        origin: dict[str, float],
        lead: dict[str, float],
        target: dict[str, float],
    ) -> tuple[float | None, float | None]:
        radius_km = 6_371.0088

        def local_delta(point: dict[str, float]) -> tuple[float, float]:
            longitude_delta = ((point["lon"] - origin["lon"] + 180.0) % 360.0) - 180.0
            mean_latitude = math.radians((origin["lat"] + point["lat"]) / 2.0)
            east = radius_km * math.radians(longitude_delta) * math.cos(mean_latitude)
            north = radius_km * math.radians(point["lat"] - origin["lat"])
            return east, north

        target_east, target_north = local_delta(target)
        target_distance = math.hypot(target_east, target_north)
        if target_distance <= 1e-12:
            return None, None
        lead_east, lead_north = local_delta(lead)
        forward_east = target_east / target_distance
        forward_north = target_north / target_distance
        forward = lead_east * forward_east + lead_north * forward_north
        lateral = -lead_east * forward_north + lead_north * forward_east
        return lateral, forward

    def _record_formation_status(self, step: int) -> None:
        if (
            self._audit_path is None
            or step % self.config.formation_audit_interval_steps != 0
        ):
            return
        current_hm = {}
        for owner in self._current_owners(step):
            own = self._observations[owner].get("self") or {}
            try:
                kind = int(own.get("type", -1))
            except (TypeError, ValueError):
                continue
            if kind in (21000, 21001) and self._alive(owner):
                current_hm[owner] = own

        for m_id, own in sorted(current_hm.items()):
            if int(own["type"]) != 21001 or self._audited_formation_steps.get(m_id) == step:
                continue
            target_id = self._assignments.get(m_id)
            group = self._communication_group(m_id, own)
            group_members = frozenset(group)
            h_ids = tuple(
                owner
                for owner, candidate in sorted(current_hm.items())
                if int(candidate["type"]) == 21000
                and owner in group_members
                and target_id is not None
                and self._assignments.get(owner) == target_id
            )
            try:
                m_position = _position(own)
            except (TypeError, ValueError):
                m_position = None
            if m_position is not None and not all(
                math.isfinite(value) for value in m_position.values()
            ):
                m_position = None

            h_positions = {}
            for h_id in h_ids:
                try:
                    position = _position(current_hm[h_id])
                except (TypeError, ValueError):
                    continue
                if position is not None and all(
                    math.isfinite(value) for value in position.values()
                ):
                    h_positions[h_id] = position
            nearest_h_id = None
            separation_km = None
            if m_position is not None and h_positions:
                nearest_h_id, separation_km = min(
                    (
                        (h_id, _distance_km(m_position, position))
                        for h_id, position in h_positions.items()
                    ),
                    key=lambda item: (item[1], item[0]),
                )

            target_track = self._owner_target_tracks(m_id, step).get(target_id)
            lateral_km = forward_km = None
            if (
                m_position is not None
                and nearest_h_id is not None
                and target_track is not None
            ):
                lateral_km, forward_km = self._formation_offsets_km(
                    m_position,
                    h_positions[nearest_h_id],
                    target_track.position,
                )
            raw_stage = own.get("stage")
            try:
                m_stage = (
                    None
                    if isinstance(raw_stage, bool) or raw_stage is None
                    else max(0, int(raw_stage))
                )
            except (TypeError, ValueError, OverflowError):
                m_stage = None
            fields = {
                "m_id": m_id,
                "target_id": target_id,
                "m_stage": m_stage,
                "m_lon": None if m_position is None else m_position["lon"],
                "m_lat": None if m_position is None else m_position["lat"],
                "m_track_source": (
                    None if target_track is None else target_track.source
                ),
                "m_track_age": (
                    None if target_track is None else target_track.age_steps
                ),
                "h_ids": h_ids,
                "nearest_h_id": nearest_h_id,
                "separation_km": separation_km,
                "lateral_km": lateral_km,
                "forward_km": forward_km,
                "same_communication_group": (
                    None
                    if nearest_h_id is None
                    else True
                ),
                "group": group,
                "movement": int(
                    self._observations[m_id].get("_orchestrator_maneuver_state", 0)
                ),
                "velocity": self._audit_velocity(own),
            }
            if not h_ids:
                fields["reason"] = "no_active_h_lead"
            self._write_audit("formation_status", step, **fields)
            self._audited_formation_steps[m_id] = step

    @staticmethod
    def _delivery_probability(evidence: DeliveryEvidence) -> float:
        if evidence.probability is not None:
            return float(evidence.probability)
        distance = evidence.minimum_target_distance_metres
        if distance is not None and distance <= 500.0:
            return 0.98
        if evidence.last_stage >= 4:
            return 0.85
        if distance is not None and distance <= 1200.0:
            return 0.65
        if evidence.disappeared:
            return 0.15
        return 0.0

    def _record_belief_changes(
        self,
        step: int,
        group: tuple[int, ...],
        tracker: TargetDamageProbabilitySkill,
    ) -> None:
        if self._audit_path is None:
            return
        for target in self.public_targets:
            belief = tracker.estimate(target.target_id)
            signature = (
                belief.destruction_probability,
                belief.projected_destruction_probability,
                belief.committed_missiles,
                belief.delivery_missiles,
            )
            key = (group, target.target_id)
            if self._audited_beliefs.get(key) == signature:
                continue
            self._write_audit(
                "belief_change",
                step,
                group=group,
                target=target.target_id,
                type=target.target_type,
                delivered_probability=belief.destruction_probability,
                projected_probability=belief.projected_destruction_probability,
                committed_ids=belief.committed_missiles,
                delivery_ids=belief.delivery_missiles,
            )
            self._audited_beliefs[key] = signature

    def _record_retarget_status(
        self,
        step: int,
        member: RetargetMemberSnapshot,
        targets: tuple[RetargetTargetSnapshot, ...],
        reason: str,
    ) -> None:
        if self._audit_path is None:
            return
        current = next(
            (target for target in targets if target.target_id == member.current_target_id),
            None,
        )
        settings = self.config.dynamic_retarget
        fields = {
            "owner": member.entity_id,
            "type": member.missile_type,
            "current_target": member.current_target_id,
            "delivered_probability": (
                None if current is None else current.destruction_probability
            ),
            "threshold": settings.destruction_threshold,
            "stage": member.stage,
            "distance_km": member.current_distance_km,
            "group": member.communication_group,
            "retarget_count": member.retarget_count,
            "last_retarget_step": member.last_retarget_step,
            "fresh_target_ids": member.fresh_target_ids,
            "velocity_x": member.velocity_x,
            "velocity_y": member.velocity_y,
            "max_target_separation_km": settings.max_target_separation_km,
            "max_turn_angle_degrees": settings.max_turn_angle_degrees,
            "max_candidate_distance_km": settings.max_candidate_distance_km,
            "max_per_old_target_per_step": settings.max_per_old_target_per_step,
            "reason": reason,
        }
        signature = (
            tuple((name, fields[name]) for name in sorted(fields))
            if self.config.dynamic_retarget_enabled
            else (reason,)
        )
        if self._audited_retarget_status.get(member.entity_id) == signature:
            return
        self._write_audit("retarget_status", step, **fields)
        self._audited_retarget_status[member.entity_id] = signature

    def _record_incomplete_retarget_status(
        self,
        step: int,
        owner: int,
        current_target: int | None,
        progress: _HMProgress | None,
        group: tuple[int, ...],
    ) -> None:
        if self._audit_path is None:
            return
        reason = (
            "insufficient_legal_data"
            if self.config.dynamic_retarget_enabled
            else "dynamic_retarget_disabled"
        )
        own = self._observations[owner].get("self") or {}
        raw_stage = _field(own, "stage")
        velocity_x, velocity_y = self._velocity_xy(own)
        fresh_target_ids = tuple(sorted(
            target_id
            for target_id, track in self._owner_detected_public_targets(
                owner, step
            ).items()
            if track.source == "detect_fresh"
        ))
        settings = self.config.dynamic_retarget
        fields = {
            "owner": owner,
            "type": 21001,
            "current_target": current_target,
            "delivered_probability": None,
            "threshold": settings.destruction_threshold,
            "stage": None if raw_stage is None else max(0, int(raw_stage)),
            "distance_km": (
                None if progress is None else progress.current_distance_km
            ),
            "group": group,
            "retarget_count": self._retarget_counts.get(owner, 0),
            "last_retarget_step": self._last_retarget_steps.get(owner),
            "fresh_target_ids": fresh_target_ids,
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "max_target_separation_km": settings.max_target_separation_km,
            "max_turn_angle_degrees": settings.max_turn_angle_degrees,
            "max_candidate_distance_km": settings.max_candidate_distance_km,
            "max_per_old_target_per_step": settings.max_per_old_target_per_step,
            "reason": reason,
        }
        signature = (
            tuple((name, fields[name]) for name in sorted(fields))
            if self.config.dynamic_retarget_enabled
            else (reason,)
        )
        if self._audited_retarget_status.get(owner) == signature:
            return
        self._write_audit("retarget_status", step, **fields)
        self._audited_retarget_status[owner] = signature

    def _tracker_from_ledger(
        self, ledger: Mapping[str, _LedgerEvent]
    ) -> TargetDamageProbabilitySkill:
        tracker = TargetDamageProbabilitySkill()
        for target in self.public_targets:
            tracker.register(TargetPriorEvidence(
                target.target_id,
                target.target_type,
                target.initial_health,
                discovered_step=0,
            ))
        operation_order = {
            "commit": 0,
            "transfer": 1,
            "delivery": 2,
            "release": 2,
            "early_loss": 3,
        }
        for event in sorted(
            ledger.values(),
            key=lambda item: (
                item.step,
                operation_order[item.operation],
                item.event_id,
            ),
        ):
            if event.operation == "commit":
                tracker.commit(event.payload)
            elif event.operation == "transfer":
                missile_id, target_id = event.payload
                tracker.transfer(missile_id, target_id, event_id=event.event_id)
            elif event.operation == "delivery":
                tracker.report_delivery(event.payload)
            elif event.operation == "release":
                tracker.release(event.payload, event_id=event.event_id)
            elif event.operation == "early_loss":
                continue
            else:
                raise RuntimeError(f"unknown owner-ledger operation {event.operation!r}")
        return tracker

    def _merge_owner_ledgers(
        self, owners: tuple[int, ...]
    ) -> TargetDamageProbabilitySkill:
        merged: dict[str, _LedgerEvent] = {}
        for owner in owners:
            for event_id, event in self._owner_ledgers[owner].items():
                previous = merged.get(event_id)
                if previous is not None and previous != event:
                    raise ValueError(f"event_id {event_id!r} has conflicting owner ledgers")
                merged[event_id] = event
        for owner in owners:
            self._owner_ledgers[owner] = dict(merged)
        shared = self._tracker_from_ledger(merged).export()
        return TargetDamageProbabilitySkill.from_snapshot(shared)

    def _owner_detected_public_targets(
        self,
        owner: int,
        step: int,
        *,
        max_age_steps: float | None = None,
    ) -> dict[int, _TargetPosition]:
        positions: dict[int, _TargetPosition] = {}
        own = self._observations[owner].get("self") or {}
        detected = own.get("detectInfo") or {}
        if not isinstance(detected, Mapping):
            return positions
        for raw_id, value in detected.items():
            try:
                target_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            original = self._public_by_id.get(target_id)
            if original is None:
                continue
            try:
                entity_id = int(_field(value, "entity_id", target_id))
                target_type = int(
                    _field(value, "entity_type", _field(value, "type", -1))
                )
            except (TypeError, ValueError):
                continue
            if entity_id != target_id or target_type != original.target_type:
                continue
            try:
                position = _position(value)
            except (TypeError, ValueError):
                continue
            if position is None or not all(math.isfinite(item) for item in position.values()):
                continue
            raw_time = _field(value, "time")
            if (
                isinstance(raw_time, bool)
                or not isinstance(raw_time, (int, float))
                or not math.isfinite(float(raw_time))
            ):
                raw_time = None
            else:
                raw_time = float(raw_time)
            fingerprint = (
                raw_time,
                round(position["lon"], 12),
                round(position["lat"], 12),
            )
            cache_key = (owner, target_id)
            cached = self._track_freshness.get(cache_key)
            if cached is None or cached.fingerprint != fingerprint:
                cached = _TrackFreshness(fingerprint, int(step))
                self._track_freshness[cache_key] = cached
            age_steps = float(step - cached.last_changed_step)
            observed_step = float(cached.last_changed_step)
            freshness_limit = (
                self.config.retarget_track_max_age_steps
                if max_age_steps is None
                else max_age_steps
            )
            fresh = age_steps <= freshness_limit
            candidate = _TargetPosition(
                position,
                "detect_fresh" if fresh else "detect_stale",
                observed_step,
                age_steps,
                raw_time,
                fingerprint,
            )
            previous = positions.get(target_id)
            if previous is None or (
                candidate.observed_step is not None
                and (
                    previous.observed_step is None
                    or candidate.observed_step > previous.observed_step
                )
            ):
                positions[target_id] = candidate
        return positions

    def _owner_target_tracks(
        self, owner: int, step: int
    ) -> dict[int, _TargetPosition]:
        tracks = {
            target.target_id: _TargetPosition(
                {"lon": target.lon, "lat": target.lat},
                "public",
                None,
                None,
                None,
                (
                    "public",
                    round(target.lon, 12),
                    round(target.lat, 12),
                ),
            )
            for target in self.public_targets
        }
        tracks.update(self._owner_detected_public_targets(owner, step))
        return tracks

    def _owner_target_positions(
        self, owner: int, step: int
    ) -> dict[int, dict[str, float]]:
        return {
            target_id: track.position
            for target_id, track in self._owner_target_tracks(owner, step).items()
        }

    def _group_fresh_target_tracks(
        self, owners: tuple[int, ...], step: int
    ) -> dict[int, _TargetPosition]:
        newest: dict[int, _TargetPosition] = {}
        for owner in sorted(owners):
            for target_id, track in self._owner_detected_public_targets(
                owner, step
            ).items():
                if track.source != "detect_fresh":
                    continue
                previous = newest.get(target_id)
                if (
                    previous is None
                    or (
                        track.source_time if track.source_time is not None
                        else track.observed_step
                    )
                    > (
                        previous.source_time if previous.source_time is not None
                        else previous.observed_step
                    )
                ):
                    newest[target_id] = track
        return newest

    @staticmethod
    def _velocity_xy(own: Mapping) -> tuple[float | None, float | None]:
        for field_name in ("velocity", "vel_lla"):
            value = own.get(field_name)
            if value is None:
                continue
            raw_x, raw_y = _field(value, "x"), _field(value, "y")
            if raw_x is None or raw_y is None:
                continue
            try:
                x, y = float(raw_x), float(raw_y)
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                return x, y
        return None, None

    @staticmethod
    def _communication_group(owner: int, own: Mapping) -> tuple[int, ...]:
        raw = own.get("commRangeInfo") or {}
        if isinstance(raw, Mapping):
            values = raw.keys()
        elif isinstance(raw, (list, tuple, set, frozenset)):
            values = raw
        else:
            values = ()
        members = {owner}
        for value in values:
            candidate = _field(value, "entity_id", _field(value, "id", value))
            try:
                candidate = int(candidate)
            except (TypeError, ValueError):
                continue
            if candidate >= 0:
                members.add(candidate)
        return tuple(sorted(members))

    def _record_hm_progress(self, step: int, current_hm: Mapping[int, dict]) -> None:
        for owner, own in sorted(current_hm.items()):
            group = self._communication_group(owner, own)
            self._last_groups[owner] = group
            position = _position(own)
            target_id = self._assignments.get(owner)
            target_track = self._owner_target_tracks(owner, step).get(target_id)
            target_position = None if target_track is None else target_track.position
            current_distance = None
            minimum_distance = None
            if position is not None and target_position is not None:
                current_distance = _distance_km(position, target_position)
                minimum_distance = current_distance * 1000.0
                previous = self._hm_progress.get(owner)
                if (
                    previous is not None
                    and previous.target_id == target_id
                    and previous.minimum_target_distance_metres is not None
                ):
                    minimum_distance = min(
                        minimum_distance, previous.minimum_target_distance_metres
                    )
            self._hm_progress[owner] = _HMProgress(
                step=step,
                missile_type=int(own["type"]),
                stage=max(0, int(own.get("stage", 0))),
                position=position,
                target_id=target_id,
                target_position=target_position,
                target_position_source=(
                    None if target_track is None else target_track.source
                ),
                target_track_age_steps=(
                    None if target_track is None else target_track.age_steps
                ),
                current_distance_km=current_distance,
                minimum_target_distance_metres=minimum_distance,
                communication_group=group,
            )

    def _record_l_progress(
        self,
        step: int,
        current_l: Mapping[int, dict],
        tracks_by_owner: Mapping[int, Mapping[int, object]],
    ) -> None:
        if self.config.l_first_wave_per_target == 0:
            return
        for owner in sorted(current_l):
            target_id = self._assignments.get(owner)
            if target_id is None:
                continue
            track = tracks_by_owner.get(owner, {}).get(target_id)
            if track is None:
                continue
            previous = self._l_progress.get(owner)
            target_position = (
                previous.target_position
                if previous is not None and previous.target_id == target_id
                else {"lon": float(track.lon), "lat": float(track.lat)}
            )
            position = self._legal_position(current_l[owner])
            distance = (
                None
                if position is None
                else _distance_km(position, target_position)
            )
            minimum_distance = None if distance is None else distance * 1000.0
            if (
                previous is not None
                and previous.target_id == target_id
                and previous.minimum_target_distance_metres is not None
            ):
                minimum_distance = (
                    previous.minimum_target_distance_metres
                    if minimum_distance is None
                    else min(
                        minimum_distance,
                        previous.minimum_target_distance_metres,
                    )
                )
            self._l_progress[owner] = _LProgress(
                step=step,
                stage=max(0, int(current_l[owner].get("stage", 0))),
                position=position,
                target_id=target_id,
                target_position=target_position,
                current_distance_km=distance,
                minimum_target_distance_metres=minimum_distance,
                communication_group=self._communication_group(
                    owner, current_l[owner]
                ),
            )

    def _update_l_catalogue(self, owner: int, step: int) -> dict[int, object]:
        """Merge only this owner's legal 9500 observations into its history."""
        catalogue = self._l_catalogues.setdefault(owner, {})
        observation = self._observations[owner]
        own = observation.get("self") or {}
        detected = own.get("detectInfo") or {}
        if not isinstance(detected, Mapping):
            return dict(catalogue)
        for raw_id, value in detected.items():
            isolated = {
                **observation,
                "self": {**own, "detectInfo": {raw_id: value}},
            }
            try:
                tracks = tracks_from_isolated_observations(
                    {owner: isolated}, step=step,
                )
            except (TypeError, ValueError, OverflowError):
                continue
            for track in tracks:
                previous = catalogue.get(track.target_id)
                if (
                    previous is None
                    or track.observed_step >= previous.observed_step
                ):
                    catalogue[track.target_id] = track
        return dict(catalogue)

    @staticmethod
    def _km_point(position: Mapping[str, float]) -> tuple[float, float]:
        latitude = float(position["lat"])
        return (
            float(position["lon"]) * 111.320 * math.cos(math.radians(latitude)),
            latitude * 110.574,
        )

    @staticmethod
    def _local_km_point(
        position: Mapping[str, float], origin: tuple[float, float]
    ) -> tuple[float, float]:
        longitude_delta = (
            (float(position["lon"]) - origin[0] + 180.0) % 360.0
        ) - 180.0
        mean_latitude = math.radians(
            (float(position["lat"]) + origin[1]) / 2.0
        )
        return (
            6_371.0088 * math.radians(longitude_delta) * math.cos(mean_latitude),
            6_371.0088 * math.radians(float(position["lat"]) - origin[1]),
        )

    @staticmethod
    def _nominal_core_damage(member_type: int, target_type: int) -> float:
        if member_type == 21000:
            return 16.0 if target_type == 9400 else 12.0
        if member_type == 21001:
            return 4.0 if target_type == 9400 else 3.0
        if member_type == 21002 and target_type == 9500:
            return 0.8
        raise ValueError(
            f"unsupported nominal damage pair {member_type}/{target_type}"
        )

    @staticmethod
    def _target_health(target_type: int) -> float:
        try:
            return {9400: 24.0, 9600: 18.0, 9500: 1.2}[target_type]
        except KeyError as exc:
            raise ValueError(f"unsupported target type {target_type}") from exc

    def _visible_early_losses(
        self,
        losses: Mapping[int, _EarlyLoss],
        group: tuple[int, ...],
    ) -> tuple[_EarlyLoss, ...]:
        visible = {
            loss.event_id: loss
            for loss in losses.values()
            if any(
                loss.event_id in self._owner_ledgers.get(owner, {})
                for owner in group
            )
        }
        return tuple(visible[event_id] for event_id in sorted(visible))

    def _hm_reserve_deficits(
        self,
        group: tuple[int, ...],
        coordinate_origin: tuple[float, float] | None = None,
    ) -> tuple[TargetDeficitSnapshot, ...]:
        visible = self._visible_early_losses(self._hm_early_losses, group)
        deficits = []
        for target_id in sorted({loss.target_id for loss in visible}):
            region = self._hm_target_regions.get(target_id)
            target = self._public_by_id.get(target_id)
            if region is None or target is None:
                continue
            target_losses = tuple(
                loss for loss in visible if loss.target_id == target_id
            )
            evidence = tuple(sorted(loss.owner for loss in target_losses))
            initial_commitment = sum(
                self._nominal_core_damage(
                    self._mission_plan.entity_types[owner], target.target_type
                )
                for owner, assigned_target in self._core_hm_assignments.items()
                if assigned_target == target_id
            )
            early_lost = sum(loss.nominal_damage for loss in target_losses)
            remaining_damage = max(0.0, initial_commitment - early_lost)
            missing = self._target_health(target.target_type) - remaining_damage
            if missing <= 0.0:
                continue
            deficits.append(TargetDeficitSnapshot(
                target_id=target_id,
                type=target.target_type,
                weight=float(target.value),
                missing_damage=missing,
                priority=float(target.value),
                position=(
                    self._km_point({"lon": target.lon, "lat": target.lat})
                    if coordinate_origin is None
                    else self._local_km_point(
                        {"lon": target.lon, "lat": target.lat},
                        coordinate_origin,
                    )
                ),
                responsibility_region=region,
                evidence_ids=evidence,
            ))
        return tuple(deficits)

    def _l_reserve_deficits(
        self,
        group: tuple[int, ...],
        coordinate_origin: tuple[float, float] | None = None,
    ) -> tuple[TargetDeficitSnapshot, ...]:
        visible = self._visible_early_losses(self._l_early_losses, group)
        deficits = []
        for target_id in sorted({loss.target_id for loss in visible}):
            evidence_losses = tuple(
                loss for loss in visible if loss.target_id == target_id
            )
            evidence = tuple(sorted(loss.owner for loss in evidence_losses))
            initial_commitment = sum(
                0.8
                for owner, assigned_target in self._l_core_assignments.items()
                if assigned_target == target_id
            )
            early_lost = sum(loss.nominal_damage for loss in evidence_losses)
            remaining_damage = max(0.0, initial_commitment - early_lost)
            missing = self._target_health(9500) - remaining_damage
            if missing <= 0.0:
                continue
            target_position = min(
                evidence_losses, key=lambda loss: (loss.step, loss.owner)
            ).target_position
            deficits.append(TargetDeficitSnapshot(
                target_id=target_id,
                type=9500,
                weight=1.0,
                missing_damage=missing,
                priority=0.0,
                position=(
                    self._km_point(target_position)
                    if coordinate_origin is None
                    else self._local_km_point(
                        target_position, coordinate_origin
                    )
                ),
                responsibility_region="l",
                evidence_ids=evidence,
            ))
        return tuple(deficits)

    def _deficit_event_ids(
        self, deficit: TargetDeficitSnapshot
    ) -> tuple[str, ...]:
        evidence_owners = set(deficit.evidence_ids)
        return tuple(sorted(
            loss.event_id
            for loss in (
                *self._hm_early_losses.values(),
                *self._l_early_losses.values(),
            )
            if loss.owner in evidence_owners
            and loss.target_id == deficit.target_id
        ))

    def _current_reserve_member(
        self,
        owner: int,
        step: int,
        current_hm: Mapping[int, dict],
        current_l: Mapping[int, dict],
        l_tracks_by_owner: Mapping[int, Mapping[int, object]],
        coordinate_origin: tuple[float, float] | None = None,
    ) -> ReserveMemberSnapshot | None:
        package = next(
            (
                candidate
                for candidate in self._reserve_packages.values()
                if owner in candidate.member_ids
            ),
            None,
        )
        if package is None:
            return None
        target_id = self._assignments.get(owner)
        held = package.state == "STAGING" and target_id is None
        if held:
            target_id = package.provisional_target_id
        if target_id is None:
            return None
        if owner in current_hm:
            own = current_hm[owner]
            legal_target_ids = tuple(sorted(self._owner_target_tracks(owner, step)))
            member_type = int(own["type"])
        elif owner in current_l:
            own = current_l[owner]
            legal_target_ids = tuple(sorted(l_tracks_by_owner.get(owner, {})))
            member_type = 21002
        else:
            return None
        position = self._legal_position(own)
        if position is None or target_id not in legal_target_ids:
            return None
        stage = self._observed_nonnegative_int(own.get("stage"))
        _, heading_east, heading_north = self._owner_local_velocity(
            owner, own, position
        )
        snapshot = ReserveMemberSnapshot(
            id=owner,
            type=member_type,
            stage=0 if stage is None else stage,
            group=self._communication_group(owner, own),
            current_target=target_id,
            legal_target_ids=legal_target_ids,
            position=(
                self._km_point(position)
                if coordinate_origin is None
                else self._local_km_point(position, coordinate_origin)
            ),
            retarget_count=self._reserve_retarget_counts.get(owner, 0),
            heading_east=heading_east,
            heading_north=heading_north,
            remaining_steps=max(0, self.config.max_steps - step),
        )
        self._reserve_member_snapshots[owner] = snapshot
        self._reserve_seen_members.add(owner)
        return snapshot

    def _reserve_package_with_fallback_position(
        self,
        package: ReservePackageSnapshot,
        l_tracks_by_owner: Mapping[int, Mapping[int, object]],
        coordinate_origin: tuple[float, float],
    ) -> ReservePackageSnapshot:
        if package.fallback_position is not None:
            return package
        if package.member_types[0] == 21000:
            target = self._public_by_id.get(package.fallback_target_id)
            if target is None:
                return package
            position = self._local_km_point(
                {"lon": target.lon, "lat": target.lat}, coordinate_origin
            )
        else:
            positions = []
            for owner in package.member_ids:
                track = l_tracks_by_owner.get(owner, {}).get(
                    package.fallback_target_id
                )
                if track is None:
                    return package
                positions.append(self._local_km_point(
                    {"lon": float(track.lon), "lat": float(track.lat)},
                    coordinate_origin,
                ))
            position = (
                sum(item[0] for item in positions) / len(positions),
                sum(item[1] for item in positions) / len(positions),
            )
        return replace(package, fallback_position=position)

    def _reserve_rejection_fields(
        self,
        package: ReservePackageSnapshot,
        members: tuple[ReserveMemberSnapshot, ...],
        deficits: tuple[TargetDeficitSnapshot, ...],
        step: int,
        reason: str,
    ) -> dict:
        if reason not in {
            "distance_exceeded",
            "turn_angle_exceeded",
            "insufficient_time",
            "missing_heading",
        }:
            return {}
        if step >= self.config.reserve_deadline_step:
            target_id = package.fallback_target_id
            target_position = package.fallback_position
        else:
            logical = tuple(
                deficit
                for deficit in deficits
                if deficit.responsibility_region == package.responsibility_region
                and deficit.target_id != package.provisional_target_id
                and all(
                    deficit.target_id in member.legal_target_ids
                    for member in members
                )
                and (
                    (members[0].type == 21000 and deficit.type in (9400, 9600))
                    or (members[0].type == 21002 and deficit.type == 9500)
                )
            )
            if not logical:
                return {}
            center_east = sum(member.position[0] for member in members) / len(members)
            center_north = sum(member.position[1] for member in members) / len(members)
            chosen = min(
                logical,
                key=lambda deficit: (
                    -(
                        deficit.weight
                        * deficit.missing_damage
                        / (
                            1.0
                            + math.hypot(
                                deficit.position[0] - center_east,
                                deficit.position[1] - center_north,
                            )
                        )
                    ),
                    -deficit.priority,
                    math.hypot(
                        deficit.position[0] - center_east,
                        deficit.position[1] - center_north,
                    ),
                    deficit.target_id,
                ),
            )
            target_id = chosen.target_id
            target_position = chosen.position
        if target_position is None:
            return {"target": target_id}

        distances = []
        angles = []
        travel_steps = []
        for member in members:
            east = target_position[0] - member.position[0]
            north = target_position[1] - member.position[1]
            distance = math.hypot(east, north)
            distances.append(distance)
            speed = 200.0 if member.type == 21000 else 100.0
            travel_steps.append(
                distance * 1000.0 / speed / self.config.seconds_per_step
            )
            if distance <= 1e-12:
                angles.append(0.0)
                continue
            if member.heading_east is None or member.heading_north is None:
                angles.append(None)
                continue
            norm = math.hypot(member.heading_east, member.heading_north)
            if norm <= 1e-12:
                angles.append(None)
                continue
            cosine = (
                member.heading_east * east + member.heading_north * north
            ) / (norm * distance)
            angles.append(math.degrees(math.acos(min(1.0, max(-1.0, cosine)))))
        member_type = members[0].type
        return {
            "target": target_id,
            "distance_km": max(distances),
            "max_distance_km": (
                self.config.reserve_h_max_distance_km
                if member_type == 21000
                else self.config.reserve_l_max_distance_km
            ),
            "turn_angle_degrees": (
                None if any(angle is None for angle in angles) else max(angles)
            ),
            "max_turn_angle_degrees": (
                self.config.reserve_h_max_turn_angle_degrees
                if member_type == 21000
                else self.config.reserve_l_max_turn_angle_degrees
            ),
            "travel_steps": max(travel_steps),
            "remaining_steps": min(
                member.remaining_steps
                for member in members
                if member.remaining_steps is not None
            ),
        }

    def _record_reserve_deficits(
        self,
        step: int,
        package: ReservePackageSnapshot,
        group: object,
        deficits: tuple[TargetDeficitSnapshot, ...],
    ) -> None:
        for deficit in deficits:
            signature = (
                package.package_id,
                deficit.target_id,
                deficit.missing_damage,
                deficit.evidence_ids,
            )
            if signature in self._audited_reserve_deficits:
                continue
            self._write_audit(
                "reserve_deficit",
                step,
                package=package.package_id,
                owners=package.member_ids,
                group=group,
                target=deficit.target_id,
                type=deficit.type,
                missing=deficit.missing_damage,
                target_health=self._target_health(deficit.type),
                initial_core_nominal=self._initial_core_nominal(deficit),
                early_lost_nominal=self._early_lost_nominal(deficit),
                responsibility_region=deficit.responsibility_region,
                evidence_ids=deficit.evidence_ids,
                event_ids=self._deficit_event_ids(deficit),
            )
            self._audited_reserve_deficits.add(signature)

    def _initial_core_nominal(self, deficit: TargetDeficitSnapshot) -> float:
        if deficit.type == 9500:
            return sum(
                0.8
                for assigned_target in self._l_core_assignments.values()
                if assigned_target == deficit.target_id
            )
        return sum(
            self._nominal_core_damage(
                self._mission_plan.entity_types[owner], deficit.type
            )
            for owner, assigned_target in self._core_hm_assignments.items()
            if assigned_target == deficit.target_id
        )

    def _early_lost_nominal(self, deficit: TargetDeficitSnapshot) -> float:
        evidence = set(deficit.evidence_ids)
        losses = (
            self._l_early_losses.values()
            if deficit.type == 9500
            else self._hm_early_losses.values()
        )
        return sum(
            loss.nominal_damage
            for loss in losses
            if loss.owner in evidence and loss.target_id == deficit.target_id
        )

    def _plan_reserve_dispatches(
        self,
        step: int,
        current_hm: Mapping[int, dict],
        current_l: Mapping[int, dict],
        l_tracks_by_owner: Mapping[int, Mapping[int, object]],
        rows: dict[int, list[tuple[float, ...]]],
    ) -> None:
        self._reserve_member_snapshots.clear()
        if not self._reserve_packages:
            return
        for package in sorted(
            self._reserve_packages.values(), key=lambda item: item.package_id
        ):
            if package.state != "STAGING":
                continue
            coordinate_origin = self._reserve_reachability_origins.get(
                package.package_id
            )
            if coordinate_origin is None:
                for owner in package.member_ids:
                    own = current_hm.get(owner) or current_l.get(owner)
                    position = None if own is None else self._legal_position(own)
                    if not self._valid_lonlat(position):
                        continue
                    assert position is not None
                    coordinate_origin = (position["lon"], position["lat"])
                    self._reserve_reachability_origins[
                        package.package_id
                    ] = coordinate_origin
                    break
            snapshots = tuple(
                self._current_reserve_member(
                    owner,
                    step,
                    current_hm,
                    current_l,
                    l_tracks_by_owner,
                    coordinate_origin,
                )
                for owner in package.member_ids
            )
            if any(snapshot is None for snapshot in snapshots):
                missing = tuple(
                    owner
                    for owner, snapshot in zip(
                        package.member_ids, snapshots, strict=True
                    )
                    if snapshot is None
                )
                self._write_audit(
                    "reserve_rejected",
                    step,
                    package=package.package_id,
                    members=package.member_ids,
                    reason="member_snapshot_unavailable",
                    unavailable_members=missing,
                )
                continue
            active = tuple(snapshot for snapshot in snapshots if snapshot is not None)
            groups = {snapshot.group for snapshot in active}
            if len(groups) != 1:
                self._write_audit(
                    "reserve_rejected",
                    step,
                    package=package.package_id,
                    members=package.member_ids,
                    reason="package_group_mismatch",
                    groups=tuple(sorted(groups, key=repr)),
                )
                continue
            if any(
                self._assignments.get(snapshot.id) is not None
                for snapshot in active
            ):
                self._write_audit(
                    "reserve_rejected",
                    step,
                    package=package.package_id,
                    members=package.member_ids,
                    reason="provisional_target_mismatch",
                )
                continue
            if any(
                package.fallback_target_id not in snapshot.legal_target_ids
                for snapshot in active
            ):
                self._write_audit(
                    "reserve_rejected",
                    step,
                    package=package.package_id,
                    members=package.member_ids,
                    reason="fallback_target_not_legal",
                )
                continue
            group = next(iter(groups))
            assert coordinate_origin is not None
            deficits = (
                self._hm_reserve_deficits(group, coordinate_origin)
                if package.member_types[0] == 21000
                else self._l_reserve_deficits(group, coordinate_origin)
            )
            deficits = tuple(sorted(deficits, key=lambda item: item.target_id))
            self._record_reserve_deficits(step, package, group, deficits)
            decision_package = self._reserve_package_with_fallback_position(
                package, l_tracks_by_owner, coordinate_origin
            )
            decision = self._reserve_echelon.decide(
                step=step,
                members=active,
                deficits=deficits,
                packages=(decision_package,),
            )
            for package_id, reason in sorted(decision.rejected_reasons.items()):
                self._write_audit(
                    "reserve_rejected",
                    step,
                    package=package_id,
                    members=package.member_ids,
                    reason=reason,
                    group=group,
                    **self._reserve_rejection_fields(
                        decision_package, active, deficits, step, reason
                    ),
                )
            for request in decision.requests:
                goals = {}
                invalid_reason = None
                for owner in request.member_ids:
                    assigned = self._assignments.get(owner)
                    if assigned is not None:
                        invalid_reason = "old_target_changed"
                        break
                    if owner in current_hm:
                        track = self._owner_target_tracks(owner, step).get(
                            request.new_target_id
                        )
                        goal = None if track is None else track.position
                    else:
                        track = l_tracks_by_owner.get(owner, {}).get(
                            request.new_target_id
                        )
                        goal = (
                            None
                            if track is None
                            else {"lon": float(track.lon), "lat": float(track.lat)}
                        )
                    if goal is None:
                        invalid_reason = "request_target_not_legal"
                        break
                    goals[owner] = goal
                if invalid_reason is not None:
                    self._write_audit(
                        "reserve_rejected",
                        step,
                        package=package.package_id,
                        members=request.member_ids,
                        reason=invalid_reason,
                        group=group,
                    )
                    continue

                applied_hm = {}
                for owner in request.member_ids:
                    self._reserve_holding.reset(owner)
                    self._reserve_holding_anchors.pop(owner, None)
                self._reserve_reachability_origins.pop(
                    package.package_id, None
                )
                for owner in request.member_ids:
                    goal = goals[owner]
                    self._assignments[owner] = request.new_target_id
                    self._goals[owner] = goal
                    self._goal_versions[(owner, request.new_target_id)] = (
                        goal["lon"], goal["lat"]
                    )
                    self._reserve_retarget_counts[owner] = (
                        self._reserve_retarget_counts.get(owner, 0) + 1
                    )
                    self._set_target_action(
                        rows, owner, request.new_target_id, priority=True
                    )
                    action_event_id = (
                        f"reserve-action:{package.package_id}:{owner}:"
                        f"{step}:{request.new_target_id}"
                    )
                    self._write_audit(
                        "reserve_action4",
                        step,
                        package=package.package_id,
                        owner=owner,
                        target=request.new_target_id,
                        action_type=4,
                        reason=request.reason,
                        event_id=action_event_id,
                    )
                    if owner in current_hm:
                        applied_hm[owner] = current_hm[owner]
                if applied_hm:
                    self._record_hm_progress(step, applied_hm)
                for decided_package in decision.packages:
                    if decided_package.package_id in self._reserve_packages:
                        self._reserve_packages[
                            decided_package.package_id
                        ] = decided_package
                request_event_id = (
                    f"reserve-request:{package.package_id}:{step}:"
                    f"{request.new_target_id}"
                )
                fields = {
                    "package": package.package_id,
                    "members": request.member_ids,
                    "old_target": request.old_target_id,
                    "new_target": request.new_target_id,
                    "reason": request.reason,
                    "group": group,
                    "event_id": request_event_id,
                }
                if request.deficit is not None:
                    fields["evidence_ids"] = request.deficit.evidence_ids
                    fields["evidence_event_ids"] = self._deficit_event_ids(
                        request.deficit
                    )
                    fields["missing"] = request.deficit.missing_damage
                    self._write_audit("reserve_dispatch", step, **fields)
                else:
                    self._write_audit("reserve_fallback", step, **fields)
            if not decision.requests:
                for decided_package in decision.packages:
                    if decided_package.package_id in self._reserve_packages:
                        self._reserve_packages[
                            decided_package.package_id
                        ] = decided_package

    def _plan_dynamic_retargets(
        self,
        step: int,
        current_hm: Mapping[int, dict],
        rows: dict[int, list[tuple[float, ...]]],
    ) -> None:
        groups: dict[tuple[int, ...], list[int]] = {}
        for owner, own in sorted(current_hm.items()):
            if self._alive(owner):
                groups.setdefault(self._communication_group(owner, own), []).append(owner)

        self._damage_by_group.clear()
        for group, raw_owners in sorted(groups.items()):
            owners = tuple(raw_owners)
            tracker = self._merge_owner_ledgers(owners)
            self._damage_by_group[group] = tracker
            self._record_belief_changes(step, group, tracker)
            if (
                not self.config.dynamic_retarget_enabled
                and self._audit_path is None
            ):
                continue

            group_tracks = self._group_fresh_target_tracks(owners, step)
            target_position_sources = {}
            targets = []
            for target in self.public_targets:
                belief = tracker.estimate(target.target_id)
                track = group_tracks.get(target.target_id)
                position = (
                    {"lon": target.lon, "lat": target.lat}
                    if track is None
                    else track.position
                )
                target_position_sources[target.target_id] = (
                    "public" if track is None else track.source
                )
                targets.append(RetargetTargetSnapshot(
                    target.target_id,
                    target.target_type,
                    target.value,
                    position["lon"],
                    position["lat"],
                    belief.destruction_probability,
                    belief.projected_destruction_probability,
                    len(belief.committed_missiles),
                ))
            target_snapshots = tuple(targets)
            targets_by_id = {
                target.target_id: target for target in target_snapshots
            }
            members = []
            incomplete_m = []
            for owner in owners:
                if owner in self._reserve_h_ids:
                    continue
                progress = self._hm_progress.get(owner)
                target_id = self._assignments.get(owner)
                own = current_hm[owner]
                owner_tracks = self._owner_target_tracks(owner, step)
                legal_targets = tuple(sorted(owner_tracks))
                fresh_targets = tuple(sorted(
                    candidate_id
                    for candidate_id, track in owner_tracks.items()
                    if track.source == "detect_fresh"
                ))
                velocity_x, velocity_y = self._velocity_xy(own)
                kind = int(own["type"])
                if (
                    progress is None
                    or target_id is None
                    or target_id not in legal_targets
                    or progress.position is None
                    or not all(math.isfinite(value) for value in progress.position.values())
                    or progress.current_distance_km is None
                ):
                    if kind == 21001:
                        incomplete_m.append((owner, target_id, progress))
                    continue
                members.append(RetargetMemberSnapshot(
                    owner,
                    kind,
                    target_id,
                    True,
                    progress.stage,
                    progress.current_distance_km,
                    group,
                    kind == 21000,
                    self._retarget_counts.get(owner, 0),
                    self._last_retarget_steps.get(owner),
                    progress.position["lon"],
                    progress.position["lat"],
                    legal_targets,
                    fresh_targets,
                    velocity_x,
                    velocity_y,
                ))
            member_snapshots = tuple(members)
            decision = None
            if self.config.dynamic_retarget_enabled:
                decision = self._dynamic_retarget.decide(
                    step=step, members=member_snapshots, targets=target_snapshots
                )
            for member in member_snapshots:
                if member.missile_type != 21001:
                    continue
                reason = (
                    "dynamic_retarget_disabled"
                    if decision is None
                    else decision.rejected_reasons.get(member.entity_id, "eligible")
                )
                self._record_retarget_status(
                    step, member, target_snapshots, reason
                )
            for owner, target_id, progress in incomplete_m:
                self._record_incomplete_retarget_status(
                    step, owner, target_id, progress, group
                )
            if decision is None:
                continue
            members_by_owner = {
                member.entity_id: member for member in member_snapshots
            }
            changed: dict[int, dict] = {}
            for request in decision.requests:
                owner = request.entity_id
                member = members_by_owner[owner]
                candidate = targets_by_id[request.new_target_id]
                event_id = f"transfer:{owner}:{step}:{request.new_target_id}"
                request_fields = {
                    "owner": owner,
                    "type": member.missile_type,
                    "old_target": request.old_target_id,
                    "new_target": request.new_target_id,
                    "probability": request.target_destruction_probability,
                    "group": member.communication_group,
                    "stage": member.stage,
                    "distance_km": member.current_distance_km,
                    "candidate_position": {
                        "lon": candidate.lon,
                        "lat": candidate.lat,
                    },
                    "candidate_position_source": target_position_sources[
                        request.new_target_id
                    ],
                    "reason": request.reason,
                    "event_id": event_id,
                }
                self._write_audit("retarget_request", step, **request_fields)
                if owner not in owners or self._assignments.get(owner) != request.old_target_id:
                    continue
                goal = {"lon": candidate.lon, "lat": candidate.lat}
                tracker.transfer(owner, request.new_target_id, event_id=event_id)
                self._add_ledger_event(
                    owners,
                    _LedgerEvent(
                        step,
                        "transfer",
                        event_id,
                        (owner, request.new_target_id),
                    ),
                )
                self._assignments[owner] = request.new_target_id
                self._goals[owner] = goal
                self._goal_versions[(owner, request.new_target_id)] = (
                    goal["lon"], goal["lat"]
                )
                self._retarget_counts[owner] = self._retarget_counts.get(owner, 0) + 1
                self._last_retarget_steps[owner] = step
                self._set_target_action(
                    rows, owner, request.new_target_id, priority=True
                )
                self._write_audit("retarget_applied", step, **request_fields)
                changed[owner] = current_hm[owner]
            if changed:
                self._record_hm_progress(step, changed)
                self._record_belief_changes(step, group, tracker)
                self._damage_by_group[group] = TargetDamageProbabilitySkill.from_snapshot(
                    tracker.export()
                )

    @staticmethod
    def _valid_lonlat(position: Mapping[str, float] | None) -> bool:
        return bool(
            position is not None
            and -180.0 <= position["lon"] <= 180.0
            and -90.0 <= position["lat"] <= 90.0
        )

    def _assigned_terminal_goal(
        self, owner: int
    ) -> tuple[int, dict[str, float]] | None:
        target_id = self._assignments.get(owner)
        goal = self._goals.get(owner)
        if target_id is None or not isinstance(goal, Mapping):
            return None
        try:
            raw_lon, raw_lat = goal.get("lon"), goal.get("lat")
            if isinstance(raw_lon, bool) or isinstance(raw_lat, bool):
                return None
            position = {"lon": float(raw_lon), "lat": float(raw_lat)}
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) for value in position.values()):
            return None
        if not self._valid_lonlat(position):
            return None
        if self._goal_versions.get((owner, target_id)) != (
            position["lon"], position["lat"]
        ):
            return None
        return target_id, position

    def _terminal_target_track(
        self, owner: int, step: int, target_id: int
    ) -> _TargetPosition | None:
        max_age = self.config.terminal_guidance.max_track_age
        if target_id in self._public_by_id:
            track = self._owner_detected_public_targets(
                owner,
                step,
                max_age_steps=max_age,
            ).get(target_id)
            if track is None or not self._valid_lonlat(track.position):
                return None
            return track

        own = self._observations[owner].get("self") or {}
        detected = own.get("detectInfo") or {}
        if not isinstance(detected, Mapping):
            return None
        for raw_id, value in detected.items():
            try:
                detected_id = int(raw_id)
                entity_id = int(_field(value, "entity_id", detected_id))
                target_type = int(
                    _field(value, "entity_type", _field(value, "type", -1))
                )
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                detected_id != target_id
                or entity_id != target_id
                or target_type != 9500
            ):
                continue
            position = self._legal_position(value)
            if not self._valid_lonlat(position):
                continue
            raw_time = _field(value, "time")
            if (
                isinstance(raw_time, bool)
                or not isinstance(raw_time, (int, float))
                or not math.isfinite(float(raw_time))
            ):
                raw_time = None
            else:
                raw_time = float(raw_time)
            fingerprint = (
                raw_time,
                round(position["lon"], 12),
                round(position["lat"], 12),
            )
            cache_key = (owner, target_id)
            cached = self._track_freshness.get(cache_key)
            if cached is None or cached.fingerprint != fingerprint:
                cached = _TrackFreshness(fingerprint, int(step))
                self._track_freshness[cache_key] = cached
            age_steps = float(step - cached.last_changed_step)
            return _TargetPosition(
                position,
                "detect_fresh" if age_steps <= max_age else "detect_stale",
                float(cached.last_changed_step),
                age_steps,
                raw_time,
                fingerprint,
            )
        return None

    def _nearest_interceptor_distance_m(
        self, owner_position: dict[str, float], own: Mapping
    ) -> float | None:
        detected = own.get("detectInfo") or {}
        if not isinstance(detected, Mapping):
            return None
        distances = []
        for raw_id, value in detected.items():
            try:
                detected_id = int(raw_id)
                entity_id = int(_field(value, "entity_id", detected_id))
                entity_type = int(
                    _field(value, "entity_type", _field(value, "type", -1))
                )
            except (TypeError, ValueError, OverflowError):
                continue
            if detected_id < 0 or entity_id != detected_id or entity_type != 24000:
                continue
            position = self._legal_position(value)
            if not self._valid_lonlat(position):
                continue
            distances.append(_distance_km(owner_position, position) * 1000.0)
        return None if not distances else min(distances)

    @staticmethod
    def _observed_nonnegative_int(value) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return max(0, result)

    def _record_terminal_guidance(
        self,
        step: int,
        owner: int,
        missile_type: int | None,
        baseline: int,
        final: int,
        *,
        stage: int | None,
        distance_m: float | None,
        track_source: str | None,
        track_age: float | None,
        nearest_interceptor_distance_m: float | None,
        threat: bool,
        latched: bool,
        overridden: bool,
        reason: str,
        velocity_source: str,
        velocity_east: float | None,
        velocity_north: float | None,
        mode: str | None = None,
        heading_error_degrees: float | None = None,
        turn_direction: str | None = None,
        predicted_miss_distance: float | None = None,
        control_state: str = "PASSTHROUGH",
        no_improvement_count: int = 0,
    ) -> None:
        self._motion_guidance[owner] = {
            "step": step,
            "baseline_action": baseline,
            "final_action": final,
            "override_reason": reason,
            "predicted_miss_distance": predicted_miss_distance,
            "control_state": control_state,
            "no_improvement_count": no_improvement_count,
            "heading_error": heading_error_degrees,
            "velocity_source": velocity_source,
        }
        signature = (
            reason,
            latched,
            final,
            control_state,
            no_improvement_count,
        )
        if self._audited_terminal_guidance.get(owner) != signature or overridden:
            details = {
                "owner": owner,
                "type": missile_type,
                "baseline": baseline,
                "final": final,
                "stage": stage,
                "distance_m": distance_m,
                "track_source": track_source,
                "track_age": track_age,
                "nearest_interceptor_distance_m": nearest_interceptor_distance_m,
                "threat": threat,
                "latched": latched,
                "overridden": overridden,
                "reason": reason,
                "velocity_source": velocity_source,
                "velocity_east": velocity_east,
                "velocity_north": velocity_north,
                "predicted_miss_distance": predicted_miss_distance,
                "control_state": control_state,
                "no_improvement_count": no_improvement_count,
                "heading_error": heading_error_degrees,
            }
            if mode is not None:
                details["mode"] = mode
            if heading_error_degrees is not None:
                details["heading_error_degrees"] = heading_error_degrees
            if turn_direction is not None:
                details["turn_direction"] = turn_direction
            self._write_audit(
                "terminal_guidance", step,
                **details,
            )
        self._audited_terminal_guidance[owner] = signature

    def _terminal_passthrough(
        self,
        step: int,
        owner: int,
        missile_type: int | None,
        baseline: int,
        reason: str,
        *,
        stage: int | None = None,
        distance_m: float | None = None,
        track_source: str | None = None,
        track_age: float | None = None,
        nearest_interceptor_distance_m: float | None = None,
        threat: bool = False,
        velocity_source: str = "none",
        velocity_east: float | None = None,
        velocity_north: float | None = None,
    ) -> int:
        self._terminal_guidance.reset(owner)
        self._record_terminal_guidance(
            step,
            owner,
            missile_type,
            baseline,
            baseline,
            stage=stage,
            distance_m=distance_m,
            track_source=track_source,
            track_age=track_age,
            nearest_interceptor_distance_m=nearest_interceptor_distance_m,
            threat=threat,
            latched=False,
            overridden=False,
            reason=reason,
            velocity_source=velocity_source,
            velocity_east=velocity_east,
            velocity_north=velocity_north,
        )
        return baseline

    def _apply_terminal_guidance(
        self, step: int, owner: int, own: Mapping, baseline: int
    ) -> int:
        missile_type = self._observed_nonnegative_int(own.get("type"))
        stage = self._observed_nonnegative_int(own.get("stage"))
        owner_position = self._legal_position(own)
        if not self._valid_lonlat(owner_position):
            owner_position = None
        velocity_source, velocity_east, velocity_north = (
            self._owner_local_velocity(owner, own, owner_position)
        )
        velocity_fields = {
            "velocity_source": velocity_source,
            "velocity_east": velocity_east,
            "velocity_north": velocity_north,
        }
        if not self.config.terminal_guidance_enabled:
            return self._terminal_passthrough(
                step,
                owner,
                missile_type,
                baseline,
                "terminal_guidance_disabled",
                stage=stage,
                **velocity_fields,
            )
        if missile_type is None or stage is None:
            return self._terminal_passthrough(
                step,
                owner,
                missile_type,
                baseline,
                "invalid_owner_metadata",
                stage=stage,
                **velocity_fields,
            )
        if missile_type != 21001:
            return self._terminal_passthrough(
                step,
                owner,
                missile_type,
                baseline,
                "missile_type_passthrough",
                stage=stage,
                **velocity_fields,
            )
        if stage < self.config.terminal_guidance.min_stage:
            return self._terminal_passthrough(
                step,
                owner,
                missile_type,
                baseline,
                "stage_below_minimum",
                stage=stage,
                **velocity_fields,
            )

        assigned = self._assigned_terminal_goal(owner)
        if assigned is None:
            return self._terminal_passthrough(
                step,
                owner,
                missile_type,
                baseline,
                "missing_assigned_goal",
                stage=stage,
                **velocity_fields,
            )
        target_id, goal = assigned
        if owner_position is None:
            return self._terminal_passthrough(
                step,
                owner,
                missile_type,
                baseline,
                "missing_owner_position",
                stage=stage,
                **velocity_fields,
            )

        distance_m = _distance_km(owner_position, goal) * 1000.0
        nearest_interceptor = self._nearest_interceptor_distance_m(
            owner_position, own
        )
        threat = bool(
            nearest_interceptor is not None
            and nearest_interceptor
            <= self.config.terminal_guidance.threat_distance_m
        )
        target_track = self._terminal_target_track(owner, step, target_id)
        if target_track is None or target_track.age_steps is None:
            return self._terminal_passthrough(
                step,
                owner,
                missile_type,
                baseline,
                "missing_owner_target_fix",
                stage=stage,
                distance_m=distance_m,
                nearest_interceptor_distance_m=nearest_interceptor,
                threat=threat,
                **velocity_fields,
            )

        if velocity_east is None or velocity_north is None:
            return self._terminal_passthrough(
                step,
                owner,
                missile_type,
                baseline,
                "missing_owner_velocity",
                stage=stage,
                distance_m=distance_m,
                nearest_interceptor_distance_m=nearest_interceptor,
                threat=threat,
                **velocity_fields,
            )

        snapshot = TerminalGuidanceSnapshot(
            entity_id=owner,
            missile_type=missile_type,
            stage=stage,
            baseline_action=baseline,
            aimpoint_lon=goal["lon"],
            aimpoint_lat=goal["lat"],
            track_source=target_track.source,
            track_age=target_track.age_steps,
            nearest_interceptor_distance_m=nearest_interceptor,
            step=step,
            position_lon=owner_position["lon"],
            position_lat=owner_position["lat"],
            velocity_east_mps=velocity_east,
            velocity_north_mps=velocity_north,
            velocity_source=velocity_source,
        )
        decision = self._terminal_guidance.decide(snapshot)
        self._record_terminal_guidance(
            step,
            owner,
            missile_type,
            decision.baseline,
            decision.final,
            stage=stage,
            distance_m=decision.distance,
            track_source=target_track.source,
            track_age=target_track.age_steps,
            nearest_interceptor_distance_m=nearest_interceptor,
            threat=decision.threat,
            latched=decision.latched,
            overridden=decision.overridden,
            reason=decision.reason,
            velocity_source=decision.velocity_source,
            velocity_east=velocity_east,
            velocity_north=velocity_north,
            mode=decision.mode,
            heading_error_degrees=decision.heading_error,
            turn_direction=decision.turn_direction,
            predicted_miss_distance=decision.predicted_miss_distance,
            control_state=decision.control_state,
            no_improvement_count=decision.no_improvement_count,
        )
        return decision.final

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

    def _reserve_package_for_owner(
        self, owner: int
    ) -> ReservePackageSnapshot | None:
        return next(
            (
                package
                for package in self._reserve_packages.values()
                if owner in package.member_ids
            ),
            None,
        )

    def _apply_reserve_holding(
        self, step: int, owner: int, own: Mapping, baseline: int
    ) -> int:
        package = self._reserve_package_for_owner(owner)
        if (
            package is None
            or package.member_types[0] != 21000
            or package.state != "STAGING"
            or self._assignments.get(owner) is not None
        ):
            if owner in self._reserve_h_ids:
                self._reserve_holding.reset(owner)
                self._reserve_holding_anchors.pop(owner, None)
            return baseline

        position = self._legal_position(own)
        if not self._valid_lonlat(position):
            self._write_audit(
                "reserve_holding",
                step,
                package=package.package_id,
                owner=owner,
                phase=None,
                cross_track=None,
                progress=None,
                baseline=baseline,
                final=baseline,
                overridden=False,
                reason="missing_owner_position",
            )
            return baseline
        assert position is not None
        anchor = self._reserve_holding_anchors.setdefault(
            owner, (position["lon"], position["lat"])
        )
        forward_target = None
        forward_target_id = None
        for target_id in (
            package.provisional_target_id,
            package.fallback_target_id,
        ):
            target = self._public_by_id.get(target_id)
            if target is None:
                continue
            candidate = (target.lon, target.lat)
            if _distance_km(
                {"lon": anchor[0], "lat": anchor[1]},
                {"lon": candidate[0], "lat": candidate[1]},
            ) > 1e-9:
                forward_target = candidate
                forward_target_id = target_id
                break
        velocity_source, velocity_east, velocity_north = self._owner_local_velocity(
            owner, own, position
        )
        if (
            forward_target is None
            or velocity_east is None
            or velocity_north is None
            or math.hypot(velocity_east, velocity_north) <= 1e-12
        ):
            self._write_audit(
                "reserve_holding",
                step,
                package=package.package_id,
                owner=owner,
                phase=None,
                cross_track=None,
                progress=None,
                baseline=baseline,
                final=baseline,
                overridden=False,
                reason=(
                    "missing_forward_target"
                    if forward_target is None
                    else "missing_owner_velocity"
                ),
                velocity_source=velocity_source,
                anchor=anchor,
                forward_target=forward_target,
                forward_target_id=forward_target_id,
            )
            return baseline
        decision = self._reserve_holding.decide(HoldingSnapshot(
            entity_id=owner,
            type=21000,
            step=step,
            position=(position["lon"], position["lat"]),
            velocity=(velocity_east, velocity_north),
            anchor=anchor,
            forward_target=forward_target,
            baseline_action=baseline,
            state=package.state,
        ))
        self._write_audit(
            "reserve_holding",
            step,
            package=package.package_id,
            owner=owner,
            phase=decision.phase,
            cross_track=decision.cross_track,
            progress=decision.forward_progress,
            baseline=baseline,
            final=decision.action,
            overridden=decision.overridden,
            reason=decision.reason,
            velocity_source=velocity_source,
            anchor=anchor,
            forward_target=forward_target,
            forward_target_id=forward_target_id,
        )
        return decision.action

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
            held = self._apply_reserve_holding(
                step, owner, observation["self"], selected
            )
            final = self._apply_terminal_guidance(step, owner, observation["self"], held)
            maneuver = float(final - 1)
            self._actions[owner] = ((0, owner, maneuver, 0), *self._actions.get(owner, ()))

    def reset_member(self, entity_id: int) -> None:
        entity_id = int(entity_id)
        if entity_id in self._reset_members:
            return
        if not self._reset_members:
            reset_step = self._planned_step if self._planned_step is not None else 0
            self._write_audit("reset", reset_step, owner=entity_id)
            self._audited_beliefs.clear()
            self._audited_retarget_status.clear()
            self._audited_formation_steps.clear()
            self._audited_missile_trace_steps.clear()
            self._audited_terminal_guidance.clear()
            self._missile_trace_history.clear()
            self._observations.clear()
            self._actions.clear()
            self._goals.clear()
            self._assignments.clear()
            self._goal_versions.clear()
            self._hm_progress.clear()
            self._hm_types.clear()
            self._core_hm_assignments.clear()
            self._reserve_h_ids = frozenset()
            self._reserve_packages.clear()
            self._hm_target_regions.clear()
            self._hm_early_losses.clear()
            self._reserve_retarget_counts.clear()
            self._reserve_member_snapshots.clear()
            self._reserve_seen_members.clear()
            self._reserve_package_terminations.clear()
            self._reserve_holding_anchors.clear()
            self._reserve_reachability_origins.clear()
            self._audited_reserve_deficits.clear()
            self._l_core_ids = None
            self._l_core_assignments.clear()
            self._l_progress.clear()
            self._l_early_losses.clear()
            self._l_catalogues = {owner: {} for owner in self.members}
            self._l_target_count = None
            self._last_groups.clear()
            self._terminalized.clear()
            self._owner_ledgers = {owner: {} for owner in self.members}
            self._damage_by_group.clear()
            self._retarget_counts.clear()
            self._last_retarget_steps.clear()
            self._track_freshness.clear()
            self._initial_attack_sent.clear()
            self._motion_guidance.clear()
            self._owner_velocity_history = {
                owner: self._new_owner_velocity_state() for owner in self.members
            }
            self._planned_step = None
            self._satellite_committed = False
            self._terminal_guidance.reset()
            self._reserve_holding.reset()
            self._restore_mission_plan_state()
            reset = getattr(self.policy, "reset_episode", None)
            if callable(reset):
                reset()
        self._reset_members.add(entity_id)
