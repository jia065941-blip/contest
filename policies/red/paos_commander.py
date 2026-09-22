"""Atomic R11-PAOS commander and audit boundary.

This module is intentionally separate from :mod:`policies.red.commander` so
the R0--R10 execution semantics remain byte-for-byte unchanged.  PAOS learns
only an eight-dimensional quota allocator; this adapter owns all platform
lifecycle, scheduling, and audit contracts around that pure policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .baselines import PlatformState, TacticalMetrics, TargetPrior, distance_km, engagement_effectiveness
from .contracts import Position
from .tracks import InitialCatalogueTrackFusion


_KIND_BY_TYPE = {21000: "H", 21001: "M", 21002: "L"}
_KINDS = ("H", "M", "L")
_KIND_INDEX = {kind: index for index, kind in enumerate(_KINDS)}
_SEED_ENV = {
    "blue": "BLUE_POLICY_SEED",
    "red": "RED_POLICY_SEED",
    "simulation": "SIMULATION_SEED",
}


def validate_paos_process_rounds(policy_name: str, total_rounds: int) -> None:
    """Enforce the one-fresh-process-per-branch PAOS contract."""

    if policy_name == "r11_paos" and int(total_rounds) != 1:
        raise ValueError("r11_paos requires --total-rounds 1 (one fresh process per branch)")


class PAOSContractError(RuntimeError):
    """A fatal R11 execution-contract violation."""


class Lifecycle(str, Enum):
    FREE = "free"
    ACTIVE_PENDING = "active_pending"
    ISSUED = "issued"
    UNAVAILABLE_UNISSUED = "unavailable_unissued"


@dataclass(frozen=True)
class LedgerRecord:
    platform_id: int
    kind: str
    state: Lifecycle = Lifecycle.FREE
    version: int = 0
    plan_id: int | None = None
    target_id: int | None = None
    target_ordinal: int | None = None
    due_step: int | None = None
    leader: bool = False
    issued_step: int | None = None
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class PlannedAssignment:
    platform_id: int
    prior_version: int
    target_id: int
    target_ordinal: int
    due_step: int
    leader: bool = False


@dataclass(frozen=True)
class QuotaPlan:
    plan_id: int
    event_class: str
    step: int
    snapshot_hash: str
    catalogue_ordinals: tuple[int, ...]
    assignments: tuple[PlannedAssignment, ...]
    reserve_counts: tuple[int, int, int]


@dataclass(frozen=True)
class SealPreview:
    plan_id: int
    event_class: str
    vector: tuple[int, ...]
    vector_hash: str
    decision_pool_counts: tuple[int, int, int]
    kind_slices: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_hash(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _qfloat(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise PAOSContractError(f"non-finite public float: {number!r}")
    return f"{number:.9f}"


def _vector_hash(vector: Sequence[int]) -> str:
    from .learning.paos_policy import canonical_vector_hash

    return canonical_vector_hash(vector)


def _theta_hash(theta: Sequence[float]) -> str:
    """Fallback canonical theta digest shared by probe and rollout audits."""

    return _json_hash({"schema_version": "r11_paos_theta_v1", "theta": [_qfloat(v) for v in theta]})


class PAOSCommander:
    """Legal event-driven quota executor for a frozen 90-D motion policy."""

    REQUEST_SCHEMA = "r11_paos_request_v1"
    PROBE_SCHEMA = "r11_paos_probe_v1"
    AUDIT_SCHEMA = "r11_paos_rollout_audit_v1"
    EVENT_CLASSES = ("initial", "new_objective", "time_045", "time_070")
    fail_fast_agent_errors = True

    def __init__(
        self,
        targets: tuple[TargetPrior, ...],
        *,
        top_model: str,
        max_steps: int,
        seed: int = 1,
        mode: str = "rollout",
        request_path: str | None = None,
        bottom_model: str | None = None,
        observation_dim: int = 90,
    ) -> None:
        if mode not in {"probe", "rollout"}:
            raise ValueError(f"Unsupported PAOS mode {mode!r}")
        if int(observation_dim) != 90:
            raise ValueError("R11-PAOS requires the frozen 90-D hierarchical interface")

        from .learning.paos_policy import PAOSConfig, PolicySnapshot, TargetCandidate, load_theta_checkpoint, project_quotas

        self._PAOSConfig = PAOSConfig
        self._PolicySnapshot = PolicySnapshot
        self._TargetCandidate = TargetCandidate
        self._project_quotas = project_quotas
        self.config = PAOSConfig()
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.mode = mode
        self.request_path = request_path
        self.actual_seed_tuple: dict[str, int] | None = None
        self.request = self._load_request(request_path)

        checkpoint_bounds = "center" if mode == "probe" or self.request.get("evaluation_stage") == "final" else "candidate"
        self.checkpoint = load_theta_checkpoint(top_model, config=self.config, bounds=checkpoint_bounds)
        self.theta = tuple(float(value) for value in self.checkpoint.theta)
        self.top_model = str(top_model)
        self.bottom_model = str(bottom_model) if bottom_model else None
        self.top_file_sha256 = _file_sha256(top_model)
        self.bottom_sha256 = _file_sha256(bottom_model)
        self.candidate_sha256 = str(self.checkpoint.theta_sha256)
        self._validate_request_static()

        self.initial_targets = tuple(targets)
        self.initial_target_ids = frozenset(target.entity_id for target in targets)
        self.targets = tuple(targets)
        self.track_fusion = InitialCatalogueTrackFusion(targets)
        self._ordinal_by_target = {
            target.entity_id: ordinal for ordinal, target in enumerate(targets)
        }
        self._target_by_ordinal = {
            ordinal: target.entity_id for ordinal, target in enumerate(targets)
        }
        self._next_ordinal = len(targets)
        self._membership_revision = 0
        self._planned_membership_revision = 0

        self.expected_platform_ids: set[int] = set()
        self.reports: dict[int, PlatformState] = {}
        self.ledger: dict[int, LedgerRecord] = {}
        self.initial_inventory: tuple[int, int, int] | None = None
        self.plan_version = 0
        self.event_latches = {
            "initial": False,
            "time_045": False,
            "time_070": False,
        }
        self.tactical_metrics = TacticalMetrics()
        self.first_discovery_step: dict[int, int] = {}
        self.first_assignment_step: dict[int, int] = {}
        self.official_return = 0.0

        self._fatal_errors: list[str] = []
        self._contract_violations: dict[str, int] = {
            "request": 0,
            "plan_validation": 0,
            "stale_callback": 0,
            "ledger_conservation": 0,
            "preview_seal_mismatch": 0,
            "leader": 0,
            "seed_identity": 0,
            "candidate_identity": 0,
        }
        self._events: list[dict[str, Any]] = []
        self._event_snapshots: list[dict[str, Any]] = []
        self._same_state_9500: list[dict[str, Any]] = []
        self._initial_legal_state_hash: str | None = None
        self._first_preview: SealPreview | None = None
        self._first_seal: SealPreview | None = None
        self._proposal_count = 0
        self._preview_count = 0
        self._seal_count = 0
        self._issued_count = 0

    def _load_request(self, path: str | None) -> dict[str, Any]:
        if path is None:
            return {}
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("PAOS request must be a JSON object")
        return value

    def _validate_request_static(self) -> None:
        if not self.request:
            return
        if self.request.get("schema_version") != self.REQUEST_SCHEMA:
            raise ValueError("Unsupported PAOS request schema")
        kind = self.request.get("kind")
        allowed = {"pair_probe", "bank_projection"} if self.mode == "probe" else {"rollout"}
        if kind not in allowed:
            raise ValueError(f"PAOS mode {self.mode!r} cannot serve request kind {kind!r}")
        frozen_sequences = {
            "radius_grid": (0.25, 0.5, 1.0, 2.0, 4.0),
            "candidate_box": (-4.0, 4.0),
            "preview_tv_range": (0.05, 0.20),
        }
        for name, expected_value in frozen_sequences.items():
            raw_value = self.request.get(name)
            if not isinstance(raw_value, list) or tuple(float(value) for value in raw_value) != expected_value:
                raise ValueError(f"PAOS request {name} differs from the frozen contract")
        if self.request.get("candidate_file_sha256") != self.top_file_sha256:
            raise ValueError("PAOS request candidate_file_sha256 does not match loaded top checkpoint")

        raw_seeds = self.request.get("seed_tuple")
        if not isinstance(raw_seeds, Mapping) or set(raw_seeds) != set(_SEED_ENV):
            raise ValueError("seed_tuple must contain exactly blue, red, simulation")
        requested_seeds = {name: int(raw_seeds[name]) for name in _SEED_ENV}
        missing = [environment for environment in _SEED_ENV.values() if os.getenv(environment) is None]
        if missing:
            raise ValueError(f"PAOS runtime seed environment is incomplete: {missing}")
        actual_seeds = {
            name: int(os.environ[environment]) for name, environment in _SEED_ENV.items()
        }
        if requested_seeds != actual_seeds:
            raise ValueError(
                f"PAOS request seed_tuple {requested_seeds} != runtime seeds {actual_seeds}"
            )
        self.actual_seed_tuple = actual_seeds

        if kind == "rollout" and self.request.get("candidate_sha256") != self.candidate_sha256:
            raise ValueError("PAOS request candidate_sha256 does not match loaded top checkpoint")
        if kind == "pair_probe" and self.request.get("center_sha256") != self.candidate_sha256:
            raise ValueError("PAOS probe center_sha256 does not match loaded top checkpoint")
        if kind == "pair_probe" and any(
            self.request.get(name) is not None for name in ("candidate_sha256", "sign", "radius")
        ):
            raise ValueError("PAOS pair probe candidate/sign/radius fields must be null")
        direction = self.request.get("direction_vector")
        if direction is not None and (
            not isinstance(direction, list)
            or len(direction) != 8
            or not all(math.isfinite(float(value)) for value in direction)
        ):
            raise ValueError("direction_vector must contain eight finite values")

        if kind == "rollout" and self.request.get("evaluation_stage") == "final":
            null_fields = ("cycle", "direction", "sign", "radius", "direction_vector")
            if any(self.request.get(name) is not None for name in null_fields):
                raise ValueError("PAOS final rollout must not carry training direction/sign/radius")
            if self.request.get("center_sha256") != self.candidate_sha256:
                raise ValueError("PAOS final center_sha256 does not match frozen q32")
            expected = self.request.get("expected")
            expected_keys = {"legal_state_hash", "preview_vector", "preview_hash"}
            if not isinstance(expected, Mapping) or set(expected) != expected_keys:
                raise ValueError("PAOS final expected object has the wrong schema")
            if any(value is not None for value in expected.values()):
                raise ValueError("PAOS final expected fields must all be null")
        elif kind == "rollout":
            if self.request.get("sign") not in {"plus", "minus"} or self.request.get("radius") is None:
                raise ValueError("PAOS training rollout requires sign and radius")
            if self.request.get("cycle") is None or self.request.get("direction") is None:
                raise ValueError("PAOS training rollout requires cycle and direction")

    def _policy_theta_hash(self, theta: Sequence[float]) -> str:
        try:
            from .learning.paos_policy import theta_sha256
        except ImportError:
            return _theta_hash(theta)
        return str(theta_sha256(theta))

    def register_platform(self, entity_id: int) -> None:
        self.expected_platform_ids.add(int(entity_id))

    @staticmethod
    def _track_field(track: Any, name: str, default: Any) -> Any:
        return track.get(name, default) if isinstance(track, dict) else getattr(track, name, default)

    def report(self, observation: dict) -> None:
        own = observation.get("self", {})
        entity_id = int(observation.get("entity_id", -1))
        kind = _KIND_BY_TYPE.get(int(own.get("type", 0)))
        position = own.get("position", {})
        if entity_id < 0 or kind is None or not isinstance(position, dict):
            return
        platform = PlatformState(
            entity_id=entity_id,
            kind=kind,
            position=Position(
                lon=float(position.get("lon", 0.0)),
                lat=float(position.get("lat", 0.0)),
                alt=float(position.get("alt", 0.0)),
            ),
            alive=float(own.get("health", 0.0)) > 0.0 and bool(own.get("isVisible", True)),
            launched=self.ledger.get(entity_id, LedgerRecord(entity_id, kind)).state is Lifecycle.ISSUED,
        )
        previous = self.ledger.get(entity_id)
        if previous is None:
            self.ledger[entity_id] = LedgerRecord(entity_id, kind)
        elif previous.kind != kind:
            self._latch("plan_validation", f"platform {entity_id} changed kind {previous.kind}->{kind}")
        elif previous.state is Lifecycle.UNAVAILABLE_UNISSUED and platform.alive:
            self._latch("plan_validation", f"unavailable platform {entity_id} reappeared")
        self.reports[entity_id] = platform
        if not platform.alive:
            self._mark_unavailable_before_issue(entity_id)

    def _mark_unavailable_before_issue(self, entity_id: int) -> None:
        record = self.ledger.get(int(entity_id))
        if record is None or record.state not in {Lifecycle.FREE, Lifecycle.ACTIVE_PENDING}:
            return
        self.ledger[int(entity_id)] = replace(
            record,
            state=Lifecycle.UNAVAILABLE_UNISSUED,
            version=record.version + 1,
            plan_id=None,
            target_id=None,
            target_ordinal=None,
            due_step=None,
            leader=False,
            unavailable_reason="own_death_before_issue",
        )

    def _ingest_catalogue(self, observation: dict, step: int) -> bool:
        before = {target.entity_id for target in self.track_fusion.targets}
        self.track_fusion.ingest(observation)
        after_targets = {target.entity_id: target for target in self.track_fusion.targets}
        new_ids = set(after_targets) - before
        if not new_ids:
            self.targets = self.track_fusion.targets
            return False

        observed_order: list[int] = []
        for raw_id, track in (observation.get("self", {}).get("detectInfo") or {}).items():
            entity_id = int(self._track_field(track, "entity_id", raw_id))
            if entity_id in new_ids and entity_id not in observed_order:
                observed_order.append(entity_id)
        observed_order.extend(sorted(new_ids - set(observed_order)))
        for entity_id in observed_order:
            self._ordinal_by_target[entity_id] = self._next_ordinal
            self._target_by_ordinal[self._next_ordinal] = entity_id
            self._next_ordinal += 1
            self._membership_revision += 1
            self.first_discovery_step[entity_id] = int(step)
        self.targets = self.track_fusion.targets
        return True

    def _synchronize(self, observations: tuple[dict, ...]) -> bool:
        observed_ids: set[int] = set()
        membership_changed = False
        step = int(observations[0].get("step", 0)) if observations else 0
        for observation in observations:
            entity_id = int(observation.get("entity_id", -1))
            if entity_id >= 0:
                observed_ids.add(entity_id)
            self.report(observation)
            membership_changed = self._ingest_catalogue(observation, step) or membership_changed

        for entity_id in sorted(self.expected_platform_ids - observed_ids):
            record = self.ledger.get(entity_id)
            prior = self.reports.get(entity_id)
            if record is None or prior is None or not prior.alive:
                continue
            self.reports[entity_id] = PlatformState(
                entity_id=prior.entity_id,
                kind=prior.kind,
                position=prior.position,
                alive=False,
                launched=record.state is Lifecycle.ISSUED,
            )
            self._mark_unavailable_before_issue(entity_id)

        if self._ready() and self.initial_inventory is None:
            self.initial_inventory = tuple(
                sum(record.kind == kind for record in self.ledger.values())
                for kind in _KINDS
            )
        self._refresh_metrics()
        self._check_conservation()
        return membership_changed

    def _ready(self) -> bool:
        return bool(self.targets) and bool(self.expected_platform_ids) and self.expected_platform_ids <= self.ledger.keys()

    def _refresh_metrics(self) -> None:
        assigned: dict[int, int] = {}
        for record in self.ledger.values():
            if record.state in {Lifecycle.ACTIVE_PENDING, Lifecycle.ISSUED} and record.target_id is not None:
                assigned[record.target_id] = assigned.get(record.target_id, 0) + 1
        alive_count = sum(report.alive for report in self.reports.values())
        issued = sum(record.state is Lifecycle.ISSUED for record in self.ledger.values())
        self.tactical_metrics = TacticalMetrics(
            launched_count=issued,
            alive_count=alive_count,
            lost_count=max(0, len(self.expected_platform_ids) - alive_count),
            total_count=len(self.expected_platform_ids),
            event_revision=self._membership_revision,
            assigned_target_counts=tuple(sorted(assigned.items())),
        )

    def begin_step(self, observations: tuple[dict, ...]) -> None:
        membership_changed = self._synchronize(observations)
        if self.mode != "rollout" or not self._ready() or self._fatal_errors:
            return
        step = int(observations[0].get("step", 0)) if observations else 0
        events: list[str] = []
        if not self.event_latches["initial"]:
            events.append("initial")
        if membership_changed and self._membership_revision > self._planned_membership_revision:
            events.append("new_objective")
        if step >= math.ceil(0.45 * self.max_steps) and not self.event_latches["time_045"]:
            events.append("time_045")
        if step >= math.ceil(0.70 * self.max_steps) and not self.event_latches["time_070"]:
            events.append("time_070")
        for event_class in events:
            if self._fatal_errors:
                break
            self._handle_plan_event(event_class, step)

    def reconcile_terminal(self, observations: tuple[dict, ...]) -> None:
        """Post-physics own-side reconciliation for the terminal frame."""

        self._synchronize(observations)
        self._check_conservation()

    def _decision_pool(self, step: int, kind: str) -> tuple[LedgerRecord, ...]:
        return tuple(
            sorted(
                (
                    record for record in self.ledger.values()
                    if record.kind == kind
                    and self.reports.get(
                        record.platform_id,
                        PlatformState(record.platform_id, kind, Position(0, 0), False),
                    ).alive
                    and (
                        record.state is Lifecycle.FREE
                        or (
                            record.state is Lifecycle.ACTIVE_PENDING
                            and record.due_step is not None
                            and record.due_step > step
                        )
                    )
                ),
                key=lambda record: record.platform_id,
            )
        )

    def _make_policy_snapshot(self, step: int):
        if self.initial_inventory is None:
            raise PAOSContractError("initial inventory is not frozen")
        pools = {kind: self._decision_pool(step, kind) for kind in _KINDS}
        own_positions = [
            self.reports[record.platform_id].position
            for kind in _KINDS for record in pools[kind]
            if record.platform_id in self.reports
        ]
        if own_positions:
            origin = Position(
                sum(position.lon for position in own_positions) / len(own_positions),
                sum(position.lat for position in own_positions) / len(own_positions),
                sum(position.alt for position in own_positions) / len(own_positions),
            )
        else:
            origin = Position(0.0, 0.0, 0.0)
        candidates = []
        target_by_id = {target.entity_id: target for target in self.targets}
        for ordinal in sorted(self._target_by_ordinal):
            target = target_by_id[self._target_by_ordinal[ordinal]]
            committed = tuple(
                sum(
                    record.kind == kind
                    and record.target_ordinal == ordinal
                    and record.state in {Lifecycle.ACTIVE_PENDING, Lifecycle.ISSUED}
                    for record in self.ledger.values()
                )
                for kind in _KINDS
            )
            candidates.append(
                self._TargetCandidate(
                    admission_ordinal=int(ordinal),
                    entity_type=int(target.entity_type),
                    distance=float(distance_km(origin, target.position)),
                    committed_counts=committed,
                )
            )
        return self._PolicySnapshot(
            time_fraction=min(1.0, max(0.0, step / max(1, self.max_steps))),
            initial_inventory=self.initial_inventory,
            decision_pool_counts=tuple(len(pools[kind]) for kind in _KINDS),
            targets=tuple(candidates),
        )

    def _legal_state_snapshot(self, step: int):
        from .learning.paos_policy import (
            LegalPlatformState,
            LegalStateSnapshot,
            LegalTargetState,
            PlanLatch,
        )

        target_by_id = {target.entity_id: target for target in self.targets}
        targets = tuple(
            LegalTargetState(
                admission_ordinal=int(ordinal),
                entity_type=int(target_by_id[self._target_by_ordinal[ordinal]].entity_type),
                position=(
                    float(target_by_id[self._target_by_ordinal[ordinal]].position.lon),
                    float(target_by_id[self._target_by_ordinal[ordinal]].position.lat),
                    float(target_by_id[self._target_by_ordinal[ordinal]].position.alt),
                ),
            )
            for ordinal in sorted(self._target_by_ordinal)
        )
        platforms = []
        for entity_id, record in sorted(self.ledger.items()):
            report = self.reports.get(entity_id)
            if report is None:
                raise PAOSContractError(f"platform {entity_id} has no public report")
            platforms.append(
                LegalPlatformState(
                    platform_id=int(entity_id),
                    kind=record.kind,
                    position=(
                        float(report.position.lon),
                        float(report.position.lat),
                        float(report.position.alt),
                    ),
                    lifecycle=record.state.value,
                    plan_id=record.plan_id,
                    target_ordinal=record.target_ordinal,
                    due_step=record.due_step,
                    leader=record.leader,
                    reason=record.unavailable_reason,
                    version=record.version,
                )
            )
        latches = [
            PlanLatch(name, bool(value))
            for name, value in sorted(self.event_latches.items())
        ]
        latches.append(PlanLatch(
            "new_objective_pending",
            self._membership_revision > self._planned_membership_revision,
        ))
        return LegalStateSnapshot(
            step=int(step),
            # E01 uses a one-second simulator tick, so the public step is time.
            time_seconds=float(step),
            targets=targets,
            platforms=tuple(platforms),
            plan_latches=tuple(latches),
        )

    def _state_payload(self, step: int) -> dict[str, Any]:
        from .learning.paos_policy import canonical_legal_state_payload

        return canonical_legal_state_payload(self._legal_state_snapshot(step))

    def legal_state_hash(self, step: int) -> str:
        from .learning.paos_policy import canonical_legal_state_hash

        return canonical_legal_state_hash(self._legal_state_snapshot(step))

    @staticmethod
    def _kind_quota(projected: Any, kind: str) -> Any:
        kinds = projected.kinds
        if isinstance(kinds, Mapping):
            return kinds[kind]
        return kinds[_KIND_INDEX[kind]]

    def _build_plan(self, event_class: str, step: int, state_hash: str, projected: Any) -> QuotaPlan:
        ordinals = tuple(int(value) for value in projected.catalogue_ordinals)
        assignments: list[PlannedAssignment] = []
        reserve_counts: list[int] = []
        for kind in _KINDS:
            pool = list(self._decision_pool(step, kind))
            quota = self._kind_quota(projected, kind)
            target_counts = tuple(int(value) for value in quota.target_counts)
            reserve = int(quota.reserve_count)
            reserve_counts.append(reserve)
            cursor = 0
            cap = max(1, math.floor(0.25 * self.initial_inventory[_KIND_INDEX[kind]]))
            for target_ordinal, count in zip(ordinals, target_counts):
                target_id = self._target_by_ordinal.get(target_ordinal)
                if target_id is None:
                    raise PAOSContractError(f"unknown target ordinal {target_ordinal}")
                for _ in range(count):
                    if cursor >= len(pool):
                        raise PAOSContractError(f"{kind} projected quota exceeds decision pool")
                    record = pool[cursor]
                    slot = cursor // cap
                    due_step = step + slot * 40
                    assignments.append(PlannedAssignment(
                        platform_id=record.platform_id,
                        prior_version=record.version,
                        target_id=target_id,
                        target_ordinal=target_ordinal,
                        due_step=due_step,
                    ))
                    cursor += 1
            if cursor + reserve != len(pool):
                raise PAOSContractError(
                    f"{kind} quota conservation failed: assigned={cursor} reserve={reserve} pool={len(pool)}"
                )

        # Exactly one persistent leader per H-containing (plan,target,slot).
        leader_keys: set[tuple[int, int]] = set()
        with_leaders: list[PlannedAssignment] = []
        for assignment in assignments:
            record = self.ledger[assignment.platform_id]
            slot = (assignment.due_step - step) // 40
            key = (assignment.target_ordinal, slot)
            leader = record.kind == "H" and key not in leader_keys
            if leader:
                leader_keys.add(key)
            with_leaders.append(replace(assignment, leader=leader))
        return QuotaPlan(
            plan_id=self.plan_version + 1,
            event_class=event_class,
            step=step,
            snapshot_hash=state_hash,
            catalogue_ordinals=ordinals,
            assignments=tuple(with_leaders),
            reserve_counts=tuple(reserve_counts),
        )

    def preview_seal(self, plan: QuotaPlan) -> SealPreview:
        """Pure whole-plan validation; no ledger, counter, or latch mutation."""

        if plan.event_class not in self.EVENT_CLASSES:
            raise PAOSContractError(f"unknown event class {plan.event_class!r}")
        if plan.plan_id != self.plan_version + 1:
            raise PAOSContractError(f"stale plan id {plan.plan_id}; expected {self.plan_version + 1}")
        if plan.snapshot_hash != self.legal_state_hash(plan.step):
            raise PAOSContractError("plan snapshot hash is stale")
        if plan.catalogue_ordinals != tuple(sorted(self._target_by_ordinal)):
            raise PAOSContractError("plan catalogue ordinals do not match legal membership")
        if len(plan.reserve_counts) != 3 or any(value < 0 for value in plan.reserve_counts):
            raise PAOSContractError("invalid reserve counts")

        seen: set[int] = set()
        target_index = {ordinal: index for index, ordinal in enumerate(plan.catalogue_ordinals)}
        rows = [[0] * len(plan.catalogue_ordinals) for _ in _KINDS]
        leader_groups: dict[tuple[int, int], int] = {}
        pools = {kind: {record.platform_id: record for record in self._decision_pool(plan.step, kind)} for kind in _KINDS}
        for assignment in plan.assignments:
            if assignment.platform_id in seen:
                raise PAOSContractError(f"duplicate platform {assignment.platform_id} in plan")
            seen.add(assignment.platform_id)
            record = self.ledger.get(assignment.platform_id)
            if record is None or assignment.platform_id not in pools.get(record.kind, {}):
                raise PAOSContractError(f"platform {assignment.platform_id} is not cancellable/free")
            if record.version != assignment.prior_version:
                raise PAOSContractError(f"platform {assignment.platform_id} version is stale")
            if assignment.target_ordinal not in target_index:
                raise PAOSContractError(f"target ordinal {assignment.target_ordinal} is unavailable")
            if self._target_by_ordinal[assignment.target_ordinal] != assignment.target_id:
                raise PAOSContractError("target ID/ordinal mapping mismatch")
            target = next(target for target in self.targets if target.entity_id == assignment.target_id)
            if engagement_effectiveness(record.kind, target.entity_type) <= 0.0:
                raise PAOSContractError(f"incompatible {record.kind}->{target.entity_type} assignment")
            if assignment.due_step < plan.step or (assignment.due_step - plan.step) % 40:
                raise PAOSContractError("assignment does not use the frozen 40-step schedule")
            rows[_KIND_INDEX[record.kind]][target_index[assignment.target_ordinal]] += 1
            if record.kind == "H":
                group = (assignment.target_ordinal, assignment.due_step)
                leader_groups[group] = leader_groups.get(group, 0) + int(assignment.leader)
            elif assignment.leader:
                raise PAOSContractError("only H records may be satellite leaders")
        if any(count != 1 for count in leader_groups.values()):
            raise PAOSContractError("each H target/slot group must retain exactly one leader")

        vector: list[int] = []
        slices: list[tuple[int, int]] = []
        cursor = 0
        for kind_index, kind in enumerate(_KINDS):
            pool_size = len(pools[kind])
            assigned = sum(rows[kind_index])
            reserve = int(plan.reserve_counts[kind_index])
            if assigned + reserve != pool_size:
                raise PAOSContractError(
                    f"{kind} plan does not conserve D_k: {assigned}+{reserve}!={pool_size}"
                )
            vector.extend(rows[kind_index])
            vector.append(reserve)
            next_cursor = cursor + len(rows[kind_index]) + 1
            slices.append((cursor, next_cursor))
            cursor = next_cursor
        return SealPreview(
            plan_id=plan.plan_id,
            event_class=plan.event_class,
            vector=tuple(vector),
            vector_hash=_vector_hash(vector),
            decision_pool_counts=tuple(len(pools[kind]) for kind in _KINDS),
            kind_slices=tuple(slices),
        )

    def seal(self, plan: QuotaPlan) -> SealPreview:
        """Atomically replace only alive, non-due cancellable pending records."""

        try:
            preview = self.preview_seal(plan)
        except Exception as error:
            self._latch("plan_validation", str(error))
            raise
        prospective = dict(self.ledger)
        for kind in _KINDS:
            for record in self._decision_pool(plan.step, kind):
                prospective[record.platform_id] = replace(
                    record,
                    state=Lifecycle.FREE,
                    version=record.version + int(record.state is Lifecycle.ACTIVE_PENDING),
                    plan_id=None,
                    target_id=None,
                    target_ordinal=None,
                    due_step=None,
                    leader=False,
                )
        for assignment in plan.assignments:
            record = prospective[assignment.platform_id]
            prospective[assignment.platform_id] = replace(
                record,
                state=Lifecycle.ACTIVE_PENDING,
                version=record.version + 1,
                plan_id=plan.plan_id,
                target_id=assignment.target_id,
                target_ordinal=assignment.target_ordinal,
                due_step=assignment.due_step,
                leader=assignment.leader,
                issued_step=None,
                unavailable_reason=None,
            )
        self.ledger = prospective
        self.plan_version = plan.plan_id
        self._seal_count += 1
        self._check_conservation()
        return preview

    def _verify_rollout_expectation(self, state_hash: str, preview: SealPreview) -> None:
        expected = self.request.get("expected") or {}
        expected_state = expected.get("legal_state_hash")
        expected_vector = expected.get("preview_vector")
        expected_hash = expected.get("preview_hash")
        if expected_state is not None and str(expected_state) != state_hash:
            self._latch("candidate_identity", "rollout legal state does not match registered probe")
        if expected_vector is not None and tuple(map(int, expected_vector)) != preview.vector:
            self._latch("candidate_identity", "rollout preview vector does not match registered probe")
        if expected_hash is not None and str(expected_hash) != preview.vector_hash:
            self._latch("candidate_identity", "rollout preview hash does not match registered probe")
        requested_candidate = self.request.get("candidate_sha256")
        if requested_candidate and str(requested_candidate) != self.candidate_sha256:
            self._latch("candidate_identity", "loaded candidate theta hash differs from request")

    def _handle_plan_event(self, event_class: str, step: int) -> None:
        try:
            state_payload = self._state_payload(step)
            state_hash = self.legal_state_hash(step)
            snapshot = self._make_policy_snapshot(step)
            projected = self._project_quotas(snapshot, self.theta, config=self.config)
            plan = self._build_plan(event_class, step, state_hash, projected)
            self._proposal_count += 1
            preview = self.preview_seal(plan)
            self._preview_count += 1
            if self._first_preview is None:
                self._initial_legal_state_hash = state_hash
                self._first_preview = preview
                self._verify_rollout_expectation(state_hash, preview)
            if self._fatal_errors:
                return
            sealed = self.seal(plan)
            if sealed.vector != preview.vector or sealed.vector_hash != preview.vector_hash:
                self._latch("preview_seal_mismatch", "mutating seal differs from pure preview")
                return
            if self._first_seal is None:
                self._first_seal = sealed
            if event_class in self.event_latches:
                self.event_latches[event_class] = True
            if event_class == "new_objective":
                self._planned_membership_revision = self._membership_revision
            self._events.append({
                "event_class": event_class,
                "step": step,
                "plan_id": plan.plan_id,
                "legal_state_hash": state_hash,
                "preview_vector": list(preview.vector),
                "preview_hash": preview.vector_hash,
                "commander_sealed_vector": list(sealed.vector),
                "commander_sealed_hash": sealed.vector_hash,
            })
            public_snapshot = self._snapshot_json(snapshot)
            public_snapshot.update({
                "snapshot_hash": state_hash,
                "event_class": event_class,
                "step": step,
                "kind_slices": [list(value) for value in preview.kind_slices],
                "sealed_vector": list(sealed.vector),
                "sealed_hash": sealed.vector_hash,
                "legal_state": state_payload,
            })
            self._event_snapshots.append(public_snapshot)
            self._record_same_state_9500(snapshot, state_hash, event_class)
            for assignment in plan.assignments:
                if assignment.target_id not in self.initial_target_ids:
                    self.first_assignment_step.setdefault(assignment.target_id, step)
        except Exception as error:
            self._latch("plan_validation", f"{event_class}@{step}: {error}")

    def mark_launch_command_issued(
        self,
        platform_id: int,
        *,
        plan_id: int,
        prior_version: int,
        step: int,
    ) -> bool:
        record = self.ledger.get(int(platform_id))
        if (
            record is None
            or record.state is not Lifecycle.ACTIVE_PENDING
            or record.plan_id != int(plan_id)
            or record.version != int(prior_version)
            or record.due_step is None
            or record.due_step > int(step)
        ):
            self._latch("stale_callback", f"stale launch callback for platform {platform_id}")
            return False
        self.ledger[int(platform_id)] = replace(
            record,
            state=Lifecycle.ISSUED,
            version=record.version + 1,
            issued_step=int(step),
            due_step=None,
        )
        self._issued_count += 1
        self._check_conservation()
        return True

    def action_for(self, entity_id: int, step: int) -> list[float] | None:
        if self._fatal_errors:
            return None
        record = self.ledger.get(int(entity_id))
        if (
            record is None
            or record.state is not Lifecycle.ACTIVE_PENDING
            or record.due_step is None
            or record.due_step > int(step)
        ):
            return None
        target = next((item for item in self.targets if item.entity_id == record.target_id), None)
        if target is None or record.plan_id is None:
            self._latch("plan_validation", f"pending platform {entity_id} lost legal target")
            return None
        if not self.mark_launch_command_issued(
            int(entity_id), plan_id=record.plan_id, prior_version=record.version, step=int(step)
        ):
            return None
        return [1.0, float(entity_id), float(target.position.lon), float(target.position.lat)]

    def target_id_for(self, platform_id: int) -> int | None:
        record = self.ledger.get(int(platform_id))
        if record is None or record.state not in {Lifecycle.ACTIVE_PENDING, Lifecycle.ISSUED}:
            return None
        return record.target_id

    def should_use_satellite(self, platform_id: int) -> bool:
        record = self.ledger.get(int(platform_id))
        return bool(record is not None and record.state is Lifecycle.ISSUED and record.leader)

    def learning_task_context(self, platform_id: int) -> tuple[float, float, float, float, float]:
        target_id = self.target_id_for(platform_id)
        target = next((item for item in self.targets if item.entity_id == target_id), None)
        target_type = target.entity_type if target is not None else -1
        return (
            float(target_type == 9400),
            float(target_type == 9500),
            float(target_type == 9600),
            self.tactical_metrics.pressure,
            self._issued_count / max(1, len(self.expected_platform_ids)),
        )

    def observe_top_reward(self, reward: float) -> None:
        self.official_return += float(reward)

    def finish_top_episode(self, *, terminal: bool = True) -> None:
        self._check_conservation()

    def save_top(self, path: str) -> None:
        raise RuntimeError("PAOS top parameters are updated only by the external frozen protocol")

    def _snapshot_json(self, snapshot: Any) -> dict[str, Any]:
        return {
            "schema_version": "r11_paos_public_snapshot_v1",
            "time_fraction": _qfloat(snapshot.time_fraction),
            "initial_inventory": [int(value) for value in snapshot.initial_inventory],
            "decision_pool_counts": [int(value) for value in snapshot.decision_pool_counts],
            "targets": [
                {
                    "admission_ordinal": int(target.admission_ordinal),
                    "entity_type": int(target.entity_type),
                    "distance": _qfloat(target.distance),
                    "committed_counts": [int(value) for value in target.committed_counts],
                }
                for target in snapshot.targets
            ],
        }

    def _snapshot_from_json(self, value: Mapping[str, Any]):
        return self._PolicySnapshot(
            time_fraction=float(value["time_fraction"]),
            initial_inventory=tuple(map(int, value["initial_inventory"])),
            decision_pool_counts=tuple(map(int, value["decision_pool_counts"])),
            targets=tuple(
                self._TargetCandidate(
                    admission_ordinal=int(target["admission_ordinal"]),
                    entity_type=int(target["entity_type"]),
                    distance=float(target["distance"]),
                    committed_counts=tuple(map(int, target["committed_counts"])),
                )
                for target in value["targets"]
            ),
        )

    def _record_same_state_9500(self, snapshot: Any, snapshot_hash: str, event_class: str) -> None:
        if not any(int(target.entity_type) == 9500 for target in snapshot.targets):
            return
        direction = self.request.get("direction_vector")
        radius = self.request.get("radius")
        sign = self.request.get("sign")
        if direction is None or radius is None or sign not in {"plus", "minus"}:
            return
        multiplier = 1.0 if sign == "plus" else -1.0
        center = tuple(
            value - multiplier * float(radius) * float(delta)
            for value, delta in zip(self.theta, direction)
        )
        plus = tuple(value + float(radius) * float(delta) for value, delta in zip(center, direction))
        minus = tuple(value - float(radius) * float(delta) for value, delta in zip(center, direction))
        p_projected = self._project_quotas(snapshot, plus, config=self.config)
        m_projected = self._project_quotas(snapshot, minus, config=self.config)
        ordinals = tuple(int(value) for value in p_projected.catalogue_ordinals)
        type_by_ordinal = {int(target.admission_ordinal): int(target.entity_type) for target in snapshot.targets}
        p_l = self._kind_quota(p_projected, "L")
        m_l = self._kind_quota(m_projected, "L")
        plus_count = sum(int(value) for ordinal, value in zip(ordinals, p_l.target_counts) if type_by_ordinal[ordinal] == 9500)
        minus_count = sum(int(value) for ordinal, value in zip(ordinals, m_l.target_counts) if type_by_ordinal[ordinal] == 9500)
        self._same_state_9500.append({
            "snapshot_hash": snapshot_hash,
            "event_class": event_class,
            "plus_l_to_type_9500": plus_count,
            "minus_l_to_type_9500": minus_count,
            "difference": plus_count - minus_count,
        })

    @staticmethod
    def _quota_tv(plus: Any, minus: Any) -> float:
        totals = []
        for kind in _KINDS:
            p = PAOSCommander._kind_quota(plus, kind)
            m = PAOSCommander._kind_quota(minus, kind)
            count = int(p.decision_pool_count)
            if count <= 0:
                continue
            p_vector = tuple(map(int, p.target_counts)) + (int(p.reserve_count),)
            m_vector = tuple(map(int, m.target_counts)) + (int(m.reserve_count),)
            totals.append(sum(abs(left - right) for left, right in zip(p_vector, m_vector)) / (2.0 * count))
        return sum(totals) / len(totals) if totals else 0.0

    def _candidate_projection(self, snapshot: Any, theta: Sequence[float]) -> dict[str, Any]:
        projected = self._project_quotas(snapshot, theta, config=self.config)
        vector = tuple(map(int, projected.vector()))
        return {
            "candidate_sha256": self._policy_theta_hash(theta),
            "preview_vector": list(vector),
            "preview_hash": _vector_hash(vector),
        }

    def _load_snapshot_bank(self, path: str | None) -> list[dict[str, Any]]:
        if not path:
            return []
        source = Path(path)
        if source.suffix == ".jsonl":
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            value = json.loads(source.read_text(encoding="utf-8"))
            rows = value if isinstance(value, list) else value.get("snapshots", value.get("event_snapshots", []))
        return [dict(row) for row in rows]

    def no_step_probe(self, observations: tuple[dict, ...]) -> dict[str, Any]:
        """Prepare legal state and project candidates without policy action/step."""

        self._synchronize(observations)
        if not self._ready():
            raise PAOSContractError("probe could not construct the complete legal initial state")
        kind = self.request.get("kind", "pair_probe")
        if kind == "bank_projection":
            return self._bank_projection_probe()
        step = int(observations[0].get("step", 0)) if observations else 0
        snapshot = self._make_policy_snapshot(step)
        state_hash = self.legal_state_hash(step)
        direction = tuple(float(value) for value in self.request["direction_vector"])
        grid = tuple(float(value) for value in self.request.get("radius_grid", (0.25, 0.5, 1, 2, 4)))
        candidate_box = tuple(float(value) for value in self.request.get("candidate_box", (-4, 4)))
        candidates = []
        bank_rows = self._load_snapshot_bank(self.request.get("snapshot_bank_path"))
        for radius in grid:
            plus_theta = tuple(value + radius * delta for value, delta in zip(self.theta, direction))
            minus_theta = tuple(value - radius * delta for value, delta in zip(self.theta, direction))
            feasible = all(candidate_box[0] <= value <= candidate_box[1] for value in (*plus_theta, *minus_theta))
            entry: dict[str, Any] = {"radius": radius, "feasible": feasible}
            if feasible:
                plus_projected = self._project_quotas(snapshot, plus_theta, config=self.config)
                minus_projected = self._project_quotas(snapshot, minus_theta, config=self.config)
                plus_plan = self._build_plan("initial", step, state_hash, plus_projected)
                minus_plan = self._build_plan("initial", step, state_hash, minus_projected)
                plus_preview = self.preview_seal(plus_plan)
                minus_preview = self.preview_seal(minus_plan)
                entry.update({
                    "plus": {
                        "candidate_sha256": self._policy_theta_hash(plus_theta),
                        "preview_vector": list(plus_preview.vector),
                        "preview_hash": plus_preview.vector_hash,
                    },
                    "minus": {
                        "candidate_sha256": self._policy_theta_hash(minus_theta),
                        "preview_vector": list(minus_preview.vector),
                        "preview_hash": minus_preview.vector_hash,
                    },
                    "tv": self._quota_tv(plus_projected, minus_projected),
                })
                if bank_rows:
                    bank_previews = []
                    tvs = []
                    for row in bank_rows:
                        bank_snapshot = self._snapshot_from_json(row)
                        p = self._project_quotas(bank_snapshot, plus_theta, config=self.config)
                        m = self._project_quotas(bank_snapshot, minus_theta, config=self.config)
                        p_vector = tuple(map(int, p.vector()))
                        m_vector = tuple(map(int, m.vector()))
                        tvs.append(self._quota_tv(p, m))
                        bank_previews.append({
                            "snapshot_hash": row.get("snapshot_hash"),
                            "event_class": row.get("event_class"),
                            "decision_pool_counts": list(bank_snapshot.decision_pool_counts),
                            "kind_slices": self._kind_slices(len(bank_snapshot.targets)),
                            "plus_vector": list(p_vector),
                            "minus_vector": list(m_vector),
                        })
                    entry["bank_tv"] = sum(tvs) / len(tvs) if tvs else 0.0
                    entry["bank_previews"] = bank_previews
            candidates.append(entry)

        response = {
            "schema_version": self.PROBE_SCHEMA,
            "kind": "pair_probe",
            "run_id": self.request.get("run_id"),
            "cycle": self.request.get("cycle"),
            "direction": self.request.get("direction"),
            "seed_tuple": self.actual_seed_tuple,
            "center_sha256": self.request.get("center_sha256") or self.candidate_sha256,
            "legal_state_hash": state_hash,
            "decision_pool_counts": list(snapshot.decision_pool_counts),
            "kind_slices": self._kind_slices(len(snapshot.targets)),
            "candidates": candidates,
        }
        registered = self.request.get("registered_candidates") or []
        if bank_rows and registered:
            response["registered_bank_previews"] = self._registered_bank_previews(bank_rows, registered)
        return response

    @staticmethod
    def _kind_slices(target_count: int) -> list[list[int]]:
        width = int(target_count) + 1
        return [[index * width, (index + 1) * width] for index in range(3)]

    def _registered_bank_previews(self, rows: list[dict[str, Any]], registered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for candidate in registered:
            plus_theta = tuple(map(float, candidate["plus_theta"]))
            minus_theta = tuple(map(float, candidate["minus_theta"]))
            snapshots = []
            for row in rows:
                snapshot = self._snapshot_from_json(row)
                plus = self._project_quotas(snapshot, plus_theta, config=self.config)
                minus = self._project_quotas(snapshot, minus_theta, config=self.config)
                snapshots.append({
                    "snapshot_hash": row.get("snapshot_hash"),
                    "event_class": row.get("event_class"),
                    "decision_pool_counts": list(snapshot.decision_pool_counts),
                    "kind_slices": self._kind_slices(len(snapshot.targets)),
                    "plus_vector": list(map(int, plus.vector())),
                    "minus_vector": list(map(int, minus.vector())),
                })
            output.append({"direction": candidate.get("direction"), "snapshots": snapshots})
        return output

    def _bank_projection_probe(self) -> dict[str, Any]:
        rows = self._load_snapshot_bank(self.request.get("snapshot_bank_path"))
        groups = []
        for group in self.request.get("candidate_groups", []):
            theta = tuple(map(float, group["theta"]))
            snapshots = []
            for row in rows:
                snapshot = self._snapshot_from_json(row)
                projected = self._project_quotas(snapshot, theta, config=self.config)
                vector = tuple(map(int, projected.vector()))
                snapshots.append({
                    "snapshot_hash": row.get("snapshot_hash"),
                    "event_class": row.get("event_class"),
                    "decision_pool_counts": list(snapshot.decision_pool_counts),
                    "kind_slices": self._kind_slices(len(snapshot.targets)),
                    "quota_vector": list(vector),
                    "quota_hash": _vector_hash(vector),
                })
            groups.append({
                "label": group.get("label"),
                "theta_sha256": self._policy_theta_hash(theta),
                "snapshots": snapshots,
            })
        return {
            "schema_version": self.PROBE_SCHEMA,
            "kind": "bank_projection",
            "run_id": self.request.get("run_id"),
            "seed_tuple": self.actual_seed_tuple,
            "candidate_groups": groups,
        }

    def _check_conservation(self) -> None:
        if self.initial_inventory is None:
            return
        for index, kind in enumerate(_KINDS):
            count = sum(record.kind == kind for record in self.ledger.values())
            if count != self.initial_inventory[index]:
                self._latch(
                    "ledger_conservation",
                    f"{kind} ledger count {count} != N0 {self.initial_inventory[index]}",
                )

    def _latch(self, category: str, message: str) -> None:
        if category not in self._contract_violations:
            self._contract_violations[category] = 0
        self._contract_violations[category] += 1
        rendered = f"{category}: {message}"
        if rendered not in self._fatal_errors:
            self._fatal_errors.append(rendered)

    def assert_contract_ok(self) -> None:
        if self._fatal_errors:
            raise PAOSContractError("; ".join(self._fatal_errors))

    def _ledger_diagnostics(self) -> dict[str, Any]:
        counts = {
            kind: {state.value: 0 for state in Lifecycle}
            for kind in _KINDS
        }
        for record in self.ledger.values():
            counts[record.kind][record.state.value] += 1
        return {
            "initial_inventory": list(self.initial_inventory or (0, 0, 0)),
            "state_counts": counts,
            "plan_version": self.plan_version,
        }

    def paos_audit(self) -> dict[str, Any]:
        first_preview = self._first_preview
        first_seal = self._first_seal
        return {
            "schema_version": self.AUDIT_SCHEMA,
            "run_id": self.request.get("run_id"),
            "cycle": self.request.get("cycle"),
            "direction": self.request.get("direction"),
            "sign": self.request.get("sign"),
            "radius": self.request.get("radius"),
            "seed_tuple": self.actual_seed_tuple,
            "center_sha256": self.request.get("center_sha256"),
            "candidate_sha256": self.candidate_sha256,
            "candidate_file_sha256": self.top_file_sha256,
            "bottom_sha256": self.bottom_sha256,
            "input_dim": 90,
            "official_return": self.official_return,
            "legal_state_hash": self._initial_legal_state_hash,
            "first_preview_vector": list(first_preview.vector) if first_preview else None,
            "first_preview_hash": first_preview.vector_hash if first_preview else None,
            "first_commander_sealed_vector": list(first_seal.vector) if first_seal else None,
            "first_commander_sealed_hash": first_seal.vector_hash if first_seal else None,
            "proposal_count": self._proposal_count,
            "preview_count": self._preview_count,
            "seal_count": self._seal_count,
            "launch_command_issued_count": self._issued_count,
            "contract_violations": dict(sorted(self._contract_violations.items())),
            "fatal_error": "; ".join(self._fatal_errors) if self._fatal_errors else None,
            "ledger": self._ledger_diagnostics(),
            "events": list(self._events),
            "event_snapshots": list(self._event_snapshots),
            "same_state_9500_differences": list(self._same_state_9500),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "initial_target_ids": sorted(self.initial_target_ids),
            "current_target_ids": sorted(target.entity_id for target in self.targets),
            "dynamic_target_ids": sorted(
                target.entity_id for target in self.targets
                if target.entity_id not in self.initial_target_ids
            ),
            "first_discovery_step": {
                str(entity_id): step for entity_id, step in sorted(self.first_discovery_step.items())
            },
            "first_assignment_step": {
                str(entity_id): step for entity_id, step in sorted(self.first_assignment_step.items())
            },
            "top_level": {"paos_audit": self.paos_audit()},
        }

    def reset(self) -> None:
        # One fresh process per PAOS branch is mandatory.  Refuse accidental
        # multi-round reuse instead of silently carrying a catalogue ordinal.
        if self.ledger or self.plan_version or self._events:
            self._latch("request", "PAOS commander cannot be reused across rounds")
            return
