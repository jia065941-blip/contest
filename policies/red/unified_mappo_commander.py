"""单网络混合动作 MAPPO 的逐步生命周期与目标选择。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from typing import Any, Mapping, Sequence

from .baselines import BaselineRules, PlatformState, TargetPrior, TacticalMetrics
from .contracts import Position
from .tracks import InitialCatalogueTrackFusion

_KIND_BY_TYPE = {21000: "H", 21001: "M", 21002: "L"}
_SEARCH_TARGET_ID = -100
_RUNTIME_SET_POSITION = 4.0
_USE_SATELLITE = 3.0


@dataclass(frozen=True)
class DecisionEvent:
    """A factual causal decision available to target-credit construction."""

    event_id: int
    event_type: str
    agent_id: int
    entity_id: int
    step: int
    target_id: int | None
    search_source: int | None
    satellite_source: int | None
    action: Any


class UnifiedMAPPOCommander:
    """将共享策略的生命周期动作映射为仿真动作。"""

    fail_fast_agent_errors = True

    def __init__(
        self,
        targets: tuple[TargetPrior, ...],
        *,
        max_steps: int,
        search_polygon: tuple[Position, ...] | None = None,
    ) -> None:
        self.max_steps = max(1, int(max_steps))
        self.search_polygon = tuple(search_polygon or ())
        self.dynamic_lifecycle = os.getenv(
            "RED_UNIFIED_DYNAMIC_LIFECYCLE", "0"
        ) == "1"
        self.catalogue_targets = tuple(targets)
        # 9500 tracks are search-only in dynamic mode.  Their initialization
        # records are not legal actor observations and must not enter masks.
        self.initial_targets = tuple(
            target
            for target in self.catalogue_targets
            if not self.dynamic_lifecycle or int(target.entity_type) != 9500
        )
        self.hidden_initial_target_ids = frozenset(
            target.entity_id for target in self.catalogue_targets
            if int(target.entity_type) == 9500
        ) if self.dynamic_lifecycle else frozenset()
        self.targets = self.initial_targets
        self.track_fusion = InitialCatalogueTrackFusion(self.initial_targets)
        self.search_target = self._make_search_target(search_polygon)
        self.rules = BaselineRules()
        self.policy = None
        self.expected_platform_ids: set[int] = set()
        self.agent_by_platform: dict[int, int] = {}
        self.reports: dict[int, PlatformState] = {}
        self.target_by_platform: dict[int, int] = {}
        self.search_source_by_platform: dict[int, int] = {}
        self.track_source_by_platform: dict[int, dict[int, int]] = {}
        self.satellite_track_source_by_platform: dict[int, dict[int, int]] = {}
        self.visible_9500_by_platform: dict[int, set[int]] = {}
        self.discovered_9500_by_platform: dict[int, set[int]] = {}
        self.pending: dict[int, tuple[int, float, float]] = {}
        self._pending_launch_positions: dict[int, Position] = {}
        self.launched_ids: set[int] = set()
        self.first_discovery_step: dict[int, int] = {}
        self._decision_ledger: list[DecisionEvent] = []
        self._pending_decision_events: list[DecisionEvent] = []
        self._next_event_id = 0
        self._planned = False
        self.wait_count = 0
        self.launch_count = 0
        self.keep_count = 0
        self.retarget_count = 0
        self.search_count = 0
        self.satellite_request_count = 0
        self.illegal_lifecycle_count = 0
        self.tactical_metrics = TacticalMetrics()

    def _make_search_target(
        self, search_polygon: tuple[Position, ...] | None
    ) -> TargetPrior | None:
        if not self.dynamic_lifecycle or not search_polygon:
            return None
        return TargetPrior(
            entity_id=_SEARCH_TARGET_ID,
            entity_type=-1,
            position=Position(
                lon=sum(point.lon for point in search_polygon)
                / len(search_polygon),
                lat=sum(point.lat for point in search_polygon)
                / len(search_polygon),
                alt=0.0,
            ),
            value=0.0,
            alive=True,
        )

    def _policy_targets(self) -> tuple[TargetPrior, ...]:
        if self.search_target is None:
            return self.targets
        return (*self.targets, self.search_target)

    def _reserved_policy_targets(self) -> tuple[TargetPrior, ...]:
        """Reserve fixed actor slots for initially hidden catalogue targets."""

        if not self.dynamic_lifecycle:
            return ()
        return tuple(
            target
            for target in self.catalogue_targets
            if target.entity_id in self.hidden_initial_target_ids
        )

    def _configure_policy_targets(self) -> None:
        if self.policy is None:
            raise RuntimeError("统一 MAPPO commander 尚未连接共享策略")
        self.policy.configure_targets(
            self._policy_targets(),
            reserved_targets=self._reserved_policy_targets(),
        )

    def _sync_actor_target_visibility(self) -> None:
        """Forward each actor's engine-fused communication-cluster tracks."""

        if self.policy is None:
            raise RuntimeError("统一 MAPPO commander 尚未连接共享策略")
        setter = getattr(self.policy, "set_actor_target_visibility", None)
        if not callable(setter):
            return
        entity_ids = (
            set(self.expected_platform_ids)
            | set(self.reports)
            | set(self.policy.active_entity_ids)
        )
        setter({
            entity_id: tuple(sorted(
                self.visible_9500_by_platform.get(entity_id, set())
            ))
            for entity_id in entity_ids
        })

    def attach_policy(self, policy: Any) -> None:
        self.policy = policy
        configure_search_polygon = getattr(
            self.policy, "configure_search_polygon", None
        )
        if callable(configure_search_polygon):
            configure_search_polygon(self.search_polygon)
        if self.dynamic_lifecycle:
            # Initial coordinates belong to the first LAUNCH timestep, never
            # to the pre-episode staging/deployment pass.
            os.environ["RED_UNIFIED_TRAIN_INITIAL"] = "0"
            # KEEP/RETARGET is a legal choice on every live timestep.
            if hasattr(self.policy, "retarget_interval"):
                self.policy.retarget_interval = 1
        self._configure_policy_targets()
        self._sync_actor_target_visibility()

    def register_platform(self, entity_id: int) -> None:
        self.expected_platform_ids.add(int(entity_id))

    def register_agent_identity(self, entity_id: int, agent_id: int) -> None:
        self.register_platform(entity_id)
        self.agent_by_platform[int(entity_id)] = int(agent_id)

    @property
    def decision_ledger(self) -> tuple[dict[str, Any], ...]:
        """Return an immutable snapshot of all factual decisions this episode."""

        return tuple(asdict(event) for event in self._decision_ledger)

    def consume_decision_events(self) -> tuple[dict[str, Any], ...]:
        """Consume factual decision events added since the previous call."""

        events = tuple(asdict(event) for event in self._pending_decision_events)
        self._pending_decision_events.clear()
        return events

    def _record_decision(
        self,
        event_type: str,
        *,
        entity_id: int,
        step: int,
        target_id: int | None,
        search_source: int | None,
        action: Any,
        satellite_source: int | None = None,
    ) -> None:
        event = DecisionEvent(
            event_id=self._next_event_id,
            event_type=str(event_type).upper(),
            agent_id=self.agent_by_platform.get(int(entity_id), -1),
            entity_id=int(entity_id),
            step=int(step),
            target_id=None if target_id is None else int(target_id),
            search_source=(
                None if search_source is None else int(search_source)
            ),
            satellite_source=(
                None if satellite_source is None else int(satellite_source)
            ),
            action=action,
        )
        self._next_event_id += 1
        self._decision_ledger.append(event)
        self._pending_decision_events.append(event)

    def set_launch_position(
        self,
        entity_id: int,
        lon: float,
        lat: float,
        alt: float = 0.0,
    ) -> None:
        """Stage a learned position for the entity's first LAUNCH only."""

        entity_id = int(entity_id)
        if entity_id in self.launched_ids:
            raise RuntimeError(f"实体 {entity_id} 已发射，不能再次设置初始坐标")
        values = (float(lon), float(lat), float(alt))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("首次 LAUNCH 坐标必须为有限数值")
        self._pending_launch_positions[entity_id] = Position(*values)

    def target_search_source_for(self, platform_id: int) -> int | None:
        return self.search_source_by_platform.get(int(platform_id))

    def record_maneuver(
        self,
        entity_id: int,
        step: int,
        maneuver: int,
        satellite_source: int | None = None,
    ) -> None:
        """Record the factual maneuver selected for an on-field entity."""

        entity_id = int(entity_id)
        target_id = self.target_by_platform.get(entity_id)
        self._record_decision(
            "MANEUVER",
            entity_id=entity_id,
            step=step,
            target_id=target_id,
            search_source=self.search_source_by_platform.get(entity_id),
            satellite_source=satellite_source,
            action=int(maneuver),
        )

    @staticmethod
    def _field(value: Any, name: str, default: Any) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    def report(self, observation: dict) -> None:
        own = observation.get("self", {})
        entity_id = int(observation.get("entity_id", -1))
        kind = _KIND_BY_TYPE.get(int(own.get("type", -1)))
        position = own.get("position", {})
        if entity_id < 0 or kind is None or not isinstance(position, dict):
            return
        known_types = {
            int(target.entity_id): int(target.entity_type)
            for target in (*self.catalogue_targets, *self.targets)
        }
        visible_9500: set[int] = set()
        track_sources: dict[int, int] = {}
        satellite_track_sources: dict[int, int] = {}
        for raw_id, track in (own.get("detectInfo") or {}).items():
            target_id = int(self._field(track, "entity_id", raw_id))
            target_type = int(
                self._field(track, "entity_type", known_types.get(target_id, -1))
            )
            if target_type == 9500:
                visible_9500.add(target_id)
                source = int(self._field(track, "detect_from", entity_id))
                track_sources[target_id] = source
                if bool(self._field(track, "via_satellite", False)):
                    satellite_track_sources[target_id] = source
        self.visible_9500_by_platform[entity_id] = visible_9500
        self.track_source_by_platform[entity_id] = track_sources
        self.satellite_track_source_by_platform[entity_id] = satellite_track_sources
        self.discovered_9500_by_platform.setdefault(entity_id, set()).update(
            visible_9500
        )
        agent_id = int(observation.get("agent_id", -1))
        if agent_id >= 0:
            self.agent_by_platform[entity_id] = agent_id
        self.reports[entity_id] = PlatformState(
            entity_id=entity_id,
            kind=kind,
            position=Position(
                lon=float(position.get("lon", 0.0)),
                lat=float(position.get("lat", 0.0)),
                alt=float(position.get("alt", 0.0)),
            ),
            alive=float(own.get("health", 0.0)) > 0.0
            and bool(own.get("isVisible", True)),
            launched=entity_id in self.launched_ids,
        )

    def begin_step(self, observations: tuple[dict, ...]) -> None:
        if self.policy is None:
            raise RuntimeError("统一 MAPPO commander 尚未连接共享策略")
        changed = False
        for observation in observations:
            self.report(observation)
            changed = self.track_fusion.ingest(observation) or changed
        if changed:
            previous_ids = {target.entity_id for target in self.targets}
            self.targets = self.track_fusion.targets
            for target in self.targets:
                if target.entity_id not in previous_ids:
                    provenance = self.track_fusion.provenance_for(
                        target.entity_id
                    )
                    self.first_discovery_step[target.entity_id] = (
                        provenance[0].step if provenance else int(
                            observations[0].get("step", 0) if observations else 0
                        )
                    )
            self._configure_policy_targets()
        self._sync_actor_target_visibility()
        active_ids = self.policy.active_entity_ids
        if (
            not self.dynamic_lifecycle
            and not self._planned
            and active_ids.issubset(self.reports)
        ):
            step = int(observations[0].get("step", 0) if observations else 0)
            self._build_initial_plan(step)
        alive_count = sum(report.alive for report in self.reports.values())
        self.tactical_metrics = TacticalMetrics(
            launched_count=len(self.launched_ids),
            alive_count=alive_count,
            lost_count=max(0, len(active_ids) - alive_count),
            total_count=len(active_ids),
        )

    def _build_initial_plan(self, step: int) -> None:
        target_by_id = {target.entity_id: target for target in self.targets}
        for entity_id in sorted(self.policy.active_entity_ids):
            report = self.reports.get(entity_id)
            target_id = self.policy.deployment_target_id(entity_id)
            target = target_by_id.get(target_id)
            if report is None or target is None:
                continue
            due_step = step + self.rules.wave_delay(report.kind)
            self.target_by_platform[entity_id] = target.entity_id
            self.pending[entity_id] = (
                due_step,
                float(target.position.lon),
                float(target.position.lat),
            )
        self._planned = True

    def action_for(
        self, entity_id: int, step: int
    ) -> list[float] | list[list[float]] | None:
        entity_id = int(entity_id)
        if self.dynamic_lifecycle:
            return self._dynamic_action_for(entity_id, int(step))
        if entity_id in self.launched_ids:
            return None
        pending = self.pending.get(entity_id)
        if pending is None or pending[0] > int(step):
            return None
        _, lon, lat = self.pending.pop(entity_id)
        self.launched_ids.add(entity_id)
        action = [1.0, float(entity_id), lon, lat]
        target_id = self.target_by_platform.get(entity_id)
        search_source = (
            self.track_fusion.source_for(target_id)
            if target_id is not None else None
        )
        self._record_decision(
            "LAUNCH",
            entity_id=entity_id,
            step=step,
            target_id=target_id,
            search_source=search_source,
            satellite_source=self._satellite_source(entity_id, target_id),
            action=tuple(action),
        )
        return action

    @staticmethod
    def _coerce_position(value: Any, fallback_alt: float) -> Position | None:
        if isinstance(value, Mapping):
            if "lon" not in value or "lat" not in value:
                return None
            raw = (
                value["lon"],
                value["lat"],
                value.get("alt", fallback_alt),
            )
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) < 2:
                return None
            raw = (
                value[0],
                value[1],
                value[2] if len(value) > 2 else fallback_alt,
            )
        else:
            return None
        values = tuple(float(item) for item in raw)
        if not all(math.isfinite(item) for item in values):
            raise ValueError("首次 LAUNCH 坐标必须为有限数值")
        return Position(*values)

    def _consume_launch_position(
        self, entity_id: int, decision: Mapping[str, Any]
    ) -> Position | None:
        report = self.reports.get(entity_id)
        fallback_alt = report.position.alt if report is not None else 0.0
        explicit = decision.get("initial_position")
        if explicit is None and {"initial_lon", "initial_lat"}.issubset(decision):
            explicit = {
                "lon": decision["initial_lon"],
                "lat": decision["initial_lat"],
                "alt": decision.get("initial_alt", fallback_alt),
            }
        position = self._coerce_position(explicit, fallback_alt)
        if position is not None:
            self._pending_launch_positions.pop(entity_id, None)
            return position
        provider = getattr(self.policy, "consume_launch_position", None)
        if callable(provider):
            position = self._coerce_position(provider(entity_id), fallback_alt)
            if position is not None:
                self._pending_launch_positions.pop(entity_id, None)
                return position
        return self._pending_launch_positions.pop(entity_id, None)

    def _target_is_legal(self, entity_id: int, target: TargetPrior) -> bool:
        report = self.reports.get(entity_id)
        if report is None:
            return False
        target_id = int(target.entity_id)
        if target_id == _SEARCH_TARGET_ID:
            return report.kind == "L"
        target_type = int(target.entity_type)
        if target_type == 9500 and self.dynamic_lifecycle:
            if target_id not in self.visible_9500_by_platform.get(
                entity_id, set()
            ):
                return False
        if report.kind == "L":
            return target_type == 9500
        # Core's hit-rate table gives H/M a zero probability against 9500.
        # Detection is still useful and is shared by the engine communication
        # graph, but a zero-effect track is not a legal attack destination.
        return target_type in {9400, 9600}

    def _search_source(self, entity_id: int, target_id: int) -> int | None:
        if target_id == _SEARCH_TARGET_ID:
            return entity_id
        target = next(
            (item for item in self.targets if item.entity_id == target_id),
            None,
        )
        if target is None or int(target.entity_type) != 9500:
            return None
        source = self.track_source_by_platform.get(entity_id, {}).get(target_id)
        if source is None:
            return None
        source_records = [
            record
            for record in self.track_fusion.provenance_for(target_id)
            if int(record.detect_from) == source
        ]
        if not source_records:
            return None
        current_source_record = source_records[-1]
        # Search credit requires an actual SEARCH by the currently fused
        # source before (or at) the detection used by the later attack.
        return source if any(
            event.event_type == "SEARCH"
            and event.entity_id == source
            and event.step <= current_source_record.step
            for event in self._decision_ledger
        ) else None

    def _satellite_source(
        self, entity_id: int, target_id: int | None
    ) -> int | None:
        if target_id is None:
            return None
        return self.satellite_track_source_by_platform.get(
            int(entity_id), {}
        ).get(int(target_id))

    def _satellite_rows(
        self,
        *,
        entity_id: int,
        step: int,
        target_id: int | None,
        requested: int,
    ) -> list[list[float]]:
        if not requested:
            return []
        action = [_USE_SATELLITE, float(entity_id), 0.0, 0.0]
        self._record_decision(
            "SATELLITE_REQUEST",
            entity_id=entity_id,
            step=step,
            target_id=target_id,
            search_source=None,
            satellite_source=entity_id,
            action=tuple(action),
        )
        self.satellite_request_count += 1
        return [action]

    def _dynamic_action_for(
        self, entity_id: int, step: int
    ) -> list[list[float]] | None:
        decision = self.policy.consume_lifecycle_decision(entity_id)
        kind = str(decision["kind"])
        satellite_request = int(decision.get("satellite_request", 0))
        if kind == "wait":
            target_id = int(decision.get("target_id", -1))
            if target_id != -1:
                self.target_by_platform[entity_id] = target_id
            self.wait_count += 1
            return None
        if kind == "keep":
            self.keep_count += 1
            target_id = self.target_by_platform.get(entity_id)
            if bool(decision.get("goal_boundary", False)):
                # Selecting the currently active target at an option boundary is
                # still a stochastic target-head decision even though its engine
                # command is the deterministic KEEP. Record that decision so
                # trajectory counterfactual credit has an auditable PPO anchor.
                self._record_decision(
                    "TARGET_SELECT",
                    entity_id=entity_id,
                    step=step,
                    target_id=target_id,
                    search_source=self.search_source_by_platform.get(entity_id),
                    satellite_source=self._satellite_source(
                        entity_id, target_id
                    ),
                    action="KEEP",
                )
            rows = self._satellite_rows(
                entity_id=entity_id,
                step=step,
                target_id=target_id,
                requested=satellite_request,
            )
            return rows or None

        target_id = int(decision["target_id"])
        target = next(
            (
                item
                for item in self._policy_targets()
                if int(item.entity_id) == target_id
            ),
            None,
        )
        if target is None or not self._target_is_legal(entity_id, target):
            self.illegal_lifecycle_count += 1
            raise RuntimeError(
                f"实体 {entity_id} 选择了当前不合法目标 {target_id}"
            )
        search_source = self._search_source(entity_id, target_id)
        if search_source is None:
            self.search_source_by_platform.pop(entity_id, None)
        else:
            self.search_source_by_platform[entity_id] = search_source
        self.target_by_platform[entity_id] = target_id
        if target_id == _SEARCH_TARGET_ID:
            self.search_count += 1

        target_lon = float(decision.get("target_lon", target.position.lon))
        target_lat = float(decision.get("target_lat", target.position.lat))
        satellite_source = self._satellite_source(entity_id, target_id)
        satellite_rows = self._satellite_rows(
            entity_id=entity_id,
            step=step,
            target_id=target_id,
            requested=satellite_request,
        )
        if kind == "launch":
            self.launch_count += 1
            self.launched_ids.add(entity_id)
            commands: list[list[float]] = []
            initial_position = self._consume_launch_position(entity_id, decision)
            if initial_position is not None:
                commands.append([
                    _RUNTIME_SET_POSITION,
                    float(entity_id),
                    float(initial_position.lon),
                    float(initial_position.lat),
                    float(initial_position.alt),
                ])
            commands.append([1.0, float(entity_id), target_lon, target_lat])
            self._record_decision(
                "LAUNCH",
                entity_id=entity_id,
                step=step,
                target_id=target_id,
                search_source=search_source,
                satellite_source=satellite_source,
                action=tuple(tuple(row) for row in commands),
            )
            if target_id == _SEARCH_TARGET_ID:
                self._record_decision(
                    "SEARCH",
                    entity_id=entity_id,
                    step=step,
                    target_id=target_id,
                    search_source=entity_id,
                    satellite_source=None,
                    action=(target_lon, target_lat),
                )
            commands.extend(satellite_rows)
            return commands

        self.retarget_count += 1
        action = [2.0, float(entity_id), target_lon, target_lat]
        commands = [action, *satellite_rows]
        self._record_decision(
            "RETARGET",
            entity_id=entity_id,
            step=step,
            target_id=target_id,
            search_source=search_source,
            satellite_source=satellite_source,
            action=tuple(action),
        )
        if target_id == _SEARCH_TARGET_ID:
            self._record_decision(
                "SEARCH",
                entity_id=entity_id,
                step=step,
                target_id=target_id,
                search_source=entity_id,
                satellite_source=None,
                action=(target_lon, target_lat),
            )
        return commands

    def target_id_for(self, platform_id: int) -> int | None:
        return self.target_by_platform.get(int(platform_id))

    def should_use_satellite(self, platform_id: int) -> bool:
        del platform_id
        return False

    def learning_task_context(
        self, platform_id: int
    ) -> tuple[float, float, float, float, float]:
        target_id = self.target_id_for(platform_id)
        target = next(
            (
                item
                for item in self._policy_targets()
                if item.entity_id == target_id
            ),
            None,
        )
        target_type = int(target.entity_type) if target is not None else -1
        return (
            float(target_type == 9400),
            float(target_type == 9500),
            float(target_type == 9600),
            float(target_id == _SEARCH_TARGET_ID),
            len(self.launched_ids)
            / max(1, len(self.policy.active_entity_ids)),
        )

    def diagnostics(self) -> dict[str, Any]:
        initial_ids = {target.entity_id for target in self.initial_targets}
        result = {
            "initial_target_ids": sorted(initial_ids),
            "hidden_initial_target_ids": sorted(self.hidden_initial_target_ids),
            "current_target_ids": sorted(
                target.entity_id for target in self.targets
            ),
            "dynamic_target_ids": sorted(
                target.entity_id
                for target in self.targets
                if target.entity_id not in initial_ids
            ),
            "first_discovery_step": {
                str(entity_id): step
                for entity_id, step in sorted(self.first_discovery_step.items())
            },
            "assigned_count": len(self.target_by_platform),
            "launched_count": len(self.launched_ids),
            "dynamic_lifecycle": self.dynamic_lifecycle,
            "wait_count": self.wait_count,
            "launch_count": self.launch_count,
            "keep_count": self.keep_count,
            "retarget_count": self.retarget_count,
            "search_count": self.search_count,
            "satellite_request_count": self.satellite_request_count,
            "illegal_lifecycle_count": self.illegal_lifecycle_count,
            "decision_event_count": len(self._decision_ledger),
            "local_9500_visibility": {
                str(entity_id): sorted(target_ids)
                for entity_id, target_ids in sorted(
                    self.visible_9500_by_platform.items()
                )
            },
            "track_provenance": {
                str(target_id): [asdict(record) for record in records]
                for target_id, records in sorted(
                    self.track_fusion.provenance.items()
                )
            },
        }
        if self.policy is not None:
            result["unified_mappo"] = self.policy.diagnostics()
        return result

    def observe_top_reward(self, reward: float) -> None:
        del reward

    def finish_top_episode(self, *, terminal: bool = True) -> None:
        del terminal

    def reset(self) -> None:
        self.targets = self.initial_targets
        self.track_fusion = InitialCatalogueTrackFusion(self.initial_targets)
        self.reports.clear()
        self.target_by_platform.clear()
        self.search_source_by_platform.clear()
        self.track_source_by_platform.clear()
        self.satellite_track_source_by_platform.clear()
        self.visible_9500_by_platform.clear()
        self.discovered_9500_by_platform.clear()
        self.pending.clear()
        self._pending_launch_positions.clear()
        self.launched_ids.clear()
        self.first_discovery_step.clear()
        self._decision_ledger.clear()
        self._pending_decision_events.clear()
        self._next_event_id = 0
        self._planned = False
        self.wait_count = 0
        self.launch_count = 0
        self.keep_count = 0
        self.retarget_count = 0
        self.search_count = 0
        self.satellite_request_count = 0
        self.illegal_lifecycle_count = 0
        self.tactical_metrics = TacticalMetrics()
        if self.policy is not None:
            self._configure_policy_targets()
            self._sync_actor_target_visibility()
