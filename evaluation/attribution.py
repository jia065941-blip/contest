"""Per-objective and per-unit causal evidence for competition evaluations.

This module deliberately runs outside the actor interface.  It may inspect
engine truth to explain a completed run, but it never writes observations,
tasks, actions, rewards, or engine state.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any


RED_TYPES = {21000: "H", 21001: "M", 21002: "L"}
OBJECTIVE_TYPES = {9400: "mother", 9500: "dark", 9600: "sentinel"}
INTERCEPTOR_TYPE = 24000
DAMAGE_RADIUS_M = 100.0


def _position(entity) -> list[float]:
    point = entity.posEcf
    return [float(point.x), float(point.y), float(point.z)]


def _distance(a: list[float], b: list[float]) -> tuple[float, float]:
    horizontal = math.hypot(a[0] - b[0], a[1] - b[1])
    return horizontal, math.hypot(horizontal, a[2] - b[2])


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return repr(value)


class AttributionTracker:
    """Join legal task history with engine-truth outcome evidence."""

    def __init__(self, *, env, scenario, output_dir, seed):
        self.env = env
        self.scenario = scenario
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.factory = env.engine.simulator_factory
        self.map_name = scenario.path.stem
        policy = scenario.reward_policy
        self.weights = dict(policy.objective_weights) if policy is not None else {}

        self.objectives: dict[int, dict[str, Any]] = {}
        self.units: dict[int, dict[str, Any]] = {}
        self.assignment_events: list[dict[str, Any]] = []
        self.discovery_events: list[dict[str, Any]] = []
        self.damage_events: list[dict[str, Any]] = []
        self._last_assignment: dict[int, tuple[Any, ...]] = {}
        self._last_track_step: dict[tuple[int, int], int] = {}
        self._relation_lengths: dict[int, int] = defaultdict(int)
        self._previous_health: dict[int, float] = {}
        self._previous_visible: dict[int, bool] = {}
        self._finished = False

    def _entities(self):
        return {
            simulator.entity_ext.entity.id: simulator.entity_ext.entity
            for simulator in self.factory.get_all_simulators()
        }

    def start(self) -> None:
        entities = self._entities()
        area = self.env.engine.profile.imagineProfile.mapArea
        bounds = [float(area.xMin), float(area.xMax), float(area.yMin), float(area.yMax)]
        for entity_id, entity in entities.items():
            health = float(entity.survivePoints)
            self._previous_health[entity_id] = health
            self._previous_visible[entity_id] = bool(entity.isVisible)
            if entity.entityType in OBJECTIVE_TYPES:
                point = _position(entity)
                outside_distance = math.hypot(
                    point[0] - min(max(point[0], bounds[0]), bounds[1]),
                    point[1] - min(max(point[1], bounds[2]), bounds[3]),
                )
                public = entity.entityType in (9400, 9600)
                self.objectives[entity_id] = {
                    "id": entity_id,
                    "name": entity.nameChn,
                    "type": int(entity.entityType),
                    "kind": OBJECTIVE_TYPES[entity.entityType],
                    "weight": float(self.weights.get(entity_id, 0.0)),
                    "position": point,
                    "initial_health": health,
                    "final_health": health,
                    "public_at_initialization": public,
                    "discovered_step": 0 if public else None,
                    "discovered_by": [],
                    "track_refresh_by_unit": {},
                    "health_changes": [],
                    "hit_events": [],
                    "outside_map_distance_m": outside_distance,
                }

        for entity_id, agent in self.env.agent_manager.get_registered_agents():
            entity = entities[entity_id]
            self.units[entity_id] = {
                "id": entity_id,
                "role": RED_TYPES[int(entity.entityType)],
                "type": int(entity.entityType),
                "initial_position": _position(entity),
                "final_position": _position(entity),
                "terminal_step": None,
                "terminal_reason": None,
                "interceptor_hits": [],
                "objective_hits": [],
                "assignments": [],
                "minimum_distance_by_target": {},
            }
        self._sample_assignments(0, entities)
        self._sample_discoveries(0, entities)

    def update(self, step: int) -> None:
        entities = self._entities()
        self._sample_assignments(step, entities)
        self._sample_discoveries(step, entities)
        self._sample_relations(step, entities)
        self._sample_health_and_terminals(step, entities)

    def _sample_assignments(self, step: int, entities) -> None:
        for entity_id, agent in self.env.agent_manager.get_registered_agents():
            task = deepcopy(getattr(agent, "task", {}) or {})
            target_id = task.get("target_id")
            try:
                target_id = None if target_id is None else int(target_id)
            except (TypeError, ValueError):
                target_id = None
            signature = (
                task.get("kind"), target_id, task.get("target_ref"),
                tuple(task.get("last_point") or ()), task.get("waypoint_index"),
            )
            if self._last_assignment.get(entity_id) != signature:
                event = {
                    "step": int(step),
                    "unit_id": entity_id,
                    "role": self.units[entity_id]["role"],
                    "kind": task.get("kind"),
                    "target_id": target_id,
                    "target_ref": task.get("target_ref"),
                    "last_point": _json_value(task.get("last_point")),
                    "waypoint_index": task.get("waypoint_index"),
                }
                self.assignment_events.append(event)
                self.units[entity_id]["assignments"].append(event)
                self._last_assignment[entity_id] = signature

            if target_id in self.objectives:
                own = _position(entities[entity_id])
                target = _position(entities[target_id])
                horizontal, spatial = _distance(own, target)
                row = self.units[entity_id]["minimum_distance_by_target"].setdefault(
                    str(target_id),
                    {"horizontal_m": horizontal, "spatial_m": spatial, "step": int(step)},
                )
                if spatial < row["spatial_m"]:
                    row.update(horizontal_m=horizontal, spatial_m=spatial, step=int(step))

    def _sample_discoveries(self, step: int, entities) -> None:
        for unit_id in self.units:
            for raw_id, track in entities[unit_id].detectInfo.items():
                try:
                    target_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                objective = self.objectives.get(target_id)
                if objective is None:
                    continue
                observed_step = int(
                    track.get("step", step) if isinstance(track, dict) else getattr(track, "step", step)
                )
                key = (unit_id, target_id)
                previous = self._last_track_step.get(key)
                if previous is not None and observed_step <= previous:
                    continue
                self._last_track_step[key] = observed_step
                refresh = objective["track_refresh_by_unit"].setdefault(str(unit_id), {
                    "first_received_step": int(step),
                    "last_received_step": int(step),
                    "first_observed_step": observed_step,
                    "last_observed_step": observed_step,
                    "refresh_count": 0,
                })
                refresh["last_received_step"] = int(step)
                refresh["last_observed_step"] = observed_step
                refresh["refresh_count"] += 1
                if unit_id not in objective["discovered_by"]:
                    objective["discovered_by"].append(unit_id)
                    self.discovery_events.append({
                        "step": int(step), "target_id": target_id, "unit_id": unit_id,
                        "role": self.units[unit_id]["role"],
                        "observed_step": observed_step,
                    })
                if objective["discovered_step"] is None:
                    objective["discovered_step"] = int(step)

    def _sample_relations(self, step: int, entities) -> None:
        relation = self.factory.target_hit_relation
        for target_id, rows in relation.items():
            offset = self._relation_lengths[target_id]
            if offset >= len(rows):
                continue
            for raw in rows[offset:]:
                source_id = int(raw["id"])
                source_type = int(raw["entity_type"])
                event = {
                    "recorded_step": int(step),
                    "engine_step": int(raw["sim_step"]),
                    "target_id": int(target_id),
                    "source_id": source_id,
                    "source_type": source_type,
                    "source_role": RED_TYPES.get(source_type),
                    "source_position": _json_value(raw["posEcf"]),
                }
                if target_id in self.objectives and source_type in RED_TYPES:
                    self.objectives[target_id]["hit_events"].append(event)
                    self.units[source_id]["objective_hits"].append(event)
                    self.damage_events.append(event)
                if target_id in self.units and source_type == INTERCEPTOR_TYPE:
                    self.units[target_id]["interceptor_hits"].append(event)
            self._relation_lengths[target_id] = len(rows)

    def _sample_health_and_terminals(self, step: int, entities) -> None:
        for target_id, objective in self.objectives.items():
            health = float(entities[target_id].survivePoints)
            previous = self._previous_health[target_id]
            if health != previous:
                objective["health_changes"].append({
                    "step": int(step), "before": previous, "after": health,
                    "damage": max(0.0, previous - health),
                })
            objective["final_health"] = health
            self._previous_health[target_id] = health

        for unit_id, unit in self.units.items():
            entity = entities[unit_id]
            visible = bool(entity.isVisible)
            unit["final_position"] = _position(entity)
            if unit["terminal_step"] is None and self._previous_visible[unit_id] and not visible:
                unit["terminal_step"] = int(step)
                task_target = self._last_assignment.get(unit_id, (None, None))[1]
                unit["terminal_target_id"] = task_target
                if task_target in self.objectives:
                    horizontal, spatial = _distance(_position(entity), _position(entities[task_target]))
                    unit["terminal_distance_to_target_m"] = {
                        "horizontal": horizontal, "spatial": spatial,
                    }
            self._previous_visible[unit_id] = visible
            self._previous_health[unit_id] = float(entity.survivePoints)

    def _policy_clamps(self) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        agents = self.env.agent_manager.get_registered_agents()
        if not agents:
            return result
        state = getattr(agents[0][1], "state", None)
        rows = getattr(state, "boundary_clamp_by_target", {}) if state is not None else {}
        for raw_id, value in rows.items():
            result[int(raw_id)] = _json_value(value)
        return result

    def _finish_unit_reasons(self) -> None:
        for unit in self.units.values():
            hit_objectives = unit["objective_hits"]
            interceptor_hits = unit["interceptor_hits"]
            if unit["terminal_step"] is None:
                unit["terminal_reason"] = "survived_to_end"
            elif hit_objectives:
                unit["terminal_reason"] = "objective_impact"
            elif interceptor_hits:
                unit["terminal_reason"] = "intercepted"
            else:
                unit["terminal_reason"] = "terminated_without_objective_hit"

    def _objective_result(self, target_id: int, clamps) -> dict[str, Any]:
        objective = self.objectives[target_id]
        assignments = [
            event for event in self.assignment_events if event["target_id"] == target_id
        ]
        assigned_ids = sorted({event["unit_id"] for event in assignments})
        assigned_units = [self.units[unit_id] for unit_id in assigned_ids]
        hit_events = objective["hit_events"]
        damage = objective["initial_health"] - objective["final_health"]
        destroyed = objective["final_health"] <= 0.0
        clamp = clamps.get(target_id, {})
        arrived = [
            unit["id"] for unit in assigned_units
            if unit["minimum_distance_by_target"].get(str(target_id), {}).get("spatial_m", math.inf)
            <= DAMAGE_RADIUS_M
        ]
        intercepted = [unit["id"] for unit in assigned_units if unit["terminal_reason"] == "intercepted"]

        if destroyed:
            primary = "destroyed"
        elif objective["discovered_step"] is None:
            primary = "undiscovered_or_uncovered"
        elif not assignments:
            primary = "discovered_but_not_dispatched"
        elif damage > 0.0:
            primary = "hit_but_insufficient_damage"
        elif hit_events:
            primary = "hit_but_role_caused_zero_damage"
        elif clamp:
            primary = "boundary_constrained_no_hit"
        elif assigned_units and len(intercepted) == len(assigned_units):
            primary = "dispatched_but_all_assigned_units_intercepted"
        elif arrived:
            primary = "arrived_but_no_effective_hit"
        else:
            primary = "dispatched_but_did_not_reach_damage_radius"

        contributing = []
        if intercepted:
            contributing.append("assigned_unit_interception")
        if objective["outside_map_distance_m"] > 0.0:
            contributing.append("target_outside_map_boundary")
        if clamp:
            contributing.append("submitted_waypoint_projection")
        if damage > 0.0 and not destroyed:
            contributing.append("insufficient_accumulated_damage")
        if objective["discovered_step"] is None:
            contributing.append("no_legal_red_detection")

        result = deepcopy(objective)
        result.update({
            "destroyed": destroyed,
            "missing_score_weight": 0.0 if destroyed else objective["weight"],
            "damage_delivered": damage,
            "assignment_count": len(assignments),
            "assigned_unit_ids": assigned_ids,
            "assigned_roles": dict(Counter(self.units[unit_id]["role"] for unit_id in assigned_ids)),
            "intercepted_assigned_unit_ids": intercepted,
            "arrived_within_100m_unit_ids": arrived,
            "boundary_clamp": clamp,
            "primary_cause": primary,
            "contributing_causes": contributing,
        })
        return result

    def finish(self, termination_reason: str) -> Path:
        if self._finished:
            return self.output_dir / "attribution.json"
        self._finished = True
        self._finish_unit_reasons()
        clamps = self._policy_clamps()
        objectives = [self._objective_result(target_id, clamps) for target_id in sorted(self.objectives)]
        missed = [row for row in objectives if not row["destroyed"]]
        causes = Counter(row["primary_cause"] for row in missed)
        missing_by_kind = Counter()
        for row in missed:
            missing_by_kind[row["kind"]] += row["missing_score_weight"]
        payload = {
            "status": "complete",
            "map": self.map_name,
            "seed": self.seed,
            "steps": int(self.env.current_step),
            "termination_reason": termination_reason,
            "truth_scope": {
                "mode": "trusted_post_hoc_evaluator",
                "fed_back_to_red_policy": False,
                "red_runtime_observation_unchanged": True,
                "note": "Engine health and hit relations are used only after actions for attribution.",
            },
            "summary": {
                "objective_count": len(objectives),
                "destroyed_count": len(objectives) - len(missed),
                "missed_count": len(missed),
                "missing_weight": sum(row["missing_score_weight"] for row in missed),
                "missing_weight_by_kind": dict(missing_by_kind),
                "missed_primary_causes": dict(causes),
                "red_terminal_reasons": dict(Counter(unit["terminal_reason"] for unit in self.units.values())),
            },
            "objectives": objectives,
            "units": [self.units[unit_id] for unit_id in sorted(self.units)],
            "assignment_events": self.assignment_events,
            "discovery_events": self.discovery_events,
            "damage_events": self.damage_events,
        }
        return self._write(payload)

    def abort(self, error: BaseException) -> Path | None:
        if self._finished:
            return None
        self._finished = True
        return self._write({
            "status": "failed", "map": self.map_name, "seed": self.seed,
            "steps": int(self.env.current_step), "error": repr(error),
            "truth_scope": {"fed_back_to_red_policy": False},
        })

    def _write(self, payload: dict[str, Any]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "attribution.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(_json_value(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path
