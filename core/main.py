import argparse
import copy
import logging
import os
import random
import subprocess
import time
import json
import sys
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import requests
import numpy as np
from urllib.parse import urlparse
from envengine import Profile, TrainingEnv
from envengine.sdk.log import LogManager
from envengine.sdk.writer import WriteConfig, init_writer, get_writer, write_immediately
from user_agents import AttackMissileAgent, DeployAgent
from evaluation import RunSummary
from experiments.unified_mappo.counterfactual_replay import (
    TRAJECTORY_REWARD_MODE,
    run_counterfactual_replays,
)

try:
    from scenarios.cases import RewardTracker, load_reward_policy
except ImportError:
    RewardTracker = load_reward_policy = None

from policies.red import (
    Position,
    RED_POLICY_CHOICES,
    build_red_commander,
    initial_targets_from_observation,
)
from policies.red.learning import (
    UNIFIED_LOCAL_OBSERVATION_DIM,
    build_learning_motion_policy,
)
from policies.red.paos_commander import validate_paos_process_rounds


def _native_target_id(commander, command: dict) -> int | None:
    target = command.get("target")
    if not target:
        return None
    candidates = tuple(getattr(commander, "catalogue_targets", ()))
    if not candidates:
        return None
    selected = min(
        candidates,
        key=lambda item: math.hypot(
            float(target["x"]) - float(item.position.lon),
            float(target["y"]) - float(item.position.lat),
        ),
    )
    distance = math.hypot(
        float(target["x"]) - float(selected.position.lon),
        float(target["y"]) - float(selected.position.lat),
    )
    return int(selected.entity_id) if distance < 1e-4 else None


def _sync_student_to_native_prefix(
    training_env,
    commander,
    prefix_steps: list[dict],
) -> None:
    learning_policy = getattr(training_env, "_shared_learning_policy", None)
    sync_external_target = getattr(
        learning_policy,
        "sync_external_target_assignment",
        None,
    )
    agents = {
        int(agent.entity_id): agent
        for agent in training_env.agent_manager.get_all_agents()
        if int(agent.entity_id) >= 0
    }
    for step_row in prefix_steps:
        step = int(step_row["step"])
        for command in step_row["actions"]:
            entity_id = int(command.get("executor_id", -1))
            agent = agents.get(entity_id)
            if agent is None:
                continue
            command_type = int(command.get("commandType_id", -1))
            if command_type == 200:
                agent.launch_step = step - 1
                commander.launched_ids.add(entity_id)
            if command_type in {200, 3014}:
                target_id = _native_target_id(commander, command)
                if target_id is not None:
                    commander.target_by_platform[entity_id] = target_id
                    if callable(sync_external_target):
                        sync_external_target(entity_id, target_id)
            elif command_type == 3007:
                acceleration = float(command.get("acc_z", 0.0))
                agent.last_learning_maneuver = (
                    -1 if acceleration < 0.0 else 1 if acceleration > 0.0 else 0
                )
            elif command_type == 3013:
                agent.sat_used = True


def _sync_native_target_reservations(
    training_env,
    commander,
    actions: list[dict],
    *,
    excluded_entity_ids: set[int],
) -> None:
    """Expose fixed same-step peer commitments to the target allocator."""

    learning_policy = getattr(training_env, "_shared_learning_policy", None)
    sync_external_target = getattr(
        learning_policy,
        "sync_external_target_assignment",
        None,
    )
    if not callable(sync_external_target):
        return
    for command in actions:
        entity_id = int(command.get("executor_id", -1))
        if entity_id in excluded_entity_ids:
            continue
        if int(command.get("commandType_id", -1)) not in {200, 3014}:
            continue
        target_id = _native_target_id(commander, command)
        if target_id is not None:
            sync_external_target(entity_id, target_id)


def _has_9500_detection(observation: dict) -> bool:
    for entity in observation.get("entities", {}).values():
        if int(entity.get("side", -1)) != 0:
            continue
        for raw_id, track in (entity.get("detectInfo") or {}).items():
            target_id = int(
                track.get("entity_id", raw_id)
                if isinstance(track, dict)
                else getattr(track, "entity_id", raw_id)
            )
            if target_id in {168, 169}:
                return True
    return False


def _native_causal_null_actions(
    actions: list[dict],
    interventions: list[dict],
    timestep: int,
) -> tuple[list[dict], list[dict]]:
    requested = {
        (
            int(row["timestep"]),
            int(row["executor_id"]),
            int(row["command_type"]),
        )
        for row in interventions
    }
    kept: list[dict] = []
    applied: list[dict] = []
    for command in actions:
        key = (
            timestep,
            int(command.get("executor_id", -1)),
            int(command.get("commandType_id", -1)),
        )
        if key in requested:
            applied.append({
                "timestep": key[0],
                "executor_id": key[1],
                "command_type": key[2],
            })
        else:
            kept.append(command)
    return kept, applied


def _native_handoff_override_actions(
    actions: list[dict],
    overrides: list[dict],
    timestep: int,
    excluded_unit_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    current = [
        row for row in overrides if int(row["timestep"]) == int(timestep)
    ]
    replaced = {
        (int(row["executor_id"]), int(command_type))
        for row in current
        for command_type in row.get(
            "teacher_command_types",
            (int(row["teacher_command_type"]),),
        )
    }
    result = [
        copy.deepcopy(command)
        for command in actions
        if (
            int(command.get("executor_id", -1)),
            int(command.get("commandType_id", -1)),
        ) not in replaced
    ]
    applied = []
    for row in current:
        unit_id = str(row["unit_id"])
        if unit_id in excluded_unit_ids:
            result.extend(copy.deepcopy(
                row.get("counterfactual_actions", ())
            ))
            applied.append({
                "unit_id": unit_id,
                "timestep": int(row["timestep"]),
                "executor_id": int(row["executor_id"]),
                "command_type": int(row["teacher_command_type"]),
            })
            continue
        result.extend(copy.deepcopy(row["actions"]))
    return result, applied


def _native_causal_branch_result(
    *,
    run_summary,
    training_env,
    final_observation: dict,
    termination_reason: str,
    native_trace: dict,
    target_id: int,
    executor_id: int,
    prefix_target_agent_rewards: dict[int, dict[int, float]],
    role: str,
    branch_step: int,
    max_steps: int,
    applied_interventions: list[dict],
) -> dict:
    launched_ids = {
        int(command["executor_id"])
        for row in native_trace["steps"]
        for command in row["actions"]
        if int(command.get("commandType_id", -1)) == 200
    }
    summary = run_summary.build(
        final_observation,
        termination_reason=termination_reason,
        red_launched=len(launched_ids),
    )
    _attach_l_resource_outcomes(summary, training_env)
    objective = next(
        row for row in summary["objectives"]
        if int(row["id"]) == target_id
    )
    initial_health = float(objective["initial_health"])
    final_health = float(objective["final_health"])
    damage_ratio = min(
        1.0,
        max(0.0, (initial_health - final_health) / initial_health),
    )
    weight = float(summary["score"]["objective_weights"][str(target_id)])
    target_return = weight * damage_ratio / 21.0
    executor_agent_id = next(
        (
            int(agent.agent_id)
            for agent in training_env.agent_manager.get_all_agents()
            if int(agent.entity_id) == int(executor_id)
        ),
        None,
    )
    if executor_agent_id is None:
        raise RuntimeError(f"反事实执行实体 {executor_id} 没有对应 agent_id")
    cumulative_direct = float(
        training_env.episode_target_agent_rewards.get(target_id, {}).get(
            executor_agent_id, 0.0
        )
    )
    prefix_direct = float(
        prefix_target_agent_rewards.get(target_id, {}).get(
            executor_agent_id, 0.0
        )
    )
    executor_target_return = max(0.0, cumulative_direct - prefix_direct)
    detection = summary.get("red", {}).get("detection", {})
    first_step_by_target = {
        int(target_id): int(step)
        for target_id, step in detection.get(
            "first_9500_step_by_target", {}
        ).items()
    }
    first_source_by_target = {
        int(target_id): int(source_id)
        for target_id, source_id in detection.get(
            "first_9500_source_by_target", {}
        ).items()
    }
    team_first_search_discoveries = sorted(
        (
            target_id,
            first_step_by_target[target_id],
        )
        for target_id, source_id in first_source_by_target.items()
        if source_id == int(executor_id)
        and first_step_by_target.get(target_id, -1) >= int(branch_step)
    )
    executor_simulator = next(
        (
            simulator
            for simulator in training_env.engine.simulator_factory
            .get_simulators_by_type(21002)
            if int(simulator.entity_ext.entity.id) == int(executor_id)
        ),
        None,
    )
    direct_search_discoveries = sorted(
        (
            int(target_id),
            int(step),
        )
        for target_id, step in getattr(
            executor_simulator,
            "direct_9500_first_detection_step",
            {},
        ).items()
        if int(step) >= int(branch_step)
    )
    remaining_steps = max(1, int(max_steps) - int(branch_step))
    def timed_detection_return(discoveries) -> float:
        return sum(
            (
                1.0
                + 0.25
                * max(
                    0.0,
                    min(
                        1.0,
                        (int(max_steps) - int(discovery_step))
                        / remaining_steps,
                    ),
                )
            )
            / 9.0
            for _, discovery_step in discoveries
        )

    search_direct_detection_return = timed_detection_return(
        direct_search_discoveries
    )
    first_distance_m_by_target = {
        int(target_id): float(distance_m)
        for target_id, distance_m in getattr(
            executor_simulator,
            "direct_9500_first_distance_m",
            {},
        ).items()
    }
    min_distance_m_by_target = {
        int(target_id): float(distance_m)
        for target_id, distance_m in getattr(
            executor_simulator,
            "direct_9500_min_distance_m",
            {},
        ).items()
    }
    min_alive_distance_m_by_target = {
        int(target_id): float(distance_m)
        for target_id, distance_m in getattr(
            executor_simulator,
            "direct_9500_min_alive_distance_m",
            {},
        ).items()
    }
    search_nearest_alive_distance_m = (
        min(min_alive_distance_m_by_target.values())
        if min_alive_distance_m_by_target else None
    )
    # Reward the route only for approaching a target while that target is
    # still alive.  The old average fractional progress rewarded nearly every
    # blueward route, including paths that crossed an already-destroyed ship.
    search_proximity_return = (
        0.25 * math.exp(-search_nearest_alive_distance_m / 100_000.0)
        if search_nearest_alive_distance_m is not None else 0.0
    )
    search_discovery_return = (
        search_direct_detection_return + search_proximity_return
    )
    team_first_search_discovery_return = timed_detection_return(
        team_first_search_discoveries
    )
    return {
        "role": role,
        "branch_step": branch_step,
        "target_id": target_id,
        "score": float(summary["score"]["score"]),
        "destroyed_ids": list(map(int, summary["score"]["destroyed_ids"])),
        "target_initial_health": initial_health,
        "target_final_health": final_health,
        "target_damage_ratio": damage_ratio,
        "target_return": target_return,
        "target_score": 100.0 * target_return,
        "executor_id": int(executor_id),
        "executor_agent_id": executor_agent_id,
        "executor_target_return": executor_target_return,
        "executor_target_score": 100.0 * executor_target_return,
        "search_discovery_ids": [
            int(target_id) for target_id, _ in direct_search_discoveries
        ],
        "search_discovery_steps": {
            str(target_id): int(step)
            for target_id, step in direct_search_discoveries
        },
        "search_discovery_count": len(direct_search_discoveries),
        "search_discovery_return": search_discovery_return,
        "search_direct_detection_return": search_direct_detection_return,
        "search_proximity_return": search_proximity_return,
        "search_nearest_alive_distance_m": (
            search_nearest_alive_distance_m
        ),
        "search_executor_simulator_found": executor_simulator is not None,
        "search_executor_final_visible": (
            bool(executor_simulator.entity_ext.entity.isVisible)
            if executor_simulator is not None else False
        ),
        "search_executor_final_health": (
            float(executor_simulator.entity_ext.entity.survivePoints)
            if executor_simulator is not None else 0.0
        ),
        "search_detection_sample_count": int(getattr(
            executor_simulator,
            "direct_9500_detection_sample_count",
            0,
        )),
        "search_last_detection_step": int(getattr(
            executor_simulator,
            "direct_9500_last_detection_step",
            -1,
        )),
        "search_executor_termination_reason": getattr(
            executor_simulator, "search_termination_reason", None
        ),
        "search_executor_termination_step": int(getattr(
            executor_simulator, "search_termination_step", -1
        )),
        "search_executor_termination_distance_m": getattr(
            executor_simulator, "search_termination_distance_m", None
        ),
        "search_executor_destroyed_by_entity_id": getattr(
            executor_simulator, "search_destroyed_by_entity_id", None
        ),
        "search_executor_destroyed_by_entity_type": getattr(
            executor_simulator, "search_destroyed_by_entity_type", None
        ),
        "search_min_distance_m_by_target": {
            str(target_id): distance_m
            for target_id, distance_m in min_distance_m_by_target.items()
        },
        "search_min_alive_distance_m_by_target": {
            str(target_id): distance_m
            for target_id, distance_m in min_alive_distance_m_by_target.items()
        },
        "search_monitor_samples": list(getattr(
            executor_simulator,
            "direct_9500_monitor_samples",
            (),
        )),
        "team_first_search_discovery_ids": [
            int(target_id)
            for target_id, _ in team_first_search_discoveries
        ],
        "team_first_search_discovery_steps": {
            str(target_id): int(step)
            for target_id, step in team_first_search_discoveries
        },
        "team_first_search_discovery_return": (
            team_first_search_discovery_return
        ),
        "applied_interventions": applied_interventions,
    }


def _rasterize_l_sensor_coverage(
    simulators,
    bounds,
    *,
    grid_width: int = 64,
    grid_height: int = 48,
    sensor_radius_m: float = 30_000.0,
) -> dict:
    """Rasterize true L sensor footprints without reading objective positions."""

    if bounds is None:
        return {
            "grid_width": int(grid_width),
            "grid_height": int(grid_height),
            "covered_cell_count": 0,
            "coverage_fraction": 0.0,
            "duplicate_cell_count": 0,
            "duplicate_coverage_ratio": 0.0,
            "first_step_by_entity_and_cell": {},
        }
    lon_min, lat_min, lon_max, lat_max = map(float, bounds)
    if not (lon_max > lon_min and lat_max > lat_min):
        raise ValueError(f"非法搜索覆盖边界: {bounds}")
    width = max(1, int(grid_width))
    height = max(1, int(grid_height))
    radius_m = max(0.0, float(sensor_radius_m))
    lon_step = (lon_max - lon_min) / width
    lat_step = (lat_max - lat_min) / height
    cells_by_entity: dict[int, set[int]] = {}
    first_step_by_entity: dict[int, dict[int, int]] = {}

    for simulator in simulators:
        entity_id = int(simulator.entity_ext.entity.id)
        covered = cells_by_entity.setdefault(entity_id, set())
        first_steps = first_step_by_entity.setdefault(entity_id, {})
        for sample in getattr(simulator, "sensor_footprint_samples", ()):
            lon = float(sample["lon"])
            lat = float(sample["lat"])
            alt = abs(float(sample.get("alt", 0.0)))
            step = int(sample["step"])
            if not all(math.isfinite(value) for value in (lon, lat, alt)):
                continue
            horizontal_radius_m = math.sqrt(max(0.0, radius_m ** 2 - alt ** 2))
            lat_radius = horizontal_radius_m / 110_570.0
            lon_scale = max(1e-6, 111_320.0 * abs(math.cos(math.radians(lat))))
            lon_radius = horizontal_radius_m / lon_scale
            col_start = max(0, int(math.floor((lon - lon_radius - lon_min) / lon_step)))
            col_end = min(width - 1, int(math.floor((lon + lon_radius - lon_min) / lon_step)))
            row_start = max(0, int(math.floor((lat - lat_radius - lat_min) / lat_step)))
            row_end = min(height - 1, int(math.floor((lat + lat_radius - lat_min) / lat_step)))
            for row in range(row_start, row_end + 1):
                cell_lat = lat_min + (row + 0.5) * lat_step
                north_m = (cell_lat - lat) * 110_570.0
                for col in range(col_start, col_end + 1):
                    cell_lon = lon_min + (col + 0.5) * lon_step
                    mean_lat = math.radians((cell_lat + lat) / 2.0)
                    east_m = (cell_lon - lon) * 111_320.0 * math.cos(mean_lat)
                    if east_m ** 2 + north_m ** 2 + alt ** 2 > radius_m ** 2:
                        continue
                    cell = row * width + col
                    covered.add(cell)
                    previous = first_steps.get(cell)
                    if previous is None or step < previous:
                        first_steps[cell] = step

    cover_count_by_cell: dict[int, int] = {}
    for cells in cells_by_entity.values():
        for cell in cells:
            cover_count_by_cell[cell] = cover_count_by_cell.get(cell, 0) + 1
    covered_cell_count = len(cover_count_by_cell)
    total_entity_cell_visits = sum(cover_count_by_cell.values())
    duplicate_count = sum(max(0, count - 1) for count in cover_count_by_cell.values())
    return {
        "grid_width": width,
        "grid_height": height,
        "covered_cell_count": covered_cell_count,
        "coverage_fraction": covered_cell_count / (width * height),
        "duplicate_cell_count": duplicate_count,
        "duplicate_coverage_ratio": (
            duplicate_count / total_entity_cell_visits
            if total_entity_cell_visits else 0.0
        ),
        "first_step_by_entity_and_cell": {
            str(entity_id): {
                str(cell): step for cell, step in sorted(first_steps.items())
            }
            for entity_id, first_steps in sorted(first_step_by_entity.items())
            if first_steps
        },
    }


def _team_detection_auc(
    direct_first_steps_by_entity,
    objective_ids,
    max_steps: int,
) -> float:
    """Area under the cumulative legal-detection curve, normalized to [0, 1]."""

    targets = {int(target_id) for target_id in objective_ids}
    if not targets:
        return 1.0
    horizon = max(1, int(max_steps))
    earliest: dict[int, int] = {}
    for target_steps in direct_first_steps_by_entity.values():
        for raw_target_id, raw_step in target_steps.items():
            target_id = int(raw_target_id)
            if target_id not in targets:
                continue
            step = min(horizon, max(0, int(raw_step)))
            earliest[target_id] = min(step, earliest.get(target_id, step))
    return sum(
        (horizon - earliest[target_id]) / horizon
        for target_id in targets
        if target_id in earliest
    ) / len(targets)


def _attach_l_resource_outcomes(summary: dict, training_env) -> None:
    """Attach a terminal audit for every L without exposing it to the Actor."""

    factory = training_env.engine.simulator_factory
    simulators = list(factory.get_simulators_by_type(21002))
    l_ids = {int(sim.entity_ext.entity.id) for sim in simulators}
    launched_ids = {
        int(sim.entity_ext.entity.id)
        for sim in simulators
        if int(getattr(sim, "launch", -1)) >= 0
    }
    events = tuple(factory.causal_event_ledger)
    intercepted_ids = {
        int(event["target_entity_id"])
        for event in events
        if str(event.get("event_type")) == "direct_hit"
        and int(event.get("attacking_entity_type", -1)) == 24000
        and int(event.get("target_entity_type", -1)) == 21002
        and float(event.get("actual_damage", 0.0)) > 0.0
    } & l_ids
    hit_9500_by_entity: dict[int, set[int]] = {}
    for event in events:
        if (
            str(event.get("event_type")) != "direct_hit"
            or int(event.get("attacking_entity_type", -1)) != 21002
            or int(event.get("target_entity_type", -1)) != 9500
            or float(event.get("actual_damage", 0.0)) <= 0.0
        ):
            continue
        attacker_id = int(event["attacking_entity_id"])
        if attacker_id in l_ids:
            hit_9500_by_entity.setdefault(attacker_id, set()).add(
                int(event["target_entity_id"])
            )
    hitter_ids = set(hit_9500_by_entity)
    termination_reason_by_entity = {
        int(sim.entity_ext.entity.id): getattr(
            sim, "search_termination_reason", None
        )
        for sim in simulators
    }
    timeout_ids = {
        entity_id
        for entity_id, reason in termination_reason_by_entity.items()
        if reason == "timeout"
    }
    completion_ids = {
        entity_id
        for entity_id, reason in termination_reason_by_entity.items()
        if reason in {
            "waypoint_completion",
            "model_completion",
            "model_completion_away_from_waypoint",
        }
    }
    completion_without_hit_ids = completion_ids - hitter_ids
    nonintercepted_ids = launched_ids - intercepted_ids
    nonintercepted_nonhitter_ids = nonintercepted_ids - hitter_ids

    direct_first_steps_by_entity = {
        int(sim.entity_ext.entity.id): {
            int(target_id): int(step)
            for target_id, step in getattr(
                sim, "direct_9500_first_detection_step", {}
            ).items()
        }
        for sim in simulators
        if getattr(sim, "direct_9500_first_detection_step", {})
    }
    discovered_by_l = {
        int(target_id)
        for raw_source_id, target_steps in direct_first_steps_by_entity.items()
        if int(raw_source_id) in l_ids
        for target_id in target_steps
    }
    discovered_by_explorer = {
        int(target_id)
        for raw_source_id, target_steps in direct_first_steps_by_entity.items()
        if int(raw_source_id) in nonintercepted_nonhitter_ids
        for target_id in target_steps
    }
    detection = summary.get("red", {}).get("detection", {})
    cluster_targets_by_source = detection.get("9500_targets_by_source", {})
    cluster_discovered_by_l = {
        int(target_id)
        for raw_source_id, target_ids in cluster_targets_by_source.items()
        if int(raw_source_id) in l_ids
        for target_id in target_ids
    }
    objective_9500_ids = {
        int(row["id"])
        for row in summary.get("objectives", ())
        if int(row.get("type", -1)) == 9500
    }
    destroyed_9500_ids = {
        int(row["id"])
        for row in summary.get("objectives", ())
        if int(row.get("type", -1)) == 9500
        and float(row.get("final_health", 0.0)) <= 0.0
    }
    required_hits = len(objective_9500_ids)
    hit_target_ids = {
        target_id
        for target_ids in hit_9500_by_entity.values()
        for target_id in target_ids
    }
    sensor_coverage = _rasterize_l_sensor_coverage(
        simulators,
        getattr(training_env, "search_coverage_bounds", None),
    )
    detection_auc = _team_detection_auc(
        direct_first_steps_by_entity,
        objective_9500_ids,
        getattr(training_env, "search_coverage_max_steps", 3000),
    )
    agent_id_by_entity = {
        int(agent.entity_id): int(agent.agent_id)
        for agent in training_env.agent_manager.get_all_agents()
        if int(agent.entity_id) in l_ids
    }
    summary.setdefault("red", {})["l_resource_outcomes"] = {
        "total": len(l_ids),
        "launched": len(launched_ids),
        "intercepted": len(intercepted_ids),
        "interception_rate": (
            len(intercepted_ids) / len(launched_ids)
            if launched_ids else 0.0
        ),
        "nonintercepted": len(nonintercepted_ids),
        "timeout": len(timeout_ids & launched_ids),
        "hit_9500_entities": len(hitter_ids),
        "hit_9500_targets": len(hit_target_ids),
        "completion_without_9500_hit": len(
            completion_without_hit_ids & launched_ids
        ),
        "nonintercepted_nonhitter": len(nonintercepted_nonhitter_ids),
        "explorer_count": len(nonintercepted_nonhitter_ids),
        "required_distinct_9500_hits": required_hits,
        "search_reserve_after_required_hits": max(
            0, len(nonintercepted_ids) - required_hits
        ),
        "search_reserve_to_required_hits_ratio": (
            max(0, len(nonintercepted_ids) - required_hits) / required_hits
            if required_hits else 1.0
        ),
        "timeout_to_required_hits_ratio": (
            len(timeout_ids & launched_ids) / required_hits
            if required_hits else 1.0
        ),
        "objective_9500_count": required_hits,
        "team_detection_auc": detection_auc,
        "sensor_coverage": sensor_coverage,
        "l_discovered_9500_count": len(discovered_by_l & objective_9500_ids),
        "explorer_discovered_9500_count": len(
            discovered_by_explorer & objective_9500_ids
        ),
        "l_9500_coverage": (
            len(discovered_by_l & objective_9500_ids) / required_hits
            if required_hits else 1.0
        ),
        "explorer_9500_coverage": (
            len(discovered_by_explorer & objective_9500_ids) / required_hits
            if required_hits else 1.0
        ),
        "intercepted_ids": sorted(intercepted_ids),
        "timeout_ids": sorted(timeout_ids & launched_ids),
        "hitter_ids": sorted(hitter_ids),
        "completion_without_9500_hit_ids": sorted(
            completion_without_hit_ids & launched_ids
        ),
        "discovered_9500_ids": sorted(discovered_by_l & objective_9500_ids),
        "explorer_discovered_9500_ids": sorted(
            discovered_by_explorer & objective_9500_ids
        ),
        "direct_9500_first_step_by_entity": {
            str(entity_id): {
                str(target_id): step
                for target_id, step in sorted(target_steps.items())
                if target_id in objective_9500_ids
            }
            for entity_id, target_steps in sorted(
                direct_first_steps_by_entity.items()
            )
        },
        "cluster_source_discovered_9500_ids": sorted(
            cluster_discovered_by_l & objective_9500_ids
        ),
        "destroyed_9500_ids": sorted(destroyed_9500_ids),
        "hit_9500_targets_by_entity": {
            str(entity_id): sorted(target_ids)
            for entity_id, target_ids in sorted(hit_9500_by_entity.items())
        },
        "agent_id_by_entity": {
            str(entity_id): agent_id
            for entity_id, agent_id in sorted(agent_id_by_entity.items())
        },
    }


def _run_native_handoff_probe(
    *,
    unit: dict,
    overrides: list[dict],
    native_trace_path: Path,
    target_id: int,
    scenario: str,
    max_steps: int,
    output_dir: Path,
    counterfactual_target_id: int | None = None,
    counterfactual_sample_index: int | None = None,
) -> dict:
    unit_id = str(unit["unit_id"])
    probe_dir = output_dir / unit_id
    if (
        counterfactual_target_id is not None
        and counterfactual_sample_index is not None
    ):
        raise ValueError("target id and sampled counterfactual are mutually exclusive")
    if counterfactual_target_id is not None:
        probe_dir = probe_dir / f"target_{int(counterfactual_target_id)}"
    elif counterfactual_sample_index is not None:
        probe_dir = probe_dir / f"sample_{int(counterfactual_sample_index):02d}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    spec_path = probe_dir / "spec.json"
    result_path = probe_dir / "result.json"
    log_path = probe_dir / "run.log"
    probe_overrides = copy.deepcopy(overrides)
    sampled_target_id = None
    if counterfactual_target_id is not None:
        matched = False
        for row in probe_overrides:
            if str(row["unit_id"]) != unit_id:
                continue
            candidates = row.get("counterfactual_actions_by_target", {})
            key = str(int(counterfactual_target_id))
            if key not in candidates:
                raise ValueError(
                    f"unit {unit_id} 没有合法候选目标 {counterfactual_target_id}"
                )
            row["counterfactual_actions"] = copy.deepcopy(candidates[key])
            matched = True
        if not matched:
            raise ValueError(f"未找到 dense target unit {unit_id}")
    elif counterfactual_sample_index is not None:
        matched = False
        for row in probe_overrides:
            if str(row["unit_id"]) != unit_id:
                continue
            candidates = {
                int(candidate["sample_index"]): candidate
                for candidate in row.get("counterfactual_samples", ())
            }
            sample_index = int(counterfactual_sample_index)
            if sample_index not in candidates:
                raise ValueError(
                    f"unit {unit_id} 没有策略反事实样本 {sample_index}"
                )
            candidate = candidates[sample_index]
            row["counterfactual_actions"] = copy.deepcopy(candidate["actions"])
            sampled_target_id = int(candidate["target_id"])
            matched = True
        if not matched:
            raise ValueError(f"未找到 sampled target unit {unit_id}")
    tracked_target_id = (
        int(counterfactual_target_id)
        if counterfactual_target_id is not None
        else int(sampled_target_id)
        if sampled_target_id is not None
        else int(target_id)
    )
    spec = {
        "schema_version": 1,
        "kind": "native_mappo_handoff_unit",
        "target_id": tracked_target_id,
        "reference_target_id": int(target_id),
        "counterfactual_target_id": counterfactual_target_id,
        "counterfactual_sample_index": counterfactual_sample_index,
        "counterfactual_only": (
            counterfactual_target_id is not None
            or counterfactual_sample_index is not None
        ),
        "branch_step": int(unit["timestep"]),
        "factual_overrides": probe_overrides,
        "interventions": [{
            "unit_id": unit_id,
            "timestep": int(unit["timestep"]),
            "executor_id": int(unit["executor_id"]),
            "command_type": int(unit["command_type"]),
        }],
    }
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("RED_NATIVE_HANDOFF_SPEC", None)
    environment.update({
        "PYTHONUNBUFFERED": "1",
        # COW probes run many concurrent reference-policy replicas.  Keep
        # every child off CUDA as required by the counterfactual design and
        # avoid multiplying one GPU context per legal-target branch.
        "CUDA_VISIBLE_DEVICES": "",
        "RED_NATIVE_ROLLOUT_DEVICE": "cpu",
        "RED_UNIFIED_DEVICE": "cpu",
        "RED_POLICY": "r9_hierarchical_learning",
        "RED_MOTION_POLICY": "mappo",
        "RED_LEARNING_TRAIN": "0",
        "RED_LEARNING_MODEL": os.environ["RED_NATIVE_TEACHER_MODEL"],
        "RED_REWARD_MODE": "weighted_damage_individual",
        "RED_NATIVE_GUIDANCE_TRACE": str(native_trace_path.resolve()),
        "RED_NATIVE_GUIDANCE_STAGE": "handoff_cow_probe",
        "RED_NATIVE_CAUSAL_GROUP_SPEC": str(spec_path.resolve()),
        "RED_NATIVE_CAUSAL_GROUP_RESULT": str(result_path.resolve()),
    })
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--scenario", str(Path(scenario).resolve()),
            "--output-dir", str((probe_dir / "sim_results").resolve()),
            "--max-steps", str(max_steps),
            "--total-rounds", "1",
            "--render-mode", "none",
            "--disable-log-color",
        ],
        cwd=Path(__file__).resolve().parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"handoff COW unit {unit_id} exited {completed.returncode}; "
            f"log={log_path}"
        )
    return json.loads(result_path.read_text(encoding="utf-8"))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Simulation Training Environment')

    parser.add_argument('--scenario',
                        type=str,
                        default='./scenarios/platform.json',
                        help='Path or URL to scenario configuration file (default: ./scenarios/platform.json)')

    parser.add_argument('--total-rounds',
                        type=int,
                        default=100,
                        help='Total simulation rounds (default: 1)')

    parser.add_argument('--max-steps',
                        type=int,
                        default=1000,
                        help='Maximum steps per round (default: 1000)')

    parser.add_argument('--render-mode',
                        type=str,
                        default="human",
                        choices=['human', 'none'],
                        help='Render mode (default: human)')

    parser.add_argument('--output-dir',
                        type=str,
                        default='results',
                        help='Output directory base path (default: results)')

    parser.add_argument('--batch-size',
                        type=int,
                        default=100,
                        help='Batch size for writer (default: 100)')

    parser.add_argument('--disable-log-color',
                        action='store_false',
                        dest='enable_log_color',
                        default=True,
                        help='Disable log color')

    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Enable verbose output')

    parser.add_argument('--enable-config',
                        action='store_true',
                        default=False,
                        help='Enable config writing (default: False)')

    parser.add_argument('--enable-state',
                        action='store_true',
                        default=False,
                        help='Enable state writing (default: False)')

    parser.add_argument('--enable-event',
                        action='store_true',
                        default=False,
                        help='Enable event writing (default: False)')

    parser.add_argument('--enable-ai-action',
                        action='store_true',
                        default=False,
                        help='Enable AI action writing (default: False)')

    return parser.parse_args()


def read_profile(url: str) -> Profile:
    """
    通过本地文件url读取或者通过远程url读取
    :param url:
    :return:
    """
    # 判断是否为有效的URL（包含协议）
    parsed_url = urlparse(url)

    if parsed_url.scheme in ['http', 'https']:  # 远程文件路径
        response = requests.get(url)
        if response.status_code == 200:
            profile_data = response.json()['data']
            return Profile.from_dict(profile_data)
        else:
            raise Exception(f"无法从远程URL获取数据: {response.status_code}")
    elif os.path.exists(url):  # 本地文件路径
        with open(url, 'r', encoding='utf-8') as file:
            profile_data = json.load(file)
            return Profile.from_dict(profile_data)
    else:
        raise ValueError(f"无效的URL或文件路径: {url}")


def main():
    # 解析命令行参数
    args = parse_args()
    if os.getenv("RED_POLICY") == "r12_unified_mappo":
        os.environ.setdefault("RED_REWARD_MODE", TRAJECTORY_REWARD_MODE)
        os.environ.setdefault("RED_UNIFIED_DYNAMIC_LIFECYCLE", "1")
    simulation_seed = os.getenv("SIMULATION_SEED")
    trajectory_counterfactual = (
        os.getenv("RED_REWARD_MODE") == TRAJECTORY_REWARD_MODE
    )
    if trajectory_counterfactual and simulation_seed is None:
        simulation_seed = os.getenv("RED_POLICY_SEED", "1")
        os.environ["SIMULATION_SEED"] = simulation_seed
    if simulation_seed is not None:
        random.seed(int(simulation_seed))
        np.random.seed(int(simulation_seed))

    # 打印配置信息
    logging.info("=" * 60)
    logging.info("Simulation Configuration:")
    logging.info(f"  Scenario file: {args.scenario}")
    logging.info(f"  Total rounds: {args.total_rounds}")
    logging.info(f"  Max steps per round: {args.max_steps}")
    logging.info(f"  Render mode: {args.render_mode}")
    logging.info(f"  Output directory: {args.output_dir}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Enable gog color: {args.enable_log_color}")
    logging.info(f"  Verbose: {args.verbose}")
    logging.info(f"  Enable state: {args.enable_state}")
    logging.info(f"  Enable event: {args.enable_event}")
    logging.info(f"  Simulation seed: {simulation_seed}")
    logging.info(f"  Enable AI action: {args.enable_ai_action}")

    logging.info("=" * 60)

    # 全局写文件配置
    write_config = WriteConfig(
        output_dir=f"{args.output_dir}/" + datetime.now().strftime("%Y%m%d%H%M%S"),
        verbose=args.verbose,
        batch_size=args.batch_size,
        enable_config=args.enable_config,
        enable_state=args.enable_state,
        enable_event=args.enable_event,
        enable_ai_action=args.enable_ai_action
    )

    # 初始化日志管理器
    LogManager(color_enabled=args.enable_log_color)

    # 初始化写入器
    init_writer(write_config)

    # 加载配置
    profile: Profile = read_profile(args.scenario)
    reward_policy = load_reward_policy(args.scenario) if load_reward_policy is not None else None
    if reward_policy is not None:
        os.environ["BLUE_ASSET_VALUES"] = json.dumps(
            dict(reward_policy.objective_weights)
        )
    # print(profile)
    # 实例化训练环境
    training_env = TrainingEnv(profile, render_mode=args.render_mode)
    # 注册智能体, 为每个实体创建一个智能体
    simulators = training_env.engine.simulator_factory.get_all_simulators()

    # 初始化需要给红方 AI 的信息
    init_observation_ship = training_env._get_init_ship_observation()
    red_policy = os.getenv("RED_POLICY", "r0_random")
    if red_policy not in RED_POLICY_CHOICES:
        raise ValueError(f"Unsupported red policy '{red_policy}'")
    validate_paos_process_rounds(red_policy, args.total_rounds)
    top_model = os.getenv("RED_TOP_MODEL")
    top_training = os.getenv("RED_TOP_TRAIN", "0") == "1"
    top_stage = os.getenv("RED_TOP_STAGE", "double_q")
    if red_policy == "r10_bc_erca" and not top_model:
        raise ValueError("r10_bc_erca requires RED_TOP_MODEL")
    if red_policy == "r11_paos" and not top_model:
        raise ValueError("r11_paos requires RED_TOP_MODEL")
    learning_model = os.getenv("RED_LEARNING_MODEL")
    map_area = profile.imagineProfile.mapArea
    training_env.search_coverage_bounds = (
        map_area.lonMin, map_area.latMin, map_area.lonMax, map_area.latMax
    )
    training_env.search_coverage_max_steps = int(args.max_steps)
    commander = build_red_commander(
        initial_targets_from_observation(init_observation_ship),
        policy_name=red_policy,
        seed=int(os.getenv("RED_POLICY_SEED", "1")),
        search_polygon=(
            Position(map_area.lonMin, map_area.latMin),
            Position(map_area.lonMax, map_area.latMin),
            Position(map_area.lonMax, map_area.latMax),
            Position(map_area.lonMin, map_area.latMax),
        ),
        top_model=top_model,
        top_training=top_training,
        top_stage=top_stage,
        top_max_steps=args.max_steps,
        paos_mode=os.getenv("RED_PAOS_MODE", "rollout"),
        paos_request=os.getenv("RED_PAOS_REQUEST"),
        bottom_model=learning_model,
    )
    red_motion_policy = os.getenv("RED_MOTION_POLICY", "reactive_evasion")
    hierarchical_learning = red_policy in {
        "r9_hierarchical_learning",
        "r10_bc_erca",
        "r11_paos",
        "r12_unified_mappo",
    }
    if hierarchical_learning and red_motion_policy not in {
        "random_masked", "ppo_baseline", "ppo_custom", "ppo", "mappo", "unified_mappo"
    }:
        raise ValueError(
            f"{red_policy} requires RED_MOTION_POLICY to be "
            "random_masked, ppo_baseline, ppo_custom, ppo, mappo, or unified_mappo"
        )
    learning_training = os.getenv("RED_LEARNING_TRAIN", "0") == "1"
    if red_policy in {"r10_bc_erca", "r11_paos"} and learning_training:
        raise ValueError(f"{red_policy} freezes the low-level motion policy")
    if red_policy == "r12_unified_mappo" and red_motion_policy != "unified_mappo":
        raise ValueError("r12_unified_mappo requires RED_MOTION_POLICY=unified_mappo")
    if red_policy == "r11_paos" and red_motion_policy != "ppo_custom":
        raise ValueError("r11_paos requires RED_MOTION_POLICY=ppo_custom")
    if red_policy == "r11_paos" and not learning_model:
        raise ValueError("r11_paos requires RED_LEARNING_MODEL")
    learning_policy = None
    if red_motion_policy in {
        "random_masked", "ppo_baseline", "ppo_custom", "ppo", "mappo", "unified_mappo"
    }:
        learning_policy = build_learning_motion_policy(
            red_motion_policy,
            seed=int(os.getenv("RED_POLICY_SEED", "1")),
            model_path=learning_model,
            max_steps=args.max_steps,
            training=learning_training,
            observation_dim=(
                UNIFIED_LOCAL_OBSERVATION_DIM if red_policy == "r12_unified_mappo"
                else 90 if hierarchical_learning else 85
            ),
        )
        if learning_training or red_policy == "r12_unified_mappo":
            training_env.set_shared_learning_policy(learning_policy)
    if red_policy == "r12_unified_mappo":
        if learning_policy is None:
            raise RuntimeError("r12_unified_mappo did not construct a shared policy")
        commander.attach_policy(learning_policy)
    counterfactual_probe = (
        trajectory_counterfactual
        and os.getenv("RED_TRAJECTORY_CF_PROBE", "0") == "1"
    )
    counterfactual_stop_step = int(
        os.getenv("RED_CF_STOP_STEP", str(args.max_steps))
    )
    if trajectory_counterfactual and red_policy != "r12_unified_mappo":
        raise ValueError("轨迹反事实奖励模式仅支持 r12_unified_mappo")
    if counterfactual_probe:
        single_request = bool(
            os.getenv("RED_CF_SPEC") and os.getenv("RED_CF_RESULT")
        )
        batch_request = bool(
            os.getenv("RED_CF_BATCH_SPEC")
            and os.getenv("RED_CF_BATCH_RESULT")
        )
        if not (single_request or batch_request):
            raise ValueError("反事实 probe 缺少单请求或批量请求规范")

    red_simulators = sorted(
        (
            simulator for simulator in simulators
            if int(simulator.entity_ext.entity.entityType) in {21000, 21001, 21002}
        ),
        key=lambda simulator: int(simulator.entity_ext.entity.id),
    )
    for agent_id, simulator in enumerate(red_simulators, start=1):
        entity_id = simulator.entity_ext.entity.id
        # 按类型注册红方飞行器智能体
        if simulator.entity_ext.entity.entityType == 21000 or simulator.entity_ext.entity.entityType == 21001 or simulator.entity_ext.entity.entityType == 21002:
            agent = AttackMissileAgent(
                agent_id,
                entity_id,
                init_observation_ship,
                commander=commander,
                motion_policy=red_motion_policy,
                learning_policy=learning_policy,
                learning_max_steps=args.max_steps,
                hierarchical_learning=hierarchical_learning,
            )
            register_entity_type = getattr(
                learning_policy,
                "register_entity_type",
                None,
            )
            if callable(register_entity_type):
                register_entity_type(
                    entity_id,
                    int(simulator.entity_ext.entity.entityType),
                )
            training_env.agent_manager.register_agent(agent)
            commander.register_platform(entity_id)
    logging.info(f"[测试] 已注册 {training_env.agent_manager.get_agent_count()} 个智能体")

    # 注册部署智能体
    deploy_agent = DeployAgent(
        -1,
        -1,
        {},
        profile.imagineProfile.redArea.coordinates,
        profile.imagineProfile.redArea.coordinatesHM,
        unified_policy=learning_policy if red_policy == "r12_unified_mappo" else None,
    )
    training_env.agent_manager.register_agent(deploy_agent)

    logging.info("[测试] 运行环境初始化完成")

    for i in range(args.total_rounds):
        if trajectory_counterfactual and simulation_seed is not None:
            round_seed = int(simulation_seed) + i
            random.seed(round_seed)
            np.random.seed(round_seed)
            os.environ["SIMULATION_SEED"] = str(round_seed)

        frozen_counterfactual_checkpoint = None
        if trajectory_counterfactual and learning_training and not counterfactual_probe:
            frozen_counterfactual_checkpoint = (
                Path(write_config.output_dir)
                / "trajectory_credit"
                / f"frozen_policy_round_{i + 1}.pt"
            )
            learning_policy.save(frozen_counterfactual_checkpoint)

        logging.info(f"[测试] 运行第 {i + 1} 轮")
        commander.reset()
        initial_observation = training_env.reset()
        run_summary = RunSummary(
            scenario=reward_policy.scenario_id if reward_policy is not None else str(args.scenario),
            policies={
                "red": red_policy,
                "red_motion": red_motion_policy,
                "red_top": (
                    "bc_erca_top" if red_policy == "r10_bc_erca"
                    else "paos_top" if red_policy == "r11_paos"
                    else "unified_hybrid_mappo" if red_policy == "r12_unified_mappo"
                    else "none"
                ),
                "blue": os.getenv("BLUE_POLICY", "engine_default"),
            },
            reward_tracker=RewardTracker(reward_policy) if reward_policy is not None else None,
        )
        run_summary.start(initial_observation)
        final_observation = initial_observation
        termination_reason = "time_limit"
        # Opt-in C0a b9 path: symmetric candidate branches with a closed-loop
        # frozen teacher. Existing curricula retain their original entry path.
        if os.getenv("RED_C0A_B9_REQUEST"):
            from c0a_goal_b9_runtime import run_request

            run_request(
                training_env, commander, learning_policy, run_summary, args,
                _sync_student_to_native_prefix, _sync_native_target_reservations,
            )
            training_env.close()
            get_writer().close()
            return
        native_guidance_trace = os.getenv("RED_NATIVE_GUIDANCE_TRACE")
        native_guidance_stage = os.getenv("RED_NATIVE_GUIDANCE_STAGE", "")
        native_causal_spec_path = os.getenv("RED_NATIVE_CAUSAL_GROUP_SPEC")
        native_causal_result_path = os.getenv("RED_NATIVE_CAUSAL_GROUP_RESULT")
        native_handoff_spec_path = os.getenv("RED_NATIVE_HANDOFF_SPEC")
        native_handoff_spec = None
        native_handoff_overrides: list[dict] = []
        native_handoff_credit = None
        native_distillation_manifest = None
        native_guidance_prefix_step = 0
        student_start_step = 1
        if native_guidance_trace:
            native_trace = json.loads(
                Path(native_guidance_trace).read_text(encoding="utf-8")
            )
            training_env.red_model_deploy()
            if native_guidance_stage == "teacher_distillation":
                from experiments.unified_mappo.teacher_distillation import (
                    TeacherDatasetWriter,
                )

                training_env.red_model_deploy_from_native(
                    native_trace["deployment_actions"]
                )
                writer = TeacherDatasetWriter(
                    os.environ["RED_TEACHER_DISTILL_DATASET_DIR"],
                    seed=int(native_trace["simulation_seed"]),
                    scenario=str(args.scenario),
                    source_trace=native_guidance_trace,
                    sample_stride=int(os.getenv(
                        "RED_TEACHER_DISTILL_SAMPLE_STRIDE", "20"
                    )),
                    chunk_size=int(os.getenv(
                        "RED_TEACHER_DISTILL_CHUNK_SIZE", "32768"
                    )),
                )
                prefix_steps = []
                for row in native_trace["steps"]:
                    step = int(row["step"])
                    _sync_native_target_reservations(
                        training_env,
                        commander,
                        row["actions"],
                        excluded_entity_ids=set(),
                    )
                    batch, batch_stats = (
                        training_env.prepare_native_distillation_batch(
                            row["actions"],
                            decision_step=step,
                            sample_stride=writer.sample_stride,
                        )
                    )
                    writer.add(batch, batch_stats)
                    obs, reward, done, info = training_env.step(
                        native_actions=row["actions"]
                    )
                    _sync_student_to_native_prefix(
                        training_env,
                        commander,
                        [row],
                    )
                    native_guidance_prefix_step = step
                    final_observation = obs
                    run_summary.update(step, obs)
                    if done:
                        termination_reason = "environment_done"
                        break
                native_distillation_manifest = writer.finalize(
                    teacher_score=float(
                        native_trace["summary"]["score"]["score"]
                    )
                )
                student_start_step = args.max_steps + 1
            elif native_causal_spec_path:
                causal_spec = json.loads(
                    Path(native_causal_spec_path).read_text(encoding="utf-8")
                )
                branch_step = int(causal_spec["branch_step"])
                target_id = int(causal_spec["target_id"])
                result_path = Path(native_causal_result_path)
                result_path.parent.mkdir(parents=True, exist_ok=True)
                training_env.red_model_deploy_from_native(
                    native_trace["deployment_actions"]
                )
                for row in native_trace["steps"]:
                    step = int(row["step"])
                    if step >= branch_step:
                        break
                    actions = row["actions"]
                    if causal_spec.get("factual_overrides"):
                        actions, _ = _native_handoff_override_actions(
                            actions,
                            causal_spec["factual_overrides"],
                            step,
                            set(),
                        )
                    obs, reward, done, info = training_env.step(
                        native_actions=actions
                    )
                    final_observation = obs
                    run_summary.update(step, obs)

                prefix_target_agent_rewards = copy.deepcopy(
                    training_env.episode_target_agent_rewards
                )

                counterfactual_only = bool(
                    causal_spec.get("counterfactual_only", False)
                )
                child_pid = None if counterfactual_only else os.fork()
                role = (
                    "counterfactual"
                    if counterfactual_only or child_pid != 0
                    else "factual"
                )
                applied_interventions: list[dict] = []
                excluded_unit_ids = {
                    str(row["unit_id"])
                    for row in causal_spec.get("interventions", ())
                } if role == "counterfactual" else set()
                for row in native_trace["steps"]:
                    step = int(row["step"])
                    if step < branch_step:
                        continue
                    actions = row["actions"]
                    if causal_spec.get("factual_overrides"):
                        actions, applied_now = _native_handoff_override_actions(
                            actions,
                            causal_spec["factual_overrides"],
                            step,
                            excluded_unit_ids,
                        )
                        applied_interventions.extend(applied_now)
                    elif role == "counterfactual":
                        actions, applied_now = _native_causal_null_actions(
                            actions,
                            causal_spec["interventions"],
                            step,
                        )
                        applied_interventions.extend(applied_now)
                    obs, reward, done, info = training_env.step(
                        native_actions=actions
                    )
                    final_observation = obs
                    run_summary.update(step, obs)
                    if done:
                        termination_reason = "environment_done"
                        break

                branch_result = _native_causal_branch_result(
                    run_summary=run_summary,
                    training_env=training_env,
                    final_observation=final_observation,
                    termination_reason=termination_reason,
                    native_trace=native_trace,
                    target_id=target_id,
                    executor_id=int(
                        causal_spec["interventions"][0]["executor_id"]
                    ),
                    prefix_target_agent_rewards=prefix_target_agent_rewards,
                    role=role,
                    branch_step=branch_step,
                    max_steps=args.max_steps,
                    applied_interventions=applied_interventions,
                )
                branch_path = Path(str(result_path) + f".{role}")
                branch_path.write_text(
                    json.dumps(branch_result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if counterfactual_only:
                    causal_result = {
                        "schema_version": 1,
                        "native_copy_on_write": True,
                        "suffix_semantics": (
                            "teacher_native_action_replay_with_fixed_mappo_overrides"
                        ),
                        "request": causal_spec,
                        "counterfactual": branch_result,
                    }
                    result_path.write_text(
                        json.dumps(causal_result, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(
                        "NATIVE_CAUSAL_PROBE "
                        + json.dumps(causal_result, ensure_ascii=False),
                        flush=True,
                    )
                    training_env.close()
                    get_writer().close()
                    return
                if child_pid == 0:
                    os._exit(0)

                _, wait_status = os.waitpid(child_pid, 0)
                child_exit_code = os.waitstatus_to_exitcode(wait_status)
                if child_exit_code != 0:
                    raise RuntimeError(
                        f"教师原生反事实事实分支退出码异常: {child_exit_code}"
                    )
                factual = json.loads(
                    Path(str(result_path) + ".factual").read_text(
                        encoding="utf-8"
                    )
                )
                counterfactual = json.loads(
                    Path(str(result_path) + ".counterfactual").read_text(
                        encoding="utf-8"
                    )
                )
                causal_result = {
                    "schema_version": 1,
                    "native_copy_on_write": True,
                    "suffix_semantics": (
                        "teacher_native_action_replay_with_fixed_mappo_overrides"
                        if causal_spec.get("factual_overrides")
                        else "teacher_native_action_replay"
                    ),
                    "request": causal_spec,
                    "factual": factual,
                    "counterfactual": counterfactual,
                    "target_return_delta": max(
                        0.0,
                        float(factual["target_return"])
                        - float(counterfactual["target_return"]),
                    ),
                    "search_discovery_return_delta": max(
                        0.0,
                        float(factual["search_discovery_return"])
                        - float(counterfactual["search_discovery_return"]),
                    ),
                }
                result_path.write_text(
                    json.dumps(causal_result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    "NATIVE_CAUSAL_PROBE "
                    + json.dumps(causal_result, ensure_ascii=False),
                    flush=True,
                )
                training_env.close()
                get_writer().close()
                return
            elif native_handoff_spec_path:
                native_handoff_spec = json.loads(
                    Path(native_handoff_spec_path).read_text(encoding="utf-8")
                )
                training_env.red_model_deploy_from_native(
                    native_trace["deployment_actions"]
                )
                rollout_device = os.getenv("RED_NATIVE_ROLLOUT_DEVICE")
                if rollout_device:
                    learning_policy.trainer.move_runtime_device(
                        learning_policy.trainer.resolve_runtime_device(
                            rollout_device
                        )
                    )
                units_by_step: dict[int, list[dict]] = {}
                for unit in native_handoff_spec["controlled_units"]:
                    units_by_step.setdefault(int(unit["timestep"]), []).append(unit)
                persistent_option_control = bool(
                    native_handoff_spec.get(
                        "persistent_student_option_control", False
                    )
                )
                for row in native_trace["steps"]:
                    step = int(row["step"])
                    controlled_units = units_by_step.get(step, [])
                    active_student_options = (
                        set(learning_policy.active_student_option_entity_ids)
                        if persistent_option_control else set()
                    )
                    if controlled_units or active_student_options:
                        persistent_inference_entity_ids = None
                        if (
                            persistent_option_control
                            and str(native_handoff_spec.get(
                                "controlled_components", "joint"
                            )) == "search"
                        ):
                            persistent_inference_entity_ids = {
                                int(unit["executor_id"])
                                for unit in controlled_units
                            } | training_env.persistent_search_boundary_entity_ids(
                                active_student_options
                            )
                        _sync_native_target_reservations(
                            training_env,
                            commander,
                            row["actions"],
                            excluded_entity_ids=(
                                {
                                    int(unit["executor_id"])
                                    for unit in controlled_units
                                }
                                | active_student_options
                            ),
                        )
                        actions, student_entities, overrides = (
                            training_env.prepare_native_handoff_actions(
                                row["actions"],
                                controlled_units,
                                str(native_handoff_spec.get(
                                    "controlled_components", "joint"
                                )),
                                persistent_option_control=(
                                    persistent_option_control
                                ),
                                target_head_counterfactual=bool(
                                    native_handoff_spec.get(
                                        "target_head_counterfactual", False
                                    )
                                ),
                                dense_target_counterfactual=bool(
                                    native_handoff_spec.get(
                                        "dense_target_counterfactual", False
                                    )
                                ),
                                target_counterfactual_samples=int(
                                    native_handoff_spec.get(
                                        "target_counterfactual_samples", 0
                                    )
                                ),
                                persistent_inference_entity_ids=(
                                    persistent_inference_entity_ids
                                ),
                            )
                        )
                        native_handoff_overrides.extend(overrides)
                        obs, reward, done, info = training_env.step(
                            native_actions=actions,
                            guided_student_entity_ids=student_entities,
                        )
                    else:
                        actions = row["actions"]
                        obs, reward, done, info = training_env.step(
                            native_actions=actions
                        )
                    _sync_student_to_native_prefix(
                        training_env,
                        commander,
                        [{"step": step, "actions": actions}],
                    )
                    native_guidance_prefix_step = step
                    final_observation = obs
                    run_summary.update(step, obs)
                    if done:
                        termination_reason = "environment_done"
                        break
                student_start_step = args.max_steps + 1
            elif native_guidance_stage == "t0":
                training_env.red_model_deploy()
                prefix_steps = []
            else:
                training_env.red_model_deploy_from_native(
                    native_trace["deployment_actions"]
                )
                native_guidance_prefix_step = int(
                    os.getenv("RED_NATIVE_GUIDANCE_PREFIX_STEP", "0")
                )
                prefix_steps = []
                dynamic_detection_anchor = (
                    native_guidance_stage == "detection"
                    and native_guidance_prefix_step < 0
                )
                source_steps = (
                    native_trace["steps"] if dynamic_detection_anchor
                    else [
                        row for row in native_trace["steps"]
                        if int(row["step"]) <= native_guidance_prefix_step
                    ]
                )
                for row in source_steps:
                    obs, reward, done, info = training_env.step(
                        native_actions=row["actions"]
                    )
                    prefix_steps.append(row)
                    native_guidance_prefix_step = int(row["step"])
                    final_observation = obs
                    run_summary.update(int(row["step"]), obs)
                    if dynamic_detection_anchor and _has_9500_detection(obs):
                        break
            if native_handoff_spec is not None or native_distillation_manifest is not None:
                prefix_steps = []
            if native_distillation_manifest is None:
                _sync_student_to_native_prefix(training_env, commander, prefix_steps)
            if native_handoff_spec is None and native_distillation_manifest is None:
                python_random_state = random.getstate()
                child_pid = os.fork()
                if child_pid == 0:
                    os._exit(0)
                _, wait_status = os.waitpid(child_pid, 0)
                exit_code = os.waitstatus_to_exitcode(wait_status)
                if exit_code != 0:
                    raise RuntimeError(
                        f"原生快照子分支退出码异常: {exit_code}"
                    )
                random.setstate(python_random_state)
                rollout_device = os.getenv("RED_NATIVE_ROLLOUT_DEVICE")
                if rollout_device:
                    learning_policy.trainer.move_runtime_device(
                        learning_policy.trainer.resolve_runtime_device(
                            rollout_device
                        )
                    )
                student_start_step = native_guidance_prefix_step + 1
        else:
            training_env.red_model_deploy()
        # time.sleep(1000)
        if red_policy == "r11_paos" and os.getenv("RED_PAOS_MODE") == "probe":
            probe = training_env.prepare_commander_no_step_probe(commander)
            print("PAOS_PROBE " + json.dumps(probe, ensure_ascii=False, allow_nan=False, sort_keys=True))
            training_env.close()
            get_writer().close()
            return
        start_time = time.perf_counter()
        # 运行仿真, 训练环境会自动调用智能体的get_action方法
        for step in range(student_start_step, args.max_steps + 1):
            obs, reward, done, info = training_env.step()
            commander.observe_top_reward(training_env.last_official_team_reward)
            final_observation = obs
            run_summary.update(step, obs)
            if (
                counterfactual_probe
                and step >= learning_policy.replay_stop_step
            ):
                termination_reason = "counterfactual_stop"
                logging.info("[反事实] 已到指定窗口终点 step=%s", step)
                break
            if step % 200 == 0:
                # logging.info(f"[测试] Step {step}: obs={obs}, reward={reward}, done={done}, info={info}")
                logging.info(f"[测试] Step {step}")
                # pass
            if done:
                logging.info("[测试] 仿真结束")
                termination_reason = "environment_done"
                break
        end_time = time.perf_counter()

        logging.info(f"[测试] 第 {i + 1} 轮结束，本轮仿真总用时: {end_time - start_time:.6f} 秒")
        # 写剩余缓冲区数据
        write_immediately()
        red_launched = sum(
            getattr(agent, "launch_step", -1) >= 0
            for agent in training_env.agent_manager.get_all_agents()
        )
        commander.finish_top_episode(terminal=True)
        summary = run_summary.build(
            final_observation,
            termination_reason=termination_reason,
            red_launched=red_launched,
        )
        _attach_l_resource_outcomes(summary, training_env)
        counterfactual_outcome = None
        if trajectory_counterfactual and learning_training and not counterfactual_probe:
            if frozen_counterfactual_checkpoint is None:
                raise RuntimeError("事实回合缺少冻结策略快照")
            factual_inputs = learning_policy.counterfactual_episode_inputs()
            counterfactual_output_dir = (
                Path(write_config.output_dir)
                / "trajectory_credit"
                / f"round_{i + 1}"
            )
            counterfactual_outcome = run_counterfactual_replays(
                factual_events=factual_inputs["factual_events"],
                factual_target_rewards=factual_inputs["target_rewards"],
                scenario=args.scenario,
                max_steps=args.max_steps,
                frozen_checkpoint=frozen_counterfactual_checkpoint,
                output_dir=counterfactual_output_dir,
                gamma=float(learning_policy.trainer.config.gamma),
                base_env=dict(os.environ),
            )
            learning_policy.apply_trajectory_credit(counterfactual_outcome.credit)
            summary["counterfactual_credit"] = {
                "manifest": str(counterfactual_outcome.manifest_path),
                "single_replay_count": counterfactual_outcome.single_replay_count,
                "joint_replay_count": counterfactual_outcome.joint_replay_count,
                "validation": dict(counterfactual_outcome.manifest["credit_validation"]),
            }
        if native_handoff_spec is not None:
            controlled_units = list(native_handoff_spec["controlled_units"])
            target_head_counterfactual = bool(
                native_handoff_spec.get("target_head_counterfactual", False)
            )
            dense_target_counterfactual = bool(
                native_handoff_spec.get("dense_target_counterfactual", False)
            )
            target_counterfactual_samples = int(
                native_handoff_spec.get("target_counterfactual_samples", 0)
            )
            sampled_target_rows = []
            search_component = (
                str(native_handoff_spec.get("controlled_components", "joint"))
                == "search"
            )
            team_search = bool(
                search_component
                and native_handoff_spec.get("team_search", False)
            )
            credit_dir = (
                Path(write_config.output_dir) / "native_handoff_credit"
            )
            shared_counterfactual = os.getenv(
                "RED_NATIVE_HANDOFF_COUNTERFACTUAL_RETURN"
            )
            if target_head_counterfactual and shared_counterfactual is not None:
                raise RuntimeError(
                    "目标头反事实必须按学生位置重新分叉，不能复用整动作缓存"
                )
            if search_component and shared_counterfactual is not None:
                raise RuntimeError(
                    "搜索坐标反事实必须按学生航路重新分叉，不能复用毁伤缓存"
                )
            if team_search:
                controlled_entity_ids = {
                    int(unit["executor_id"]) for unit in controlled_units
                }
                resources = summary["red"]["l_resource_outcomes"]
                first_steps_by_source = resources.get(
                    "direct_9500_first_step_by_entity", {}
                )
                first_l_source_by_target: dict[int, tuple[int, int]] = {}
                for raw_source_id, target_steps in first_steps_by_source.items():
                    source_id = int(raw_source_id)
                    if source_id not in controlled_entity_ids:
                        continue
                    for raw_target_id, raw_step in target_steps.items():
                        target_id = int(raw_target_id)
                        candidate = (int(raw_step), source_id)
                        previous = first_l_source_by_target.get(target_id)
                        if previous is None or candidate < previous:
                            first_l_source_by_target[target_id] = candidate
                objective_9500_ids = {
                    int(row["id"])
                    for row in summary.get("objectives", ())
                    if int(row.get("type", -1)) == 9500
                }
                first_l_source_by_target = {
                    target_id: value
                    for target_id, value in first_l_source_by_target.items()
                    if target_id in objective_9500_ids
                }
                required_count = max(1, len(objective_9500_ids))
                contribution_by_entity = {
                    entity_id: sum(
                        source_id == entity_id
                        for _, source_id in first_l_source_by_target.values()
                    ) / required_count
                    for entity_id in controlled_entity_ids
                }
                deltas = [
                    float(contribution_by_entity[int(unit["executor_id"])])
                    for unit in controlled_units
                ]
                factual_target_return = sum(deltas)
                counterfactual_returns = [
                    max(0.0, factual_target_return - delta)
                    for delta in deltas
                ]
                probe_results = []
                probe_directory = None
                dense_target_rows = []
            elif (
                target_head_counterfactual
                and target_counterfactual_samples > 0
            ):
                override_by_unit = {
                    str(row["unit_id"]): row
                    for row in native_handoff_overrides
                }
                sampled_requests = []
                candidates_without_replacement = (
                    os.getenv(
                        "RED_C0A_GOAL_CANDIDATES_WITHOUT_REPLACEMENT", "0"
                    ) == "1"
                )
                for unit in controlled_units:
                    unit_id = str(unit["unit_id"])
                    candidates = override_by_unit[unit_id].get(
                        "counterfactual_samples", ()
                    )
                    invalid_sample_count = (
                        len(candidates) > target_counterfactual_samples
                        if candidates_without_replacement
                        else len(candidates) != target_counterfactual_samples
                    )
                    if invalid_sample_count:
                        raise RuntimeError(
                            f"unit {unit_id} 的策略反事实样本数 "
                            f"{len(candidates)} != {target_counterfactual_samples}"
                        )
                    sampled_requests.extend(
                        (unit, int(candidate["sample_index"]))
                        for candidate in candidates
                    )
                worker_count = min(
                    max(1, len(sampled_requests)),
                    int(os.getenv("RED_NATIVE_HANDOFF_COW_WORKERS", "8")),
                )
                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    futures = [
                        (
                            str(unit["unit_id"]),
                            sample_index,
                            pool.submit(
                                _run_native_handoff_probe,
                                unit=unit,
                                overrides=native_handoff_overrides,
                                native_trace_path=Path(native_guidance_trace),
                                target_id=int(native_handoff_spec["target_id"]),
                                scenario=args.scenario,
                                max_steps=args.max_steps,
                                output_dir=credit_dir,
                                counterfactual_sample_index=sample_index,
                            ),
                        )
                        for unit, sample_index in sampled_requests
                    ]
                    sampled_results = {
                        (unit_id, sample_index): future.result()
                        for unit_id, sample_index, future in futures
                    }
                factual_score = float(summary["score"]["score"])
                deltas = []
                counterfactual_returns = []
                probe_results = []
                for unit in controlled_units:
                    unit_id = str(unit["unit_id"])
                    override = override_by_unit[unit_id]
                    student_target_id = int(override["student_target_id"])
                    candidate_rows = []
                    for candidate in override["counterfactual_samples"]:
                        sample_index = int(candidate["sample_index"])
                        target_id = int(candidate["target_id"])
                        result = sampled_results[(unit_id, sample_index)]
                        outcome = result["counterfactual"]
                        score = float(outcome["score"])
                        if (
                            target_id == student_target_id
                            and abs(score - factual_score) > 1e-8
                        ):
                            raise RuntimeError(
                                "同目标策略反事实未复现事实分数: "
                                f"unit={unit_id}, sample={sample_index}, "
                                f"factual={factual_score}, replay={score}"
                            )
                        candidate_rows.append({
                            "sample_index": sample_index,
                            "target_index": int(candidate["target_index"]),
                            "target_id": target_id,
                            "score": score,
                            "score_delta_from_student": (
                                score - factual_score
                            ) / 100.0,
                            "target_return": float(outcome["target_return"]),
                        })
                    if candidate_rows:
                        baseline_score = sum(
                            row["score"] for row in candidate_rows
                        ) / len(candidate_rows)
                        deltas.append((factual_score - baseline_score) / 100.0)
                        counterfactual_returns.append(
                            sum(row["target_return"] for row in candidate_rows)
                            / len(candidate_rows)
                        )
                        probe_results.append(
                            sampled_results[(
                                unit_id,
                                int(candidate_rows[0]["sample_index"]),
                            )]
                        )
                    else:
                        baseline_score = factual_score
                        deltas.append(0.0)
                        counterfactual_returns.append(0.0)
                    sampled_target_rows.append({
                        "unit_id": unit_id,
                        "executor_id": int(unit["executor_id"]),
                        "timestep": int(unit["timestep"]),
                        "student_target_id": student_target_id,
                        "factual_score": factual_score,
                        "baseline_score": baseline_score,
                        "samples": candidate_rows,
                    })
                # Sampled probes are counterfactual-only.  The exact factual
                # all-objective suffix target is reconstructed by the
                # curriculum runner from the main factual summary.
                factual_target_return = factual_score / 100.0
                probe_directory = str(credit_dir.resolve())
                dense_target_rows = []
            elif shared_counterfactual is None:
                worker_count = min(
                    len(controlled_units),
                    int(os.getenv("RED_NATIVE_HANDOFF_COW_WORKERS", "8")),
                )
                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    futures = [
                        pool.submit(
                            _run_native_handoff_probe,
                            unit=unit,
                            overrides=native_handoff_overrides,
                            native_trace_path=Path(native_guidance_trace),
                            target_id=int(native_handoff_spec["target_id"]),
                            scenario=args.scenario,
                            max_steps=args.max_steps,
                            output_dir=credit_dir,
                        )
                        for unit in controlled_units
                    ]
                probe_results = [future.result() for future in futures]
                dense_target_rows = []
                if dense_target_counterfactual:
                    override_by_unit = {
                        str(row["unit_id"]): row
                        for row in native_handoff_overrides
                    }
                    teacher_results = {
                        str(unit["unit_id"]): result
                        for unit, result in zip(controlled_units, probe_results)
                    }
                    dense_requests = []
                    for unit in controlled_units:
                        override = override_by_unit[str(unit["unit_id"])]
                        student_target_id = int(override["student_target_id"])
                        for raw_target_id in override.get(
                            "counterfactual_actions_by_target", {}
                        ):
                            candidate_target_id = int(raw_target_id)
                            dense_requests.append((unit, candidate_target_id))
                    worker_count = min(
                        max(1, len(dense_requests)),
                        int(os.getenv("RED_NATIVE_HANDOFF_COW_WORKERS", "8")),
                    )
                    with ThreadPoolExecutor(max_workers=worker_count) as pool:
                        futures = [
                            (
                                str(unit["unit_id"]),
                                candidate_target_id,
                                pool.submit(
                                    _run_native_handoff_probe,
                                    unit=unit,
                                    overrides=native_handoff_overrides,
                                    native_trace_path=Path(native_guidance_trace),
                                    target_id=int(native_handoff_spec["target_id"]),
                                    scenario=args.scenario,
                                    max_steps=args.max_steps,
                                    output_dir=credit_dir,
                                    counterfactual_target_id=candidate_target_id,
                                ),
                            )
                            for unit, candidate_target_id in dense_requests
                        ]
                        extra_results = {
                            (unit_id, candidate_target_id): future.result()
                            for unit_id, candidate_target_id, future in futures
                        }
                    factual_score = float(summary["score"]["score"])
                    teacher_target_id = int(native_handoff_spec["target_id"])
                    for unit in controlled_units:
                        unit_id = str(unit["unit_id"])
                        override = override_by_unit[unit_id]
                        student_target_id = int(override["student_target_id"])
                        if not override.get("counterfactual_actions_by_target"):
                            continue
                        candidate_rows = []
                        for raw_target_id in override.get(
                            "counterfactual_actions_by_target", {}
                        ):
                            candidate_target_id = int(raw_target_id)
                            outcome = extra_results[
                                (unit_id, candidate_target_id)
                            ]["counterfactual"]
                            score = float(outcome["score"])
                            if (
                                candidate_target_id == student_target_id
                                and abs(score - factual_score) > 1e-8
                            ):
                                raise RuntimeError(
                                    "学生目标稠密重放未复现事实分数: "
                                    f"unit={unit_id}, factual={factual_score}, "
                                    f"replay={score}"
                                )
                            candidate_rows.append({
                                "target_id": candidate_target_id,
                                "score": score,
                                "score_delta_from_student": (
                                    score - factual_score
                                ) / 100.0,
                                "target_return": float(
                                    outcome["target_return"]
                                ),
                                "target_score": float(outcome["target_score"]),
                                "executor_target_return": float(
                                    outcome["executor_target_return"]
                                ),
                                "executor_target_score": float(
                                    outcome["executor_target_score"]
                                ),
                            })
                        dense_target_rows.append({
                            "unit_id": unit_id,
                            "executor_id": int(unit["executor_id"]),
                            "timestep": int(unit["timestep"]),
                            "student_target_id": student_target_id,
                            "factual_score": factual_score,
                            "candidates": candidate_rows,
                        })
                if search_component:
                    counterfactual_returns = [
                        float(
                            result["counterfactual"]
                            ["search_discovery_return"]
                        )
                        for result in probe_results
                    ]
                    deltas = [
                        max(
                            0.0,
                            float(result["search_discovery_return_delta"]),
                        )
                        for result in probe_results
                    ]
                    factual_target_return = (
                        float(
                            probe_results[0]["factual"]
                            ["search_discovery_return"]
                        )
                        if probe_results else 0.0
                    )
                else:
                    counterfactual_returns = [
                        float(result["counterfactual"]["target_return"])
                        for result in probe_results
                    ]
                    deltas = (
                        [
                            (
                                float(result["factual"]["score"])
                                - float(result["counterfactual"]["score"])
                            ) / 100.0
                            for result in probe_results
                        ]
                        if target_head_counterfactual
                        else [
                            max(0.0, float(result["target_return_delta"]))
                            for result in probe_results
                        ]
                    )
                    factual_target_return = (
                        float(probe_results[0]["factual"]["target_return"])
                        if probe_results else 0.0
                    )
                probe_directory = str(credit_dir.resolve())
            else:
                target_id = int(native_handoff_spec["target_id"])
                objective = next(
                    row for row in summary["objectives"]
                    if int(row["id"]) == target_id
                )
                initial_health = float(objective["initial_health"])
                damage_ratio = min(
                    1.0,
                    max(
                        0.0,
                        (initial_health - float(objective["final_health"]))
                        / initial_health,
                    ),
                )
                factual_target_return = (
                    float(summary["score"]["objective_weights"][str(target_id)])
                    * damage_ratio
                    / 21.0
                )
                counterfactual_returns = [
                    float(shared_counterfactual)
                    for _ in controlled_units
                ]
                deltas = [
                    max(0.0, factual_target_return - value)
                    for value in counterfactual_returns
                ]
                probe_directory = None
                dense_target_rows = []
            student_target_by_unit = {
                str(row["unit_id"]): int(row["student_target_id"])
                for row in native_handoff_overrides
                if "student_target_id" in row
            }
            if target_head_counterfactual and any(
                str(unit["unit_id"]) not in student_target_by_unit
                for unit in controlled_units
            ):
                raise RuntimeError("目标头反事实缺少学生目标映射")
            delta_sum = sum(deltas)
            credits = [
                {
                    "unit_id": str(unit["unit_id"]),
                    "executor_id": int(unit["executor_id"]),
                    "timestep": int(unit["timestep"]),
                    "target_id": (
                        student_target_by_unit[str(unit["unit_id"])]
                        if target_head_counterfactual
                        else (
                            -100
                            if search_component
                            else int(native_handoff_spec["target_id"])
                        )
                    ),
                    "credit": (
                        delta
                        if target_head_counterfactual
                        else (
                            factual_target_return * delta / delta_sum
                            if delta_sum > 1e-12 else 0.0
                        )
                    ),
                }
                for unit, delta in zip(controlled_units, deltas)
            ]
            if not team_search:
                learning_policy.apply_native_handoff_credit(credits)
            native_handoff_credit = {
                "curriculum_stage": str(native_handoff_spec["curriculum_stage"]),
                "target_id": int(native_handoff_spec["target_id"]),
                "target_head_counterfactual": target_head_counterfactual,
                "student_target_ids": [
                    int(row["target_id"]) for row in credits
                ],
                "controlled_unit_count": len(controlled_units),
                "positive_unit_count": sum(delta > 1e-12 for delta in deltas),
                "negative_unit_count": sum(delta < -1e-12 for delta in deltas),
                "return_kind": (
                    "team_legal_9500_coverage_event_difference"
                    if team_search
                    else "legal_direct_9500_detection_plus_alive_proximity"
                    if search_component
                    else "sampled_old_policy_joint_e01"
                    if target_counterfactual_samples > 0
                    else "weighted_objective_damage"
                ),
                "team_search": team_search,
                "factual_target_return": factual_target_return,
                "counterfactual_target_returns": counterfactual_returns,
                "counterfactual_deltas": deltas,
                "dense_target_counterfactual": dense_target_counterfactual,
                "dense_target_rows": dense_target_rows,
                "target_counterfactual_samples": target_counterfactual_samples,
                "sampled_target_rows": sampled_target_rows,
                "factual_search_discovery_ids": (
                    sorted(first_l_source_by_target)
                    if team_search
                    else list(
                        probe_results[0]["factual"]
                        ["search_discovery_ids"]
                    )
                    if search_component and probe_results else []
                ),
                "factual_search_discovery_steps": (
                    {
                        str(target_id): int(step_and_source[0])
                        for target_id, step_and_source in sorted(
                            first_l_source_by_target.items()
                        )
                    }
                    if team_search
                    else dict(
                        probe_results[0]["factual"]
                        ["search_discovery_steps"]
                    )
                    if search_component and probe_results else {}
                ),
                "factual_search_direct_detection_return": (
                    factual_target_return
                    if team_search
                    else float(
                        probe_results[0]["factual"]
                        ["search_direct_detection_return"]
                    )
                    if search_component and probe_results else 0.0
                ),
                "factual_search_proximity_return": (
                    0.0
                    if team_search
                    else float(
                        probe_results[0]["factual"]
                        ["search_proximity_return"]
                    )
                    if search_component and probe_results else 0.0
                ),
                "factual_search_nearest_alive_distance_m": (
                    None
                    if team_search
                    else probe_results[0]["factual"].get(
                        "search_nearest_alive_distance_m"
                    )
                    if search_component and probe_results else None
                ),
                "factual_team_first_search_discovery_ids": (
                    sorted(first_l_source_by_target)
                    if team_search
                    else list(
                        probe_results[0]["factual"]
                        ["team_first_search_discovery_ids"]
                    )
                    if search_component and probe_results else []
                ),
                "factual_team_first_search_discovery_steps": (
                    {
                        str(target_id): int(step_and_source[0])
                        for target_id, step_and_source in sorted(
                            first_l_source_by_target.items()
                        )
                    }
                    if team_search
                    else dict(
                        probe_results[0]["factual"]
                        ["team_first_search_discovery_steps"]
                    )
                    if search_component and probe_results else {}
                ),
                "counterfactual_reused": (
                    shared_counterfactual is not None and not team_search
                ),
                "credited_reward_sum": sum(row["credit"] for row in credits),
                "teacher_action_count": sum(
                    len(row["actions"]) for row in native_trace["steps"]
                ),
                "mappo_action_count": (
                    sum(
                        len(row.get("actions", ()))
                        for row in native_handoff_overrides
                    )
                    if bool(native_handoff_spec.get(
                        "persistent_student_option_control", False
                    ))
                    else len(controlled_units)
                ),
                "mappo_decision_step_count": (
                    sum(
                        bool(row.get("actions"))
                        for row in native_handoff_overrides
                    )
                    if bool(native_handoff_spec.get(
                        "persistent_student_option_control", False
                    ))
                    else len(controlled_units)
                ),
                "teacher_steps_in_ppo_rollout": 0,
                "probe_directory": probe_directory,
            }
            summary["native_causal_handoff"] = native_handoff_credit
        episode_metrics: dict[str, float] = {}
        finish_unified_episode = getattr(learning_policy, "finish_episode", None)
        if callable(finish_unified_episode):
            episode_metrics = finish_unified_episode(float(summary.get("score", {}).get("score", 0.0)))
        if counterfactual_probe:
            if learning_policy.replay_branch_role == "counterfactual":
                result_path = Path(learning_policy.replay_result_path)
                result_path.parent.mkdir(parents=True, exist_ok=True)
                applied_interventions = list(
                    learning_policy.applied_interventions
                )
                if not applied_interventions:
                    raise RuntimeError(
                        "反事实子分支没有执行任何 NULL 干预"
                    )
                probe_result = {
                    "success": True,
                    "status": "complete",
                    "target_rewards": list(
                        training_env.target_reward_history
                    ),
                    "applied_interventions": applied_interventions,
                    "branch_timestep": min(
                        int(row["timestep"])
                        for row in applied_interventions
                    ),
                    "executed_until": int(training_env.current_step),
                    "terminated": (
                        termination_reason == "environment_done"
                    ),
                }
                result_path.write_text(
                    json.dumps(
                        probe_result,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                os._exit(0)
            if learning_policy.replay_branch_role != "prefix":
                raise RuntimeError("反事实 probe 分支角色无效")
            learning_policy.wait_for_replay_children()
            batch_result_path = os.getenv("RED_CF_BATCH_RESULT")
            if batch_result_path:
                Path(batch_result_path).write_text(
                    json.dumps(
                        {"success": True, "status": "complete"},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        summary.setdefault("red", {})["dynamic_catalogue"] = commander.diagnostics()
        summary["red"]["reward_mode"] = training_env.reward_mode
        summary["red"]["official_reward_return"] = training_env.episode_official_return
        summary["red"]["agent_reward_sum"] = training_env.episode_agent_reward_sum
        summary["red"]["credit_conservation_error"] = (
            training_env.max_credit_conservation_error
        )
        summary["red"]["rewarded_agent_count"] = len(
            training_env.episode_rewarded_agent_ids
        )
        if native_handoff_credit is not None:
            summary["red"]["agent_reward_sum"] = float(
                episode_metrics.get("rollout_reward_sum", 0.0)
            )
            summary["red"]["rewarded_agent_count"] = int(
                native_handoff_credit["positive_unit_count"]
            )
        if native_guidance_trace:
            summary["native_snapshot_guidance"] = {
                "trace": str(Path(native_guidance_trace).resolve()),
                "stage": native_guidance_stage,
                "snapshot_prefix_step": native_guidance_prefix_step,
                "student_on_policy_start_step": student_start_step,
            }
        if native_distillation_manifest is not None:
            dataset_summary = json.loads(
                native_distillation_manifest.read_text(encoding="utf-8")
            )
            summary["teacher_distillation"] = {
                "manifest": str(native_distillation_manifest.resolve()),
                "sample_count": int(dataset_summary["sample_count"]),
                "factor_counts": dict(dataset_summary["factor_counts"]),
                "skipped_target_count": int(
                    dataset_summary["skipped_target_count"]
                ),
                "teacher_steps_in_ppo_rollout": 0,
            }
        native_trace_path = os.getenv("RED_NATIVE_TRAJECTORY_PATH")
        if native_trace_path:
            trace_path = Path(native_trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "scenario": str(args.scenario),
                    "blue_seed": int(os.getenv("BLUE_POLICY_SEED", "1")),
                    "red_seed": int(os.getenv("RED_POLICY_SEED", "1")),
                    "simulation_seed": int(os.getenv("SIMULATION_SEED", "1")),
                    "deployment_actions": training_env.native_deployment_actions,
                    "steps": training_env.native_trajectory_steps,
                    "summary": summary,
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            summary["native_trajectory"] = {
                "path": str(trace_path.resolve()),
                "deployment_action_count": len(
                    training_env.native_deployment_actions
                ),
                "step_count": len(training_env.native_trajectory_steps),
            }
        summary_filename = "summary.json" if args.total_rounds == 1 else f"summary_round_{i + 1}.json"
        if trajectory_counterfactual and learning_training:
            policy_diagnostics = commander.diagnostics()["unified_mappo"]
            credit_validation = policy_diagnostics.get(
                "trajectory_credit_validation", {}
            )
            summary["red"]["event_time_agent_reward_sum"] = (
                training_env.episode_agent_reward_sum
            )
            summary["red"]["agent_reward_sum"] = float(
                episode_metrics.get("rollout_reward_sum", 0.0)
            )
            summary["red"]["credit_conservation_error"] = float(
                credit_validation.get("max_event_conservation_error", 0.0)
            )
            summary["red"]["rewarded_agent_count"] = int(
                policy_diagnostics.get("decision_anchored_agent_count", 0)
            )
        summary_path = run_summary.write(write_config.output_dir, summary, summary_filename)
        print("FINAL_SUMMARY " + json.dumps(summary, ensure_ascii=False))
        logging.info(f"[测试] 单局汇总已写入: {summary_path}")

    if learning_training and learning_policy is not None:
        update = getattr(learning_policy, "update", None)
        # The original PPO saved immediately after the final round and did not
        # flush its remainder buffer.  Preserve that lifecycle for a faithful
        # baseline; custom PPO/MAPPO keep the corrected final flush.
        if callable(update) and red_motion_policy != "ppo_baseline":
            update()
        logging.info(
            "[训练] device=%s, updates=%s, transitions=%s, metrics=%s",
            getattr(learning_policy, "device", "n/a"),
            getattr(learning_policy, "update_count", 0),
            getattr(learning_policy, "transition_count", 0),
            getattr(learning_policy, "last_metrics", {}),
        )
        if learning_model:
            Path(learning_model).parent.mkdir(parents=True, exist_ok=True)
            learning_policy.save(learning_model)
            logging.info(f"[测试] 红方学习模型已保存: {learning_model}")
    if red_policy == "r10_bc_erca" and top_training and top_model:
        commander.save_top(top_model)
        logging.info(
            "[训练] BC-ERCA 顶层已保存: %s, diagnostics=%s",
            top_model, commander.diagnostics().get("top_level", {}),
        )

    training_env.close()
    # 关闭写入器
    get_writer().close()


if __name__ == '__main__':
    main()
