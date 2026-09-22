"""Train MAPPO with reconstructed start states and temporal attack options."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_native_guidance_manifest import scenario_targets, trajectory_record

TEACHER_MODEL = ROOT / "models/r9_mappo_e01.pt"
ATTACK_COMMANDS = {200, 3014}
RED_TYPES = {21000, 21001, 21002}
REQUIRED_TARGETS = (2551, 2552, 169)
QUALITY_TARGETS = (51, 52, 53, 54, 106, 168, 169)
BASE_TARGET_SLOT_IDS = (51, 52, 53, 54, 106, 168, 169, -100)
STAGES = ("C0", "C1", "C2", "C3a", "C3", "C4")
C0_COMPONENTS = (
    "lifecycle", "goal", "search", "position", "goal_position", "joint"
)
C4_BOUNDARIES = ("satellite", "search", "initial_launch", "deployment", "t0")
SUPPORT_PROBABILITY = {
    "C0": 1.0,
    "C1": 0.75,
    "C2": 0.5,
    "C3a": 0.25,
    "C3": 0.25,
    "C4": 0.0,
}

C1_TRAINABLE_PARAMETER_PREFIXES = (
    "presence_head.",
    "initial_mean_head.",
    "initial_log_std",
    "target_head.",
    "agent_embedding.",
    "critic_encoder.",
    "value_head.",
    "target_q_head.",
)
C1_SHARED_TARGET_PARAMETER_PREFIXES = (
    "target_item_encoder.",
    "target_query.",
    "target_score.",
    "target_transformer_item_encoder.",
    "target_transformer_query.",
    "target_transformer_encoder.",
    "target_transformer_score.",
)
C0_GOAL_PARAMETER_PREFIXES = (
    "target_transformer_item_encoder.",
    "target_transformer_query.",
    "target_transformer_encoder.",
    "target_transformer_score.",
)
C1_TARGET_RESIDUAL_PARAMETER_PREFIXES = (
    "target_teacher_item_encoder.",
    "target_teacher_query.",
    "target_teacher_score.",
)
C0_SEARCH_PARAMETER_PREFIXES = ("search_head.",)


def search_segment_credits(
    boundary_steps: Sequence[int],
    monitor_samples: Sequence[Mapping[str, Any]],
    discovery_steps: Mapping[str | int, int] | None = None,
) -> list[dict[str, float | int]]:
    """Assign each SEARCH boundary only the outcome of its own route leg."""

    steps = sorted(set(map(int, boundary_steps)))
    discoveries = {
        int(target_id): int(step)
        for target_id, step in (discovery_steps or {}).items()
    }

    def nearest_alive_distance(sample: Mapping[str, Any]) -> float | None:
        distances = [
            float(target["distance_m"])
            for target in sample.get("targets", ())
            if bool(target.get("alive", False))
            and math.isfinite(float(target.get("distance_m", math.inf)))
        ]
        return min(distances) if distances else None

    samples = sorted(
        (
            sample for sample in monitor_samples
            if isinstance(sample, Mapping) and "step" in sample
        ),
        key=lambda sample: int(sample["step"]),
    )
    credits: list[dict[str, float | int]] = []
    for index, boundary_step in enumerate(steps):
        next_step = steps[index + 1] if index + 1 < len(steps) else None
        segment = [
            sample for sample in samples
            if int(sample["step"]) > boundary_step
            and (next_step is None or int(sample["step"]) <= next_step)
        ]
        distance_rows = [
            (sample, nearest_alive_distance(sample)) for sample in segment
        ]
        distance_rows = [
            (sample, distance) for sample, distance in distance_rows
            if distance is not None
        ]
        start_distance = (
            float(distance_rows[0][1]) if distance_rows else math.inf
        )
        minimum_distance = (
            min(float(distance) for _, distance in distance_rows)
            if distance_rows else math.inf
        )
        start_potential = (
            0.25 * math.exp(-start_distance / 100_000.0)
            if math.isfinite(start_distance) else 0.0
        )
        closest_potential = (
            0.25 * math.exp(-minimum_distance / 100_000.0)
            if math.isfinite(minimum_distance) else 0.0
        )
        proximity_gain = max(0.0, closest_potential - start_potential)
        discovered_ids = {
            target_id for target_id, step in discoveries.items()
            if step > boundary_step and (next_step is None or step <= next_step)
        }
        direct_return = len(discovered_ids) / 9.0
        credits.append({
            "step": boundary_step,
            "return": direct_return + proximity_gain,
            "direct_return": direct_return,
            "proximity_gain": proximity_gain,
            "start_distance_m": start_distance,
            "minimum_distance_m": minimum_distance,
        })
    return credits


def team_search_boundary_credits(
    boundary_steps: Sequence[int],
    boundary_agent_ids: Sequence[int],
    direct_first_steps_by_entity: Mapping[str | int, Mapping[str | int, int]],
    agent_id_by_entity: Mapping[str | int, int],
    objective_9500_count: int,
    boundary_search_indices: Sequence[int] | None = None,
    search_cell_count: int = 192,
) -> list[dict[str, float | int]]:
    """Credit unique legal discoveries plus non-hidden team grid novelty."""

    keys = sorted(set(zip(
        map(int, boundary_agent_ids),
        map(int, boundary_steps),
    )))
    boundaries_by_agent: dict[int, list[int]] = {}
    for agent_id, step in keys:
        boundaries_by_agent.setdefault(agent_id, []).append(step)
    entity_by_agent = {
        int(agent_id): int(entity_id)
        for entity_id, agent_id in agent_id_by_entity.items()
    }
    first_l_source_by_target: dict[int, tuple[int, int]] = {}
    for raw_entity_id, target_steps in direct_first_steps_by_entity.items():
        entity_id = int(raw_entity_id)
        agent_id = next((
            candidate_agent_id
            for candidate_agent_id, candidate_entity_id in entity_by_agent.items()
            if candidate_entity_id == entity_id
        ), None)
        if agent_id is None or agent_id not in boundaries_by_agent:
            continue
        for raw_target_id, raw_step in target_steps.items():
            target_id = int(raw_target_id)
            candidate = (int(raw_step), int(agent_id))
            previous = first_l_source_by_target.get(target_id)
            if previous is None or candidate < previous:
                first_l_source_by_target[target_id] = candidate

    required = max(1, int(objective_9500_count))
    credit_by_key = {key: 0.0 for key in keys}
    discovery_count_by_key = {key: 0 for key in keys}
    novelty_by_key = {key: 0.0 for key in keys}
    for detection_step, agent_id in first_l_source_by_target.values():
        eligible = [
            step for step in boundaries_by_agent[agent_id]
            if step <= detection_step
        ]
        if not eligible:
            continue
        key = (agent_id, max(eligible))
        credit_by_key[key] += 1.0 / required
        discovery_count_by_key[key] += 1

    if boundary_search_indices is not None:
        if not (
            len(boundary_steps)
            == len(boundary_agent_ids)
            == len(boundary_search_indices)
        ):
            raise ValueError("团队搜索边界与格点动作数量不一致")
        cell_by_key = {
            (int(agent_id), int(step)): int(cell)
            for agent_id, step, cell in zip(
                boundary_agent_ids,
                boundary_steps,
                boundary_search_indices,
            )
        }
        visited_cells: set[int] = set()
        for step in sorted({key[1] for key in keys}):
            keys_at_step = [key for key in keys if key[1] == step]
            by_cell: dict[int, list[tuple[int, int]]] = {}
            for key in keys_at_step:
                cell = cell_by_key[key]
                if 0 <= cell < int(search_cell_count):
                    by_cell.setdefault(cell, []).append(key)
            for cell, cell_keys in by_cell.items():
                if cell in visited_cells:
                    continue
                shared_novelty = 1.0 / (
                    max(1, int(search_cell_count)) * len(cell_keys)
                )
                for key in cell_keys:
                    novelty_by_key[key] += shared_novelty
                    credit_by_key[key] += shared_novelty
            visited_cells.update(by_cell)
    return [
        {
            "agent_id": agent_id,
            "step": step,
            "return": credit_by_key[(agent_id, step)],
            "unique_discovery_count": discovery_count_by_key[(agent_id, step)],
            "grid_novelty_return": novelty_by_key[(agent_id, step)],
        }
        for agent_id, step in keys
    ]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-start-states", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Train an isolated candidate without advancing curriculum state.",
    )
    parser.add_argument("--start-stage", choices=STAGES, default="C0")
    parser.add_argument(
        "--start-c0-component", choices=C0_COMPONENTS, default="lifecycle"
    )
    parser.add_argument("--anchor-replicas", type=int, default=8)
    parser.add_argument("--c0-anchor-seeds", type=str, default="")
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--start-batch-index", type=int, default=0)
    parser.add_argument(
        "--require-distinct-batch-seeds",
        action="store_true",
        help="Fail before rollout if any seed is repeated within a batch.",
    )
    parser.add_argument(
        "--item-index-offset",
        type=int,
        default=0,
        help=(
            "Offset batch-local item ids and policy RNG seeds; useful for "
            "paired validation of a slice from a historical larger batch."
        ),
    )
    parser.add_argument(
        "--c0-search-selector-offset",
        type=int,
        default=None,
        help=(
            "Fix the per-seed L selector to seed+offset. By default the "
            "batch index is used, which rotates the controlled L each batch."
        ),
    )
    parser.add_argument(
        "--c0-team-search",
        action="store_true",
        help=(
            "At C0 search, hand a bounded rotating cohort of legal L launch "
            "boundaries to the shared student search policy."
        ),
    )
    parser.add_argument(
        "--c0-team-search-max-controlled-l",
        type=int,
        default=0,
        help=(
            "Maximum student-controlled L per episode. Zero means every L "
            "available after the attack reserve; restructured C0s starts at 8."
        ),
    )
    parser.add_argument(
        "--c0-team-search-attack-reserve",
        type=int,
        default=9,
        help=(
            "Minimum rotating teacher-controlled L reserve. Selection never "
            "uses future damage anchors or hidden target coordinates."
        ),
    )
    parser.add_argument(
        "--c0-search-option-chaining",
        action="store_true",
        help=(
            "Keep a student-owned SEARCH option active and reopen a legal "
            "target/search boundary near waypoint completion."
        ),
    )
    parser.add_argument(
        "--c0-search-option-dwell-steps", type=int, default=120
    )
    parser.add_argument(
        "--c0-search-option-reopen-distance-km", type=float, default=15.0
    )
    parser.add_argument(
        "--c0-search-option-emergency-reopen-distance-km",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--c0-search-reachable-mask",
        action="store_true",
        help=(
            "At each C0_search boundary, mask grid cells outside a legal "
            "distance ring around the controlled L's current position."
        ),
    )
    parser.add_argument(
        "--c0-search-leg-min-distance-km", type=float, default=50.0
    )
    parser.add_argument(
        "--c0-search-leg-max-distance-km", type=float, default=180.0
    )
    parser.add_argument(
        "--c0-deterministic-evasion",
        action="store_true",
        help=(
            "During persistent C0 search, replace the controlled L maneuver "
            "with a local-track CPA rule excluded from PPO probability."
        ),
    )
    parser.add_argument(
        "--attack-option-min-dwell-steps", type=int, default=60
    )
    parser.add_argument(
        "--disable-option-control-handoff", action="store_true"
    )
    parser.add_argument("--share-c0-counterfactual", action="store_true")
    parser.add_argument(
        "--share-c1-single-counterfactual", action="store_true"
    )
    parser.add_argument(
        "--c0-target-head-counterfactual",
        action="store_true",
        help=(
            "At C0_goal, replay the same boundary with only the target "
            "restored to the teacher target and use the signed E01 delta."
        ),
    )
    parser.add_argument(
        "--c0-dense-target-counterfactual",
        action="store_true",
        help=(
            "At C0_goal, additionally replay every locally legal target "
            "while holding all other actions and random streams fixed."
        ),
    )
    parser.add_argument(
        "--c0-goal-counterfactual-samples",
        type=int,
        default=0,
        help=(
            "At C0a_goal, sample K legal target references independently "
            "with replacement from the same frozen old policy."
        ),
    )
    parser.add_argument(
        "--c1-target-head-counterfactual",
        action="store_true",
        help=(
            "Fork the same C1 boundary with student non-target actions fixed, "
            "restore only the teacher target, and use signed E01 score delta."
        ),
    )
    parser.add_argument(
        "--c1-dense-target-counterfactual",
        action="store_true",
        help=(
            "At each C1 target boundary, replay every locally legal target "
            "while holding all other student actions fixed."
        ),
    )
    parser.add_argument("--c0-counterfactual-cache", type=Path)
    parser.add_argument("--c1-counterfactual-cache", type=Path)
    parser.add_argument("--c0-factor-actor-lr-scale", type=float, default=0.1)
    parser.add_argument("--c0-factor-critic-lr-scale", type=float, default=0.1)
    parser.add_argument("--c0-actor-update-epochs", type=int, default=1)
    parser.add_argument(
        "--c0-target-supervision-coef",
        type=float,
        default=0.0,
        help=(
            "At C0_goal only, add cross-entropy supervision for the legal "
            "assigned attack target at the controlled causal boundary."
        ),
    )
    parser.add_argument("--c1-actor-lr-scale", type=float, default=0.1)
    parser.add_argument("--c1-actor-update-epochs", type=int, default=1)
    parser.add_argument("--c1-heads-only", action="store_true")
    parser.add_argument(
        "--c1-shared-target-only",
        action="store_true",
        help=(
            "Freeze the legacy actor and critic; update only the shared "
            "per-target scorer from exact target counterfactual returns."
        ),
    )
    parser.add_argument(
        "--c1-target-residual-only",
        action="store_true",
        help=(
            "Freeze the historical policy and update only the independently "
            "gated target residual from exact target counterfactual returns."
        ),
    )
    parser.add_argument(
        "--c1-controlled-components",
        choices=("goal", "joint"),
        default="joint",
    )
    parser.add_argument("--c1-target-supervision-coef", type=float, default=0.0)
    parser.add_argument("--c1-initial-supervision-coef", type=float, default=0.0)
    parser.add_argument("--teacher-decision-dataset-manifest", type=Path)
    parser.add_argument("--actor-position-kl-limit", type=float, default=0.01)
    parser.add_argument("--actor-joint-kl-limit", type=float, default=0.02)
    parser.add_argument("--actor-backtrack-factor", type=float, default=0.5)
    parser.add_argument(
        "--deterministic-actor",
        action="store_true",
        help=(
            "record the normal guided rollout while selecting every active "
            "student action deterministically"
        ),
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help=(
            "Run one C1 evaluation batch without nested counterfactual probes "
            "or a PPO update. E01 score and factual suffix return come from "
            "the completed main episode."
        ),
    )
    parser.add_argument(
        "--resume-completed-episodes",
        action="store_true",
        help=(
            "Reuse an episode when both its rollout and FINAL_SUMMARY log "
            "already exist; rerun only incomplete episodes."
        ),
    )
    return parser.parse_args()


def _batch_update_kwargs(
    args: argparse.Namespace,
    stage: str,
    c0_component: str | None,
) -> dict[str, Any]:
    c0_factorized = (
        stage == "C0"
        and c0_component in {"search", "position", "goal_position", "joint"}
    )
    c0_or_c1 = stage in {"C0", "C1"}
    actor_learning_rate_scale = 1.0
    if stage == "C0":
        actor_learning_rate_scale = args.c0_factor_actor_lr_scale
    elif stage == "C1":
        actor_learning_rate_scale = args.c1_actor_lr_scale
    return {
        "actor_update_epochs": (
            args.c1_actor_update_epochs
            if stage == "C1"
            else (args.c0_actor_update_epochs if stage == "C0" else None)
        ),
        "critic_update_epochs": 4 if c0_or_c1 else None,
        "actor_loss_scale": 1.0,
        "actor_learning_rate_scale": actor_learning_rate_scale,
        "critic_learning_rate_scale": (
            args.c0_factor_critic_lr_scale if c0_factorized else 1.0
        ),
        "critic_beta1": 0.0 if c0_factorized else None,
        "actor_beta1": 0.0 if c0_factorized else None,
        "actor_position_kl_limit": args.actor_position_kl_limit,
        "actor_joint_kl_limit": args.actor_joint_kl_limit,
        "actor_backtrack_factor": args.actor_backtrack_factor,
    }


def ordered_trajectories(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rank = {target_id: index for index, target_id in enumerate(REQUIRED_TARGETS)}
    return sorted(
        manifest["trajectories"],
        key=lambda row: (
            rank.get(int(row["assigned_target_id"]), len(rank)),
            -float(row["teacher_score"]),
            int(row["seed"]),
        ),
    )


def _unit(step: int, action: dict[str, Any]) -> dict[str, Any]:
    entity_id = int(action["executor_id"])
    command_type = int(action["commandType_id"])
    return {
        "unit_id": f"s{step:04d}_e{entity_id}_c{command_type}",
        "timestep": int(step),
        "executor_id": entity_id,
        "command_type": command_type,
    }


def causal_attack_groups(row: dict[str, Any]) -> list[list[dict[str, Any]]]:
    target_id = int(row["assigned_target_id"])
    trace = json.loads(Path(row["trace"]).read_text(encoding="utf-8"))
    anchor = row["damage_anchors"][str(target_id)]
    damage_step = int(anchor["damage_step"])
    attackers = {
        int(event["attacking_entity_id"])
        for step_row in trace["steps"]
        for event in step_row["causal_events"]
        if int(event.get("target_entity_id", -1)) == target_id
        and int(event.get("attacking_entity_type", -1)) in RED_TYPES
        and float(event.get("actual_damage", 0.0)) > 0.0
    }
    units_by_step: dict[int, dict[str, dict[str, Any]]] = {}
    for step_row in trace["steps"]:
        step = int(step_row["step"])
        if step > damage_step:
            break
        for action in step_row["actions"]:
            if int(action.get("executor_id", -1)) not in attackers:
                continue
            if int(action.get("commandType_id", -1)) not in ATTACK_COMMANDS:
                continue
            item = _unit(step, action)
            units_by_step.setdefault(step, {})[item["unit_id"]] = item

    anchor_step = int(anchor["decision_step"])
    anchor_unit = {
        "unit_id": (
            f"s{anchor_step:04d}_e{int(anchor['attacking_entity_id'])}_"
            f"c{int(anchor['decision_type'])}"
        ),
        "timestep": anchor_step,
        "executor_id": int(anchor["attacking_entity_id"]),
        "command_type": int(anchor["decision_type"]),
    }
    units_by_step.setdefault(anchor_step, {})[anchor_unit["unit_id"]] = anchor_unit
    ordered_steps = [
        anchor_step,
        *sorted((step for step in units_by_step if step != anchor_step), reverse=True),
    ]
    groups = [list(units_by_step[step].values()) for step in ordered_steps]
    groups[0] = [
        anchor_unit,
        *(unit for unit in groups[0] if unit["unit_id"] != anchor_unit["unit_id"]),
    ]
    return groups


def units_for_stage(
    groups: list[list[dict[str, Any]]], stage: str
) -> list[dict[str, Any]]:
    if stage == "C0":
        return [groups[0][0]]
    if stage == "C1":
        return [unit for group in groups[:2] for unit in group]
    if stage == "C2":
        return [unit for group in groups for unit in group]
    if stage == "C3a":
        # A single student attack boundary starts a sticky option; subsequent
        # goal/maneuver/satellite decisions remain student-controlled to beta.
        return [groups[0][0]]
    return []


def scenario_entity_types(path: Path) -> dict[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, int] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "id" in value and "entityType" in value:
                result[int(value["id"])] = int(value["entityType"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return result


def c0_search_units(
    row: dict[str, Any],
    entity_types: dict[int, int],
) -> list[dict[str, Any]]:
    """Return every genuine L launch boundary without reading hidden targets."""

    trace = json.loads(Path(row["trace"]).read_text(encoding="utf-8"))
    launches_by_entity: dict[int, dict[str, Any]] = {}
    for step_row in trace["steps"]:
        step = int(step_row["step"])
        # Current pku final20 scenarios use an absolute simTime while a fresh
        # simulator instance starts at time zero.  A launch on the very first
        # frame therefore trips the simulator's 1800-second timeout before it
        # can scan once.  Keep the upstream simulator unchanged and train only
        # on physically executable L launch boundaries after its first clock
        # synchronization frame.
        if step <= 1:
            continue
        for action in step_row["actions"]:
            entity_id = int(action.get("executor_id", -1))
            if int(action.get("commandType_id", -1)) != 200:
                continue
            if int(entity_types.get(entity_id, -1)) != 21002:
                continue
            launches_by_entity.setdefault(entity_id, _unit(step, action))
    launches = sorted(
        launches_by_entity.values(),
        key=lambda unit: (int(unit["timestep"]), int(unit["executor_id"])),
    )
    if not launches:
        raise RuntimeError(f"seed={int(row['seed'])} 的教师轨迹没有 L 发射边界")
    return launches


def c0_team_search_partition(
    row: dict[str, Any],
    entity_types: dict[int, int],
    required_attack_reserve: int = 9,
    max_controlled_l: int = 0,
    selector_offset: int = 0,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Choose a rotating SEARCH cohort without future-outcome leakage."""

    launches = c0_search_units(row, entity_types)
    reserve_count = min(
        max(0, int(required_attack_reserve)),
        max(0, len(launches) - 1),
    )
    search_capacity = len(launches) - reserve_count
    requested = int(max_controlled_l)
    search_count = (
        search_capacity if requested <= 0
        else min(requested, search_capacity)
    )
    if search_count <= 0:
        raise RuntimeError(
            f"seed={int(row['seed'])} 预留攻击 L 后没有搜索单位"
        )
    offset = (int(row["seed"]) + int(selector_offset)) % len(launches)
    rotated = launches[offset:] + launches[:offset]
    search_units = rotated[:search_count]
    reserved_ids = sorted(
        int(unit["executor_id"])
        for unit in rotated[search_count:search_count + reserve_count]
    )
    return search_units, reserved_ids


def c0_search_unit(
    row: dict[str, Any],
    entity_types: dict[int, int],
    selector_index: int,
) -> dict[str, Any]:
    """Choose one rotating L from the legal launch-boundary catalogue."""

    launches = c0_search_units(row, entity_types)
    return launches[int(selector_index) % len(launches)]


def c0_counterfactual_cache_key(row: dict[str, Any]) -> str:
    unit = causal_attack_groups(row)[0][0]
    return (
        f"s{int(row['seed']):04d}:"
        f"t{int(row['assigned_target_id'])}:"
        f"u{unit['unit_id']}"
    )


def trace_reference_suffix_return(
    row: dict[str, Any], prefix_step: int, target_id: int | None = None
) -> float:
    trace = json.loads(Path(row["trace"]).read_text(encoding="utf-8"))
    weights = {
        int(key): float(value)
        for key, value in trace["summary"]["score"]["objective_weights"].items()
    }
    total_weight = float(trace["summary"]["score"]["total_objective_weight"])
    target_ids = (int(target_id),) if target_id is not None else tuple(weights)
    returns = 0.0
    for current_target in target_ids:
        events = [
            event
            for step_row in trace["steps"]
            for event in step_row["causal_events"]
            if int(event.get("target_entity_id", -1)) == current_target
            and "target_health_before" in event
            and "target_health_after" in event
        ]
        if not events:
            continue
        initial_health = max(float(event["target_health_before"]) for event in events)
        boundary_health = initial_health
        final_health = initial_health
        for event in sorted(events, key=lambda value: int(value["step"])):
            final_health = min(final_health, float(event["target_health_after"]))
            if int(event["step"]) <= int(prefix_step):
                boundary_health = min(
                    boundary_health, float(event["target_health_after"])
                )
        returns += (
            weights[current_target]
            / total_weight
            * max(0.0, boundary_health - final_health)
            / initial_health
        )
    return returns


def formal_joint_suffix_return(
    row: Mapping[str, Any],
    prefix_step: int,
    summary: Mapping[str, Any],
) -> float:
    """Official all-objective factual return after a native prefix boundary."""

    trace = json.loads(Path(row["trace"]).read_text(encoding="utf-8"))
    final_objectives = {
        int(objective["id"]): objective
        for objective in summary["objectives"]
    }
    weights = {
        int(key): float(value)
        for key, value in summary["score"]["objective_weights"].items()
    }
    total_weight = float(summary["score"]["total_objective_weight"])
    if total_weight <= 0.0:
        raise ValueError("正式计分目标总权重必须为正数")
    boundary_health = {
        target_id: float(final_objectives[target_id]["initial_health"])
        for target_id in weights
    }
    for step_row in trace["steps"]:
        if int(step_row["step"]) > int(prefix_step):
            continue
        for event in step_row["causal_events"]:
            target_id = int(event.get("target_entity_id", -1))
            if target_id not in boundary_health:
                continue
            if "target_health_after" in event:
                boundary_health[target_id] = min(
                    boundary_health[target_id],
                    float(event["target_health_after"]),
                )

    factual_return = 0.0
    for target_id, weight in weights.items():
        objective = final_objectives[target_id]
        initial_health = float(objective["initial_health"])
        if initial_health <= 0.0:
            continue
        final_health = float(objective["final_health"])
        suffix_damage = max(
            0.0,
            min(1.0, (boundary_health[target_id] - final_health) / initial_health),
        )
        factual_return += weight / total_weight * suffix_damage
    return factual_return


def weighted_type_score(summary: dict[str, Any], types: set[int]) -> float:
    """Return this objective-type group's contribution to the E01 score."""

    total_weight = float(summary["score"]["total_objective_weight"])
    weighted_damage = 0.0
    for objective in summary["objectives"]:
        if int(objective.get("type", -1)) not in types:
            continue
        initial_health = float(objective["initial_health"])
        final_health = float(objective["final_health"])
        damage_ratio = (
            max(0.0, min(1.0, (initial_health - final_health) / initial_health))
            if initial_health > 0.0 else 0.0
        )
        weighted_damage += float(objective["weight"]) * damage_ratio
    return 100.0 * weighted_damage / total_weight


def c4_prefix_step(row: dict[str, Any], boundary: str) -> int:
    trace = json.loads(Path(row["trace"]).read_text(encoding="utf-8"))
    steps = trace["steps"]
    if boundary == "satellite":
        candidates = [
            int(step_row["step"])
            for step_row in steps
            for action in step_row["actions"]
            if int(action.get("commandType_id", -1)) == 3013
        ]
        return max(0, min(candidates) - 1) if candidates else 0
    if boundary == "search":
        candidates = [
            int(step_row["step"])
            for step_row in steps
            for action in step_row["actions"]
            if int(action.get("commandType_id", -1)) == 3014
        ]
        return max(0, min(candidates) - 1) if candidates else 0
    if boundary == "initial_launch":
        candidates = [
            int(step_row["step"])
            for step_row in steps
            for action in step_row["actions"]
            if int(action.get("commandType_id", -1)) == 200
        ]
        return max(0, min(candidates) - 1)
    return 0


def _concat(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tuple(value.cpu() for value in values), dim=0)


def load_teacher_initial_supervision(
    dataset_manifest_path: Path,
) -> dict[tuple[int, int, int], tuple[int, tuple[float, float]]]:
    """Map a causal launch boundary to its policy-agent row and teacher XY."""

    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    lookup: dict[tuple[int, int, int], tuple[int, tuple[float, float]]] = {}
    for replay in dataset["replays"]:
        seed = int(replay["seed"])
        replay_manifest_path = Path(replay["manifest"])
        replay_manifest = json.loads(
            replay_manifest_path.read_text(encoding="utf-8")
        )
        for chunk_record in replay_manifest["chunks"]:
            chunk = torch.load(
                Path(chunk_record["path"]),
                map_location="cpu",
                weights_only=False,
            )
            initial_rows = torch.nonzero(
                chunk["factor_mask"][:, 1].to(dtype=torch.bool),
                as_tuple=False,
            ).flatten()
            for row_index in initial_rows.tolist():
                key = (
                    seed,
                    int(chunk["entity_ids"][row_index].item()),
                    int(chunk["steps"][row_index].item()),
                )
                value = (
                    int(chunk["agent_ids"][row_index].item()),
                    tuple(
                        float(coordinate)
                        for coordinate in chunk["initial_xy"][row_index].tolist()
                    ),
                )
                previous = lookup.get(key)
                if previous is not None and previous != value:
                    raise ValueError(f"教师初始位置键冲突: {key}")
                lookup[key] = value
    return lookup


def merge_rollouts(
    paths: Sequence[Path],
    anchor_group_ids: Sequence[str] | None = None,
    normalize_actor_advantages: bool = True,
    actor_update_epochs: int | None = None,
    critic_update_epochs: int | None = None,
    actor_loss_scale: float = 1.0,
    actor_learning_rate_scale: float = 1.0,
    critic_learning_rate_scale: float = 1.0,
    critic_beta1: float | None = None,
    actor_beta1: float | None = None,
    actor_position_kl_limit: float | None = None,
    actor_joint_kl_limit: float | None = None,
    actor_backtrack_factor: float = 0.5,
    target_supervision: Sequence[tuple[int, int] | None] | None = None,
    target_supervision_coef: float = 0.0,
    initial_supervision: Sequence[
        Sequence[tuple[int, int, tuple[float, float], int]]
    ] | None = None,
    initial_supervision_coef: float = 0.0,
    episode_actor_returns: Sequence[float] | None = None,
    episode_actor_advantages: Sequence[float] | None = None,
    rollout_rewards_as_actor_advantages: bool = False,
    causal_advantage_projection: bool = True,
    episode_search_segment_credits: Sequence[
        Sequence[Mapping[str, Any]]
    ] | None = None,
    episode_value_targets: Sequence[float] | None = None,
):
    from experiments.unified_mappo.model import (
        HybridAction,
        HybridActionMask,
        HybridRolloutBatch,
    )

    batches = [
        torch.load(path, map_location="cpu", weights_only=False)["rollout"]
        for path in paths
    ]
    search_grid_size = 16 * 12
    if anchor_group_ids is not None and len(anchor_group_ids) != len(batches):
        raise ValueError("anchor_group_ids 必须与 rollout 一一对应")
    if rollout_rewards_as_actor_advantages and (
        episode_actor_returns is not None
        or episode_actor_advantages is not None
    ):
        raise ValueError(
            "rollout reward actor advantage 不能与 episode advantage 同时提供"
        )
    if episode_actor_returns is not None:
        if anchor_group_ids is None:
            raise ValueError("episode_actor_returns 需要 anchor_group_ids")
        if len(episode_actor_returns) != len(batches):
            raise ValueError("episode_actor_returns 必须与 rollout 一一对应")
    if episode_actor_advantages is not None:
        if episode_actor_returns is not None:
            raise ValueError(
                "episode_actor_returns 与 episode_actor_advantages 不能同时提供"
            )
        if len(episode_actor_advantages) != len(batches):
            raise ValueError("episode_actor_advantages 必须与 rollout 一一对应")
    if episode_search_segment_credits is not None:
        if episode_actor_advantages is not None or episode_actor_returns is not None:
            raise ValueError("逐航段优势不能与回合级 Actor 回报/优势同时使用")
        if rollout_rewards_as_actor_advantages:
            raise ValueError("逐航段优势不能与 rollout reward 优势同时使用")
        if len(episode_search_segment_credits) != len(batches):
            raise ValueError("episode_search_segment_credits 必须与 rollout 一一对应")
    if (
        episode_value_targets is not None
        and len(episode_value_targets) != len(batches)
    ):
        raise ValueError("episode_value_targets 必须与 rollout 一一对应")
    if target_supervision is not None and len(target_supervision) != len(batches):
        raise ValueError("target_supervision 必须与 rollout 一一对应")
    if initial_supervision is not None and len(initial_supervision) != len(batches):
        raise ValueError("initial_supervision 必须与 rollout 一一对应")
    target_supervision_indices: list[torch.Tensor] = []
    target_supervision_masks: list[torch.Tensor] = []
    initial_supervision_coordinates: list[torch.Tensor] = []
    initial_supervision_masks: list[torch.Tensor] = []
    for batch_index, batch in enumerate(batches):
        labels = torch.full((batch.batch_size,), -1, dtype=torch.long)
        supervision_mask = torch.zeros(batch.batch_size, dtype=torch.bool)
        specification = (
            None
            if target_supervision is None
            else target_supervision[batch_index]
        )
        if specification is not None:
            anchor_step, target_index = map(int, specification)
            if not 0 <= target_index < batch.target_valid_mask.shape[1]:
                raise ValueError(
                    f"target supervision 槽位 {target_index} 超出 rollout 容量"
                )
            supervision_mask = (
                batch.steps.cpu().to(dtype=torch.long).eq(anchor_step)
                & batch.action_mask.target.cpu().to(dtype=torch.bool)
                & batch.target_valid_mask[:, target_index]
                .cpu()
                .to(dtype=torch.bool)
            )
            labels[supervision_mask] = target_index
        target_supervision_indices.append(labels)
        target_supervision_masks.append(supervision_mask)
        initial_coordinates = torch.zeros(
            (batch.batch_size, 2), dtype=batch.observations.dtype
        )
        initial_mask = torch.zeros(batch.batch_size, dtype=torch.bool)
        initial_specifications = (
            ()
            if initial_supervision is None
            else initial_supervision[batch_index]
        )
        for (
            anchor_step,
            agent_id,
            coordinates,
            target_index,
        ) in initial_specifications:
            if not 0 <= int(target_index) < batch.target_valid_mask.shape[1]:
                raise ValueError(
                    f"initial supervision 目标槽位 {target_index} 超出 rollout 容量"
                )
            matching_rows = (
                batch.steps.cpu().to(dtype=torch.long).eq(int(anchor_step))
                & batch.agent_ids.cpu().to(dtype=torch.long).eq(int(agent_id))
                & batch.action_mask.initial_position.cpu().to(dtype=torch.bool)
                & batch.target_valid_mask[:, int(target_index)]
                .cpu()
                .to(dtype=torch.bool)
            )
            matching_count = int(matching_rows.sum().item())
            if matching_count > 1:
                raise ValueError(
                    "同一教师初始位置在单个 rollout 中命中多个合法动作行"
                )
            if matching_count == 1:
                initial_coordinates[matching_rows] = torch.tensor(
                    coordinates, dtype=initial_coordinates.dtype
                )
                initial_mask |= matching_rows
        initial_supervision_coordinates.append(initial_coordinates)
        initial_supervision_masks.append(initial_mask)
    actor_advantage_masks = None
    if episode_search_segment_credits is not None:
        advantage_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        for batch, credits in zip(batches, episode_search_segment_credits):
            active = batch.action_mask.search_position.cpu().to(dtype=torch.bool)
            values = torch.zeros(batch.batch_size, dtype=batch.rewards.dtype)
            team_credit = bool(credits) and all(
                "agent_id" in row for row in credits
            )
            if team_credit:
                credit_by_key = {
                    (int(row["agent_id"]), int(row["step"])): float(
                        row["return"]
                    )
                    for row in credits
                }
                active_keys = set(zip(
                    map(int, batch.agent_ids.cpu()[active].tolist()),
                    map(int, batch.steps.cpu()[active].tolist()),
                ))
                if set(credit_by_key) != active_keys:
                    raise ValueError(
                        "团队航段信用与 SEARCH 边界不一致: "
                        f"credit={len(credit_by_key)}, active={len(active_keys)}"
                    )
                for (agent_id, step), value in credit_by_key.items():
                    matching = (
                        active
                        & batch.agent_ids.cpu().to(dtype=torch.long).eq(agent_id)
                        & batch.steps.cpu().to(dtype=torch.long).eq(step)
                    )
                    if int(matching.sum().item()) != 1:
                        raise ValueError(
                            "团队航段信用没有唯一匹配 rollout 行: "
                            f"agent={agent_id}, step={step}"
                        )
                    values[matching] = value
            else:
                credit_by_step = {
                    int(row["step"]): float(row["return"]) for row in credits
                }
                active_steps = set(map(
                    int,
                    batch.steps.cpu()[active].to(dtype=torch.long).tolist(),
                ))
                if set(credit_by_step) != active_steps:
                    raise ValueError(
                        "逐航段信用与 SEARCH 边界不一致: "
                        f"credit={sorted(credit_by_step)}, "
                        f"active={sorted(active_steps)}"
                    )
                for step, value in credit_by_step.items():
                    values[
                        active
                        & batch.steps.cpu().to(dtype=torch.long).eq(step)
                    ] = value
            advantage_rows.append(values)
            mask_rows.append(active)
        actor_advantages = _concat(advantage_rows)
        actor_advantage_masks = _concat(mask_rows)
    else:
        actor_advantages = (
            _concat([batch.rewards for batch in batches])
            if rollout_rewards_as_actor_advantages
            else (
                _concat([
                    torch.full(
                        (batch.batch_size,),
                        float(episode_actor_advantages[index]),
                        dtype=batch.rewards.dtype,
                    )
                    for index, batch in enumerate(batches)
                ])
                if episode_actor_advantages is not None
                else None
            )
        )
    value_targets = None
    if anchor_group_ids is not None:
        grouped_indices: dict[str, list[int]] = {}
        for index, group_id in enumerate(anchor_group_ids):
            grouped_indices.setdefault(str(group_id), []).append(index)
        reward_returns = [
            float(batch.rewards.sum().item()) for batch in batches
        ]
        actor_returns = (
            list(map(float, episode_actor_returns))
            if episode_actor_returns is not None
            else reward_returns
        )
        episode_advantages = [0.0] * len(batches)
        resolved_value_targets = [0.0] * len(batches)
        for indices in grouped_indices.values():
            actor_group_sum = sum(actor_returns[index] for index in indices)
            reward_group_mean = sum(
                reward_returns[index] for index in indices
            ) / len(indices)
            for index in indices:
                episode_advantages[index] = (
                    actor_returns[index]
                    - (
                        actor_group_sum - actor_returns[index]
                    ) / (len(indices) - 1)
                    if len(indices) > 1
                    else 0.0
                )
                resolved_value_targets[index] = (
                    float(episode_value_targets[index])
                    if episode_value_targets is not None
                    else reward_group_mean
                )
        if actor_advantages is None:
            actor_advantages = _concat([
                torch.full(
                    (batch.batch_size,),
                    episode_advantages[index],
                    dtype=batch.rewards.dtype,
                )
                for index, batch in enumerate(batches)
            ])
        value_targets = _concat([
            torch.full(
                (batch.batch_size,),
                resolved_value_targets[index],
                dtype=batch.rewards.dtype,
            )
            for index, batch in enumerate(batches)
        ])
    elif episode_value_targets is not None:
        value_targets = _concat([
            torch.full(
                (batch.batch_size,),
                float(episode_value_targets[index]),
                dtype=batch.rewards.dtype,
            )
            for index, batch in enumerate(batches)
        ])
    return HybridRolloutBatch(
        observations=_concat([batch.observations for batch in batches]),
        target_coordinates=_concat([batch.target_coordinates for batch in batches]),
        target_valid_mask=_concat([batch.target_valid_mask for batch in batches]),
        target_features=(
            _concat([batch.target_features for batch in batches])
            if all(getattr(batch, "target_features", None) is not None for batch in batches)
            else None
        ),
        action_mask=HybridActionMask(**{
            name: _concat([getattr(batch.action_mask, name) for batch in batches])
            for name in (
                "presence", "initial_position", "search_position", "retarget",
                "target", "maneuver", "satellite",
            )
        }),
        actions=HybridAction(**{
            name: _concat([getattr(batch.actions, name) for batch in batches])
            for name in (
                "presence", "retarget", "initial_xy", "search_xy", "target_xy",
                "maneuver", "initial_raw", "search_index", "target_index",
                "maneuver_index", "satellite",
            )
        }),
        old_log_probs=_concat([batch.old_log_probs for batch in batches]),
        old_values=_concat([batch.old_values for batch in batches]),
        rewards=_concat([batch.rewards for batch in batches]),
        dones=_concat([batch.dones for batch in batches]),
        next_observations=_concat([batch.next_observations for batch in batches]),
        critic_states=_concat([batch.critic_states for batch in batches]),
        next_critic_states=_concat([batch.next_critic_states for batch in batches]),
        agent_ids=_concat([batch.agent_ids for batch in batches]),
        steps=_concat([batch.steps for batch in batches]),
        search_valid_mask=_concat([
            (
                batch.search_valid_mask
                if getattr(batch, "search_valid_mask", None) is not None
                else torch.ones(
                    batch.batch_size, search_grid_size, dtype=torch.bool
                )
            )
            for batch in batches
        ]),
        target_credits=_concat([batch.target_credits for batch in batches]),
        credit_anchor_mask=_concat([batch.credit_anchor_mask for batch in batches]),
        joint_target_returns=tuple(
            group for batch in batches for group in batch.joint_target_returns
        ),
        joint_target_mode=batches[0].joint_target_mode,
        per_agent_advantage_normalization=True,
        actor_advantages=actor_advantages,
        actor_advantage_mask=actor_advantage_masks,
        value_targets=value_targets,
        normalize_actor_advantages=normalize_actor_advantages,
        actor_update_epochs=actor_update_epochs,
        critic_update_epochs=critic_update_epochs,
        actor_loss_scale=actor_loss_scale,
        actor_learning_rate_scale=actor_learning_rate_scale,
        critic_learning_rate_scale=critic_learning_rate_scale,
        critic_beta1=critic_beta1,
        actor_beta1=actor_beta1,
        actor_position_kl_limit=actor_position_kl_limit,
        actor_joint_kl_limit=actor_joint_kl_limit,
        actor_backtrack_factor=actor_backtrack_factor,
        target_supervision_indices=_concat(target_supervision_indices),
        target_supervision_mask=_concat(target_supervision_masks),
        target_supervision_coef=float(target_supervision_coef),
        initial_supervision_xy=_concat(initial_supervision_coordinates),
        initial_supervision_mask=_concat(initial_supervision_masks),
        initial_supervision_coef=float(initial_supervision_coef),
        causal_advantage_projection=causal_advantage_projection,
    )


def apply_batch_update(
    checkpoint: Path,
    rollout_paths: Sequence[Path],
    anchor_group_ids: Sequence[str] | None = None,
    normalize_actor_advantages: bool = True,
    actor_update_epochs: int | None = None,
    critic_update_epochs: int | None = None,
    actor_loss_scale: float = 1.0,
    actor_learning_rate_scale: float = 1.0,
    critic_learning_rate_scale: float = 1.0,
    critic_beta1: float | None = None,
    actor_beta1: float | None = None,
    actor_position_kl_limit: float | None = None,
    actor_joint_kl_limit: float | None = None,
    actor_backtrack_factor: float = 0.5,
    target_supervision: Sequence[tuple[int, int] | None] | None = None,
    target_supervision_coef: float = 0.0,
    initial_supervision: Sequence[
        Sequence[tuple[int, int, tuple[float, float], int]]
    ] | None = None,
    initial_supervision_coef: float = 0.0,
    trainable_parameter_prefixes: Sequence[str] | None = None,
    episode_actor_returns: Sequence[float] | None = None,
    episode_actor_advantages: Sequence[float] | None = None,
    rollout_rewards_as_actor_advantages: bool = False,
    causal_advantage_projection: bool = True,
    episode_search_segment_credits: Sequence[
        Sequence[Mapping[str, Any]]
    ] | None = None,
    episode_value_targets: Sequence[float] | None = None,
) -> dict[str, float]:
    from experiments.unified_mappo.model import HybridMAPPOTrainer

    trainer = HybridMAPPOTrainer.load(checkpoint, device="cuda")
    original_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in trainer.model.named_parameters()
    }
    if trainable_parameter_prefixes is not None:
        prefixes = tuple(trainable_parameter_prefixes)
        for name, parameter in trainer.model.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
    metrics = trainer.update(merge_rollouts(
        rollout_paths,
        anchor_group_ids,
        normalize_actor_advantages,
        actor_update_epochs,
        critic_update_epochs,
        actor_loss_scale,
        actor_learning_rate_scale,
        critic_learning_rate_scale,
        critic_beta1,
        actor_beta1,
        actor_position_kl_limit,
        actor_joint_kl_limit,
        actor_backtrack_factor,
        target_supervision,
        target_supervision_coef,
        initial_supervision,
        initial_supervision_coef,
        episode_actor_returns,
        episode_actor_advantages,
        rollout_rewards_as_actor_advantages,
        causal_advantage_projection,
        episode_search_segment_credits,
        episode_value_targets,
    ))
    restricted_parameter_count = 0
    if trainable_parameter_prefixes is not None:
        prefixes = tuple(trainable_parameter_prefixes)
        restricted_parameter_count = sum(
            parameter.numel()
            for name, parameter in trainer.model.named_parameters()
            if name.startswith(prefixes)
        )
    for name, parameter in trainer.model.named_parameters():
        parameter.requires_grad_(original_requires_grad[name])
    metrics["restricted_trainable_parameter_count"] = float(
        restricted_parameter_count
    )
    trainer.save(checkpoint)
    return metrics


def run_episode(
    *,
    batch_index: int,
    item_index: int,
    stage: str,
    c0_component: str | None,
    controlled_action_components: str,
    anchor_group_id: str,
    c4_boundary: str | None,
    row: dict[str, Any],
    manifest: dict[str, Any],
    entity_types: dict[int, int],
    source_checkpoint: Path,
    output_dir: Path,
    attack_option_min_dwell_steps: int,
    option_control_handoff: bool,
    c0_search_option_chaining: bool,
    c0_search_option_dwell_steps: int,
    c0_search_option_reopen_distance_km: float,
    c0_search_option_emergency_reopen_distance_km: float,
    c0_search_reachable_mask: bool,
    c0_search_leg_min_distance_km: float,
    c0_search_leg_max_distance_km: float,
    c0_deterministic_evasion: bool,
    deterministic_actor: bool,
    c0_team_search: bool = False,
    c0_team_search_max_controlled_l: int = 0,
    c0_team_search_attack_reserve: int = 9,
    c0_search_selector_offset: int | None = None,
    target_head_counterfactual: bool = False,
    dense_target_counterfactual: bool = False,
    target_counterfactual_samples: int = 0,
    counterfactual_target_return: float | None = None,
    resume_completed: bool = False,
) -> dict[str, Any]:
    seed = int(row["seed"])
    component_tag = (
        controlled_action_components if stage in {"C0", "C1"} else "full"
    )
    episode_dir = output_dir / "episodes" / (
        f"b{batch_index:05d}_i{item_index:02d}_{stage}_"
        f"{component_tag}_s{seed:04d}"
    )
    episode_dir.mkdir(parents=True, exist_ok=True)
    local_checkpoint = episode_dir / "policy.pt"
    rollout_path = episode_dir / "on_policy_rollout.pt"
    log_path = episode_dir / "run.log"
    student_trace_path = episode_dir / "student_native_trace.json"

    environment = os.environ.copy()
    persistent_search_option = bool(
        stage == "C0"
        and c0_component == "search"
        and c0_search_option_chaining
        and option_control_handoff
    )
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "BLUE_POLICY": "b0_fixed_ratio_random",
        "BLUE_POLICY_SEED": str(seed),
        "SIMULATION_SEED": str(seed),
        "RED_POLICY": "r12_unified_mappo",
        "RED_POLICY_SEED": str(seed + batch_index * 10000 + item_index),
        "RED_MOTION_POLICY": "unified_mappo",
        "RED_LEARNING_TRAIN": "1",
        "RED_LEARNING_MODEL": str(local_checkpoint.resolve()),
        "RED_UNIFIED_DYNAMIC_LIFECYCLE": "1",
        "RED_UNIFIED_TEMPORAL_ATTACK_OPTIONS": "1",
        "RED_UNIFIED_ATTACK_OPTION_MIN_DWELL_STEPS": str(attack_option_min_dwell_steps),
        "RED_UNIFIED_SEARCH_OPTION_CHAINING": (
            "1" if persistent_search_option else "0"
        ),
        "RED_UNIFIED_SEARCH_OPTION_DWELL_STEPS": str(
            c0_search_option_dwell_steps
        ),
        "RED_UNIFIED_SEARCH_OPTION_REOPEN_DISTANCE_KM": str(
            c0_search_option_reopen_distance_km
        ),
        "RED_UNIFIED_SEARCH_OPTION_EMERGENCY_REOPEN_DISTANCE_KM": str(
            c0_search_option_emergency_reopen_distance_km
        ),
        "RED_UNIFIED_SEARCH_REACHABLE_MASK": (
            "1" if c0_search_reachable_mask and persistent_search_option else "0"
        ),
        "RED_UNIFIED_SEARCH_LEG_MIN_DISTANCE_KM": str(
            c0_search_leg_min_distance_km
        ),
        "RED_UNIFIED_SEARCH_LEG_MAX_DISTANCE_KM": str(
            c0_search_leg_max_distance_km
        ),
        "RED_UNIFIED_DETERMINISTIC_EVASION": (
            "1" if c0_deterministic_evasion and persistent_search_option else "0"
        ),
        "RED_UNIFIED_OPTION_CONTROL_HANDOFF": (
            "1" if (
                (stage == "C3a" and option_control_handoff)
                or persistent_search_option
            ) else "0"
        ),
        "RED_NATIVE_SNAPSHOT_GUIDANCE": "1",
        "RED_NATIVE_GUIDANCE_TRACE": str(Path(row["trace"]).resolve()),
        "RED_NATIVE_TEACHER_MODEL": str(TEACHER_MODEL),
        "RED_UNIFIED_ROLLOUT_OUTPUT": str(rollout_path.resolve()),
        "RED_UNIFIED_UPDATE_DEVICE": "cuda",
        "RED_UNIFIED_FORCE_DETERMINISTIC_ACTOR": (
            "1" if deterministic_actor else "0"
        ),
        "RED_UNIFIED_ROLLOUT_ACTIVE_HEADS_ONLY": (
            "1" if c0_team_search and persistent_search_option else "0"
        ),
    })

    target_id = int(row["assigned_target_id"])
    prefix_step = int(row["damage_snapshot_prefix_step"])
    reserved_attack_l_entity_ids: list[int] = []
    if stage in {"C0", "C1", "C2", "C3a"}:
        if stage == "C0" and c0_component == "search":
            if c0_team_search:
                controlled_units, reserved_attack_l_entity_ids = (
                    c0_team_search_partition(
                        row,
                        entity_types,
                        required_attack_reserve=(
                            c0_team_search_attack_reserve
                        ),
                        max_controlled_l=(
                            c0_team_search_max_controlled_l
                        ),
                        selector_offset=(
                            int(batch_index)
                            if c0_search_selector_offset is None
                            else int(c0_search_selector_offset)
                        ),
                    )
                )
            else:
                controlled_units = [c0_search_unit(
                    row,
                    entity_types,
                    int(row["seed"]) + (
                        int(batch_index)
                        if c0_search_selector_offset is None
                        else int(c0_search_selector_offset)
                    ),
                )]
        else:
            controlled_units = units_for_stage(
                causal_attack_groups(row), stage
            )
        spec_path = episode_dir / "policy_control_mask.json"
        spec_path.write_text(
            json.dumps({
                "schema_version": 1,
                "curriculum_stage": stage,
                "controlled_components": controlled_action_components,
                "persistent_student_option_control": bool(
                    (stage == "C3a" and option_control_handoff)
                    or persistent_search_option
                ),
                "search_option_chaining": persistent_search_option,
                "team_search": bool(
                    c0_team_search and persistent_search_option
                ),
                "required_attack_l_reserve": (
                    c0_team_search_attack_reserve
                    if c0_team_search and persistent_search_option else 0
                ),
                "max_student_controlled_l": (
                    c0_team_search_max_controlled_l
                    if c0_team_search and persistent_search_option else 0
                ),
                "reserved_attack_l_entity_ids": (
                    reserved_attack_l_entity_ids
                ),
                "search_option_dwell_steps": c0_search_option_dwell_steps,
                "search_option_reopen_distance_km": (
                    c0_search_option_reopen_distance_km
                ),
                "search_option_emergency_reopen_distance_km": (
                    c0_search_option_emergency_reopen_distance_km
                ),
                "search_reachable_mask": bool(
                    c0_search_reachable_mask and persistent_search_option
                ),
                "search_leg_min_distance_km": c0_search_leg_min_distance_km,
                "search_leg_max_distance_km": c0_search_leg_max_distance_km,
                "deterministic_evasion": bool(
                    c0_deterministic_evasion and persistent_search_option
                ),
                "target_head_counterfactual": bool(
                    target_head_counterfactual
                ),
                "dense_target_counterfactual": bool(
                    dense_target_counterfactual
                ),
                "target_counterfactual_samples": int(
                    target_counterfactual_samples
                ),
                "seed": seed,
                "target_id": target_id,
                "controlled_units": controlled_units,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        environment.update({
            "RED_REWARD_MODE": "weighted_damage_guided_cow",
            "RED_NATIVE_GUIDANCE_STAGE": "causal_handoff",
            "RED_NATIVE_HANDOFF_SPEC": str(spec_path.resolve()),
            "RED_NATIVE_ROLLOUT_DEVICE": "cuda",
            "RED_NATIVE_HANDOFF_COW_WORKERS": "4",
        })
        if stage == "C0" and c0_component == "search":
            if c0_team_search:
                if len(controlled_units) < 2:
                    raise RuntimeError("C0 团队搜索要求至少两个合法 L 发射边界")
            else:
                if len(controlled_units) != 1:
                    raise RuntimeError("C0_search 路径监控要求恰好一个受控 L")
                environment["RED_SEARCH_MONITOR_ENTITY_ID"] = str(
                    int(controlled_units[0]["executor_id"])
                )
        if (
            counterfactual_target_return is not None
            and not target_head_counterfactual
        ):
            environment["RED_NATIVE_HANDOFF_COUNTERFACTUAL_RETURN"] = str(
                counterfactual_target_return
            )
    else:
        if stage == "C4":
            if c4_boundary is None:
                raise ValueError("C4 requires an explicit boundary")
            prefix_step = c4_prefix_step(row, c4_boundary)
        environment.update({
            "RED_REWARD_MODE": "weighted_damage_trajectory_counterfactual",
            "RED_NATIVE_GUIDANCE_STAGE": "t0" if c4_boundary == "t0" else stage,
            "RED_NATIVE_GUIDANCE_PREFIX_STEP": str(prefix_step),
            "RED_NATIVE_ROLLOUT_DEVICE": "cpu",
            "RED_RECORD_NATIVE_TRAJECTORY": "1",
            "RED_NATIVE_TRAJECTORY_PATH": str(student_trace_path.resolve()),
        })

    libraries = [
        str(ROOT / "core" / "envengine" / "simulator" / "models" / "HXDMissileModel"),
        str(Path(sys.prefix) / "lib"),
    ]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    summaries = []
    if resume_completed and rollout_path.is_file() and log_path.is_file():
        summaries = [
            json.loads(line.removeprefix("FINAL_SUMMARY "))
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("FINAL_SUMMARY ")
        ]
    if len(summaries) != 1:
        shutil.copy2(source_checkpoint, local_checkpoint)
        completed = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--scenario", str(Path(manifest["scenario"]).resolve()),
                "--output-dir", str((episode_dir / "sim_results").resolve()),
                "--max-steps", "3000",
                "--total-rounds", "1",
                "--render-mode", "none",
                "--disable-log-color",
            ],
            cwd=ROOT / "core",
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"batch {batch_index} item {item_index} exited "
                f"{completed.returncode}; log={log_path}"
            )
        summaries = [
            json.loads(line.removeprefix("FINAL_SUMMARY "))
            for line in completed.stdout.splitlines()
            if line.startswith("FINAL_SUMMARY ")
        ]
    if len(summaries) != 1:
        raise RuntimeError(
            f"batch {batch_index} item {item_index} has "
            f"{len(summaries)} FINAL_SUMMARY records; log={log_path}"
        )
    summary = summaries[0]
    teacher_summary = json.loads(
        Path(row["trace"]).read_text(encoding="utf-8")
    )["summary"]
    fixed_target_score = weighted_type_score(summary, {9400, 9600})
    teacher_fixed_target_score = weighted_type_score(
        teacher_summary, {9400, 9600}
    )
    episode_search_segment_credits: list[dict[str, float | int]] = []
    if stage in {"C0", "C1", "C2", "C3a"}:
        handoff = summary["native_causal_handoff"]
        if (
            stage == "C0"
            and c0_component == "goal"
            and target_head_counterfactual
            and target_counterfactual_samples > 0
        ):
            student_suffix_return = formal_joint_suffix_return(
                row, prefix_step, summary
            )
            reference_suffix_return = formal_joint_suffix_return(
                row, prefix_step, teacher_summary
            )
        else:
            student_suffix_return = float(handoff["factual_target_return"])
            reference_suffix_return = (
                1.0
                if stage == "C0" and c0_component == "search"
                else trace_reference_suffix_return(row, prefix_step, target_id)
            )
        controlled_count = int(handoff["controlled_unit_count"])
        positive_count = int(handoff["positive_unit_count"])
        cow_reward_sum = float(handoff["credited_reward_sum"])
        counterfactual_returns = list(map(
            float, handoff["counterfactual_target_returns"]
        ))
        negative_count = int(handoff.get("negative_unit_count", 0))
        student_target_ids = list(map(
            int, handoff.get("student_target_ids", ())
        ))
        target_head_score_deltas = (
            list(map(float, handoff["counterfactual_deltas"]))
            if bool(handoff.get("target_head_counterfactual", False))
            else []
        )
        dense_target_rows = list(handoff.get("dense_target_rows", ()))
        sampled_target_rows = list(
            handoff.get("sampled_target_rows", ())
        )
        return_kind = str(
            handoff.get("return_kind", "weighted_objective_damage")
        )
        search_discovery_ids = list(map(
            int, handoff.get("factual_search_discovery_ids", ())
        ))
        search_discovery_steps = {
            str(target_id): int(step)
            for target_id, step in handoff.get(
                "factual_search_discovery_steps", {}
            ).items()
        }
        search_direct_detection_return = float(
            handoff.get("factual_search_direct_detection_return", 0.0)
        )
        search_proximity_return = float(
            handoff.get("factual_search_proximity_return", 0.0)
        )
        search_nearest_alive_distance_m = handoff.get(
            "factual_search_nearest_alive_distance_m"
        )
        if search_nearest_alive_distance_m is not None:
            search_nearest_alive_distance_m = float(
                search_nearest_alive_distance_m
            )
        team_first_search_discovery_ids = list(map(
            int,
            handoff.get("factual_team_first_search_discovery_ids", ()),
        ))
        team_first_search_discovery_steps = {
            str(target_id): int(step)
            for target_id, step in handoff.get(
                "factual_team_first_search_discovery_steps", {}
            ).items()
        }
        if stage == "C0" and c0_component == "search":
            episode_rollout = torch.load(
                rollout_path, map_location="cpu", weights_only=False
            )["rollout"]
            search_rows = episode_rollout.action_mask.search_position.cpu().to(
                dtype=torch.bool
            )
            boundary_steps = episode_rollout.steps.cpu()[search_rows].tolist()
            if c0_team_search:
                resources = summary["red"]["l_resource_outcomes"]
                boundary_agent_ids = (
                    episode_rollout.agent_ids.cpu()[search_rows].tolist()
                )
                boundary_search_indices = (
                    episode_rollout.actions.search_index.cpu()[
                        search_rows
                    ].tolist()
                )
                episode_search_segment_credits = team_search_boundary_credits(
                    boundary_steps,
                    boundary_agent_ids,
                    resources["direct_9500_first_step_by_entity"],
                    resources["agent_id_by_entity"],
                    int(resources["objective_9500_count"]),
                    boundary_search_indices=boundary_search_indices,
                )
            else:
                probe_directory = Path(handoff["probe_directory"])
                factual_candidates = [probe_directory / "result.json"]
                if not factual_candidates[0].is_file():
                    factual_candidates = list(
                        probe_directory.glob("*/result.json")
                    )
                if len(factual_candidates) != 1:
                    raise RuntimeError(
                        "C0_search 分段信用要求唯一事实 result.json: "
                        f"probe={probe_directory}, "
                        f"candidates={factual_candidates}"
                    )
                factual = json.loads(
                    factual_candidates[0].read_text(encoding="utf-8")
                )["factual"]
                episode_search_segment_credits = search_segment_credits(
                    boundary_steps,
                    factual.get("search_monitor_samples", ()),
                    factual.get("search_discovery_steps", {}),
                )
    else:
        student_suffix_return = float(summary["red"]["official_reward_return"])
        reference_suffix_return = trace_reference_suffix_return(row, prefix_step)
        controlled_count = int(
            summary["red"]["dynamic_catalogue"]["unified_mappo"]
            ["step_inference_count"]
        )
        positive_count = int(summary["red"]["rewarded_agent_count"])
        cow_reward_sum = student_suffix_return
        counterfactual_returns = []
        negative_count = 0
        student_target_ids = []
        target_head_score_deltas = []
        dense_target_rows = []
        sampled_target_rows = []
        return_kind = "official_reward"
        search_discovery_ids = []
        search_discovery_steps = {}
        search_direct_detection_return = 0.0
        search_proximity_return = 0.0
        search_nearest_alive_distance_m = None
        team_first_search_discovery_ids = []
        team_first_search_discovery_steps = {}
    l_resource_outcomes = dict(
        summary.get("red", {}).get("l_resource_outcomes", {})
    )
    reserved_attack_ids = set(reserved_attack_l_entity_ids)
    reserved_intercepted_ids = reserved_attack_ids & set(map(
        int, l_resource_outcomes.get("intercepted_ids", ())
    ))
    reserved_hitter_ids = reserved_attack_ids & set(map(
        int, l_resource_outcomes.get("hitter_ids", ())
    ))
    local_checkpoint.unlink(missing_ok=True)
    return {
        "start_state_id": f"native-s{seed:04d}-tau{prefix_step + 1:04d}",
        "batch": batch_index,
        "item": item_index,
        "stage": stage,
        "c0_component": c0_component,
        "controlled_components": controlled_action_components,
        "team_search": bool(c0_team_search and persistent_search_option),
        "reserved_attack_l_entity_ids": sorted(reserved_attack_ids),
        "reserved_attack_l_intercepted_count": len(
            reserved_intercepted_ids
        ),
        "reserved_attack_l_hitter_count": len(reserved_hitter_ids),
        "anchor_group_id": anchor_group_id,
        "c4_boundary": c4_boundary,
        "seed": seed,
        "assigned_target_id": target_id,
        "snapshot_prefix_step": prefix_step,
        "controlled_sample_count": controlled_count,
        "positive_causal_unit_count": positive_count,
        "negative_causal_unit_count": negative_count,
        "student_target_ids": student_target_ids,
        "target_head_score_deltas": target_head_score_deltas,
        "target_head_counterfactual": target_head_counterfactual,
        "dense_target_counterfactual": dense_target_counterfactual,
        "dense_target_rows": dense_target_rows,
        "target_counterfactual_samples": target_counterfactual_samples,
        "sampled_target_rows": sampled_target_rows,
        "cow_reward_sum": cow_reward_sum,
        "return_kind": return_kind,
        "search_discovery_ids": search_discovery_ids,
        "search_discovery_steps": search_discovery_steps,
        "search_direct_detection_return": search_direct_detection_return,
        "search_proximity_return": search_proximity_return,
        "search_nearest_alive_distance_m": (
            search_nearest_alive_distance_m
        ),
        "search_segment_credits": episode_search_segment_credits,
        "team_first_search_discovery_ids": (
            team_first_search_discovery_ids
        ),
        "team_first_search_discovery_steps": (
            team_first_search_discovery_steps
        ),
        "l_resource_outcomes": l_resource_outcomes,
        "counterfactual_target_returns": counterfactual_returns,
        "counterfactual_reused": bool(
            summary.get("native_causal_handoff", {}).get(
                "counterfactual_reused", False
            )
        ),
        "score": float(summary["score"]["score"]),
        "teacher_score": float(row["teacher_score"]),
        "fixed_target_score": fixed_target_score,
        "teacher_fixed_target_score": teacher_fixed_target_score,
        "destroyed_ids": list(map(int, summary["score"]["destroyed_ids"])),
        "student_suffix_return": student_suffix_return,
        "reference_suffix_return": reference_suffix_return,
        "rollout": str(rollout_path.resolve()),
        "student_trace": (
            str(student_trace_path.resolve()) if student_trace_path.is_file() else None
        ),
        "teacher_steps_in_ppo_rollout": 0,
        "deterministic_actor": deterministic_actor,
        "attack_options": dict(
            summary["red"]["dynamic_catalogue"]["unified_mappo"]
        ),
        "log": str(log_path.resolve()),
    }


def quality_vector(result: dict[str, Any]) -> tuple[float, ...]:
    destroyed = set(map(int, result["destroyed_ids"]))
    diagnostics = result["attack_options"]
    total = max(1, int(diagnostics.get("deployment_count", 0)))
    remaining = 1.0 - float(diagnostics.get("presence_rate", 0.0))
    return (
        float(result["score"]),
        *(float(target_id in destroyed) for target_id in QUALITY_TARGETS),
        float(remaining),
    )


def pareto_insert(
    frontier: list[dict[str, Any]],
    candidate: dict[str, Any],
    reference_qualities: Sequence[tuple[float, ...]],
) -> bool:
    candidate_quality = tuple(candidate["quality"])
    dominates = lambda first, second: (
        all(left >= right for left, right in zip(first, second))
        and any(left > right for left, right in zip(first, second))
    )
    existing_qualities = [
        *reference_qualities,
        *(tuple(existing["quality"]) for existing in frontier),
    ]
    if existing_qualities and not any(
        dominates(candidate_quality, existing)
        for existing in existing_qualities
    ):
        return False
    frontier[:] = [
        existing
        for existing in frontier
        if not dominates(candidate_quality, tuple(existing["quality"]))
    ]
    frontier.append(candidate)
    return True


def teacher_quality(row: dict[str, Any]) -> tuple[float, ...]:
    destroyed = set(map(int, row["destroyed_ids"]))
    return (
        float(row["teacher_score"]),
        *(float(target_id in destroyed) for target_id in QUALITY_TARGETS),
        0.0,
    )


def student_frontier_candidate(
    result: dict[str, Any],
    targets: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    trace_path = result.get("student_trace")
    if not trace_path:
        return None
    row = trajectory_record({
        "seed": int(result["seed"]),
        "score": float(result["score"]),
        "destroyed_ids": list(result["destroyed_ids"]),
        "trace": str(trace_path),
    }, targets)
    available = sorted(map(int, row["damage_anchors"]))
    if not available:
        return None
    target_id = (
        int(result["assigned_target_id"])
        if int(result["assigned_target_id"]) in available
        else available[0]
    )
    row["assigned_target_id"] = target_id
    row["damage_snapshot_prefix_step"] = int(
        row["damage_anchors"][str(target_id)]["snapshot_prefix_step"]
    )
    return {
        "source": "student_pareto_frontier",
        "quality": list(quality_vector(result)),
        "row": row,
        "source_episode": {
            "batch": int(result["batch"]),
            "seed": int(result["seed"]),
            "score": float(result["score"]),
        },
    }
def main() -> None:
    args = parse_args()
    if args.c0_goal_counterfactual_samples < 0:
        raise ValueError("c0-goal-counterfactual-samples 不能为负数")
    if args.c0_goal_counterfactual_samples > 0:
        args.c0_target_head_counterfactual = True
    if args.c0_dense_target_counterfactual:
        args.c0_target_head_counterfactual = True
    if args.c1_dense_target_counterfactual:
        args.c1_target_head_counterfactual = True
    if (
        args.c0_target_head_counterfactual
        and args.share_c0_counterfactual
    ):
        raise ValueError("C0_goal 目标头反事实不能复用整动作缓存")
    if (
        args.c0_target_head_counterfactual
        and args.c0_target_supervision_coef > 0.0
    ):
        raise ValueError(
            "C0_goal 精确反事实训练不能同时模仿教师目标标签"
        )
    if (
        args.c0_goal_counterfactual_samples > 0
        and args.c0_dense_target_counterfactual
    ):
        raise ValueError("K 样本策略基线不能与全目标稠密重放同时启用")
    if (
        args.c1_target_head_counterfactual
        and args.share_c1_single_counterfactual
    ):
        raise ValueError("目标头反事实不能复用 C1 整动作缓存")
    target_only_modes = sum((
        bool(args.c1_shared_target_only),
        bool(args.c1_target_residual_only),
        bool(args.c1_heads_only),
    ))
    if target_only_modes > 1:
        raise ValueError("C1 参数隔离选项不能同时使用")
    if (
        args.c1_shared_target_only or args.c1_target_residual_only
    ) and not args.c1_target_head_counterfactual:
        raise ValueError(
            "C1 目标专训必须配合 c1-target-head-counterfactual"
        )
    if args.item_index_offset < 0:
        raise ValueError("item-index-offset 不能为负数")
    if args.c0_search_leg_min_distance_km < 0.0:
        raise ValueError("c0-search-leg-min-distance-km 不能为负数")
    if (
        args.c0_search_leg_max_distance_km
        < args.c0_search_leg_min_distance_km
    ):
        raise ValueError(
            "c0-search-leg-max-distance-km 不能小于最小距离"
        )
    if args.c0_actor_update_epochs <= 0:
        raise ValueError("c0-actor-update-epochs 必须为正整数")
    teacher_supervision = {
        "c0-target-supervision-coef": args.c0_target_supervision_coef,
        "c1-target-supervision-coef": args.c1_target_supervision_coef,
        "c1-initial-supervision-coef": args.c1_initial_supervision_coef,
    }
    enabled_teacher_supervision = {
        name: value
        for name, value in teacher_supervision.items()
        if value > 0.0
    }
    if any(value < 0.0 for value in teacher_supervision.values()):
        raise ValueError("教师监督系数不能为负数")
    if enabled_teacher_supervision:
        raise ValueError(
            "课程学习禁止用教师动作作监督标签；教师只允许作为未接管"
            f"动作的兜底和反事实参考: {enabled_teacher_supervision}"
        )
    if args.evaluation_only:
        if args.max_batches != 1:
            raise ValueError("evaluation-only 要求 max-batches=1")
        if (
            args.c0_target_head_counterfactual
            or args.c1_target_head_counterfactual
        ):
            raise ValueError("evaluation-only 不生成目标头反事实训练标签")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entity_types = scenario_entity_types(Path(manifest["scenario"]))
    support_rows = ordered_trajectories(manifest)
    c0_seed_order = [
        int(value)
        for value in args.c0_anchor_seeds.split(",")
        if value.strip()
    ]
    rows_by_seed = {int(row["seed"]): row for row in support_rows}
    c0_support_rows = (
        [rows_by_seed[seed] for seed in c0_seed_order]
        if c0_seed_order
        else support_rows
    )
    objective_ids = {
        int(key)
        for key in json.loads(Path(support_rows[0]["trace"]).read_text(
            encoding="utf-8"
        ))["summary"]["score"]["objective_weights"]
    }
    targets = scenario_targets(Path(manifest["scenario"]), objective_ids)
    target_slot_ids = list(BASE_TARGET_SLOT_IDS)
    target_slot_ids.extend(sorted(
        int(target_id) for target_id in targets
        if int(target_id) not in target_slot_ids
    ))
    target_slot_by_id = {target_id: index for index, target_id in enumerate(target_slot_ids)}
    if (
        args.c1_initial_supervision_coef > 0.0
        and args.teacher_decision_dataset_manifest is None
    ):
        raise ValueError(
            "启用 C1 初始位置监督时必须提供 --teacher-decision-dataset-manifest"
        )
    teacher_initial_lookup = (
        load_teacher_initial_supervision(
            args.teacher_decision_dataset_manifest
        )
        if args.teacher_decision_dataset_manifest is not None
        else {}
    )
    support_qualities = tuple(teacher_quality(row) for row in support_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "training_progress.json"
    stage_index = STAGES.index(args.start_stage)
    c0_index = C0_COMPONENTS.index(args.start_c0_component)
    c4_index = 0
    cursor = int(args.start_offset) % len(support_rows)
    c0_cursor = int(args.start_offset) % len(c0_support_rows)
    frontier_cursor = 0
    batch_index = int(args.start_batch_index)
    completed_batches = 0
    episodes: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    frontier: list[dict[str, Any]] = []
    counterfactual_cache_path = (
        args.c0_counterfactual_cache
        if args.c0_counterfactual_cache is not None
        else args.output_dir / "c0_counterfactual_cache.json"
    )
    c0_counterfactual_cache = (
        {
            str(key): float(value)
            for key, value in json.loads(
                counterfactual_cache_path.read_text(encoding="utf-8")
            ).items()
        }
        if counterfactual_cache_path.is_file()
        else {}
    )
    c1_counterfactual_cache_path = (
        args.c1_counterfactual_cache
        if args.c1_counterfactual_cache is not None
        else args.output_dir / "c1_single_counterfactual_cache.json"
    )
    c1_counterfactual_cache = (
        {
            str(key): float(value)
            for key, value in json.loads(
                c1_counterfactual_cache_path.read_text(encoding="utf-8")
            ).items()
        }
        if c1_counterfactual_cache_path.is_file()
        else {}
    )

    while stage_index < len(STAGES):
        if args.max_batches and completed_batches >= args.max_batches:
            break
        batch_index += 1
        completed_batches += 1
        stage = STAGES[stage_index]
        c0_component = C0_COMPONENTS[c0_index] if stage == "C0" else None
        c1_component = args.c1_controlled_components if stage == "C1" else None
        controlled_components = (
            c0_component if stage == "C0"
            else c1_component if stage == "C1" else "joint"
        )
        boundary = C4_BOUNDARIES[c4_index] if stage == "C4" else None
        anchor_group_ids: list[str] = []
        if stage == "C0":
            selected = []
            remaining = args.batch_start_states
            group_index = 0
            while remaining > 0:
                row = c0_support_rows[c0_cursor]
                c0_cursor = (c0_cursor + 1) % len(c0_support_rows)
                replica_count = min(args.anchor_replicas, remaining)
                group_id = (
                    f"b{batch_index:05d}_{c0_component}_"
                    f"s{int(row['seed']):04d}_g{group_index:02d}"
                )
                selected.extend([row] * replica_count)
                anchor_group_ids.extend([group_id] * replica_count)
                remaining -= replica_count
                group_index += 1
        else:
            support_count = args.batch_start_states
            if frontier:
                support_count = round(
                    args.batch_start_states * SUPPORT_PROBABILITY[stage]
                )
            frontier_count = args.batch_start_states - support_count
            selected = [
                support_rows[(cursor + offset) % len(support_rows)]
                for offset in range(support_count)
            ]
            cursor = (cursor + support_count) % len(support_rows)
            if frontier_count:
                selected.extend(
                    frontier[(frontier_cursor + offset) % len(frontier)]["row"]
                    for offset in range(frontier_count)
                )
                frontier_cursor = (
                    frontier_cursor + frontier_count
                ) % len(frontier)
            anchor_group_ids = [
                f"b{batch_index:05d}_i{index + args.item_index_offset:02d}"
                for index in range(1, len(selected) + 1)
            ]
        selected_seeds = [int(row["seed"]) for row in selected]
        distinct_seed_count = len(set(selected_seeds))
        if (
            args.require_distinct_batch_seeds
            and distinct_seed_count != len(selected_seeds)
        ):
            raise ValueError(
                "当前批次包含重复 seed: "
                f"total={len(selected_seeds)}, distinct={distinct_seed_count}"
            )
        snapshot_checkpoint = args.output_dir / "batch_input" / f"batch_{batch_index:05d}.pt"
        snapshot_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.checkpoint, snapshot_checkpoint)
        batch_started_at = time.perf_counter()
        batch_episodes: list[dict[str, Any] | None] = [None] * len(selected)

        def run_selected(
            indices: Sequence[int],
            shared_returns: dict[str, float],
        ) -> None:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    (
                        index,
                        pool.submit(
                            run_episode,
                            batch_index=batch_index,
                            item_index=index + 1 + args.item_index_offset,
                            stage=stage,
                            c0_component=c0_component,
                            controlled_action_components=controlled_components,
                            anchor_group_id=anchor_group_ids[index],
                            c4_boundary=boundary,
                            row=selected[index],
                            manifest=manifest,
                            entity_types=entity_types,
                            source_checkpoint=snapshot_checkpoint,
                            output_dir=args.output_dir,
                            attack_option_min_dwell_steps=(
                                args.attack_option_min_dwell_steps
                            ),
                            option_control_handoff=not args.disable_option_control_handoff,
                            c0_search_option_chaining=(
                                args.c0_search_option_chaining
                            ),
                            c0_search_option_dwell_steps=(
                                args.c0_search_option_dwell_steps
                            ),
                            c0_search_option_reopen_distance_km=(
                                args.c0_search_option_reopen_distance_km
                            ),
                            c0_search_option_emergency_reopen_distance_km=(
                                args.c0_search_option_emergency_reopen_distance_km
                            ),
                            c0_search_reachable_mask=(
                                args.c0_search_reachable_mask
                            ),
                            c0_search_leg_min_distance_km=(
                                args.c0_search_leg_min_distance_km
                            ),
                            c0_search_leg_max_distance_km=(
                                args.c0_search_leg_max_distance_km
                            ),
                            c0_deterministic_evasion=(
                                args.c0_deterministic_evasion
                            ),
                            deterministic_actor=args.deterministic_actor,
                            c0_team_search=args.c0_team_search,
                            c0_team_search_max_controlled_l=(
                                args.c0_team_search_max_controlled_l
                            ),
                            c0_team_search_attack_reserve=(
                                args.c0_team_search_attack_reserve
                            ),
                            c0_search_selector_offset=(
                                args.c0_search_selector_offset
                            ),
                            target_head_counterfactual=(
                                (
                                    stage == "C0"
                                    and c0_component == "goal"
                                    and args.c0_target_head_counterfactual
                                )
                                or (
                                    stage == "C1"
                                    and args.c1_target_head_counterfactual
                                )
                            ),
                            dense_target_counterfactual=(
                                (
                                    stage == "C0"
                                    and c0_component == "goal"
                                    and args.c0_dense_target_counterfactual
                                )
                                or (
                                    stage == "C1"
                                    and args.c1_dense_target_counterfactual
                                )
                            ),
                            target_counterfactual_samples=(
                                args.c0_goal_counterfactual_samples
                                if (
                                    stage == "C0"
                                    and c0_component == "goal"
                                )
                                else 0
                            ),
                            counterfactual_target_return=(
                                trace_reference_suffix_return(
                                    selected[index],
                                    int(selected[index]["damage_snapshot_prefix_step"]),
                                    int(selected[index]["assigned_target_id"]),
                                )
                                if args.evaluation_only
                                else shared_returns.get(anchor_group_ids[index])
                            ),
                            resume_completed=args.resume_completed_episodes,
                        ),
                    )
                    for index in indices
                ]
                for index, future in futures:
                    batch_episodes[index] = future.result()

        if (
            args.share_c0_counterfactual
            and stage == "C0"
            and c0_component != "search"
        ):
            group_first_indices = list(dict.fromkeys(
                anchor_group_ids.index(group_id)
                for group_id in anchor_group_ids
            ))
            group_cache_keys = {
                anchor_group_ids[index]: c0_counterfactual_cache_key(
                    selected[index]
                )
                for index in group_first_indices
            }
            primer_indices = [
                index for index in group_first_indices
                if group_cache_keys[anchor_group_ids[index]]
                not in c0_counterfactual_cache
            ]
            run_selected(primer_indices, {})
            for index in primer_indices:
                c0_counterfactual_cache[
                    group_cache_keys[anchor_group_ids[index]]
                ] = float(
                    batch_episodes[index]["counterfactual_target_returns"][0]
                )
            counterfactual_cache_path.parent.mkdir(parents=True, exist_ok=True)
            counterfactual_cache_path.write_text(
                json.dumps(
                    c0_counterfactual_cache,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            shared_returns = {
                group_id: c0_counterfactual_cache[cache_key]
                for group_id, cache_key in group_cache_keys.items()
            }
            run_selected(
                [
                    index for index in range(len(selected))
                    if index not in primer_indices
                ],
                shared_returns,
            )
        elif (
            args.share_c1_single_counterfactual
            and stage == "C1"
        ):
            c1_cache_keys_by_index: dict[int, str] = {}
            for index, row in enumerate(selected):
                controlled_units = units_for_stage(
                    causal_attack_groups(row), stage
                )
                if len(controlled_units) != 1:
                    continue
                unit = controlled_units[0]
                c1_cache_keys_by_index[index] = (
                    f"c1-single:s{int(row['seed']):04d}:"
                    f"t{int(row['assigned_target_id'])}:"
                    f"u{unit['unit_id']}"
                )
            first_index_by_cache_key: dict[str, int] = {}
            for index, cache_key in c1_cache_keys_by_index.items():
                first_index_by_cache_key.setdefault(cache_key, index)
            primer_indices = [
                index
                for cache_key, index in first_index_by_cache_key.items()
                if cache_key not in c1_counterfactual_cache
            ]
            run_selected(primer_indices, {})
            for index in primer_indices:
                c1_counterfactual_cache[
                    c1_cache_keys_by_index[index]
                ] = float(
                    batch_episodes[index]["counterfactual_target_returns"][0]
                )
            c1_counterfactual_cache_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            c1_counterfactual_cache_path.write_text(
                json.dumps(
                    c1_counterfactual_cache,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            shared_returns = {
                anchor_group_ids[index]: c1_counterfactual_cache[cache_key]
                for index, cache_key in c1_cache_keys_by_index.items()
            }
            run_selected(
                [
                    index for index in range(len(selected))
                    if index not in primer_indices
                ],
                shared_returns,
            )
        else:
            run_selected(range(len(selected)), {})
        batch_episodes = [
            result for result in batch_episodes if result is not None
        ]
        batch_rollout_wall_seconds = time.perf_counter() - batch_started_at
        score_deltas = [
            float(result["score"]) - float(result["teacher_score"])
            for result in batch_episodes
        ]
        paired_tolerance = 1e-9
        c0_actor_advantages = None
        c0_search_segment_credits = None
        c0_actor_advantage_mode = "leave_one_out"
        if stage == "C0" and args.anchor_replicas == 1:
            if not (
                c0_component == "search" and args.c0_team_search
            ) and any(
                len(result["counterfactual_target_returns"]) != 1
                for result in batch_episodes
            ):
                raise RuntimeError(
                    "C0 单 seed 边际优势要求每回合恰好一个同状态反事实"
                )
            if c0_component == "lifecycle":
                c0_actor_advantages = [
                    float(result["student_suffix_return"])
                    - float(result["counterfactual_target_returns"][0])
                    for result in batch_episodes
                ]
                c0_actor_advantage_mode = "signed_same_state_marginal"
            elif c0_component == "search":
                c0_search_segment_credits = [
                    result["search_segment_credits"]
                    for result in batch_episodes
                ]
                c0_actor_advantage_mode = (
                    "per_l_boundary_unique_team_detection"
                    if args.c0_team_search
                    else "per_search_boundary_detection_plus_potential_gain"
                )
            elif (
                c0_component == "goal"
                and args.c0_target_head_counterfactual
            ):
                c0_actor_advantage_mode = (
                    "signed_same_state_target_vs_old_policy_mc_e01"
                    if args.c0_goal_counterfactual_samples > 0
                    else "signed_same_state_target_vs_teacher_e01"
                )
            else:
                c0_actor_advantages = [
                    float(delta) / 100.0 for delta in score_deltas
                ]
                c0_actor_advantage_mode = "paired_teacher_e01"
        c0_lifecycle_decision_count = 0
        c0_lifecycle_launch_count = 0
        if stage == "C0" and c0_component == "lifecycle":
            for result in batch_episodes:
                rollout = torch.load(
                    Path(result["rollout"]),
                    map_location="cpu",
                    weights_only=False,
                )["rollout"]
                active = rollout.action_mask.presence.to(dtype=torch.bool)
                c0_lifecycle_decision_count += int(active.sum().item())
                c0_lifecycle_launch_count += int(
                    rollout.actions.presence[active].sum().item()
                )
        c0_lifecycle_launch_rate = (
            c0_lifecycle_launch_count / c0_lifecycle_decision_count
            if c0_lifecycle_decision_count else None
        )

        target_supervision_enabled = (
            (
                stage == "C0"
                and c0_component == "goal"
                and args.c0_target_supervision_coef > 0.0
            )
            or (
                stage == "C1"
                and args.c1_target_supervision_coef > 0.0
            )
        )
        target_supervision = (
            [
                (
                    int(
                        row["damage_anchors"][str(row["assigned_target_id"])]
                        ["decision_step"]
                    ),
                    target_slot_by_id[int(row["assigned_target_id"])],
                )
                for row in selected
            ]
            if target_supervision_enabled
            else None
        )
        initial_supervision = None
        if stage == "C1" and args.c1_initial_supervision_coef > 0.0:
            initial_supervision = []
            for row in selected:
                specifications: list[
                    tuple[int, int, tuple[float, float], int]
                ] = []
                for unit in units_for_stage(
                    causal_attack_groups(row), stage
                ):
                    if int(unit["command_type"]) != 200:
                        continue
                    key = (
                        int(row["seed"]),
                        int(unit["executor_id"]),
                        int(unit["timestep"]),
                    )
                    if key not in teacher_initial_lookup:
                        raise KeyError(f"教师初始位置数据缺少因果锚点: {key}")
                    agent_id, coordinates = teacher_initial_lookup[key]
                    specifications.append(
                        (
                            int(unit["timestep"]),
                            agent_id,
                            coordinates,
                            target_slot_by_id[int(row["assigned_target_id"])],
                        )
                    )
                initial_supervision.append(specifications)
        metrics = (
            {"evaluation_only": 1.0, "samples": float(sum(
                torch.load(
                    Path(result["rollout"]),
                    map_location="cpu",
                    weights_only=False,
                )["rollout"].batch_size
                for result in batch_episodes
            ))}
            if args.evaluation_only
            else apply_batch_update(
                args.checkpoint,
                [Path(result["rollout"]) for result in batch_episodes],
                anchor_group_ids=(anchor_group_ids if stage == "C0" else None),
                normalize_actor_advantages=not (
                    stage == "C0"
                    and c0_component == "goal"
                    and args.c0_target_head_counterfactual
                ),
                episode_actor_advantages=c0_actor_advantages,
                episode_search_segment_credits=c0_search_segment_credits,
                episode_value_targets=(
                    [
                        float(result["student_suffix_return"])
                        for result in batch_episodes
                    ]
                    if (
                        stage == "C0"
                        and c0_component == "goal"
                        and args.c0_goal_counterfactual_samples > 0
                    )
                    else None
                ),
                rollout_rewards_as_actor_advantages=(
                    (
                        stage == "C0"
                        and c0_component == "goal"
                        and args.c0_target_head_counterfactual
                    )
                    or (
                        stage == "C1"
                        and args.c1_target_head_counterfactual
                    )
                ),
                causal_advantage_projection=not (
                    (stage == "C1" and args.c1_target_head_counterfactual)
                    or (
                        stage == "C0"
                        and c0_component == "goal"
                        and args.c0_target_head_counterfactual
                    )
                    or c0_actor_advantage_mode == "paired_teacher_e01"
                ),
                target_supervision=target_supervision,
                target_supervision_coef=(
                    args.c1_target_supervision_coef
                    if stage == "C1"
                    else args.c0_target_supervision_coef
                    if stage == "C0" and c0_component == "goal"
                    else 0.0
                ),
                initial_supervision=initial_supervision,
                initial_supervision_coef=(
                    args.c1_initial_supervision_coef if stage == "C1" else 0.0
                ),
                trainable_parameter_prefixes=(
                    C0_GOAL_PARAMETER_PREFIXES
                    if (
                        stage == "C0"
                        and c0_component == "goal"
                        and args.c0_target_head_counterfactual
                    )
                    else (
                        C0_SEARCH_PARAMETER_PREFIXES
                        if stage == "C0" and c0_component == "search"
                        else (
                            C1_TARGET_RESIDUAL_PARAMETER_PREFIXES
                            if stage == "C1" and args.c1_target_residual_only
                            else (
                                C1_SHARED_TARGET_PARAMETER_PREFIXES
                                if stage == "C1" and args.c1_shared_target_only
                                else (
                                    C1_TRAINABLE_PARAMETER_PREFIXES
                                    if stage == "C1" and args.c1_heads_only
                                    else None
                                )
                            )
                        )
                    )
                ),
                **_batch_update_kwargs(args, stage, c0_component),
            )
        )
        student_mean = sum(
            result["student_suffix_return"] for result in batch_episodes
        ) / len(batch_episodes)
        reference_mean = sum(
            result["reference_suffix_return"] for result in batch_episodes
        ) / len(batch_episodes)
        suffix_ratio = student_mean / reference_mean
        grouped_cow_rewards: dict[str, list[float]] = {}
        for result in batch_episodes:
            grouped_cow_rewards.setdefault(
                str(result["anchor_group_id"]), []
            ).append(float(result["cow_reward_sum"]))
        cow_reward_sum = sum(
            reward
            for rewards in grouped_cow_rewards.values()
            for reward in rewards
        )
        positive_cow_episode_count = sum(
            reward > 1e-12
            for rewards in grouped_cow_rewards.values()
            for reward in rewards
        )
        cow_reference_sum = sum(
            max(rewards) * len(rewards)
            for rewards in grouped_cow_rewards.values()
        )
        positive_cow_group_count = sum(
            max(rewards) > 1e-12
            for rewards in grouped_cow_rewards.values()
        )
        positive_cow_group_fraction = (
            positive_cow_group_count / len(grouped_cow_rewards)
        )
        search_direct_detection_fraction = sum(
            bool(result.get("search_discovery_ids"))
            for result in batch_episodes
        ) / len(batch_episodes)
        search_alive_distances = [
            float(result["search_nearest_alive_distance_m"])
            for result in batch_episodes
            if result.get("search_nearest_alive_distance_m") is not None
        ]
        search_proximity_return_mean = sum(
            float(result.get("search_proximity_return", 0.0))
            for result in batch_episodes
        ) / len(batch_episodes)
        search_segment_returns = [
            float(row["return"])
            for result in batch_episodes
            for row in result.get("search_segment_credits", ())
        ]
        search_grid_novelty_return_sum = sum(
            float(row.get("grid_novelty_return", 0.0))
            for result in batch_episodes
            for row in result.get("search_segment_credits", ())
        )
        search_unique_discovery_credit_count = sum(
            int(row.get("unique_discovery_count", 0))
            for result in batch_episodes
            for row in result.get("search_segment_credits", ())
        )
        cow_return_ratio = (
            cow_reward_sum / cow_reference_sum
            if cow_reference_sum > 1e-12
            else 0.0
        )
        score_mean = sum(
            float(result["score"]) for result in batch_episodes
        ) / len(batch_episodes)
        teacher_score_mean = sum(
            float(result["teacher_score"]) for result in batch_episodes
        ) / len(batch_episodes)
        fixed_target_score_mean = sum(
            float(result["fixed_target_score"])
            for result in batch_episodes
        ) / len(batch_episodes)
        teacher_fixed_target_score_mean = sum(
            float(result["teacher_fixed_target_score"])
            for result in batch_episodes
        ) / len(batch_episodes)
        fixed_target_score_delta_mean = (
            fixed_target_score_mean - teacher_fixed_target_score_mean
        )
        paired_score_ratio = score_mean / teacher_score_mean
        l_resources = [
            result.get("l_resource_outcomes", {})
            for result in batch_episodes
        ]
        team_l_coverage_mean = sum(
            float(row.get("l_9500_coverage", 0.0)) for row in l_resources
        ) / len(l_resources)
        team_explorer_coverage_mean = sum(
            float(row.get("explorer_9500_coverage", 0.0))
            for row in l_resources
        ) / len(l_resources)
        team_full_coverage_fraction = sum(
            float(row.get("l_9500_coverage", 0.0)) >= 1.0 - 1e-12
            for row in l_resources
        ) / len(l_resources)
        team_explorer_full_coverage_fraction = sum(
            float(row.get("explorer_9500_coverage", 0.0))
            >= 1.0 - 1e-12
            for row in l_resources
        ) / len(l_resources)
        l_interception_rate_mean = sum(
            float(row.get("interception_rate", 0.0)) for row in l_resources
        ) / len(l_resources)
        l_timeout_mean = sum(
            int(row.get("timeout", 0)) for row in l_resources
        ) / len(l_resources)
        l_search_reserve_mean = sum(
            int(row.get("search_reserve_after_required_hits", 0))
            for row in l_resources
        ) / len(l_resources)
        l_timeout_to_required_hits_ratio_mean = sum(
            float(row.get("timeout_to_required_hits_ratio", 0.0))
            for row in l_resources
        ) / len(l_resources)
        explorer_resource_sufficient_fraction = sum(
            int(row.get("search_reserve_after_required_hits", 0))
            >= int(row.get("required_distinct_9500_hits", 9))
            for row in l_resources
        ) / len(l_resources)
        destroyed_9500_mean = sum(
            len(row.get("destroyed_9500_ids", ())) for row in l_resources
        ) / len(l_resources)
        destroyed_all_9500_fraction = sum(
            len(row.get("destroyed_9500_ids", ()))
            >= int(row.get("objective_9500_count", 9))
            for row in l_resources
        ) / len(l_resources)
        reserved_attack_intercepted_mean = sum(
            int(result.get("reserved_attack_l_intercepted_count", 0))
            for result in batch_episodes
        ) / len(batch_episodes)
        reserved_attack_hitter_mean = sum(
            int(result.get("reserved_attack_l_hitter_count", 0))
            for result in batch_episodes
        ) / len(batch_episodes)
        ratio = (
            (
                (
                    team_explorer_full_coverage_fraction
                    if args.c0_team_search
                    else search_direct_detection_fraction
                )
                if c0_component == "search"
                else (
                    suffix_ratio
                    if c0_component == "lifecycle"
                    else paired_score_ratio
                )
            )
            if stage == "C0" and args.anchor_replicas == 1
            else cow_return_ratio if stage == "C0" else suffix_ratio
        )
        paired_non_loss_fraction = sum(
            delta >= -paired_tolerance for delta in score_deltas
        ) / len(score_deltas)
        c0_acceptance_coverage = (
            c0_lifecycle_launch_rate
            if stage == "C0" and c0_component == "lifecycle"
            else (
                search_direct_detection_fraction
                if (
                    stage == "C0"
                    and c0_component == "search"
                    and not args.c0_team_search
                )
                else team_explorer_full_coverage_fraction
                if stage == "C0" and c0_component == "search"
                else positive_cow_group_fraction
                if not (stage == "C0" and args.anchor_replicas == 1)
                else paired_non_loss_fraction
            )
        )
        advanced = (
            not args.evaluation_only
            and not args.candidate_only
            and ratio >= args.threshold
            and (stage != "C1" or c1_component == "joint")
            and (
                stage != "C0"
                or c0_acceptance_coverage >= 0.75
            )
            and (
                stage != "C0"
                or c0_component != "search"
                or not args.c0_team_search
                or explorer_resource_sufficient_fraction >= 1.0
            )
            and (
                stage != "C0"
                or c0_component != "search"
                or fixed_target_score_delta_mean >= -paired_tolerance
            )
        )
        frontier_additions = 0
        if stage in {"C3a", "C3", "C4"}:
            for result in batch_episodes:
                candidate = student_frontier_candidate(result, targets)
                if candidate is not None:
                    frontier_additions += int(pareto_insert(
                        frontier, candidate, support_qualities
                    ))
        batch_record = {
            "batch": batch_index,
            "stage": stage,
            "c0_component": c0_component,
            "c0_team_search": bool(
                args.c0_team_search
                and stage == "C0"
                and c0_component == "search"
            ),
            "c0_team_search_max_controlled_l": (
                args.c0_team_search_max_controlled_l
                if args.c0_team_search
                and stage == "C0"
                and c0_component == "search"
                else None
            ),
            "c0_team_search_attack_reserve": (
                args.c0_team_search_attack_reserve
                if args.c0_team_search
                and stage == "C0"
                and c0_component == "search"
                else None
            ),
            "c0_search_selector_offset": (
                args.c0_search_selector_offset
                if stage == "C0" and c0_component == "search"
                else None
            ),
            "c1_component": c1_component,
            "c1_target_supervision_coef": (
                args.c1_target_supervision_coef if stage == "C1" else 0.0
            ),
            "c0_target_supervision_coef": (
                args.c0_target_supervision_coef
                if stage == "C0" and c0_component == "goal"
                else 0.0
            ),
            "c0_target_head_counterfactual": bool(
                stage == "C0"
                and c0_component == "goal"
                and args.c0_target_head_counterfactual
            ),
            "c0_dense_target_counterfactual": bool(
                stage == "C0"
                and c0_component == "goal"
                and args.c0_dense_target_counterfactual
            ),
            "c0_goal_counterfactual_samples": (
                args.c0_goal_counterfactual_samples
                if stage == "C0" and c0_component == "goal"
                else 0
            ),
            "c1_initial_supervision_coef": (
                args.c1_initial_supervision_coef if stage == "C1" else 0.0
            ),
            "anchor_replica_count": (
                args.anchor_replicas if stage == "C0" else None
            ),
            "c4_boundary": boundary,
            "student_suffix_return_mean": student_mean,
            "reference_suffix_return_mean": reference_mean,
            "suffix_return_ratio": suffix_ratio,
            "score_mean": score_mean,
            "score_min": min(
                float(result["score"]) for result in batch_episodes
            ),
            "score_max": max(
                float(result["score"]) for result in batch_episodes
            ),
            "teacher_score_mean": teacher_score_mean,
            "fixed_target_score_mean": fixed_target_score_mean,
            "teacher_fixed_target_score_mean": (
                teacher_fixed_target_score_mean
            ),
            "fixed_target_score_delta_mean": fixed_target_score_delta_mean,
            "paired_score_ratio": paired_score_ratio,
            "paired_score_delta_mean": sum(score_deltas) / len(score_deltas),
            "paired_score_wins": sum(
                delta > paired_tolerance for delta in score_deltas
            ),
            "paired_score_ties": sum(
                abs(delta) <= paired_tolerance for delta in score_deltas
            ),
            "paired_score_losses": sum(
                delta < -paired_tolerance for delta in score_deltas
            ),
            "paired_score_non_loss_fraction": paired_non_loss_fraction,
            "batch_seed_count": len(selected_seeds),
            "batch_distinct_seed_count": distinct_seed_count,
            "rollout_wall_seconds": batch_rollout_wall_seconds,
            "cow_reward_sum": cow_reward_sum,
            "positive_cow_episode_count": positive_cow_episode_count,
            "cow_empirical_reference_sum": cow_reference_sum,
            "cow_probe_episode_count": sum(
                not bool(result["counterfactual_reused"])
                for result in batch_episodes
            ),
            "cow_reused_episode_count": sum(
                bool(result["counterfactual_reused"])
                for result in batch_episodes
            ),
            "positive_cow_group_count": positive_cow_group_count,
            "positive_cow_group_fraction": positive_cow_group_fraction,
            "search_direct_detection_fraction": (
                search_direct_detection_fraction
            ),
            "team_l_9500_coverage_mean": team_l_coverage_mean,
            "team_explorer_9500_coverage_mean": (
                team_explorer_coverage_mean
            ),
            "team_full_9500_coverage_fraction": (
                team_full_coverage_fraction
            ),
            "team_explorer_full_9500_coverage_fraction": (
                team_explorer_full_coverage_fraction
            ),
            "l_interception_rate_mean": l_interception_rate_mean,
            "l_timeout_mean": l_timeout_mean,
            "l_search_reserve_after_9_hits_mean": l_search_reserve_mean,
            "l_timeout_to_required_9_hits_ratio_mean": (
                l_timeout_to_required_hits_ratio_mean
            ),
            "explorer_resource_at_least_9_fraction": (
                explorer_resource_sufficient_fraction
            ),
            "destroyed_9500_mean": destroyed_9500_mean,
            "destroyed_all_9500_fraction": destroyed_all_9500_fraction,
            "reserved_attack_l_intercepted_mean": (
                reserved_attack_intercepted_mean
            ),
            "reserved_attack_l_hitter_mean": reserved_attack_hitter_mean,
            "search_nearest_alive_distance_m_mean": (
                sum(search_alive_distances) / len(search_alive_distances)
                if search_alive_distances else None
            ),
            "search_nearest_alive_distance_m_min": (
                min(search_alive_distances) if search_alive_distances else None
            ),
            "search_nearest_alive_distance_m_max": (
                max(search_alive_distances) if search_alive_distances else None
            ),
            "search_proximity_return_mean": search_proximity_return_mean,
            "search_segment_credit_count": len(search_segment_returns),
            "search_segment_credit_mean": (
                sum(search_segment_returns) / len(search_segment_returns)
                if search_segment_returns else None
            ),
            "search_segment_credit_positive_count": sum(
                value > 1e-12 for value in search_segment_returns
            ),
            "search_segment_credit_zero_count": sum(
                abs(value) <= 1e-12 for value in search_segment_returns
            ),
            "search_grid_novelty_return_sum": (
                search_grid_novelty_return_sum
            ),
            "search_unique_discovery_credit_count": (
                search_unique_discovery_credit_count
            ),
            "cow_return_ratio": cow_return_ratio,
            "c0_actor_advantage_mode": c0_actor_advantage_mode,
            "c0_actor_advantage_mean": (
                sum(c0_actor_advantages) / len(c0_actor_advantages)
                if c0_actor_advantages is not None else None
            ),
            "c0_actor_advantage_positive_count": (
                sum(value > 1e-12 for value in c0_actor_advantages)
                if c0_actor_advantages is not None else None
            ),
            "c0_actor_advantage_zero_count": (
                sum(abs(value) <= 1e-12 for value in c0_actor_advantages)
                if c0_actor_advantages is not None else None
            ),
            "c0_actor_advantage_negative_count": (
                sum(value < -1e-12 for value in c0_actor_advantages)
                if c0_actor_advantages is not None else None
            ),
            "c0_lifecycle_decision_count": c0_lifecycle_decision_count,
            "c0_lifecycle_launch_count": c0_lifecycle_launch_count,
            "c0_lifecycle_launch_rate": c0_lifecycle_launch_rate,
            "c0_acceptance_coverage": c0_acceptance_coverage,
            "c0_acceptance_coverage_kind": (
                "legal_launch_rate"
                if stage == "C0" and c0_component == "lifecycle"
                else (
                    "nonintercepted_nonhitter_l_full_9_of_9_detection_fraction"
                    if (
                        stage == "C0"
                        and c0_component == "search"
                        and args.c0_team_search
                    )
                    else "direct_legal_detection_fraction"
                    if stage == "C0" and c0_component == "search"
                    else "positive_causal_group_fraction"
                    if not (stage == "C0" and args.anchor_replicas == 1)
                    else "paired_e01_non_loss_fraction"
                )
            ),
            "return_ratio": ratio,
            "threshold": args.threshold,
            "curriculum_advanced": advanced,
            "deterministic_actor": args.deterministic_actor,
            "evaluation_only": args.evaluation_only,
            "candidate_only": args.candidate_only,
            "ppo": metrics,
            "frontier_additions": frontier_additions,
        }
        batches.append(batch_record)
        episodes.extend(batch_episodes)
        if advanced:
            if stage == "C0" and c0_index + 1 < len(C0_COMPONENTS):
                c0_index += 1
            elif stage == "C4" and c4_index + 1 < len(C4_BOUNDARIES):
                c4_index += 1
            else:
                stage_index += 1
                c0_index = 0
                c4_index = 0
        payload = {
            "schema_version": 1,
            "status": "complete" if stage_index >= len(STAGES) else "running",
            "checkpoint": str(args.checkpoint.resolve()),
            "current_stage": STAGES[stage_index] if stage_index < len(STAGES) else "complete",
            "current_c0_component": (
                C0_COMPONENTS[c0_index]
                if stage_index < len(STAGES) and STAGES[stage_index] == "C0"
                else None
            ),
            "current_c1_component": (
                args.c1_controlled_components
                if stage_index < len(STAGES) and STAGES[stage_index] == "C1"
                else None
            ),
            "current_c4_boundary": (
                C4_BOUNDARIES[c4_index]
                if stage_index < len(STAGES) and STAGES[stage_index] == "C4"
                else None
            ),
            "policy_control_mask": "student_only",
            "teacher_steps_in_ppo_rollout": 0,
            "formal_reward_only": True,
            "require_distinct_batch_seeds": args.require_distinct_batch_seeds,
            "deterministic_actor": args.deterministic_actor,
            "c1_target_supervision_coef": args.c1_target_supervision_coef,
            "c0_target_supervision_coef": args.c0_target_supervision_coef,
            "c0_target_head_counterfactual": (
                args.c0_target_head_counterfactual
            ),
            "c0_dense_target_counterfactual": (
                args.c0_dense_target_counterfactual
            ),
            "c0_goal_counterfactual_samples": (
                args.c0_goal_counterfactual_samples
            ),
            "candidate_only": args.candidate_only,
            "c1_initial_supervision_coef": args.c1_initial_supervision_coef,
            "teacher_decision_dataset_manifest": (
                None
                if args.teacher_decision_dataset_manifest is None
                else str(args.teacher_decision_dataset_manifest.resolve())
            ),
            "temporal_attack_options": True,
            "dense_target_counterfactual": (
                args.c0_dense_target_counterfactual
                or args.c1_dense_target_counterfactual
            ),
            "attack_option_min_dwell_steps": args.attack_option_min_dwell_steps,
            "c0_search_option_chaining": args.c0_search_option_chaining,
            "c0_team_search": args.c0_team_search,
            "c0_team_search_max_controlled_l": (
                args.c0_team_search_max_controlled_l
            ),
            "c0_team_search_attack_reserve": (
                args.c0_team_search_attack_reserve
            ),
            "c0_search_option_dwell_steps": args.c0_search_option_dwell_steps,
            "c0_search_option_reopen_distance_km": (
                args.c0_search_option_reopen_distance_km
            ),
            "c0_search_option_emergency_reopen_distance_km": (
                args.c0_search_option_emergency_reopen_distance_km
            ),
            "c0_search_reachable_mask": args.c0_search_reachable_mask,
            "c0_search_leg_min_distance_km": (
                args.c0_search_leg_min_distance_km
            ),
            "c0_search_leg_max_distance_km": (
                args.c0_search_leg_max_distance_km
            ),
            "c0_deterministic_evasion": args.c0_deterministic_evasion,
            "c3a_persistent_student_option_control": (
                not args.disable_option_control_handoff
            ),
            "c0_components": list(C0_COMPONENTS),
            "c0_search_selector_offset": args.c0_search_selector_offset,
            "c0_anchor_seeds": [
                int(row["seed"]) for row in c0_support_rows
            ],
            "c0_actor_advantage_mode": c0_actor_advantage_mode,
            "empirical_leave_one_out_advantage": (
                stage == "C0" and args.anchor_replicas > 1
            ),
            "teacher_support_probability": SUPPORT_PROBABILITY[stage],
            "pareto_frontier": frontier,
            "batches": batches,
            "episodes": episodes,
        }
        progress_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output_dir / "training_records.json").write_text(
            json.dumps([
                {
                    "start_state_id": result["start_state_id"],
                    "curriculum_stage": result["stage"],
                    "c0_component": result["c0_component"],
                    "anchor_group_id": result["anchor_group_id"],
                    "prefix_step": result["snapshot_prefix_step"],
                    "controlled_unit_count": result["controlled_sample_count"],
                    "suffix_official_reward": result["student_suffix_return"],
                    "reference_suffix_official_reward": result[
                        "reference_suffix_return"
                    ],
                    "score": result["score"],
                    "seed": result["seed"],
                    "teacher_steps_in_ppo_rollout": 0,
                }
                for result in episodes
            ], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(batch_record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
