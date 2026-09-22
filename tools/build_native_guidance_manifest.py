"""Build curriculum anchors from native positive teacher trajectories."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


LAUNCH = 200
RETARGET = 3014
RED_TYPES = {21000, 21001, 21002}
REQUIRED_TARGETS = (2551, 2552, 169)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection-manifest",
        type=Path,
        action="append",
        required=True,
        help="teacher collection manifest; repeat to merge disjoint seeds",
    )
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scenario_targets(path: Path, objective_ids: set[int]) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    found: dict[int, dict] = {}

    def visit(value) -> None:
        if isinstance(value, dict):
            entity_id = value.get("id")
            if entity_id in objective_ids:
                found[int(entity_id)] = {
                    "type": int(value["entityType"]),
                    "lon": float(value["lla"]["x"]),
                    "lat": float(value["lla"]["y"]),
                }
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return found


def nearest_target(command: dict, targets: dict[int, dict]) -> int | None:
    point = command.get("target")
    if not point:
        return None
    target_id, distance = min(
        (
            entity_id,
            math.hypot(
                float(point["x"]) - target["lon"],
                float(point["y"]) - target["lat"],
            ),
        )
        for entity_id, target in targets.items()
    )
    return target_id if distance < 1e-4 else None


def damage_vector(
    payload: dict,
    objective_ids: set[int],
    through_step: int,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for target_id in objective_ids:
        all_events = [
            event
            for step in payload["steps"]
            for event in step["causal_events"]
            if int(event.get("target_entity_id", -1)) == target_id
            and "target_health_before" in event
            and "target_health_after" in event
        ]
        initial_health = (
            max(float(event["target_health_before"]) for event in all_events)
            if all_events else 1.0
        )
        observed = [
            event for event in all_events
            if int(event["step"]) <= int(through_step)
        ]
        final_health = (
            min(float(event["target_health_after"]) for event in observed)
            if observed else initial_health
        )
        values[str(target_id)] = max(
            0.0, min(1.0, (initial_health - final_health) / initial_health)
        )
    return values


def trajectory_record(row: dict, targets: dict[int, dict]) -> dict:
    path = Path(row["trace"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    objective_ids = set(targets)
    actions = [
        (int(step["step"]), action)
        for step in payload["steps"]
        for action in step["actions"]
    ]
    events = [
        event
        for step in payload["steps"]
        for event in step["causal_events"]
        if int(event.get("target_entity_id", -1)) in objective_ids
        and int(event.get("attacking_entity_type", -1)) in RED_TYPES
        and float(event.get("actual_damage", 0.0)) > 0.0
    ]
    anchors: dict[str, dict] = {}
    for target_id in row["destroyed_ids"]:
        target_events = [
            event for event in events
            if int(event["target_entity_id"]) == int(target_id)
        ]
        if not target_events:
            continue
        event = max(target_events, key=lambda item: int(item["step"]))
        attacker = int(event["attacking_entity_id"])
        decisions = [
            (step, action)
            for step, action in actions
            if int(action.get("executor_id", -1)) == attacker
            and int(action.get("commandType_id", -1)) in {LAUNCH, RETARGET}
            and step <= int(event["step"])
        ]
        if not decisions:
            continue
        decision_step, decision = max(decisions, key=lambda item: item[0])
        anchors[str(target_id)] = {
            "snapshot_prefix_step": max(0, decision_step - 1),
            "decision_step": decision_step,
            "decision_type": int(decision["commandType_id"]),
            "attacking_entity_id": attacker,
            "damage_step": int(event["step"]),
        }

    detected_steps = [
        step
        for step, action in actions
        if int(action.get("commandType_id", -1)) == LAUNCH
        and nearest_target(action, targets) in {
            entity_id for entity_id, target in targets.items()
            if target["type"] == 9500
        }
    ]
    return {
        "seed": int(row["seed"]),
        "teacher_score": float(row["score"]),
        "destroyed_ids": list(map(int, row["destroyed_ids"])),
        "trace": str(path.resolve()),
        "damage_anchors": anchors,
        "detection_snapshot_prefix_step": (
            max(0, min(detected_steps) - 1) if detected_steps else 0
        ),
        "random_seed_tuple": {
            "blue": int(payload.get("blue_seed", row["seed"])),
            "red": int(payload.get("red_seed", row["seed"])),
            "simulation": int(payload.get("simulation_seed", row["seed"])),
        },
        "terminal_damage_vector": damage_vector(
            payload, objective_ids, payload["steps"][-1]["step"]
        ),
    }


def main() -> None:
    args = parse_args()
    collections = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.collection_manifest
    ]
    trajectories = [
        row
        for collection in collections
        for row in collection["trajectories"]
    ]
    seeds = [int(row["seed"]) for row in trajectories]
    duplicate_seeds = sorted(
        seed for seed, count in Counter(seeds).items() if count > 1
    )
    if duplicate_seeds:
        raise ValueError(
            f"collection manifests contain duplicate seeds: {duplicate_seeds}"
        )
    objective_ids = {
        int(key)
        for row in trajectories
        for key in json.loads(Path(row["trace"]).read_text(encoding="utf-8"))
        ["summary"]["score"]["objective_weights"]
    }
    targets = scenario_targets(args.scenario, objective_ids)
    records = [trajectory_record(row, targets) for row in trajectories]
    positive = [record for record in records if record["teacher_score"] > 0.0]
    counts: Counter[int] = Counter()
    required_owner: dict[int, int] = {}
    for target_id in REQUIRED_TARGETS:
        candidates = [
            record for record in positive
            if str(target_id) in record["damage_anchors"]
        ]
        owner = max(candidates, key=lambda item: item["teacher_score"])
        required_owner[owner["seed"]] = target_id
    for record in sorted(positive, key=lambda item: item["teacher_score"], reverse=True):
        available = list(map(int, record["damage_anchors"]))
        target_id = required_owner.get(record["seed"])
        if target_id is None:
            target_id = min(available, key=lambda item: (counts[item], item))
        record["assigned_target_id"] = target_id
        record["damage_snapshot_prefix_step"] = record["damage_anchors"][str(target_id)]["snapshot_prefix_step"]
        prefix_step = int(record["damage_snapshot_prefix_step"])
        trace = json.loads(Path(record["trace"]).read_text(encoding="utf-8"))
        target_ids = set(targets)
        prefix_actions = [
            action
            for step in trace["steps"]
            if int(step["step"]) <= prefix_step
            for action in step["actions"]
        ]
        launched_ids = {
            int(action["executor_id"])
            for action in prefix_actions
            if int(action.get("commandType_id", -1)) == LAUNCH
        }
        discovered_ids = sorted({
            matched
            for action in prefix_actions
            if int(action.get("commandType_id", -1)) in {LAUNCH, RETARGET}
            for matched in [nearest_target(action, targets)]
            if matched is not None
        })
        deployment_ids = {
            int(action["executor_id"])
            for action in trace["deployment_actions"]
            if int(action.get("executor_id", -1)) >= 0
        }
        record["start_state_id"] = (
            f"native-s{int(record['seed']):04d}-tau{prefix_step + 1:04d}"
        )
        record["start_state_descriptor"] = {
            "native_trajectory": record["trace"],
            "random_seed_tuple": record["random_seed_tuple"],
            "handoff_step": prefix_step + 1,
            "boundary_damage_vector": damage_vector(
                trace, target_ids, prefix_step
            ),
            "terminal_damage_vector": record["terminal_damage_vector"],
            "scheduler_features": {
                "discovered_target_ids": discovered_ids,
                "launched_entity_ids": sorted(launched_ids),
                "remaining_attack_resource_fraction": (
                    (len(deployment_ids) - len(launched_ids)) / len(deployment_ids)
                    if deployment_ids else 0.0
                ),
            },
            "effective_causal_units": [
                {
                    "timestep": int(step["step"]),
                    "executor_id": int(action["executor_id"]),
                    "command_type": int(action["commandType_id"]),
                }
                for step in trace["steps"]
                if prefix_step < int(step["step"])
                <= int(record["damage_anchors"][str(target_id)]["damage_step"])
                for action in step["actions"]
                if int(action.get("commandType_id", -1)) in {LAUNCH, RETARGET}
            ],
        }
        counts[target_id] += 1
    result = {
        "schema_version": 1,
        "scenario": str(args.scenario.resolve()),
        "trajectory_count": len(positive),
        "mean_teacher_score": sum(row["teacher_score"] for row in positive) / len(positive),
        "covered_targets": sorted({target for row in positive for target in row["destroyed_ids"]}),
        "required_targets": list(REQUIRED_TARGETS),
        "assigned_target_histogram": dict(sorted(counts.items())),
        "trajectories": sorted(positive, key=lambda item: item["seed"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("trajectory_count", "mean_teacher_score", "covered_targets", "assigned_target_histogram")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
