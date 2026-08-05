"""Standalone B0--B3 scenario runner using the existing engine public interfaces."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))

from red_strategy_lab import B0RandomPolicy, B1PriorityPolicy, B2StaticAssignmentPolicy, B3RollingRulePolicy, Decision, GlobalRules, Position, Target
from red_strategy_lab.adapter import launch_rows, observation_from_engine
from red_strategy_lab.evaluation import EpisodeMetrics


def load_scenario(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def targets_from_scenario(scenario: Mapping[str, Any]) -> tuple[Target, ...]:
    entities = scenario["imagineProfile"]["entityList"]
    targets: list[Target] = []
    for entry in entities:
        entity = entry["entity"]
        if entity.get("sideId") == 0 or entity.get("entityType") != 9400:
            continue
        position = entity["lla"]
        targets.append(Target(
            entity_id=int(entity["id"]),
            position=Position(float(position["x"]), float(position["y"]), float(position.get("z", 0.0))),
            value=max(1.0, float(entity.get("survivePoints", 1.0))),
            health=float(entity.get("survivePoints", 1.0)),
        ))
    if not targets:
        raise ValueError("scenario does not contain blue-side type-9400 targets")
    return tuple(targets)


def initial_observation_from_scenario(scenario: Mapping[str, Any], targets: tuple[Target, ...]):
    """Build the same red-platform observation used at the first engine step."""
    entities: dict[int, dict[str, Any]] = {}
    for entry in scenario["imagineProfile"]["entityList"]:
        entity = entry["entity"]
        position = entity["lla"]
        entities[int(entity["id"])] = {
            "position": {"lon": position["x"], "lat": position["y"], "alt": position.get("z", 0.0)},
            "health": entity.get("survivePoints", 0.0),
            "type": entity["entityType"],
            "side": entity["sideId"],
        }
    return observation_from_engine({"step": 0, "entities": entities}, targets)


def choose_policy(name: str, rules: GlobalRules, seed: int):
    policies = {
        "b0": B0RandomPolicy(rules, seed),
        "b1": B1PriorityPolicy(rules),
        "b2": B2StaticAssignmentPolicy(rules),
        "b3": B3RollingRulePolicy(rules),
    }
    return policies[name]


def engine_observation(engine: Any) -> dict[str, Any]:
    entities: dict[int, dict[str, Any]] = {}
    for simulator in engine.simulator_factory.get_all_simulators():
        entity = simulator.entity_ext.entity
        entities[entity.id] = {
            "position": {"lon": entity.lla.x, "lat": entity.lla.y, "alt": entity.lla.z},
            "health": entity.survivePoints,
            "type": entity.entityType,
            "side": entity.sideId,
        }
    return {"step": engine.current_step, "entities": entities}


def run(args: argparse.Namespace) -> Path:
    vendor_path = os.environ.get("RED_STRATEGY_VENDOR")
    if vendor_path and Path(vendor_path).exists():
        vendor = Path(vendor_path)
        sys.path.insert(0, str(vendor))

    # The package initializer imports the renderer eagerly.  The standalone
    # runner requires only engine modules, so it exposes the package path
    # without importing the renderer or TrainingEnv.
    package = types.ModuleType("envengine")
    package.__path__ = [str(ROOT.parent / "envengine")]
    sys.modules.setdefault("envengine", package)
    from envengine.engine.engine import Engine
    from envengine.agent_manager.actions.aircraft_action.missile_launch import MissileLaunchAction
    from envengine.sdk.base_struct.Basic.Vector3d import Vector3d
    from envengine.sdk.base_struct.Message.Command import Command
    from envengine.sdk.base_struct.profile.profile import Profile

    scenario = load_scenario(Path(args.scenario))
    profile = Profile.from_dict(scenario)
    engine = Engine(profile, speed_multiplier=1_000_000)
    targets = targets_from_scenario(scenario)
    rules = GlobalRules(target_capacity=args.target_capacity, min_launch_interval=args.min_launch_interval, replan_interval=args.replan_interval)
    policy = choose_policy(args.baseline, rules, args.seed)
    metrics = EpisodeMetrics(initial_target_value=sum(target.value for target in targets))
    launched_ids: set[int] = set()
    planned = None

    for _ in range(args.max_steps):
        observation = observation_from_engine(engine_observation(engine), targets, launched_ids)
        if args.baseline == "b3" or planned is None:
            planned = policy.decide(observation)
        rows = launch_rows(planned, targets, observation.step)
        launched_now = {int(row[1]) for row in rows}
        metrics.record_decision(
            Decision(assignments=tuple(item for item in planned.assignments if item.platform_id in launched_now)),
            rules.target_capacity,
        )
        launched_ids.update(int(row[1]) for row in rows)
        commands = []
        for row in rows:
            action = MissileLaunchAction(
                executor_id=int(row[1]), target=Vector3d(float(row[2]), float(row[3]), 0.0)
            ).to_dict()
            attributes = {key: value for key, value in action.items() if key not in ("executor_id", "commandType_id")}
            commands.append(Command(
                executorId=action["executor_id"], commandTypeId=action["commandType_id"], commandAttributes=attributes
            ))
        engine.step(commands)

    final_observation = engine_observation(engine)
    final_entities = final_observation["entities"]
    metrics.loss_count = sum(1 for platform_id in launched_ids if float(final_entities.get(platform_id, {}).get("health", 0.0)) <= 0)
    metrics.destroyed_target_value = sum(target.value for target in targets if float(final_entities.get(target.entity_id, {}).get("health", 0.0)) <= 0)
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = ROOT.parent / "results" / "red_strategy_lab" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": args.baseline,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "launched_platform_ids": sorted(launched_ids),
        "metrics": metrics.summary(),
        "assignments": [assignment.__dict__ for assignment in (planned.assignments if planned else ())],
    }
    output = output_dir / "summary.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=str(ROOT.parent / "scenarios" / "platform.json"))
    parser.add_argument("--baseline", choices=("b0", "b1", "b2", "b3"), default="b1")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--target-capacity", type=int, default=2)
    parser.add_argument("--min-launch-interval", type=int, default=1)
    parser.add_argument("--replan-interval", type=int, default=10)
    args = parser.parse_args()
    output = run(args)
    print(output)


if __name__ == "__main__":
    main()
