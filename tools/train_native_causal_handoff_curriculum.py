"""Train MAPPO by progressively taking over causal teacher action units."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEACHER_MODEL = ROOT / "models/r9_mappo_e01.pt"
ATTACK_COMMANDS = {200, 3014}
RED_TYPES = {21000, 21001, 21002}
REQUIRED_TARGETS = (2551, 2552, 169)
STAGES = ("single_unit", "same_step_group", "two_groups", "four_groups", "full_chain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def ordered_trajectories(manifest: dict) -> list[dict]:
    rank = {target_id: index for index, target_id in enumerate(REQUIRED_TARGETS)}
    return sorted(
        manifest["trajectories"],
        key=lambda row: (
            rank.get(int(row["assigned_target_id"]), len(rank)),
            -float(row["teacher_score"]),
            int(row["seed"]),
        ),
    )


def _unit(step: int, action: dict) -> dict:
    entity_id = int(action["executor_id"])
    command_type = int(action["commandType_id"])
    return {
        "unit_id": f"s{step:04d}_e{entity_id}_c{command_type}",
        "timestep": int(step),
        "executor_id": entity_id,
        "command_type": command_type,
    }


def causal_attack_groups(row: dict) -> list[list[dict]]:
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
    units_by_step: dict[int, dict[str, dict]] = {}
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
    anchor_entity = int(anchor["attacking_entity_id"])
    anchor_type = int(anchor["decision_type"])
    anchor_id = f"s{anchor_step:04d}_e{anchor_entity}_c{anchor_type}"
    anchor_unit = {
        "unit_id": anchor_id,
        "timestep": anchor_step,
        "executor_id": anchor_entity,
        "command_type": anchor_type,
    }
    units_by_step.setdefault(anchor_step, {})[anchor_id] = anchor_unit
    ordered_steps = [
        anchor_step,
        *sorted((step for step in units_by_step if step != anchor_step), reverse=True),
    ]
    groups = [
        list(units_by_step[step].values()) for step in ordered_steps
    ]
    groups[0] = [
        anchor_unit,
        *(unit for unit in groups[0] if unit["unit_id"] != anchor_id),
    ]
    return groups


def units_for_stage(groups: list[list[dict]], stage: str) -> list[dict]:
    if stage == "single_unit":
        return [groups[0][0]]
    if stage == "same_step_group":
        return list(groups[0])
    group_count = {
        "two_groups": 2,
        "four_groups": 4,
        "full_chain": len(groups),
    }[stage]
    return [unit for group in groups[:group_count] for unit in group]


def run_episode(
    *,
    episode: int,
    attempt: int,
    stage: str,
    row: dict,
    groups: list[list[dict]],
    manifest: dict,
    checkpoint: Path,
    output_dir: Path,
) -> dict:
    seed = int(row["seed"])
    target_id = int(row["assigned_target_id"])
    controlled_units = units_for_stage(groups, stage)
    episode_dir = output_dir / "episodes" / f"{episode:05d}_{stage}_s{seed:04d}_a{attempt:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    spec_path = episode_dir / "handoff_spec.json"
    spec_path.write_text(
        json.dumps({
            "schema_version": 1,
            "curriculum_stage": stage,
            "seed": seed,
            "target_id": target_id,
            "damage_step": int(row["damage_anchors"][str(target_id)]["damage_step"]),
            "controlled_units": controlled_units,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "BLUE_POLICY": "b0_fixed_ratio_random",
        "BLUE_POLICY_SEED": str(seed),
        "SIMULATION_SEED": str(seed),
        "RED_POLICY": "r12_unified_mappo",
        "RED_POLICY_SEED": str(seed + attempt - 1),
        "RED_MOTION_POLICY": "unified_mappo",
        "RED_LEARNING_TRAIN": "1",
        "RED_LEARNING_MODEL": str(checkpoint.resolve()),
        "RED_REWARD_MODE": "weighted_damage_guided_cow",
        "RED_UNIFIED_DYNAMIC_LIFECYCLE": "1",
        "RED_NATIVE_SNAPSHOT_GUIDANCE": "1",
        "RED_NATIVE_GUIDANCE_TRACE": str(Path(row["trace"]).resolve()),
        "RED_NATIVE_GUIDANCE_STAGE": "causal_handoff",
        "RED_NATIVE_HANDOFF_SPEC": str(spec_path.resolve()),
        "RED_NATIVE_TEACHER_MODEL": str(TEACHER_MODEL),
        "RED_NATIVE_ROLLOUT_DEVICE": "cuda",
        "RED_UNIFIED_UPDATE_DEVICE": "cuda",
        "RED_NATIVE_HANDOFF_COW_WORKERS": "8",
    })
    libraries = [
        str(ROOT / "core" / "envengine" / "simulator" / "models" / "HXDMissileModel"),
        str(Path(sys.prefix) / "lib"),
    ]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--scenario", str(Path(manifest["scenario"]).resolve()),
            "--output-dir", str((episode_dir / "sim_results").resolve()),
            "--max-steps", "1200",
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
    log_path = episode_dir / "run.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"episode {episode} exited {completed.returncode}; log={log_path}"
        )
    summaries = [
        json.loads(line.removeprefix("FINAL_SUMMARY "))
        for line in completed.stdout.splitlines()
        if line.startswith("FINAL_SUMMARY ")
    ]
    summary = summaries[0]
    handoff = summary["native_causal_handoff"]
    return {
        "episode": episode,
        "attempt": attempt,
        "stage": stage,
        "seed": seed,
        "assigned_target_id": target_id,
        "controlled_unit_count": int(handoff["controlled_unit_count"]),
        "positive_unit_count": int(handoff["positive_unit_count"]),
        "score": float(summary["score"]["score"]),
        "destroyed_ids": list(map(int, summary["score"]["destroyed_ids"])),
        "official_reward_return": float(summary["red"]["official_reward_return"]),
        "credited_reward_sum": float(handoff["credited_reward_sum"]),
        "factual_target_return": float(handoff["factual_target_return"]),
        "counterfactual_deltas": list(map(float, handoff["counterfactual_deltas"])),
        "teacher_steps_in_ppo_rollout": int(handoff["teacher_steps_in_ppo_rollout"]),
        "ppo": dict(summary["red"]["dynamic_catalogue"]["unified_mappo"]["last_metrics"]),
        "log": str(log_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = ordered_trajectories(manifest)
    groups_by_seed = {
        int(row["seed"]): causal_attack_groups(row) for row in rows
    }
    stage_by_seed = {int(row["seed"]): 0 for row in rows}
    attempts = {int(row["seed"]): 0 for row in rows}
    progress: list[dict] = []
    progress_path = args.output_dir / "training_progress.json"
    episode = 0
    while any(index < len(STAGES) for index in stage_by_seed.values()):
        for row in rows:
            seed = int(row["seed"])
            stage_index = stage_by_seed[seed]
            if stage_index >= len(STAGES):
                continue
            stage = STAGES[stage_index]
            attempts[seed] += 1
            episode += 1
            result = run_episode(
                episode=episode,
                attempt=attempts[seed],
                stage=stage,
                row=row,
                groups=groups_by_seed[seed],
                manifest=manifest,
                checkpoint=args.checkpoint,
                output_dir=args.output_dir,
            )
            mastered = (
                result["positive_unit_count"] > 0
                and result["credited_reward_sum"] > 0.0
            )
            if mastered:
                stage_by_seed[seed] += 1
                attempts[seed] = 0
            result["stage_mastered"] = mastered
            result["next_stage"] = (
                STAGES[stage_by_seed[seed]]
                if stage_by_seed[seed] < len(STAGES) else "complete"
            )
            progress.append(result)
            progress_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "status": "running",
                    "episode_count": len(progress),
                    "stage_by_seed": {
                        str(key): (
                            STAGES[value] if value < len(STAGES) else "complete"
                        )
                        for key, value in stage_by_seed.items()
                    },
                    "episodes": progress,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    payload["status"] = "complete"
    progress_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
