"""Find the latest causally necessary teacher attack group for each trajectory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/opt/conda/envs/competition/bin/python")
TEACHER_MODEL = ROOT / "models/r9_mappo_e01.pt"
ATTACK_TYPES = {200, 3014}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def attack_groups(trace: dict, damage_step: int) -> list[tuple[int, list[dict]]]:
    groups: list[tuple[int, list[dict]]] = []
    for row in trace["steps"]:
        step = int(row["step"])
        if step > damage_step:
            break
        interventions = [
            {
                "timestep": step,
                "executor_id": int(action["executor_id"]),
                "command_type": int(action["commandType_id"]),
            }
            for action in row["actions"]
            if int(action.get("commandType_id", -1)) in ATTACK_TYPES
        ]
        if interventions:
            groups.append((step, interventions))
    suffix: list[dict] = []
    candidates: list[tuple[int, list[dict]]] = []
    for step, interventions in reversed(groups):
        suffix = interventions + suffix
        candidates.append((step, list(suffix)))
    return candidates


def run_probe(
    *,
    seed: int,
    trace_path: Path,
    target_id: int,
    branch_step: int,
    interventions: list[dict],
    output_dir: Path,
    scenario: Path,
) -> dict:
    probe_dir = output_dir / f"seed_{seed:04d}" / f"step_{branch_step:04d}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    spec_path = probe_dir / "spec.json"
    result_path = probe_dir / "result.json"
    log_path = probe_dir / "run.log"
    spec = {
        "schema_version": 1,
        "seed": seed,
        "trace": str(trace_path.resolve()),
        "target_id": target_id,
        "branch_step": branch_step,
        "null_action": "JOINT_SUFFIX_LAUNCH_TO_WAIT_RETARGET_TO_KEEP",
        "interventions": interventions,
    }
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "BLUE_POLICY": "b0_fixed_ratio_random",
        "BLUE_POLICY_SEED": str(seed),
        "SIMULATION_SEED": str(seed),
        "RED_POLICY": "r9_hierarchical_learning",
        "RED_POLICY_SEED": str(seed),
        "RED_MOTION_POLICY": "mappo",
        "RED_LEARNING_TRAIN": "0",
        "RED_LEARNING_MODEL": str(TEACHER_MODEL),
        "RED_NATIVE_GUIDANCE_TRACE": str(trace_path.resolve()),
        "RED_NATIVE_GUIDANCE_STAGE": "causal_boundary",
        "RED_NATIVE_CAUSAL_GROUP_SPEC": str(spec_path.resolve()),
        "RED_NATIVE_CAUSAL_GROUP_RESULT": str(result_path.resolve()),
    })
    libraries = [
        str(ROOT / "core" / "envengine" / "simulator" / "models" / "HXDMissileModel"),
        str(Path(sys.prefix) / "lib"),
    ]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    command = [
        str(PYTHON),
        "main.py",
        "--scenario", str(scenario.resolve()),
        "--output-dir", str((probe_dir / "sim_results").resolve()),
        "--max-steps", "1200",
        "--total-rounds", "1",
        "--render-mode", "none",
        "--disable-log-color",
    ]
    completed = subprocess.run(
        command,
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
            f"seed {seed} step {branch_step} exited {completed.returncode}; "
            f"log={log_path}"
        )
    return json.loads(result_path.read_text(encoding="utf-8"))


def analyze_trajectory(row: dict, output_dir: Path, scenario: Path) -> dict:
    seed = int(row["seed"])
    target_id = int(row["assigned_target_id"])
    trace_path = Path(row["trace"])
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    damage_step = int(row["damage_anchors"][str(target_id)]["damage_step"])
    probes: list[dict] = []
    for branch_step, interventions in attack_groups(trace, damage_step):
        result = run_probe(
            seed=seed,
            trace_path=trace_path,
            target_id=target_id,
            branch_step=branch_step,
            interventions=interventions,
            output_dir=output_dir,
            scenario=scenario,
        )
        probe = {
            "branch_step": branch_step,
            "intervention_count": len(interventions),
            "factual_target_return": float(result["factual"]["target_return"]),
            "counterfactual_target_return": float(
                result["counterfactual"]["target_return"]
            ),
            "target_return_delta": float(result["target_return_delta"]),
            "result": str(
                (output_dir / f"seed_{seed:04d}" / f"step_{branch_step:04d}" / "result.json").resolve()
            ),
        }
        probes.append(probe)
        if probe["target_return_delta"] > 1e-12:
            updated = dict(row)
            updated["damage_snapshot_prefix_step"] = branch_step - 1
            updated["causal_damage_boundary"] = {
                "branch_step": branch_step,
                "snapshot_prefix_step": branch_step - 1,
                "target_id": target_id,
                "joint_intervention_count": len(interventions),
                "target_return_delta": probe["target_return_delta"],
                "searched_steps_descending": [item["branch_step"] for item in probes],
                "evidence": probe["result"],
                "native_copy_on_write": True,
            }
            return updated
    raise RuntimeError(
        f"seed {seed} target {target_id} has no causally necessary attack group"
    )


def write_progress(path: Path, base: dict, rows: list[dict]) -> None:
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "trajectory_count": len(base["trajectories"]),
        "completed": len(rows),
        "trajectories": sorted(rows, key=lambda row: int(row["seed"])),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scenario = Path(manifest["scenario"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "causal_boundary_progress.json"
    completed_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                analyze_trajectory,
                row,
                args.output_dir,
                scenario,
            ): int(row["seed"])
            for row in manifest["trajectories"]
        }
        for future in as_completed(futures):
            row = future.result()
            completed_rows.append(row)
            write_progress(progress_path, manifest, completed_rows)
            boundary = row["causal_damage_boundary"]
            print(json.dumps({
                "completed": len(completed_rows),
                "seed": int(row["seed"]),
                "target_id": int(row["assigned_target_id"]),
                "branch_step": int(boundary["branch_step"]),
                "snapshot_prefix_step": int(boundary["snapshot_prefix_step"]),
                "target_return_delta": float(boundary["target_return_delta"]),
            }, ensure_ascii=False), flush=True)

    result = dict(manifest)
    result["schema_version"] = 2
    result["causal_boundary_method"] = {
        "candidate_set": "all_LAUNCH_RETARGET_groups_before_assigned_target_damage",
        "counterfactual": "typed_NULL_joint_suffix_removal_from_candidate_boundary",
        "state_snapshot": "native_OS_copy_on_write",
        "suffix": "teacher_actions_replayed_in_current_simulator",
    }
    result["trajectories"] = sorted(
        completed_rows,
        key=lambda row: int(row["seed"]),
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["status"] = "complete"
    progress["output_manifest"] = str(args.output_manifest.resolve())
    progress_path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
