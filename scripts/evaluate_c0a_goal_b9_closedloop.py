#!/usr/bin/env python3
"""Evaluate fixed b9 decisions with per-branch frozen-teacher continuation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import torch

from train_c0a_goal_b9 import SIM_ROOT, subprocess_environment, write_json

if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from experiments.unified_mappo.model import (  # noqa: E402
    HybridAction,
    HybridActionMask,
    HybridMAPPOConfig,
    HybridPolicyOutput,
    HybridRolloutBatch,
)
from tools.train_start_state_option_curriculum import weighted_type_score  # noqa: E402


TARGET_SLOT_IDS = (
    51, 52, 53, 54, 106, 168, 169, -100,
    2551, 2552, 2553, 2555, 2557, 2559, 2561, 2563, 2565,
    2567, 2569, 2570, 2571, 2572, 2573, 2574, 2575,
)
SEARCH_TARGET_ID = -100
EXPECTED_B9_SHA256 = (
    "0ba282d7fdfac0039b959cde17add9f4da99edd7a7e830f0c0afa989e8af6965"
)
T_CRITICAL_ONE_SIDED_95_DF63 = 1.6694022217068127


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--old-progress", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--branch-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--required-lower", type=float, default=-0.5)
    parser.add_argument("--required-suffix-ratio", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--only-seed", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_load_rollout(path: Path) -> HybridRolloutBatch:
    allowed = [
        HybridMAPPOConfig,
        HybridActionMask,
        HybridAction,
        HybridPolicyOutput,
        HybridRolloutBatch,
    ]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)["rollout"]


def extract_student_decision(row: dict[str, Any]) -> dict[str, Any]:
    batch = safe_load_rollout(Path(row["rollout"]))
    active = batch.action_mask.target.to(dtype=torch.bool)
    if int(active.sum().item()) != 1:
        raise RuntimeError(
            f"seed {row['seed']} does not contain exactly one target decision"
        )
    active_index = int(active.nonzero(as_tuple=False)[0].item())
    slot = int(batch.actions.target_index[active_index].item())
    if not 0 <= slot < len(TARGET_SLOT_IDS):
        raise RuntimeError(f"seed {row['seed']} selected unknown target slot {slot}")
    if not bool(batch.target_valid_mask[active_index, slot].item()):
        raise RuntimeError(f"seed {row['seed']} selected an illegal target slot")
    return {
        "target_id": int(TARGET_SLOT_IDS[slot]),
        "target_index": slot,
        "coordinate": [
            float(value)
            for value in batch.actions.target_xy[active_index].tolist()
        ],
        "decision_step": int(batch.steps[active_index].item()),
        "agent_id": int(batch.agent_ids[active_index].item()),
    }


def boundary_teacher_command(
    trace: dict[str, Any], decision_step: int, executor_id: int
) -> dict[str, Any]:
    rows = [row for row in trace["steps"] if int(row["step"]) == decision_step]
    if len(rows) != 1:
        raise RuntimeError(f"trace has {len(rows)} rows for step {decision_step}")
    commands = [
        action for action in rows[0]["actions"]
        if int(action.get("executor_id", -1)) == executor_id
        and int(action.get("commandType_id", -1)) == 200
    ]
    if len(commands) != 1:
        raise RuntimeError(
            f"step {decision_step} executor {executor_id} has {len(commands)} launches"
        )
    return commands[0]


def build_jobs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    progress = json.loads(args.old_progress.read_text(encoding="utf-8"))
    episodes = progress["episodes"]
    if len(episodes) != 64 or len({int(row["seed"]) for row in episodes}) != 64:
        raise RuntimeError("source b9 evaluation must contain 64 distinct seeds")
    checkpoint_sha = file_sha256(args.checkpoint)
    if Path(progress["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise RuntimeError("source evaluation checkpoint path differs from requested b9")
    if checkpoint_sha != EXPECTED_B9_SHA256:
        raise RuntimeError("requested checkpoint is not the evaluated b9 checkpoint")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    trajectories = {int(row["seed"]): row for row in manifest["trajectories"]}
    jobs = []
    selected_episodes = episodes
    if args.only_seed is not None:
        selected_episodes = [
            row for row in episodes if int(row["seed"]) == args.only_seed
        ]
    selected_episodes = selected_episodes[:args.limit]
    for order, episode in enumerate(selected_episodes, start=1):
        seed = int(episode["seed"])
        source = trajectories[seed]
        assigned = int(episode["assigned_target_id"])
        anchor = source["damage_anchors"][str(assigned)]
        executor = int(anchor["attacking_entity_id"])
        boundary = int(anchor["decision_step"])
        if boundary != int(episode["snapshot_prefix_step"]) + 1:
            raise RuntimeError(f"seed {seed} boundary differs from old evaluation")
        decision = extract_student_decision(episode)
        if decision["decision_step"] != boundary:
            raise RuntimeError(f"seed {seed} rollout decision step differs from anchor")
        trace = json.loads(Path(source["trace"]).read_text(encoding="utf-8"))
        teacher_command = boundary_teacher_command(trace, boundary, executor)
        teacher_coordinate = [
            float(teacher_command["target"]["x"]),
            float(teacher_command["target"]["y"]),
        ]
        output = args.output_dir / "episodes" / f"i{order:02d}_s{seed}"
        jobs.append({
            "order": order,
            "seed": seed,
            "start_state_id": episode["start_state_id"],
            "trace": source["trace"],
            "timestep": boundary,
            "executor_id": executor,
            "assigned_target_id": assigned,
            "student_target_id": decision["target_id"],
            "student_target_index": decision["target_index"],
            "student_coordinate": decision["coordinate"],
            "teacher_coordinate": teacher_coordinate,
            "old_rollout": episode["rollout"],
            "output": output,
        })
    return manifest, jobs


def run_job(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if args.resume and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if int(result["seed"]) == int(job["seed"]):
            return result

    teacher_slot = TARGET_SLOT_IDS.index(int(job["assigned_target_id"]))
    item = {
        "seed": int(job["seed"]),
        "start_state_id": job["start_state_id"],
        "boundary_source": "formal_damage_anchor",
        "timestep": int(job["timestep"]),
        "executor_id": int(job["executor_id"]),
        "candidate_target_indices": torch.tensor(
            [int(job["student_target_index"]), teacher_slot], dtype=torch.long
        ),
        "candidate_target_ids": torch.tensor(
            [int(job["student_target_id"]), int(job["assigned_target_id"])],
            dtype=torch.long,
        ),
        "candidate_coordinates": torch.tensor(
            [job["student_coordinate"], job["teacher_coordinate"]],
            dtype=torch.float64,
        ),
        "candidate_roles": ["student", "teacher"],
        "physical_state_sha256": None,
        "source_rollout": job["old_rollout"],
        "snapshot_model_sha256": file_sha256(args.checkpoint),
        "sampling_seed": int(job["seed"]),
    }
    input_path = output / "branch_input.pt"
    torch.save(item, input_path)
    request = {
        "mode": "branches",
        "trace": job["trace"],
        "seed": int(job["seed"]),
        "start_state_id": job["start_state_id"],
        "boundary_source": "formal_damage_anchor",
        "timestep": int(job["timestep"]),
        "executor_id": int(job["executor_id"]),
        "capture_path": str(input_path.resolve()),
        "item_path": str((output / "branch_output.pt").resolve()),
        "result_dir": str((output / "branches").resolve()),
        "branch_workers": int(args.branch_workers),
        "evaluation_closed_loop": True,
    }
    request_path = output / "request.json"
    write_json(request_path, request)
    environment = subprocess_environment(request, "branches", args.checkpoint)
    environment["RED_C0A_B9_REQUEST"] = str(request_path.resolve())
    start = time.monotonic()
    with (output / "run.log").open("w") as log:
        completed = subprocess.run(
            [
                sys.executable,
                str(SIM_ROOT / "core/main.py"),
                "--scenario", str(Path(manifest["scenario"]).resolve()),
                "--output-dir", str((output / "sim").resolve()),
                "--max-steps", str(args.max_steps),
                "--total-rounds", "1",
                "--render-mode", "none",
                "--disable-log-color",
            ],
            cwd=SIM_ROOT / "core",
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"seed {job['seed']} failed; see {output / 'run.log'}")
    payload = torch.load(
        output / "branch_output.pt", map_location="cpu", weights_only=True
    )
    branches = payload["branch_results"]
    if len(branches) != 2:
        raise RuntimeError(f"seed {job['seed']} did not produce two branches")
    student, teacher = branches
    if student["physical_state_sha256"] != teacher["physical_state_sha256"]:
        raise RuntimeError(f"seed {job['seed']} branches did not share a state")
    if student["public_rng_sha256"] != teacher["public_rng_sha256"]:
        raise RuntimeError(f"seed {job['seed']} branches did not share RNG state")
    if student["non_target_boundary_changes"] or teacher["non_target_boundary_changes"]:
        raise RuntimeError(f"seed {job['seed']} changed non-target boundary fields")
    expected_student_semantics = (
        "frozen_teacher_closed_loop_search_then_release"
        if int(job["student_target_id"]) == SEARCH_TARGET_ID
        else "frozen_teacher_closed_loop_target_locked"
    )
    if (
        student["suffix_semantics"] != expected_student_semantics
        or teacher["suffix_semantics"]
        != "frozen_teacher_closed_loop_target_locked"
    ):
        raise RuntimeError(f"seed {job['seed']} used incorrect suffix semantics")
    result = {
        "order": int(job["order"]),
        "seed": int(job["seed"]),
        "start_state_id": job["start_state_id"],
        "decision_step": int(job["timestep"]),
        "executor_id": int(job["executor_id"]),
        "student_target_id": int(job["student_target_id"]),
        "teacher_target_id": int(job["assigned_target_id"]),
        "student_coordinate": job["student_coordinate"],
        "teacher_coordinate": job["teacher_coordinate"],
        "student_score": float(student["score"]),
        "teacher_score": float(teacher["score"]),
        "student_fixed_target_score": weighted_type_score(
            student["summary"], {9400, 9600}
        ),
        "teacher_fixed_target_score": weighted_type_score(
            teacher["summary"], {9400, 9600}
        ),
        "student_suffix_return": float(student["official_joint_return"]),
        "teacher_suffix_return": float(teacher["official_joint_return"]),
        "student_end_step": int(student["end_step"]),
        "teacher_end_step": int(teacher["end_step"]),
        "student_semantics": student["suffix_semantics"],
        "teacher_semantics": teacher["suffix_semantics"],
        "physical_state_sha256": student["physical_state_sha256"],
        "public_rng_sha256": student["public_rng_sha256"],
        "elapsed_seconds": time.monotonic() - start,
    }
    write_json(result_path, result)
    return result


def outcomes(deltas: list[float], tolerance: float = 1e-12) -> dict[str, Any]:
    return {
        "wins": sum(value > tolerance for value in deltas),
        "ties": sum(abs(value) <= tolerance for value in deltas),
        "losses": sum(value < -tolerance for value in deltas),
        "mean_delta": statistics.fmean(deltas),
    }


def paired_gate(
    episodes: list[dict[str, Any]], student_key: str, teacher_key: str,
    required_lower: float,
) -> dict[str, Any]:
    student = [float(row[student_key]) for row in episodes]
    teacher = [float(row[teacher_key]) for row in episodes]
    deltas = [left - right for left, right in zip(student, teacher, strict=True)]
    standard_deviation = statistics.stdev(deltas)
    standard_error = standard_deviation / math.sqrt(len(deltas))
    mean_delta = statistics.fmean(deltas)
    lower = mean_delta - T_CRITICAL_ONE_SIDED_95_DF63 * standard_error
    return {
        "student_mean": statistics.fmean(student),
        "teacher_mean": statistics.fmean(teacher),
        "mean_delta": mean_delta,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "one_sided_95_lower": lower,
        "required_lower": required_lower,
        "pass": lower >= required_lower,
    }


def summarize(args: argparse.Namespace, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if len(episodes) != 64:
        return {
            "status": "smoke_complete",
            "episodes": len(episodes),
            "unique_seeds": len({row["seed"] for row in episodes}),
        }
    score_gate = paired_gate(
        episodes, "student_score", "teacher_score", args.required_lower
    )
    fixed_gate = paired_gate(
        episodes,
        "student_fixed_target_score",
        "teacher_fixed_target_score",
        args.required_lower,
    )
    suffix_student = statistics.fmean(
        row["student_suffix_return"] for row in episodes
    )
    suffix_teacher = statistics.fmean(
        row["teacher_suffix_return"] for row in episodes
    )
    suffix_ratio = suffix_student / suffix_teacher
    suffix_gate = {
        "student_mean": suffix_student,
        "teacher_mean": suffix_teacher,
        "ratio": suffix_ratio,
        "required_min": args.required_suffix_ratio,
        "pass": suffix_ratio >= args.required_suffix_ratio,
    }
    target_legality_gate = {
        "controlled_decisions": len(episodes),
        "illegal_samples": 0,
        "source": "source b9 rollout target_valid_mask",
        "pass": len(episodes) == 64,
    }
    integrity_gate = {
        "episodes": len(episodes),
        "unique_seeds": len({row["seed"] for row in episodes}),
        "common_state_pairs": sum(bool(row["physical_state_sha256"]) for row in episodes),
        "common_rng_pairs": sum(bool(row["public_rng_sha256"]) for row in episodes),
        "pass": len({row["seed"] for row in episodes}) == 64,
    }
    gates = {
        "e01_score_noninferiority": score_gate,
        "fixed_target_score_noninferiority": fixed_gate,
        "suffix_return_ratio": suffix_gate,
        "target_legality": target_legality_gate,
        "integrity": integrity_gate,
    }
    promotion = all(gate["pass"] for gate in gates.values())
    failure_reasons = []
    if not score_gate["pass"]:
        failure_reasons.append(
            "E01 paired one-sided 95% lower bound is below the required limit."
        )
    if not fixed_gate["pass"]:
        failure_reasons.append(
            "Fixed-target paired one-sided 95% lower bound is below the required limit."
        )
    if not suffix_gate["pass"]:
        failure_reasons.append("Formal joint suffix-return ratio is below the minimum.")
    score_deltas = [row["student_score"] - row["teacher_score"] for row in episodes]
    fixed_deltas = [
        row["student_fixed_target_score"] - row["teacher_fixed_target_score"]
        for row in episodes
    ]
    return {
        "schema_version": 2,
        "stage": "C0a_goal",
        "candidate": "b9",
        "evaluation_semantics": "paired_frozen_teacher_closed_loop_per_branch",
        "decision": "PASS" if promotion else "FAIL",
        "promotion": promotion,
        "episodes": 64,
        "unique_seeds": 64,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "confidence": {
            "method": "paired Student t one-sided 95% lower bound",
            "degrees_of_freedom": 63,
            "critical_value": T_CRITICAL_ONE_SIDED_95_DF63,
        },
        "gates": gates,
        "score_outcomes": outcomes(score_deltas),
        "fixed_target_score_outcomes": outcomes(fixed_deltas),
        "target_diagnostics": {
            "matches_teacher": sum(
                row["student_target_id"] == row["teacher_target_id"]
                for row in episodes
            ),
            "differs_from_teacher": sum(
                row["student_target_id"] != row["teacher_target_id"]
                for row in episodes
            ),
            "search_choices": sum(
                row["student_target_id"] == SEARCH_TARGET_ID for row in episodes
            ),
        },
        "failure_reasons": failure_reasons,
    }


def main() -> None:
    args = parse_args()
    args.old_progress = args.old_progress.resolve()
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    if not 1 <= args.limit <= 64:
        raise ValueError("limit must be in [1, 64]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise ValueError("output directory is not empty; use --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, jobs = build_jobs(args)
    runtime_path = Path(__file__).with_name("c0a_goal_b9_runtime.py").resolve()
    protocol = {
        "schema_version": 1,
        "evaluation_semantics": "paired_frozen_teacher_closed_loop_per_branch",
        "old_progress": str(args.old_progress),
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "student_action_source": "completed_b9_rollout_target_and_coordinate",
        "teacher_continuation": "frozen_r9_closed_loop_from_each_branch_state",
        "common_randomness": "same_prefork_python_numpy_torch_rng_state",
        "physical_target_option": "locked_until_executor_or_target_termination",
        "search_option": "boundary_search_then_immediate_teacher_release",
        "workers": args.workers,
        "branch_workers": args.branch_workers,
        "max_steps": args.max_steps,
        "seed_order": [job["seed"] for job in jobs],
        "code_sha256": {
            str(Path(__file__).resolve()): file_sha256(Path(__file__).resolve()),
            str(runtime_path): file_sha256(runtime_path),
        },
    }
    write_json(args.output_dir / "protocol.json", protocol)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, args, manifest, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            results.sort(key=lambda row: row["order"])
            write_json(args.output_dir / "episodes.json", results)
            write_json(args.output_dir / "progress.json", {
                "status": "running" if len(results) < len(jobs) else "complete",
                "completed": len(results),
                "total": len(jobs),
                "latest_seed": result["seed"],
                "mean_elapsed_seconds": statistics.fmean(
                    row["elapsed_seconds"] for row in results
                ),
            })
            print(json.dumps({
                "event": "episode_complete",
                "completed": len(results),
                "total": len(jobs),
                "seed": result["seed"],
                "student_target": result["student_target_id"],
                "teacher_target": result["teacher_target_id"],
                "score_delta": result["student_score"] - result["teacher_score"],
                "seconds": result["elapsed_seconds"],
            }, ensure_ascii=False), flush=True)
    summary = summarize(args, results)
    write_json(args.output_dir / "promotion_metrics_64.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
