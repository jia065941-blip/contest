#!/usr/bin/env python3
"""Summarize one 64-seed C0a_goal promotion evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import torch


SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from experiments.unified_mappo.model import (  # noqa: E402
    HybridAction,
    HybridActionMask,
    HybridMAPPOConfig,
    HybridPolicyOutput,
    HybridRolloutBatch,
)


T_CRITICAL_ONE_SIDED_95_DF63 = 1.6694022217068127
TARGET_SLOT_IDS = (
    51, 52, 53, 54, 106, 168, 169, -100,
    2551, 2552, 2553, 2555, 2557, 2559, 2561, 2563, 2565,
    2567, 2569, 2570, 2571, 2572, 2573, 2574, 2575,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", default="b8_iter_r4")
    parser.add_argument("--compare-progress", type=Path)
    parser.add_argument("--required-lower", type=float, default=-0.5)
    parser.add_argument("--required-suffix-ratio", type=float, default=0.8)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_gate(
    episodes: list[dict[str, Any]],
    student_key: str,
    teacher_key: str,
    required_lower: float,
) -> dict[str, float | bool]:
    student = [float(row[student_key]) for row in episodes]
    teacher = [float(row[teacher_key]) for row in episodes]
    deltas = [a - b for a, b in zip(student, teacher, strict=True)]
    mean_delta = statistics.fmean(deltas)
    standard_error = statistics.stdev(deltas) / math.sqrt(len(deltas))
    lower = mean_delta - T_CRITICAL_ONE_SIDED_95_DF63 * standard_error
    return {
        "student_mean": statistics.fmean(student),
        "teacher_mean": statistics.fmean(teacher),
        "mean_delta": mean_delta,
        "sample_standard_deviation": statistics.stdev(deltas),
        "standard_error": standard_error,
        "one_sided_95_lower": lower,
        "required_lower": required_lower,
        "pass": lower >= required_lower,
    }


def outcomes(deltas: list[float], tolerance: float = 1e-12) -> dict[str, float | int]:
    return {
        "wins": sum(value > tolerance for value in deltas),
        "ties": sum(abs(value) <= tolerance for value in deltas),
        "losses": sum(value < -tolerance for value in deltas),
        "mean_delta": statistics.fmean(deltas) if deltas else 0.0,
    }


def safe_load_rollout(path: Path) -> HybridRolloutBatch:
    allowed = [
        HybridMAPPOConfig,
        HybridActionMask,
        HybridAction,
        HybridPolicyOutput,
        HybridRolloutBatch,
    ]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["rollout"]


def target_legality(episodes: list[dict[str, Any]]) -> dict[str, int | bool]:
    controlled = 0
    illegal = 0
    for row in episodes:
        batch = safe_load_rollout(Path(row["rollout"]))
        active = batch.action_mask.target.to(dtype=torch.bool)
        indices = batch.actions.target_index.to(dtype=torch.long)
        legal = batch.target_valid_mask.gather(1, indices.unsqueeze(-1)).squeeze(-1)
        controlled += int(active.sum().item())
        illegal += int((active & ~legal).sum().item())
    return {
        "controlled_decisions": controlled,
        "illegal_samples": illegal,
        "pass": controlled == len(episodes) and illegal == 0,
    }


def integrity(episodes: list[dict[str, Any]]) -> dict[str, int | bool]:
    seeds = [int(row["seed"]) for row in episodes]
    rollouts = [Path(row["rollout"]) for row in episodes]
    logs = [Path(row["log"]) for row in episodes]
    error_logs = 0
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Traceback (most recent call last)" in text or " - ERROR - " in text:
            error_logs += 1
    nonempty_rollouts = all(path.is_file() and path.stat().st_size > 0 for path in rollouts)
    nonempty_logs = all(path.is_file() and path.stat().st_size > 0 for path in logs)
    return {
        "unique_seeds": len(set(seeds)),
        "nonempty_rollouts": nonempty_rollouts,
        "nonempty_logs": nonempty_logs,
        "error_logs": error_logs,
        "pass": (
            len(episodes) == 64
            and len(set(seeds)) == 64
            and nonempty_rollouts
            and nonempty_logs
            and error_logs == 0
        ),
    }


def actual_target_id(row: dict[str, Any]) -> int:
    batch = safe_load_rollout(Path(row["rollout"]))
    active = batch.action_mask.target.to(dtype=torch.bool)
    if int(active.sum().item()) != 1:
        raise ValueError(
            f"expected one controlled target decision for seed {row['seed']}"
        )
    slot = int(batch.actions.target_index[active][0].item())
    return TARGET_SLOT_IDS[slot]


def main() -> None:
    args = parse_args()
    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    episodes = progress["episodes"]
    if len(episodes) != 64:
        raise ValueError(f"expected 64 episodes, found {len(episodes)}")

    score_deltas = [
        float(row["score"]) - float(row["teacher_score"]) for row in episodes
    ]
    fixed_deltas = [
        float(row["fixed_target_score"]) - float(row["teacher_fixed_target_score"])
        for row in episodes
    ]
    score_gate = paired_gate(
        episodes, "score", "teacher_score", args.required_lower
    )
    fixed_gate = paired_gate(
        episodes,
        "fixed_target_score",
        "teacher_fixed_target_score",
        args.required_lower,
    )
    suffix_student = statistics.fmean(
        float(row["student_suffix_return"]) for row in episodes
    )
    suffix_teacher = statistics.fmean(
        float(row["reference_suffix_return"]) for row in episodes
    )
    suffix_ratio = suffix_student / suffix_teacher
    suffix_gate = {
        "student_mean": suffix_student,
        "teacher_mean": suffix_teacher,
        "ratio": suffix_ratio,
        "required_min": args.required_suffix_ratio,
        "pass": suffix_ratio >= args.required_suffix_ratio,
    }
    legality_gate = target_legality(episodes)
    integrity_gate = integrity(episodes)

    actual_targets = {
        int(row["seed"]): actual_target_id(row) for row in episodes
    }
    matches = [
        row for row in episodes
        if actual_targets[int(row["seed"])] == int(row["assigned_target_id"])
    ]
    mismatches = [
        row for row in episodes
        if actual_targets[int(row["seed"])] != int(row["assigned_target_id"])
    ]
    diagnostics: dict[str, Any] = {
        "actual_policy_target_matches_teacher": len(matches),
        "actual_policy_target_mismatches_teacher": len(mismatches),
        "matching_target_score_outcomes": outcomes([
            float(row["score"]) - float(row["teacher_score"]) for row in matches
        ]),
        "mismatching_target_score_outcomes": outcomes([
            float(row["score"]) - float(row["teacher_score"]) for row in mismatches
        ]),
    }
    if args.compare_progress is not None:
        previous = json.loads(args.compare_progress.read_text(encoding="utf-8"))
        previous_by_seed = {
            int(row["seed"]): row for row in previous["episodes"]
        }
        previous_actual_targets = {
            int(row["seed"]): actual_target_id(row)
            for row in previous["episodes"]
        }
        paired_previous = [
            (row, previous_by_seed[int(row["seed"])]) for row in episodes
        ]
        previous_deltas = [
            float(current["score"]) - float(old["score"])
            for current, old in paired_previous
        ]
        diagnostics.update({
            "same_as_previous_scores": sum(
                abs(delta) <= 1e-12 for delta in previous_deltas
            ),
            "same_as_previous_actual_targets": sum(
                actual_targets[int(current["seed"])]
                == previous_actual_targets[int(old["seed"])]
                for current, old in paired_previous
            ),
            "same_as_previous_destroyed_sets": sum(
                set(current.get("destroyed_ids", []))
                == set(old.get("destroyed_ids", []))
                for current, old in paired_previous
            ),
            "candidate_vs_previous_score_outcomes": outcomes(previous_deltas),
        })

    gates = {
        "e01_score_noninferiority": score_gate,
        "fixed_target_score_noninferiority": fixed_gate,
        "suffix_return_ratio": suffix_gate,
        "target_legality": legality_gate,
        "integrity": integrity_gate,
    }
    promotion = all(bool(gate["pass"]) for gate in gates.values())
    failure_reasons: list[str] = []
    if not score_gate["pass"]:
        failure_reasons.append(
            "E01 paired one-sided 95% lower bound is below "
            f"{args.required_lower} points."
        )
    if not fixed_gate["pass"]:
        failure_reasons.append(
            "Fixed 9400/9600 paired one-sided 95% lower bound is below "
            f"{args.required_lower} points."
        )
    if not suffix_gate["pass"]:
        failure_reasons.append("Suffix return ratio is below the required minimum.")
    if not legality_gate["pass"]:
        failure_reasons.append("Target legality gate failed.")
    if not integrity_gate["pass"]:
        failure_reasons.append("Evaluation integrity gate failed.")

    payload = {
        "schema_version": 1,
        "stage": "C0a_goal",
        "candidate": args.candidate,
        "decision": "PASS" if promotion else "FAIL",
        "promotion": promotion,
        "episodes": len(episodes),
        "unique_seeds": len({int(row["seed"]) for row in episodes}),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "confidence": {
            "method": "paired Student t one-sided 95% lower bound",
            "degrees_of_freedom": 63,
            "critical_value": T_CRITICAL_ONE_SIDED_95_DF63,
        },
        "gates": gates,
        "score_outcomes": outcomes(score_deltas),
        "fixed_target_score_outcomes": outcomes(fixed_deltas),
        "goal_diagnostics": diagnostics,
        "failure_reasons": failure_reasons,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
