#!/usr/bin/env python3
"""Fit and cross-validate a shared target value head on dense COW scores."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unified_mappo.model import HybridMAPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--progress", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--score-scale", type=float, default=5.0)
    parser.add_argument(
        "--context",
        choices=("shared", "teacher"),
        default="shared",
    )
    parser.add_argument(
        "--ridge-values",
        default="0.001,0.01,0.1,1,10,100",
    )
    parser.add_argument(
        "--policy-scales",
        default="0.02,0.05,0.1,0.2,0.3,0.5,1.0",
    )
    parser.add_argument("--score-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--label-field",
        choices=("score", "target_score", "executor_target_score"),
        default="score",
    )
    return parser.parse_args()


def load_states(
    paths: list[Path],
    trainer: HybridMAPPOTrainer,
    *,
    score_scale: float,
    label_field: str = "score",
) -> list[dict[str, Any]]:
    states = []
    model = trainer.model
    for path in paths:
        progress = json.loads(path.read_text(encoding="utf-8"))
        for episode in progress["episodes"]:
            if not bool(episode.get("dense_target_counterfactual", False)):
                continue
            payload = torch.load(
                episode["rollout"], map_location="cpu", weights_only=False
            )
            rollout = payload["rollout"] if isinstance(payload, dict) else payload
            diagnostics = episode["attack_options"]
            slots = [
                None if value is None else int(value)
                for value in diagnostics["target_slot_ids"]
            ]
            identity = {
                int(entity_id): int(agent_id)
                for entity_id, agent_id in diagnostics["agent_identity_map"].items()
            }
            for dense in episode.get("dense_target_rows", ()):
                agent_id = identity[int(dense["executor_id"])]
                matches = (
                    rollout.agent_ids.eq(agent_id)
                    & rollout.steps.eq(int(dense["timestep"]))
                    & rollout.action_mask.target
                ).nonzero(as_tuple=False).flatten()
                if matches.numel() != 1:
                    raise ValueError(
                        f"dense unit {dense['unit_id']} 匹配 {matches.numel()} 个 rollout 行"
                    )
                row_index = int(matches.item())
                valid = rollout.target_valid_mask[row_index].to(dtype=torch.bool)
                missing_labels = [
                    int(row["target_id"])
                    for row in dense["candidates"]
                    if label_field not in row
                ]
                if missing_labels:
                    raise ValueError(
                        f"dense unit {dense['unit_id']} 缺少 {label_field}: "
                        f"targets={missing_labels}"
                    )
                candidate_scores = {
                    int(row["target_id"]): float(row[label_field])
                    for row in dense["candidates"]
                }
                legal_ids = {
                    int(slots[index])
                    for index in valid.nonzero(as_tuple=False).flatten().tolist()
                    if slots[index] is not None
                }
                if legal_ids != set(candidate_scores):
                    raise ValueError(
                        f"dense candidates 与合法掩码不一致: unit={dense['unit_id']}"
                    )
                if len(legal_ids) < 2:
                    continue
                observations = rollout.observations[row_index:row_index + 1]
                features = rollout.target_features[row_index:row_index + 1]
                valid_batch = valid.unsqueeze(0)
                with torch.no_grad():
                    latent = model.encoder(observations)
                    context = model.target_cf_context(
                        latent, features, valid_batch
                    )[0]
                    base_logits = model.distribution_parameters(
                        observations, features, valid_batch
                    )["target_logits"][0]
                legal_indices = valid.nonzero(as_tuple=False).flatten()
                scores = torch.tensor(
                    [candidate_scores[int(slots[index])] for index in legal_indices],
                    dtype=torch.float64,
                )
                x = context.index_select(0, legal_indices).to(dtype=torch.float64)
                # Only within-state differences affect target ranking.  Remove
                # state constants before fitting the shared linear head.
                x = x - x.mean(dim=0, keepdim=True)
                y = (scores - scores.mean()) / score_scale
                states.append({
                    "seed": int(episode["seed"]),
                    "unit_id": str(dense["unit_id"]),
                    "x": x,
                    "y": y,
                    "scores": scores,
                    "target_ids": torch.tensor(
                        [int(slots[index]) for index in legal_indices],
                        dtype=torch.long,
                    ),
                    "base_logits": base_logits.index_select(
                        0, legal_indices
                    ).to(dtype=torch.float64),
                    "observations": observations.detach().cpu(),
                    "target_features": features.detach().cpu(),
                    "target_valid_mask": valid_batch.detach().cpu(),
                    "legal_indices": legal_indices.detach().cpu(),
                })
    if len({state["seed"] for state in states}) < 3:
        raise ValueError("dense leave-one-seed-out 至少需要 3 个不同 seed")
    return states


def fit_ridge(states: list[dict[str, Any]], ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([state["x"] for state in states], dim=0)
    y = torch.cat([state["y"] for state in states], dim=0)
    scale = x.square().mean(dim=0).sqrt().clamp_min(1e-6)
    z = x / scale
    kernel = z @ z.T
    dual = torch.linalg.solve(
        kernel + ridge * torch.eye(kernel.shape[0], dtype=kernel.dtype),
        y,
    )
    return z.T @ dual, scale


def state_metrics(
    state: dict[str, Any],
    weights: torch.Tensor,
    feature_scale: torch.Tensor,
    *,
    policy_scale: float,
    tolerance: float,
) -> dict[str, float | int | bool]:
    prediction = (state["x"] / feature_scale) @ weights
    scores = state["scores"]
    adjusted_logits = state["base_logits"] + policy_scale * prediction
    predicted_index = int(adjusted_logits.argmax().item())
    base_index = int(state["base_logits"].argmax().item())
    best_score = float(scores.max().item())
    predicted_score = float(scores[predicted_index].item())
    base_score = float(scores[base_index].item())
    correct = 0
    base_correct = 0
    pair_count = 0
    for left, right in itertools.combinations(range(scores.numel()), 2):
        score_delta = float(scores[left].item() - scores[right].item())
        if abs(score_delta) <= tolerance:
            continue
        predicted_delta = float(
            adjusted_logits[left].item() - adjusted_logits[right].item()
        )
        correct += int(score_delta * predicted_delta > 0.0)
        base_delta = float(
            state["base_logits"][left].item()
            - state["base_logits"][right].item()
        )
        base_correct += int(score_delta * base_delta > 0.0)
        pair_count += 1
    return {
        "seed": int(state["seed"]),
        "unit_id": str(state["unit_id"]),
        "candidate_count": int(scores.numel()),
        "pair_count": pair_count,
        "correct_pairs": correct,
        "base_correct_pairs": base_correct,
        "pair_accuracy": correct / pair_count if pair_count else 1.0,
        "predicted_target_id": int(state["target_ids"][predicted_index].item()),
        "base_target_id": int(state["target_ids"][base_index].item()),
        "optimal_hit": best_score - predicted_score <= tolerance,
        "base_optimal_hit": best_score - base_score <= tolerance,
        "regret": best_score - predicted_score,
        "base_regret": best_score - base_score,
        "expected_regret": best_score - float(
            (adjusted_logits.softmax(dim=0) * scores).sum().item()
        ),
        "base_expected_regret": best_score - float(
            (state["base_logits"].softmax(dim=0) * scores).sum().item()
        ),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    pair_count = sum(int(row["pair_count"]) for row in rows)
    return {
        "state_count": len(rows),
        "pair_count": pair_count,
        "pair_accuracy": (
            sum(int(row["correct_pairs"]) for row in rows) / pair_count
            if pair_count else 1.0
        ),
        "base_pair_accuracy": (
            sum(int(row["base_correct_pairs"]) for row in rows) / pair_count
            if pair_count else 1.0
        ),
        "optimal_hit_rate": sum(bool(row["optimal_hit"]) for row in rows) / len(rows),
        "base_optimal_hit_rate": (
            sum(bool(row["base_optimal_hit"]) for row in rows) / len(rows)
        ),
        "mean_regret": sum(float(row["regret"]) for row in rows) / len(rows),
        "base_mean_regret": (
            sum(float(row["base_regret"]) for row in rows) / len(rows)
        ),
        "mean_expected_regret": (
            sum(float(row["expected_regret"]) for row in rows) / len(rows)
        ),
        "base_mean_expected_regret": (
            sum(float(row["base_expected_regret"]) for row in rows) / len(rows)
        ),
        "max_regret": max(float(row["regret"]) for row in rows),
        "base_max_regret": max(float(row["base_regret"]) for row in rows),
    }


def main() -> None:
    args = parse_args()
    if args.score_scale <= 0.0 or args.score_tolerance < 0.0:
        raise ValueError("score-scale 必须为正，score-tolerance 必须非负")
    ridges = [float(value) for value in args.ridge_values.split(",")]
    policy_scales = [float(value) for value in args.policy_scales.split(",")]
    if not ridges or any(value <= 0.0 for value in ridges):
        raise ValueError("ridge-values 必须全部为正")
    if not policy_scales or any(value <= 0.0 for value in policy_scales):
        raise ValueError("policy-scales 必须全部为正")

    trainer = HybridMAPPOTrainer.load(args.checkpoint, device="cpu")
    trainer.model.set_target_cf_policy_scale(0.0)
    trainer.model.set_target_cf_context(args.context)
    states = load_states(
        args.progress,
        trainer,
        score_scale=args.score_scale,
        label_field=args.label_field,
    )
    seeds = sorted({int(state["seed"]) for state in states})
    candidates = []
    for ridge in ridges:
        fold_models = []
        for seed in seeds:
            train = [state for state in states if int(state["seed"]) != seed]
            held = [state for state in states if int(state["seed"]) == seed]
            weights, feature_scale = fit_ridge(train, ridge)
            fold_models.append((held, weights, feature_scale))
        for policy_scale in policy_scales:
            rows = []
            for held, weights, feature_scale in fold_models:
                rows.extend(
                    state_metrics(
                        state,
                        weights,
                        feature_scale,
                        policy_scale=policy_scale,
                        tolerance=args.score_tolerance,
                    )
                    for state in held
                )
            candidates.append({
                "ridge": ridge,
                "policy_scale": policy_scale,
                "aggregate": aggregate(rows),
                "folds": rows,
            })
    safe_candidates = [
        row for row in candidates
        if (
            float(row["aggregate"]["mean_expected_regret"])
            < float(row["aggregate"]["base_mean_expected_regret"])
            and float(row["aggregate"]["mean_regret"])
            <= float(row["aggregate"]["base_mean_regret"])
            and float(row["aggregate"]["max_regret"])
            <= float(row["aggregate"]["base_max_regret"])
            and float(row["aggregate"]["pair_accuracy"])
            >= float(row["aggregate"]["base_pair_accuracy"])
        )
    ]
    selected = min(
        safe_candidates if safe_candidates else candidates,
        key=lambda row: (
            float(row["aggregate"]["mean_expected_regret"]),
            float(row["aggregate"]["mean_regret"]),
            -float(row["aggregate"]["pair_accuracy"]),
            float(row["ridge"]),
            float(row["policy_scale"]),
        ),
    )
    accepted = bool(safe_candidates)
    result = {
        "schema_version": 1,
        "experiment": "TSA005_dense_target_counterfactual_value",
        "status": "accepted" if accepted else "rejected",
        "reason": None if accepted else "no_safe_leave_one_seed_out_gain",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "output_checkpoint": str(args.output.resolve()) if accepted else None,
        "progress_files": [str(path.resolve()) for path in args.progress],
        "state_count": len(states),
        "seed_count": len(seeds),
        "label_count": sum(int(state["scores"].numel()) for state in states),
        "score_scale": args.score_scale,
        "context": args.context,
        "label_field": args.label_field,
        "ridge_candidates": candidates,
        "selected_ridge": float(selected["ridge"]),
        "selected_policy_scale": float(selected["policy_scale"]),
        "cross_validation": selected["aggregate"],
        "selected_folds": selected["folds"],
    }
    if accepted:
        weights, feature_scale = fit_ridge(states, float(selected["ridge"]))
        effective = (weights / feature_scale).to(dtype=torch.float32)
        with torch.no_grad():
            trainer.model.target_cf_value_head.weight.copy_(effective.unsqueeze(0))
            trainer.model.target_cf_value_head.bias.zero_()
            trainer.model.set_target_cf_policy_scale(
                float(selected["policy_scale"])
            )
        trainer.last_metrics.update({
            "dense_cf_state_count": float(len(states)),
            "dense_cf_label_count": float(result["label_count"]),
            "dense_cf_selected_ridge": float(selected["ridge"]),
            "dense_cf_policy_scale": float(selected["policy_scale"]),
            "dense_cf_cv_pair_accuracy": float(
                selected["aggregate"]["pair_accuracy"]
            ),
            "dense_cf_cv_mean_regret": float(
                selected["aggregate"]["mean_regret"]
            ),
        })
        trainer.optimizer = torch.optim.Adam(
            trainer.model.parameters(), lr=trainer.config.learning_rate
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(args.output)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
