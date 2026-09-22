#!/usr/bin/env python3
"""Evaluate a frozen target checkpoint on unseen dense counterfactual rows."""

from __future__ import annotations

import argparse
import importlib.util
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


def _dense_module():
    path = ROOT / "tools" / "train_dense_target_cf_value.py"
    spec = importlib.util.spec_from_file_location("dense_target_cf_value", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--progress", type=Path, action="append", required=True)
    parser.add_argument(
        "--label-field",
        choices=("score", "target_score", "executor_target_score"),
        required=True,
    )
    parser.add_argument("--score-scale", type=float, default=5.0)
    parser.add_argument("--score-tolerance", type=float, default=1e-8)
    parser.add_argument("--metrics", type=Path, required=True)
    return parser.parse_args()


def evaluate_state(
    state: dict[str, Any],
    candidate: HybridMAPPOTrainer,
    *,
    tolerance: float,
) -> dict[str, Any]:
    with torch.no_grad():
        logits = candidate.model.distribution_parameters(
            state["observations"],
            state["target_features"],
            state["target_valid_mask"],
        )["target_logits"][0].index_select(
            0, state["legal_indices"]
        ).to(torch.float64)
    base_logits = state["base_logits"]
    scores = state["scores"]
    selected = int(logits.argmax().item())
    base_selected = int(base_logits.argmax().item())
    best_score = float(scores.max().item())
    correct_pairs = 0
    base_correct_pairs = 0
    pair_count = 0
    for left, right in itertools.combinations(range(scores.numel()), 2):
        delta = float(scores[left].item() - scores[right].item())
        if abs(delta) <= tolerance:
            continue
        correct_pairs += int(
            delta * float(logits[left].item() - logits[right].item()) > 0.0
        )
        base_correct_pairs += int(
            delta * float(
                base_logits[left].item() - base_logits[right].item()
            ) > 0.0
        )
        pair_count += 1
    return {
        "seed": int(state["seed"]),
        "unit_id": str(state["unit_id"]),
        "pair_count": pair_count,
        "correct_pairs": correct_pairs,
        "base_correct_pairs": base_correct_pairs,
        "target_id": int(state["target_ids"][selected].item()),
        "base_target_id": int(state["target_ids"][base_selected].item()),
        "regret": best_score - float(scores[selected].item()),
        "base_regret": best_score - float(scores[base_selected].item()),
        "expected_regret": best_score - float(
            (logits.softmax(dim=0) * scores).sum().item()
        ),
        "base_expected_regret": best_score - float(
            (base_logits.softmax(dim=0) * scores).sum().item()
        ),
    }


def main() -> None:
    args = parse_args()
    dense = _dense_module()
    base = HybridMAPPOTrainer.load(args.base_checkpoint, device="cpu")
    base.model.set_target_cf_policy_scale(0.0)
    candidate = HybridMAPPOTrainer.load(args.candidate_checkpoint, device="cpu")
    states = dense.load_states(
        args.progress,
        base,
        score_scale=args.score_scale,
        label_field=args.label_field,
    )
    rows = [
        evaluate_state(state, candidate, tolerance=args.score_tolerance)
        for state in states
    ]
    pair_count = sum(int(row["pair_count"]) for row in rows)
    metrics = {
        "state_count": len(rows),
        "seed_count": len({int(row["seed"]) for row in rows}),
        "label_count": sum(int(state["scores"].numel()) for state in states),
        "pair_count": pair_count,
        "pair_accuracy": sum(int(row["correct_pairs"]) for row in rows) / pair_count,
        "base_pair_accuracy": sum(int(row["base_correct_pairs"]) for row in rows) / pair_count,
        "mean_regret": sum(float(row["regret"]) for row in rows) / len(rows),
        "base_mean_regret": sum(float(row["base_regret"]) for row in rows) / len(rows),
        "mean_expected_regret": sum(float(row["expected_regret"]) for row in rows) / len(rows),
        "base_mean_expected_regret": sum(float(row["base_expected_regret"]) for row in rows) / len(rows),
        "max_regret": max(float(row["regret"]) for row in rows),
        "base_max_regret": max(float(row["base_regret"]) for row in rows),
        "top1_change_count": sum(
            int(row["target_id"] != row["base_target_id"]) for row in rows
        ),
    }
    accepted = (
        metrics["mean_expected_regret"] < metrics["base_mean_expected_regret"]
        and metrics["max_regret"] <= metrics["base_max_regret"]
    )
    result = {
        "schema_version": 1,
        "experiment": "TSA007_frozen_dense_target_holdout",
        "status": "accepted" if accepted else "rejected",
        "reason": None if accepted else "no_safe_external_holdout_gain",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
        "progress_files": [str(path.resolve()) for path in args.progress],
        "label_field": args.label_field,
        "metrics": metrics,
        "rows": rows,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
