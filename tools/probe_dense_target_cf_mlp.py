#!/usr/bin/env python3
"""Cross-validate a small nonlinear shared target residual on dense COW rows.

This is deliberately an offline probe: it freezes the deployed M5 policy and
does not write a deployable checkpoint.  A compact payload is saved only when
leave-one-seed-out expected regret improves over the frozen policy.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


# These fits contain fewer than 200 rows; BLAS thread fan-out is substantially
# slower than a single worker and needlessly competes with simulator processes.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--progress", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--score-scale", type=float, default=5.0)
    parser.add_argument("--hidden-dims", default="8,16,32")
    parser.add_argument("--weight-decays", default="0.0001,0.001,0.01")
    parser.add_argument("--policy-scales", default="0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--score-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--label-field",
        choices=("score", "target_score", "executor_target_score"),
        default="score",
    )
    return parser.parse_args()


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


def fit_model(
    states: list[dict[str, Any]],
    *,
    hidden_dim: int,
    weight_decay: float,
    epochs: int,
    learning_rate: float,
    init_seed: int,
) -> tuple[ResidualMLP, torch.Tensor]:
    torch.manual_seed(init_seed)
    x = torch.cat([state["x"] for state in states], dim=0).to(torch.float32)
    y = torch.cat([state["y"] for state in states], dim=0).to(torch.float32)
    feature_scale = x.square().mean(dim=0).sqrt().clamp_min(1e-4)
    x = x / feature_scale
    model = ResidualMLP(x.shape[1], hidden_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for _ in range(epochs):
        prediction = model(x)
        loss = torch.nn.functional.smooth_l1_loss(prediction, y, beta=0.5)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval(), feature_scale


def state_metrics(
    state: dict[str, Any],
    model: ResidualMLP,
    feature_scale: torch.Tensor,
    *,
    policy_scale: float,
    tolerance: float,
) -> dict[str, float | int | bool]:
    with torch.no_grad():
        prediction = model(
            state["x"].to(torch.float32) / feature_scale
        ).to(torch.float64)
    scores = state["scores"]
    adjusted_logits = state["base_logits"] + policy_scale * prediction
    predicted_index = int(adjusted_logits.argmax().item())
    base_index = int(state["base_logits"].argmax().item())
    best_score = float(scores.max().item())
    correct = 0
    pair_count = 0
    for left, right in itertools.combinations(range(scores.numel()), 2):
        score_delta = float(scores[left].item() - scores[right].item())
        if abs(score_delta) <= tolerance:
            continue
        logit_delta = float(adjusted_logits[left].item() - adjusted_logits[right].item())
        correct += int(score_delta * logit_delta > 0.0)
        pair_count += 1
    predicted_score = float(scores[predicted_index].item())
    base_score = float(scores[base_index].item())
    return {
        "seed": int(state["seed"]),
        "unit_id": str(state["unit_id"]),
        "candidate_count": int(scores.numel()),
        "pair_count": pair_count,
        "correct_pairs": correct,
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
        "pair_accuracy": sum(int(row["correct_pairs"]) for row in rows) / pair_count,
        "optimal_hit_rate": sum(bool(row["optimal_hit"]) for row in rows) / len(rows),
        "base_optimal_hit_rate": sum(bool(row["base_optimal_hit"]) for row in rows) / len(rows),
        "mean_regret": sum(float(row["regret"]) for row in rows) / len(rows),
        "base_mean_regret": sum(float(row["base_regret"]) for row in rows) / len(rows),
        "mean_expected_regret": sum(float(row["expected_regret"]) for row in rows) / len(rows),
        "base_mean_expected_regret": sum(float(row["base_expected_regret"]) for row in rows) / len(rows),
        "max_regret": max(float(row["regret"]) for row in rows),
        "base_max_regret": max(float(row["base_regret"]) for row in rows),
    }


def main() -> None:
    args = parse_args()
    hidden_dims = [int(value) for value in args.hidden_dims.split(",")]
    weight_decays = [float(value) for value in args.weight_decays.split(",")]
    policy_scales = [float(value) for value in args.policy_scales.split(",")]
    if args.epochs <= 0 or args.learning_rate <= 0.0:
        raise ValueError("epochs 和 learning-rate 必须为正")
    dense = _dense_module()
    trainer = HybridMAPPOTrainer.load(args.checkpoint, device="cpu")
    trainer.model.set_target_cf_policy_scale(0.0)
    states = dense.load_states(
        args.progress,
        trainer,
        score_scale=args.score_scale,
        label_field=args.label_field,
    )
    seeds = sorted({int(state["seed"]) for state in states})
    candidates = []
    for hidden_dim in hidden_dims:
        for weight_decay in weight_decays:
            fold_models = []
            for fold_index, seed in enumerate(seeds):
                train = [state for state in states if int(state["seed"]) != seed]
                held = [state for state in states if int(state["seed"]) == seed]
                model, feature_scale = fit_model(
                    train,
                    hidden_dim=hidden_dim,
                    weight_decay=weight_decay,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    init_seed=10_000 + hidden_dim * 100 + fold_index,
                )
                fold_models.append((held, model, feature_scale))
            for policy_scale in policy_scales:
                rows = []
                for held, model, feature_scale in fold_models:
                    rows.extend(
                        state_metrics(
                            state,
                            model,
                            feature_scale,
                            policy_scale=policy_scale,
                            tolerance=args.score_tolerance,
                        )
                        for state in held
                    )
                candidates.append({
                    "hidden_dim": hidden_dim,
                    "weight_decay": weight_decay,
                    "policy_scale": policy_scale,
                    "aggregate": aggregate(rows),
                    "folds": rows,
                })
    selected = min(
        candidates,
        key=lambda row: (
            float(row["aggregate"]["mean_expected_regret"]),
            float(row["aggregate"]["mean_regret"]),
            -float(row["aggregate"]["pair_accuracy"]),
        ),
    )
    metrics = selected["aggregate"]
    accepted = (
        float(metrics["mean_expected_regret"])
        < float(metrics["base_mean_expected_regret"])
        and float(metrics["max_regret"]) <= float(metrics["base_max_regret"])
    )
    result = {
        "schema_version": 1,
        "experiment": "TSA006_dense_target_counterfactual_mlp_probe",
        "status": "accepted" if accepted else "rejected",
        "reason": None if accepted else "no_safe_leave_one_seed_out_gain",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "output_payload": str(args.output.resolve()) if accepted else None,
        "state_count": len(states),
        "seed_count": len(seeds),
        "label_count": sum(int(state["scores"].numel()) for state in states),
        "label_field": args.label_field,
        "selected_hidden_dim": int(selected["hidden_dim"]),
        "selected_weight_decay": float(selected["weight_decay"]),
        "selected_policy_scale": float(selected["policy_scale"]),
        "cross_validation": metrics,
        "selected_folds": selected["folds"],
        "candidates": candidates,
    }
    if accepted:
        model, feature_scale = fit_model(
            states,
            hidden_dim=int(selected["hidden_dim"]),
            weight_decay=float(selected["weight_decay"]),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            init_seed=20_260_914,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema_version": 1,
            "input_dim": int(states[0]["x"].shape[1]),
            "hidden_dim": int(selected["hidden_dim"]),
            "feature_scale": feature_scale,
            "policy_scale": float(selected["policy_scale"]),
            "model": model.state_dict(),
        }, args.output)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "reason", "state_count", "seed_count", "label_count",
        "selected_hidden_dim", "selected_weight_decay", "selected_policy_scale",
        "cross_validation", "output_payload",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
