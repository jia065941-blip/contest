#!/usr/bin/env python3
"""Fit a low-capacity shared per-target value head from exact E01 returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unified_mappo.model import HybridMAPPOTrainer
from tools.train_counterfactual_target_pairs import load_rows, rollout_paths, target_slots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--return-scale", type=float, default=0.05)
    parser.add_argument("--zero-weight", type=float, default=0.25)
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--holdout-stride", type=int, default=5)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--min-abs-delta", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def prepare(
    trainer: HybridMAPPOTrainer,
    rows: dict[str, torch.Tensor],
    *,
    device: torch.device,
    return_scale: float,
    zero_weight: float,
    min_abs_delta: float,
) -> dict[str, torch.Tensor]:
    observations = rows["observations"].to(device)
    features = rows["features"].to(device)
    valid = rows["valid"].to(device)
    with torch.no_grad():
        latent = trainer.model.encoder(observations)
        context = trainer.model.shared_target_cf_context(latent, features, valid)
    raw_target = rows["delta"].to(device)
    nonzero = raw_target.abs().gt(min_abs_delta)
    weights = torch.where(
        nonzero,
        torch.ones_like(raw_target),
        torch.full_like(raw_target, zero_weight),
    )
    return {
        "context": context.detach(),
        "valid": valid,
        "selected": rows["student"].to(device),
        "raw_target": raw_target,
        "target": raw_target / return_scale,
        "nonzero": nonzero,
        "weights": weights,
    }


def selected_prediction(
    trainer: HybridMAPPOTrainer,
    rows: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    values = trainer.model.target_cf_value_head(rows["context"]).squeeze(-1)
    selected = values.gather(1, rows["selected"].unsqueeze(1)).squeeze(1)
    return selected, values


def weighted_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    losses = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


@torch.no_grad()
def evaluate(
    trainer: HybridMAPPOTrainer,
    rows: dict[str, torch.Tensor],
    *,
    return_scale: float,
    huber_beta: float,
) -> dict[str, float]:
    prediction, all_values = selected_prediction(trainer, rows)
    target = rows["target"]
    nonzero = rows["nonzero"]
    error = prediction - target
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    denominator = (
        centered_prediction.square().sum().sqrt()
        * centered_target.square().sum().sqrt()
    )
    correlation = (
        (centered_prediction * centered_target).sum() / denominator.clamp_min(1e-8)
    )
    valid_values = all_values[rows["valid"]]
    return {
        "weighted_huber": float(
            weighted_huber(
                prediction,
                target,
                rows["weights"],
                beta=huber_beta,
            ).cpu()
        ),
        "raw_mae": float((error.abs().mean() * return_scale).cpu()),
        "raw_rmse": float((error.square().mean().sqrt() * return_scale).cpu()),
        "correlation": float(correlation.cpu()),
        "nonzero_sign_accuracy": float(
            prediction[nonzero].sign().eq(target[nonzero].sign()).float().mean().cpu()
        ),
        "selected_prediction_mean": float(prediction.mean().cpu()),
        "selected_prediction_std": float(prediction.std(unbiased=False).cpu()),
        "valid_prediction_std": float(valid_values.std(unbiased=False).cpu()),
    }


def head_state(trainer: HybridMAPPOTrainer) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in trainer.model.target_cf_value_head.state_dict().items()
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.learning_rate <= 0.0:
        raise ValueError("epochs 和 learning-rate 必须为正")
    if args.return_scale <= 0.0 or not 0.0 <= args.zero_weight <= 1.0:
        raise ValueError("return-scale 必须为正，zero-weight 必须位于 [0,1]")
    if args.huber_beta <= 0.0 or args.holdout_stride < 2:
        raise ValueError("huber-beta 必须为正，holdout-stride 至少为 2")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 CUDA 不可用")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = HybridMAPPOTrainer.load(args.checkpoint, device=args.device)
    model = trainer.model
    model.set_target_cf_policy_scale(0.0)
    paths = rollout_paths(args.rollout_root)
    train_raw, holdout_raw, counts = load_rows(
        paths,
        slot_ids=target_slots(args.manifest),
        target_slots_count=model.config.target_slots,
        target_feature_dim=model.config.target_feature_dim,
        holdout_stride=args.holdout_stride,
        split_mode="seed",
        min_abs_delta=args.min_abs_delta,
    )
    device = next(model.parameters()).device
    train = prepare(
        trainer,
        train_raw,
        device=device,
        return_scale=args.return_scale,
        zero_weight=args.zero_weight,
        min_abs_delta=args.min_abs_delta,
    )
    holdout = prepare(
        trainer,
        holdout_raw,
        device=device,
        return_scale=args.return_scale,
        zero_weight=args.zero_weight,
        min_abs_delta=args.min_abs_delta,
    )
    initial = {
        "train": evaluate(
            trainer, train, return_scale=args.return_scale, huber_beta=args.huber_beta
        ),
        "holdout": evaluate(
            trainer, holdout, return_scale=args.return_scale, huber_beta=args.huber_beta
        ),
    }

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = list(model.target_cf_value_head.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_epoch = 0
    best_loss = initial["holdout"]["weighted_huber"]
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = selected_prediction(trainer, train)
        loss = weighted_huber(
            prediction,
            train["target"],
            train["weights"],
            beta=args.huber_beta,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if epoch % args.validation_interval != 0 and epoch != args.epochs:
            continue
        train_metrics = evaluate(
            trainer, train, return_scale=args.return_scale, huber_beta=args.huber_beta
        )
        holdout_metrics = evaluate(
            trainer, holdout, return_scale=args.return_scale, huber_beta=args.huber_beta
        )
        eligible = (
            train_metrics["weighted_huber"] < initial["train"]["weighted_huber"]
            and holdout_metrics["weighted_huber"] < best_loss
            and holdout_metrics["nonzero_sign_accuracy"] > 0.5
        )
        history.append({
            "epoch": epoch,
            "train_weighted_huber": train_metrics["weighted_huber"],
            "holdout_weighted_huber": holdout_metrics["weighted_huber"],
            "holdout_sign_accuracy": holdout_metrics["nonzero_sign_accuracy"],
            "eligible": eligible,
        })
        if eligible:
            best_epoch = epoch
            best_loss = holdout_metrics["weighted_huber"]
            best_state = head_state(trainer)

    common = {
        "schema_version": 1,
        "experiment": "TSA004_shared_target_counterfactual_value",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "rollout_roots": [str(path.resolve()) for path in args.rollout_root],
        "split_mode": "seed",
        "trainable_parameter_count": sum(p.numel() for p in parameters),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "return_scale": args.return_scale,
        "zero_weight": args.zero_weight,
        "huber_beta": args.huber_beta,
        "counts": counts,
        "initial": initial,
        "history": history,
    }
    if best_state is None:
        result = {
            **common,
            "status": "rejected",
            "reason": "no_seed_holdout_value_generalization",
            "output_checkpoint": None,
            "best_epoch": 0,
            "terminal": {
                "train": evaluate(
                    trainer, train, return_scale=args.return_scale, huber_beta=args.huber_beta
                ),
                "holdout": evaluate(
                    trainer, holdout, return_scale=args.return_scale, huber_beta=args.huber_beta
                ),
            },
        }
    else:
        model.target_cf_value_head.load_state_dict(best_state)
        final = {
            "train": evaluate(
                trainer, train, return_scale=args.return_scale, huber_beta=args.huber_beta
            ),
            "holdout": evaluate(
                trainer, holdout, return_scale=args.return_scale, huber_beta=args.huber_beta
            ),
        }
        trainer.last_metrics.update({
            "target_cf_value_best_epoch": float(best_epoch),
            "target_cf_value_return_scale": float(args.return_scale),
            "target_cf_value_holdout_huber": final["holdout"]["weighted_huber"],
            "target_cf_value_holdout_sign_accuracy": final["holdout"]["nonzero_sign_accuracy"],
        })
        trainer.optimizer = torch.optim.Adam(
            trainer.model.parameters(), lr=trainer.config.learning_rate
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(args.output)
        result = {
            **common,
            "status": "accepted",
            "output_checkpoint": str(args.output.resolve()),
            "best_epoch": best_epoch,
            "final": final,
        }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
