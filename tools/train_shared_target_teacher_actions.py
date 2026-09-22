#!/usr/bin/env python3
"""Train the shared target scorer from legal native-teacher target actions."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unified_mappo.model import HybridMAPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--target-feature-dim", type=int, default=26)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--holdout-stride", type=int, default=4)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_directories(root: Path) -> list[Path]:
    directories = sorted(path for path in root.glob("dataset/seed_*") if path.is_dir())
    if not directories:
        directories = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if len(directories) < 2:
        raise ValueError("教师数据至少需要两个独立 seed")
    return directories


def load_partition(
    directories: list[Path],
    *,
    target_slots: int,
    target_feature_dim: int,
) -> dict[str, torch.Tensor]:
    rows: dict[str, list[torch.Tensor]] = {
        "observations": [],
        "features": [],
        "valid": [],
        "labels": [],
    }
    for directory in directories:
        for path in sorted((directory / "chunks").glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            required = {
                "observations",
                "target_features",
                "target_valid_mask",
                "factor_mask",
                "target_index",
            }
            missing = required - set(payload)
            if missing:
                raise ValueError(f"{path} 缺少字段 {sorted(missing)}")
            active = payload["factor_mask"][:, 4].to(dtype=torch.bool)
            if not bool(active.any()):
                continue
            observations = payload["observations"][active].to(dtype=torch.float32)
            features = payload["target_features"][active].to(dtype=torch.float32)
            valid = payload["target_valid_mask"][active].to(dtype=torch.bool)
            labels = payload["target_index"][active].to(dtype=torch.long)
            if features.shape[1:] != (target_slots, target_feature_dim):
                raise ValueError(
                    f"{path} 目标特征形状 {tuple(features.shape)} 与 "
                    f"(*,{target_slots},{target_feature_dim}) 不兼容"
                )
            label_legal = valid.gather(1, labels.unsqueeze(1)).squeeze(1)
            if not bool(label_legal.all()):
                raise ValueError(f"{path} 包含被掩码的教师目标标签")
            rows["observations"].append(observations)
            rows["features"].append(features)
            rows["valid"].append(valid)
            rows["labels"].append(labels)
    if not rows["observations"]:
        raise ValueError("分区没有合法教师目标动作")
    return {key: torch.cat(values, dim=0) for key, values in rows.items()}


def prepare(
    trainer: HybridMAPPOTrainer,
    rows: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    device = trainer.device
    observations = rows["observations"].to(device)
    with torch.no_grad():
        latent = trainer.model.encoder(observations)
    return {
        "latent": latent.detach(),
        "features": rows["features"].to(device),
        "valid": rows["valid"].to(device),
        "labels": rows["labels"].to(device),
    }


def logits_for(
    trainer: HybridMAPPOTrainer,
    rows: dict[str, torch.Tensor],
) -> torch.Tensor:
    return trainer.model.teacher_target_logits(
        rows["latent"], rows["features"], rows["valid"]
    ).masked_fill(~rows["valid"], -1e9)


@torch.no_grad()
def evaluate(
    trainer: HybridMAPPOTrainer,
    rows: dict[str, torch.Tensor],
) -> dict[str, float]:
    logits = logits_for(trainer, rows)
    losses = F.cross_entropy(logits, rows["labels"], reduction="none")
    predictions = logits.argmax(dim=-1)
    probabilities = F.softmax(logits, dim=-1)
    label_probability = probabilities.gather(
        1, rows["labels"].unsqueeze(1)
    ).squeeze(1)
    reserved = rows["features"][:, :, 22].amax(dim=1) > 0
    bda = rows["features"][:, :, 20].amax(dim=1) > 0
    multi = rows["valid"].sum(dim=1) > 1

    def subset_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        return float(values[mask].mean().cpu()) if bool(mask.any()) else 0.0

    correct = (predictions == rows["labels"]).to(dtype=torch.float32)
    return {
        "rows": float(len(losses)),
        "nll": float(losses.mean().cpu()),
        "top1_accuracy": float(correct.mean().cpu()),
        "teacher_action_probability": float(label_probability.mean().cpu()),
        "multi_target_row_fraction": float(multi.float().mean().cpu()),
        "multi_target_top1_accuracy": subset_mean(correct, multi),
        "multi_target_nll": subset_mean(losses, multi),
        "multi_target_teacher_action_probability": subset_mean(
            label_probability, multi
        ),
        "reservation_row_fraction": float(reserved.float().mean().cpu()),
        "reservation_top1_accuracy": subset_mean(correct, reserved),
        "reservation_nll": subset_mean(losses, reserved),
        "bda_row_fraction": float(bda.float().mean().cpu()),
        "bda_top1_accuracy": subset_mean(correct, bda),
    }


def main() -> None:
    args = parse_args()
    if args.target_feature_dim <= 0:
        raise ValueError("target-feature-dim 必须为正")
    if args.epochs <= 0 or args.learning_rate <= 0.0:
        raise ValueError("epochs 和 learning-rate 必须为正")
    if args.holdout_stride < 2 or args.validation_interval <= 0:
        raise ValueError("holdout-stride 至少为 2 且 validation-interval 必须为正")
    if args.patience <= 0 or args.minibatch_size <= 0:
        raise ValueError("patience 和 minibatch-size 必须为正")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 CUDA 不可用")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = HybridMAPPOTrainer.load(
        args.checkpoint,
        device=args.device,
        target_feature_dim=args.target_feature_dim,
    )
    directories = seed_directories(args.dataset_root)
    holdout_dirs = [
        path for index, path in enumerate(directories)
        if index % args.holdout_stride == 0
    ]
    train_dirs = [path for path in directories if path not in holdout_dirs]
    train = prepare(trainer, load_partition(
        train_dirs,
        target_slots=trainer.config.target_slots,
        target_feature_dim=trainer.config.target_feature_dim,
    ))
    holdout = prepare(trainer, load_partition(
        holdout_dirs,
        target_slots=trainer.config.target_slots,
        target_feature_dim=trainer.config.target_feature_dim,
    ))

    for parameter in trainer.model.parameters():
        parameter.requires_grad_(False)
    trainer.model.target_teacher_item_encoder.load_state_dict(
        trainer.model.target_item_encoder.state_dict()
    )
    trainer.model.target_teacher_query.load_state_dict(
        trainer.model.target_query.state_dict()
    )
    trainer.model.target_teacher_score.load_state_dict(
        trainer.model.target_score.state_dict()
    )
    parameters = [
        *trainer.model.target_teacher_item_encoder.parameters(),
        *trainer.model.target_teacher_query.parameters(),
        *trainer.model.target_teacher_score.parameters(),
    ]
    for parameter in parameters:
        parameter.requires_grad_(True)
    trainer.model.set_target_teacher_policy_scale(0.0)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    initial = {
        "train": evaluate(trainer, train),
        "holdout": evaluate(trainer, holdout),
    }
    best_state = copy.deepcopy(trainer.model.state_dict())
    best_epoch = 0
    best_holdout = initial["holdout"]["nll"]
    stale_validations = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(train["labels"]), device=trainer.device)
        trainer.model.train()
        for start in range(0, len(permutation), args.minibatch_size):
            indices = permutation[start:start + args.minibatch_size]
            minibatch = {
                key: value.index_select(0, indices)
                for key, value in train.items()
            }
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(logits_for(trainer, minibatch), minibatch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        if epoch % args.validation_interval != 0 and epoch != args.epochs:
            continue
        trainer.model.eval()
        train_metrics = evaluate(trainer, train)
        holdout_metrics = evaluate(trainer, holdout)
        history.append({
            "epoch": float(epoch),
            "train_nll": train_metrics["nll"],
            "holdout_nll": holdout_metrics["nll"],
            "holdout_top1": holdout_metrics["top1_accuracy"],
        })
        if holdout_metrics["nll"] < best_holdout - 1e-6:
            best_holdout = holdout_metrics["nll"]
            best_epoch = epoch
            best_state = copy.deepcopy(trainer.model.state_dict())
            stale_validations = 0
        else:
            stale_validations += 1
            if stale_validations >= args.patience:
                break
    trainer.model.load_state_dict(best_state)
    trainer.model.set_target_teacher_policy_scale(0.0)
    trainer.model.eval()
    final = {
        "train": evaluate(trainer, train),
        "holdout": evaluate(trainer, holdout),
    }
    trainer.last_metrics.update({
        "shared_teacher_best_epoch": float(best_epoch),
        "shared_teacher_holdout_nll": final["holdout"]["nll"],
        "shared_teacher_holdout_top1": final["holdout"]["top1_accuracy"],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.output)
    result = {
        "schema_version": 1,
        "experiment": "shared_target_teacher_actions_v1",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "output_checkpoint": str(args.output.resolve()),
        "target_feature_dim": trainer.config.target_feature_dim,
        "train_seeds": [path.name for path in train_dirs],
        "holdout_seeds": [path.name for path in holdout_dirs],
        "best_epoch": best_epoch,
        "initial": initial,
        "final": final,
        "history": history,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
