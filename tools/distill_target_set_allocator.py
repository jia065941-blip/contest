#!/usr/bin/env python3
"""Function-preserving distillation for the shared per-target scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unified_mappo.model import HybridMAPPOTrainer


SELF_FEATURES = 21
IDENTITY_FEATURES = 210
DETECTION_FEATURES = 4
TARGET_GEOMETRY_FEATURES = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, action="append", default=[])
    parser.add_argument("--allocator-trace", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--holdout-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rich-features", action="store_true")
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--validation-interval", type=int, default=10)
    return parser.parse_args()


def rollout_paths(roots: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        if root.is_file():
            paths.add(root.resolve())
        else:
            paths.update(path.resolve() for path in root.rglob("on_policy_rollout.pt"))
    if not paths:
        raise ValueError("没有找到 on_policy_rollout.pt")
    return sorted(paths)


def allocator_trace_paths(inputs: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for value in inputs:
        if value.is_file():
            paths.add(value.resolve())
        else:
            paths.update(path.resolve() for path in value.rglob("*.pt"))
    return sorted(paths)


def derive_geometry_features(
    observations: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    target_slots: int,
    target_feature_dim: int,
) -> torch.Tensor:
    """Recover only the legal historical geometry; rich columns stay neutral."""

    observation_dim = int(observations.shape[1])
    target_start = SELF_FEATURES + IDENTITY_FEATURES
    target_stride, remainder = divmod(
        observation_dim - target_start - DETECTION_FEATURES,
        target_slots,
    )
    if remainder or target_stride < TARGET_GEOMETRY_FEATURES:
        raise ValueError(
            f"无法从 observation_dim={observation_dim} 恢复目标块"
        )
    features = torch.zeros(
        observations.shape[0],
        target_slots,
        target_feature_dim,
        dtype=torch.float32,
    )
    for slot in range(target_slots):
        start = target_start + slot * target_stride
        features[:, slot, :TARGET_GEOMETRY_FEATURES] = observations[
            :, start:start + TARGET_GEOMETRY_FEATURES
        ]
    features *= valid_mask.to(dtype=features.dtype).unsqueeze(-1)
    return features


def load_rows(
    paths: list[Path],
    *,
    target_slots: int,
    target_feature_dim: int,
    holdout_stride: int,
    rich_features: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, int]]:
    partitions: dict[str, dict[str, list[torch.Tensor]]] = {
        "train": {"observations": [], "features": [], "valid": []},
        "holdout": {"observations": [], "features": [], "valid": []},
    }
    file_counts = {"train_files": 0, "holdout_files": 0}
    for file_index, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "observations" in payload:
            observations = payload["observations"].to(dtype=torch.float32)
            valid = payload["target_valid_mask"].to(dtype=torch.bool)
            stored_features = payload.get("target_features")
        else:
            batch = payload["rollout"] if isinstance(payload, dict) else payload
            active = batch.action_mask.target.to(dtype=torch.bool)
            if not bool(active.any()):
                continue
            observations = batch.observations[active].to(dtype=torch.float32)
            valid = batch.target_valid_mask[active].to(dtype=torch.bool)
            batch_features = getattr(batch, "target_features", None)
            stored_features = (
                batch_features[active] if batch_features is not None else None
            )
        if observations.shape[0] == 0:
            continue
        if rich_features:
            if stored_features is None:
                raise ValueError(f"rich-features 输入缺少 target_features: {path}")
            features = stored_features.to(dtype=torch.float32)
            if features.shape != (
                observations.shape[0], target_slots, target_feature_dim
            ):
                raise ValueError(f"target_features 形状不兼容: {path}")
        else:
            features = derive_geometry_features(
                observations,
                valid,
                target_slots=target_slots,
                target_feature_dim=target_feature_dim,
            )
        partition = "holdout" if file_index % holdout_stride == 0 else "train"
        partitions[partition]["observations"].append(observations)
        partitions[partition]["features"].append(features)
        partitions[partition]["valid"].append(valid)
        file_counts[f"{partition}_files"] += 1

    packed: dict[str, dict[str, torch.Tensor]] = {}
    for name, rows in partitions.items():
        if not rows["observations"]:
            raise ValueError(f"{name} 分区没有目标边界样本")
        packed[name] = {
            key: torch.cat(values, dim=0)
            for key, values in rows.items()
        }
    counts = {
        **file_counts,
        "train_rows": int(packed["train"]["observations"].shape[0]),
        "holdout_rows": int(packed["holdout"]["observations"].shape[0]),
        "train_multi_target_rows": int((packed["train"]["valid"].sum(-1) > 1).sum()),
        "holdout_multi_target_rows": int((packed["holdout"]["valid"].sum(-1) > 1).sum()),
    }
    return packed["train"], packed["holdout"], counts


def prepare(
    trainer: HybridMAPPOTrainer,
    rows: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    observations = rows["observations"].to(device)
    features = rows["features"].to(device)
    valid = rows["valid"].to(device)
    with torch.no_grad():
        latent = trainer.model.encoder(observations)
        teacher_logits = trainer.model.target_head(latent).masked_fill(~valid, -1e9)
        teacher_prob = F.softmax(teacher_logits, dim=-1)
        teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
    return {
        "latent": latent.detach(),
        "features": features,
        "valid": valid,
        "teacher_prob": teacher_prob,
        "teacher_log_prob": teacher_log_prob,
    }


def loss_for(trainer: HybridMAPPOTrainer, rows: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = trainer.model.shared_target_logits(
        rows["latent"], rows["features"], rows["valid"]
    ).masked_fill(~rows["valid"], -1e9)
    student_log_prob = F.log_softmax(logits, dim=-1)
    return (
        rows["teacher_prob"]
        * (rows["teacher_log_prob"] - student_log_prob)
    ).sum(dim=-1).mean()


@torch.no_grad()
def evaluate(
    trainer: HybridMAPPOTrainer,
    rows: dict[str, torch.Tensor],
) -> dict[str, float]:
    logits = trainer.model.shared_target_logits(
        rows["latent"], rows["features"], rows["valid"]
    ).masked_fill(~rows["valid"], -1e9)
    student_log_prob = F.log_softmax(logits, dim=-1)
    kl = (
        rows["teacher_prob"]
        * (rows["teacher_log_prob"] - student_log_prob)
    ).sum(dim=-1)
    multi = rows["valid"].sum(dim=-1) > 1
    agreements = logits.argmax(dim=-1) == rows["teacher_prob"].argmax(dim=-1)
    return {
        "mean_kl": float(kl.mean().cpu()),
        "max_kl": float(kl.max().cpu()),
        "multi_target_mean_kl": float(kl[multi].mean().cpu()) if bool(multi.any()) else 0.0,
        "multi_target_top1_agreement": float(agreements[multi].float().mean().cpu()) if bool(multi.any()) else 1.0,
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.learning_rate <= 0.0:
        raise ValueError("epochs 和 learning-rate 必须为正")
    if args.holdout_stride < 2:
        raise ValueError("holdout-stride 必须至少为 2")
    if args.minibatch_size <= 0 or args.validation_interval <= 0:
        raise ValueError("minibatch-size 和 validation-interval 必须为正")
    if not args.rollout_root and not args.allocator_trace:
        raise ValueError("必须提供 rollout-root 或 allocator-trace")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 CUDA 不可用")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = HybridMAPPOTrainer.load(args.checkpoint, device=args.device)
    model = trainer.model
    if float(model.target_allocator_mix.item()) != 0.0:
        raise ValueError("M2 必须从 target_allocator_mix=0 的函数保持检查点开始")
    paths = sorted({
        *(
            rollout_paths(args.rollout_root)
            if args.rollout_root else ()
        ),
        *allocator_trace_paths(args.allocator_trace),
    })
    if not paths:
        raise ValueError("没有找到蒸馏输入")
    train_raw, holdout_raw, counts = load_rows(
        paths,
        target_slots=model.config.target_slots,
        target_feature_dim=model.config.target_feature_dim,
        holdout_stride=args.holdout_stride,
        rich_features=args.rich_features,
    )
    device = next(model.parameters()).device
    train = prepare(trainer, train_raw, device)
    holdout = prepare(trainer, holdout_raw, device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    allocator_parameters = [
        *model.target_item_encoder.parameters(),
        *model.target_query.parameters(),
        *model.target_score.parameters(),
    ]
    for parameter in allocator_parameters:
        parameter.requires_grad_(True)
    if not args.rich_features:
        with torch.no_grad():
            model.target_item_encoder[0].weight[
                :, TARGET_GEOMETRY_FEATURES:
            ].zero_()

    optimizer = torch.optim.Adam(allocator_parameters, lr=args.learning_rate)
    initial = {
        "train": evaluate(trainer, train),
        "holdout": evaluate(trainer, holdout),
    }
    best_state = None
    best_holdout = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(
            train["latent"].shape[0], device=device
        )
        for start in range(0, len(permutation), args.minibatch_size):
            indices = permutation[start:start + args.minibatch_size]
            minibatch = {
                key: value.index_select(0, indices)
                for key, value in train.items()
            }
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(trainer, minibatch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(allocator_parameters, 1.0)
            optimizer.step()
        if epoch % args.validation_interval == 0 or epoch == args.epochs:
            holdout_kl = evaluate(trainer, holdout)["mean_kl"]
            if holdout_kl < best_holdout:
                best_holdout = holdout_kl
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                    if name.startswith((
                        "target_item_encoder.", "target_query.", "target_score."
                    ))
                }
    if best_state is None:
        raise RuntimeError("蒸馏没有产生候选")
    model.load_state_dict(best_state, strict=False)
    final = {
        "train": evaluate(trainer, train),
        "holdout": evaluate(trainer, holdout),
    }
    model.set_target_allocator_mix(0.0)
    trainer.last_metrics.update({
        "target_allocator_distill_holdout_kl": final["holdout"]["mean_kl"],
        "target_allocator_distill_holdout_top1": final["holdout"]["multi_target_top1_agreement"],
        "target_allocator_distill_best_epoch": float(best_epoch),
    })
    trainer.save(args.output)
    result = {
        "schema_version": 1,
        "experiment": "TSA002_shared_target_scorer_distillation",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "output_checkpoint": str(args.output.resolve()),
        "rollout_roots": [str(path.resolve()) for path in args.rollout_root],
        "allocator_traces": [
            str(path.resolve()) for path in args.allocator_trace
        ],
        "geometry_only": not args.rich_features,
        "target_allocator_mix": 0.0,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "minibatch_size": args.minibatch_size,
        "validation_interval": args.validation_interval,
        "best_epoch": best_epoch,
        "counts": counts,
        "initial": initial,
        "final": final,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
