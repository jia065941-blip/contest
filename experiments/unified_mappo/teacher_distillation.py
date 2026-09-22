"""R9 teacher-action datasets and actor-only distillation for unified MAPPO."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from experiments.unified_mappo.model import HybridMAPPOConfig, HybridMAPPOTrainer
from policies.red.learning.red_policy import (
    GlobalStateEncoder,
    UNIFIED_LOCAL_OBSERVATION_DIM,
    UNIFIED_TARGET_SLOTS,
)


TENSOR_KEYS = (
    "observations",
    "target_coordinates",
    "target_features",
    "target_valid_mask",
    "factor_mask",
    "presence",
    "initial_xy",
    "search_index",
    "retarget",
    "target_index",
    "maneuver_index",
    "satellite",
    "agent_ids",
    "entity_ids",
    "steps",
)
FACTOR_NAMES = (
    "presence",
    "initial_position",
    "search_position",
    "retarget",
    "target",
    "maneuver",
    "satellite",
)


class TeacherDatasetWriter:
    """Write bounded tensor chunks while a native teacher trace is replayed."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: int,
        scenario: str,
        source_trace: str,
        sample_stride: int,
        chunk_size: int = 32768,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.chunk_dir = self.output_dir / "chunks"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.scenario = str(scenario)
        self.source_trace = str(Path(source_trace).resolve())
        self.sample_stride = max(1, int(sample_stride))
        self.chunk_size = max(1, int(chunk_size))
        self.buffers: dict[str, list[torch.Tensor]] = {key: [] for key in TENSOR_KEYS}
        self.buffered_samples = 0
        self.sample_count = 0
        self.factor_counts = [0] * len(FACTOR_NAMES)
        self.skipped_target_count = 0
        self.target_feature_dim: int | None = None
        self.chunks: list[dict[str, Any]] = []

    def add(
        self,
        tensors: Mapping[str, torch.Tensor],
        stats: Mapping[str, int],
    ) -> None:
        count = int(tensors["observations"].shape[0])
        if count:
            feature_dim = int(tensors["target_features"].shape[-1])
            if self.target_feature_dim not in {None, feature_dim}:
                raise ValueError("教师数据中的目标特征维度不一致")
            self.target_feature_dim = feature_dim
            for key in TENSOR_KEYS:
                self.buffers[key].append(tensors[key].detach().cpu())
            self.buffered_samples += count
            self.sample_count += count
            factor_counts = tensors["factor_mask"].sum(dim=0).tolist()
            self.factor_counts = [
                current + int(value)
                for current, value in zip(self.factor_counts, factor_counts)
            ]
        self.skipped_target_count += int(stats.get("skipped_target_count", 0))
        if self.buffered_samples >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffered_samples:
            return
        payload = {
            key: torch.cat(values, dim=0)
            for key, values in self.buffers.items()
        }
        filename = f"chunk_{len(self.chunks):05d}.pt"
        destination = self.chunk_dir / filename
        torch.save(payload, destination)
        count = int(payload["observations"].shape[0])
        self.chunks.append({
            "path": str(destination.resolve()),
            "sample_count": count,
        })
        self.buffers = {key: [] for key in TENSOR_KEYS}
        self.buffered_samples = 0

    def finalize(self, *, teacher_score: float) -> Path:
        self.flush()
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "kind": "r9_to_unified_mappo_teacher_actions",
            "scenario": self.scenario,
            "seed": self.seed,
            "teacher_score": float(teacher_score),
            "source_trace": self.source_trace,
            "sample_stride": self.sample_stride,
            "observation_dim": UNIFIED_LOCAL_OBSERVATION_DIM,
            "target_slots": UNIFIED_TARGET_SLOTS,
            "target_feature_dim": self.target_feature_dim,
            "sample_count": self.sample_count,
            "factor_counts": dict(zip(FACTOR_NAMES, self.factor_counts)),
            "skipped_target_count": self.skipped_target_count,
            "chunks": self.chunks,
        }
        path = self.output_dir / "manifest.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    selected = values[mask]
    return selected.mean() if selected.numel() else None


def distillation_objective(
    trainer: HybridMAPPOTrainer,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, tuple[float, int]]]:
    device = trainer.device
    observations = batch["observations"].to(device=device, dtype=torch.float32)
    target_valid = batch["target_valid_mask"].to(device=device, dtype=torch.bool)
    factor_mask = batch["factor_mask"].to(device=device, dtype=torch.bool)
    presence = batch["presence"].to(device=device, dtype=torch.float32)
    initial_xy = batch["initial_xy"].to(device=device, dtype=torch.float32)
    search_index = batch["search_index"].to(device=device, dtype=torch.long)
    retarget = batch["retarget"].to(device=device, dtype=torch.float32)
    target_index = batch["target_index"].to(device=device, dtype=torch.long)
    maneuver_index = batch["maneuver_index"].to(device=device, dtype=torch.long)
    satellite = batch["satellite"].to(device=device, dtype=torch.float32)
    parameters = trainer.model.distribution_parameters(observations)

    losses: list[torch.Tensor] = []
    metrics: dict[str, tuple[float, int]] = {}

    presence_mask = factor_mask[:, 0]
    presence_values = F.binary_cross_entropy_with_logits(
        parameters["presence_logits"], presence, reduction="none"
    )
    presence_loss = _masked_mean(presence_values, presence_mask)
    if presence_loss is not None:
        losses.append(presence_loss)
        correct = (
            (parameters["presence_logits"] >= 0).to(torch.long)
            == presence.to(torch.long)
        ) & presence_mask
        metrics["presence_accuracy"] = (
            float(correct.sum().item()), int(presence_mask.sum().item())
        )

    initial_mask = factor_mask[:, 1]
    initial_values = F.smooth_l1_loss(
        torch.tanh(parameters["initial_mean"]), initial_xy, reduction="none"
    ).mean(dim=-1)
    initial_loss = _masked_mean(initial_values, initial_mask)
    if initial_loss is not None:
        losses.append(initial_loss)
        absolute_error = (
            torch.tanh(parameters["initial_mean"]) - initial_xy
        ).abs().mean(dim=-1)
        metrics["initial_xy_mae"] = (
            float(absolute_error[initial_mask].sum().item()),
            int(initial_mask.sum().item()),
        )

    search_mask = factor_mask[:, 2]
    search_values = F.cross_entropy(
        parameters["search_logits"], search_index, reduction="none"
    )
    search_loss = _masked_mean(search_values, search_mask)
    if search_loss is not None:
        losses.append(search_loss)
        correct = (
            parameters["search_logits"].argmax(dim=-1) == search_index
        ) & search_mask
        metrics["search_accuracy"] = (
            float(correct.sum().item()),
            int(search_mask.sum().item()),
        )

    retarget_mask = factor_mask[:, 3]
    retarget_values = F.binary_cross_entropy_with_logits(
        parameters["retarget_logits"], retarget, reduction="none"
    )
    retarget_loss = _masked_mean(retarget_values, retarget_mask)
    if retarget_loss is not None:
        losses.append(retarget_loss)
        correct = (
            (parameters["retarget_logits"] >= 0).to(torch.long)
            == retarget.to(torch.long)
        ) & retarget_mask
        metrics["retarget_accuracy"] = (
            float(correct.sum().item()), int(retarget_mask.sum().item())
        )

    target_mask = factor_mask[:, 4]
    if bool(target_mask.any().item()):
        active_target_logits = parameters["target_logits"][target_mask]
        active_target_valid = target_valid[target_mask]
        active_target_logits = active_target_logits.masked_fill(
            ~active_target_valid,
            torch.finfo(active_target_logits.dtype).min,
        )
        active_target_index = target_index[target_mask]
        target_loss = F.cross_entropy(
            active_target_logits,
            active_target_index,
        )
        losses.append(target_loss)
        correct = active_target_logits.argmax(dim=-1) == active_target_index
        metrics["target_accuracy"] = (
            float(correct.sum().item()), int(target_mask.sum().item())
        )

    maneuver_mask = factor_mask[:, 5]
    maneuver_values = F.cross_entropy(
        parameters["maneuver_logits"], maneuver_index, reduction="none"
    )
    maneuver_loss = _masked_mean(maneuver_values, maneuver_mask)
    if maneuver_loss is not None:
        losses.append(maneuver_loss)
        correct = (
            parameters["maneuver_logits"].argmax(dim=-1) == maneuver_index
        ) & maneuver_mask
        metrics["maneuver_accuracy"] = (
            float(correct.sum().item()), int(maneuver_mask.sum().item())
        )

    satellite_mask = factor_mask[:, 6]
    satellite_values = F.binary_cross_entropy_with_logits(
        parameters["satellite_logits"], satellite, reduction="none"
    )
    satellite_loss = _masked_mean(satellite_values, satellite_mask)
    if satellite_loss is not None:
        losses.append(satellite_loss)
        correct = (
            (parameters["satellite_logits"] >= 0).to(torch.long)
            == satellite.to(torch.long)
        ) & satellite_mask
        metrics["satellite_accuracy"] = (
            float(correct.sum().item()), int(satellite_mask.sum().item())
        )

    if not losses:
        return parameters["presence_logits"].sum() * 0.0, metrics
    return torch.stack(losses).sum(), metrics


def _dataset_manifests(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("seed_*/manifest.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def create_initial_checkpoint(
    path: str | Path,
    *,
    max_steps: int = 3000,
    hidden_dim: int = 256,
    learning_rate: float = 1e-4,
    seed: int = 7,
) -> Path:
    trainer = HybridMAPPOTrainer(HybridMAPPOConfig(
        observation_dim=UNIFIED_LOCAL_OBSERVATION_DIM,
        critic_state_dim=GlobalStateEncoder(
            max_steps=max_steps,
            target_slots=UNIFIED_TARGET_SLOTS,
        ).state_dim,
        critic_focal_observation_dim=UNIFIED_LOCAL_OBSERVATION_DIM,
        target_slots=UNIFIED_TARGET_SLOTS,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
        device="cpu",
    ))
    destination = Path(path)
    trainer.save(destination)
    return destination


def train_from_dataset(args: argparse.Namespace) -> dict[str, Any]:
    manifests = _dataset_manifests(args.dataset_root)
    chunk_paths = [
        Path(chunk["path"])
        for manifest in manifests
        for chunk in manifest["chunks"]
    ]
    if args.source_checkpoint:
        trainer = HybridMAPPOTrainer.load(args.source_checkpoint, device="cpu")
    else:
        config = HybridMAPPOConfig(
            observation_dim=UNIFIED_LOCAL_OBSERVATION_DIM,
            critic_state_dim=GlobalStateEncoder(
                max_steps=args.max_steps,
                target_slots=UNIFIED_TARGET_SLOTS,
            ).state_dim,
            critic_focal_observation_dim=UNIFIED_LOCAL_OBSERVATION_DIM,
            target_slots=UNIFIED_TARGET_SLOTS,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device="cpu",
        )
        trainer = HybridMAPPOTrainer(config)
    trainer.move_runtime_device(args.device)
    for group in trainer.optimizer.param_groups:
        group["lr"] = float(args.learning_rate)

    randomizer = random.Random(args.seed)
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    total_samples = sum(int(row["sample_count"]) for row in manifests)
    for epoch in range(1, args.epochs + 1):
        ordered_chunks = list(chunk_paths)
        randomizer.shuffle(ordered_chunks)
        totals: dict[str, list[float]] = {}
        loss_sum = 0.0
        minibatches = 0
        trainer.model.train(True)
        for chunk_path in ordered_chunks:
            chunk = torch.load(chunk_path, map_location="cpu", weights_only=True)
            count = int(chunk["observations"].shape[0])
            generator = torch.Generator().manual_seed(args.seed + epoch + count)
            order = torch.randperm(count, generator=generator)
            for start in range(0, count, args.batch_size):
                indices = order[start:start + args.batch_size]
                batch = {key: value[indices] for key, value in chunk.items()}
                loss, metrics = distillation_objective(trainer, batch)
                trainer.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainer.model.parameters(), trainer.config.max_grad_norm
                )
                trainer.optimizer.step()
                optimizer_steps += 1
                minibatches += 1
                loss_sum += float(loss.item())
                for name, (numerator, denominator) in metrics.items():
                    accumulator = totals.setdefault(name, [0.0, 0.0])
                    accumulator[0] += numerator
                    accumulator[1] += denominator
        epoch_metrics = {
            "epoch": epoch,
            "loss": loss_sum / max(1, minibatches),
            **{
                name: numerator / max(1.0, denominator)
                for name, (numerator, denominator) in totals.items()
            },
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False), flush=True)

    trainer.move_runtime_device("cpu")
    trainer.config = replace(trainer.config, device="cpu")
    trainer.model.config = trainer.config
    trainer.update_count += optimizer_steps
    trainer.transition_count += total_samples * args.epochs
    trainer.last_metrics = {
        "distillation_epochs": float(args.epochs),
        "distillation_seed_count": float(len(manifests)),
        "distillation_sample_count": float(total_samples),
        "distillation_optimizer_steps": float(optimizer_steps),
        **{f"distillation_{key}": float(value) for key, value in history[-1].items() if key != "epoch"},
    }
    trainer.save(args.output_checkpoint)
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root.resolve()),
        "seed_count": len(manifests),
        "sample_count": total_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "checkpoint": str(args.output_checkpoint.resolve()),
        "history": history,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    result = train_from_dataset(parse_args())
    print(json.dumps({
        "checkpoint": result["checkpoint"],
        "seed_count": result["seed_count"],
        "sample_count": result["sample_count"],
        "final": result["history"][-1],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
