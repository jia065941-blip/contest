#!/usr/bin/env python3
"""Short C0a_goal update from advantages over the current factual target.

Only candidates with a strictly better official E01 return than the action
sampled by the frozen source policy receive a positive training signal.  The
complete target Transformer is updated; the shared actor trunk, critic, and
all other action heads stay bitwise frozen.  Every invocation creates a fresh
Adam optimizer and is intended to be followed by candidate regeneration.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F


SIM_ROOT = Path(__file__).resolve().parents[1]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from experiments.unified_mappo.model import (  # noqa: E402
    HybridMAPPOConfig,
    HybridMAPPOModel,
)


TARGET_PREFIXES = (
    "target_transformer_item_encoder.",
    "target_transformer_query.",
    "target_transformer_encoder.",
    "target_transformer_score.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-batch", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--advantage-epsilon", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260918)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(
    checkpoint: Path, device: str
) -> tuple[dict[str, Any], HybridMAPPOConfig, HybridMAPPOModel]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("algorithm") != "target_conditioned_ctde_mappo_v11":
        raise ValueError(f"unexpected algorithm: {payload.get('algorithm')!r}")
    values = dict(payload["config"])
    values["device"] = device
    config = HybridMAPPOConfig(**values)
    if config.target_selector_arch != "transformer":
        raise ValueError("source checkpoint does not use target Transformer")
    model = HybridMAPPOModel(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return payload, config, model


def load_items(
    paths: list[Path], device: str, expected_model_sha: str
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    batch_metadata: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("stage") != "C0a_goal":
            raise ValueError(f"not a C0a_goal batch: {path}")
        snapshot_sha = str(payload.get("snapshot_model_sha256"))
        if snapshot_sha != expected_model_sha:
            raise ValueError(
                f"candidate/source model mismatch for {path}: "
                f"{snapshot_sha} != {expected_model_sha}"
            )
        candidate_counts = [
            int(item["candidate_target_indices"].numel()) for item in payload["items"]
        ]
        if any(count < 1 or count > 10 for count in candidate_counts):
            raise ValueError(f"candidate count outside [1, 10] in {path}: {candidate_counts}")
        batch_metadata.append({
            "path": str(path.resolve()),
            "batch_index": int(payload["batch_index"]),
            "boundary_source": str(payload.get("boundary_source")),
            "state_count": len(payload["items"]),
            "singleton_state_count": sum(count == 1 for count in candidate_counts),
            "snapshot_model_sha256": snapshot_sha,
        })
        for item in payload["items"]:
            items.append(item)
            provenance.append({
                "batch_index": int(payload["batch_index"]),
                "boundary_source": str(payload.get("boundary_source")),
                "seed": int(item["seed"]),
                "start_state_id": str(item["start_state_id"]),
                "candidate_count": int(item["candidate_target_indices"].numel()),
            })
    if not items:
        raise ValueError("no candidate items")
    max_candidates = max(int(x["candidate_target_indices"].numel()) for x in items)

    def pad_with_factual(item: dict[str, Any], key: str) -> torch.Tensor:
        value = item[key]
        missing = max_candidates - int(value.numel())
        if missing <= 0:
            return value
        return torch.cat((value, value[:1].repeat(missing)), dim=0)

    data = {
        "observation": torch.cat([x["observation"] for x in items], dim=0).to(device),
        "features": torch.cat([x["target_features"] for x in items], dim=0).to(device),
        "valid": torch.cat([x["target_valid_mask"] for x in items], dim=0).to(
            device=device, dtype=torch.bool
        ),
        "indices": torch.stack([
            pad_with_factual(x, "candidate_target_indices") for x in items
        ]).to(
            device=device, dtype=torch.long
        ),
        "returns": torch.stack([
            pad_with_factual(x, "official_joint_returns") for x in items
        ]).to(
            device=device, dtype=torch.float32
        ),
    }
    return data, provenance, batch_metadata


def target_logits(
    model: HybridMAPPOModel,
    observations: torch.Tensor,
    features: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    latent = model.encoder(observations)
    return model.transformer_target_logits(latent, features, valid).masked_fill(
        ~valid, float("-inf")
    )


def metrics(
    model: HybridMAPPOModel,
    data: dict[str, torch.Tensor],
    source_probabilities: torch.Tensor,
    epsilon: float,
) -> dict[str, float | int]:
    with torch.no_grad():
        logits = target_logits(
            model, data["observation"], data["features"], data["valid"]
        )
        probabilities = torch.softmax(logits, dim=-1)
        candidate_probabilities = probabilities.gather(1, data["indices"])
        advantages = data["returns"] - data["returns"][:, :1]
        positive = advantages > epsilon
        positive_advantages = advantages.clamp_min(0.0) * positive
        captured_advantage = (
            candidate_probabilities * positive_advantages
        ).sum(dim=-1)
        positive_probability = (
            candidate_probabilities * positive.to(candidate_probabilities.dtype)
        ).sum(dim=-1)
        factual_probability = candidate_probabilities[:, 0]
        top = probabilities.argmax(dim=-1)
        top_is_positive = (
            (data["indices"] == top.unsqueeze(-1)) & positive
        ).any(dim=-1)
        tv = 0.5 * (probabilities - source_probabilities).abs().sum(dim=-1)
    return {
        "state_count": int(data["returns"].shape[0]),
        "improving_state_count": int(positive.any(dim=-1).sum().item()),
        "improving_candidate_count": int(positive.sum().item()),
        "mean_captured_positive_advantage": float(captured_advantage.mean().item()),
        "mean_probability_on_improving_candidates": float(
            positive_probability.mean().item()
        ),
        "mean_factual_target_probability": float(factual_probability.mean().item()),
        "global_top1_improves_over_factual": int(top_is_positive.sum().item()),
        "mean_tv_from_round_source": float(tv.mean().item()),
        "max_tv_from_round_source": float(tv.max().item()),
    }


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.learning_rate <= 0.0:
        raise ValueError("steps and learning-rate must be positive")
    if args.advantage_epsilon < 0.0:
        raise ValueError("advantage-epsilon must be nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source, config, model = load_model(args.source_checkpoint, args.device)
    _, _, anchor = load_model(args.source_checkpoint, args.device)
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    source_model_sha = model_sha256(source["model"])
    data, provenance, batch_metadata = load_items(
        args.candidate_batch, args.device, source_model_sha
    )

    before = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    trainable_names: list[str] = []
    trainable: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(TARGET_PREFIXES)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable_names.append(name)
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("no target Transformer parameters selected")
    model.eval()

    with torch.no_grad():
        source_probabilities = torch.softmax(target_logits(
            anchor, data["observation"], data["features"], data["valid"]
        ), dim=-1)
    baseline = metrics(
        model, data, source_probabilities, args.advantage_epsilon
    )
    advantages = data["returns"] - data["returns"][:, :1]
    positive = advantages > args.advantage_epsilon
    if not bool(positive.any()):
        raise RuntimeError("batch has no candidate better than its factual target")
    positive_weights = advantages.clamp_min(0.0) * positive
    positive_weights = positive_weights / positive_weights.sum().clamp_min(1e-12)

    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    named_parameters = dict(model.named_parameters())
    best_parameters = {
        name: named_parameters[name].detach().clone() for name in trainable_names
    }
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    best_step = 0
    best_value = float(baseline["mean_captured_positive_advantage"])
    best_top1 = int(baseline["global_top1_improves_over_factual"])
    history: list[dict[str, Any]] = []

    for step in range(1, args.steps + 1):
        logits = target_logits(
            model, data["observation"], data["features"], data["valid"]
        )
        candidate_logits = logits.gather(1, data["indices"])
        factual_logits = candidate_logits[:, :1]
        pairwise_losses = F.softplus(-(candidate_logits - factual_logits))
        loss = (positive_weights.detach() * pairwise_losses).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            trainable, args.max_grad_norm
        ).item())
        optimizer.step()

        current = metrics(
            model, data, source_probabilities, args.advantage_epsilon
        )
        current_value = float(current["mean_captured_positive_advantage"])
        current_top1 = int(current["global_top1_improves_over_factual"])
        improved = (
            current_value > best_value + 1e-12
            or (
                abs(current_value - best_value) <= 1e-12
                and current_top1 > best_top1
            )
        )
        if improved:
            best_value = current_value
            best_top1 = current_top1
            best_step = step
            best_parameters = {
                name: named_parameters[name].detach().clone()
                for name in trainable_names
            }
            best_optimizer = copy.deepcopy(optimizer.state_dict())
        history.append({
            "step": step,
            "pairwise_positive_advantage_loss": float(loss.detach().item()),
            "gradient_norm_before_clip": gradient_norm,
            "metrics": current,
            "selected_so_far": best_step,
        })
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)

    if best_step <= 0:
        raise RuntimeError("short update did not improve captured positive advantage")
    for name, value in best_parameters.items():
        named_parameters[name].data.copy_(value)
    optimizer.load_state_dict(best_optimizer)
    selected = metrics(
        model, data, source_probabilities, args.advantage_epsilon
    )

    changed_target: list[str] = []
    changed_frozen: list[str] = []
    after = model.state_dict()
    for name, old_value in before.items():
        if torch.equal(old_value, after[name].detach().cpu()):
            continue
        if name.startswith(TARGET_PREFIXES):
            changed_target.append(name)
        else:
            changed_frozen.append(name)
    if changed_frozen:
        raise RuntimeError(f"frozen tensors changed: {changed_frozen}")

    optimizer_steps = sorted({
        int(state["step"].item())
        for state in optimizer.state.values()
        if "step" in state
    })
    checkpoint = {
        "algorithm": source["algorithm"],
        "config": asdict(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update_count": int(source.get("update_count", 0)) + best_step,
        "transition_count": int(source.get("transition_count", 0)) + len(provenance),
        "metrics": {
            "method": "positive_advantage_pairwise_short_update",
            "selected_step": best_step,
            "baseline": baseline,
            "selected": selected,
            "captured_positive_advantage_delta_points": 100.0 * (
                float(selected["mean_captured_positive_advantage"])
                - float(baseline["mean_captured_positive_advantage"])
            ),
            "changed_target_transformer_tensors": len(changed_target),
            "changed_frozen_tensors": changed_frozen,
            "optimizer_step_values": optimizer_steps,
            "old_optimizer_state_loaded": 0,
            "scheduler_used": 0,
            "ppo_used": 0,
            "critic_updated": 0,
            "actor_trunk_updated": 0,
            "other_action_heads_updated": 0,
        },
        "continuation": {
            "stage": "C0a_goal_b8_iterative",
            "source_checkpoint": str(args.source_checkpoint.resolve()),
            "source_checkpoint_sha256": file_sha256(args.source_checkpoint),
            "source_model_sha256": source_model_sha,
            "candidate_batches": batch_metadata,
            "training_states": provenance,
            "optimizer": "fresh Adam",
            "optimizer_state_inherited": False,
            "objective": (
                "advantage-weighted pairwise ranking against the frozen "
                "policy's factual target; non-positive candidates have zero weight"
            ),
            "trainable_parameter_prefixes": list(TARGET_PREFIXES),
        },
    }
    output_checkpoint = args.output_dir / "checkpoints" / "working.pt"
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_checkpoint)
    summary = {
        "status": "complete",
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": file_sha256(args.source_checkpoint),
        "source_model_sha256": source_model_sha,
        "output_checkpoint": str(output_checkpoint.resolve()),
        "output_checkpoint_sha256": file_sha256(output_checkpoint),
        "output_model_sha256": model_sha256(checkpoint["model"]),
        "candidate_batches": batch_metadata,
        "state_count": len(provenance),
        "selected_step": best_step,
        "baseline": baseline,
        "selected": selected,
        "captured_positive_advantage_delta_points": checkpoint["metrics"][
            "captured_positive_advantage_delta_points"
        ],
        "changed_target_transformer_tensors": changed_target,
        "changed_frozen_tensors": changed_frozen,
        "optimizer_step_values": optimizer_steps,
        "hyperparameters": {
            "requested_steps": args.steps,
            "learning_rate": args.learning_rate,
            "advantage_epsilon": args.advantage_epsilon,
            "max_grad_norm": args.max_grad_norm,
            "seed": args.seed,
        },
    }
    write_json(args.output_dir / "training_summary.json", summary)
    write_json(args.output_dir / "training_history.json", history)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
