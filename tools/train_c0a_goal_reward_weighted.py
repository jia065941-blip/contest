#!/usr/bin/env python3
"""C0a_goal reward-weighted continuation from a frozen policy snapshot."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unified_mappo.model import (  # noqa: E402
    HybridMAPPOConfig,
    HybridMAPPOModel,
)
from tools.train_start_state_option_curriculum import (  # noqa: E402
    ordered_trajectories,
    run_episode,
    scenario_entity_types,
)


TARGET_TRANSFORMER_PREFIXES = (
    "target_transformer_item_encoder.",
    "target_transformer_query.",
    "target_transformer_encoder.",
    "target_transformer_score.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train only the complete target Transformer from official-score "
            "candidate rankings; no PPO, GAE, critic update, or optimizer restore."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-index", type=int, default=34)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--goal-learning-rate", type=float, default=1e-5)
    parser.add_argument("--optimizer-steps", type=int, default=1)
    parser.add_argument("--target-kl-limit", type=float, default=0.02)
    parser.add_argument("--backtrack-factor", type=float, default=0.5)
    parser.add_argument(
        "--boundary-source",
        choices=("damage_anchors", "all_launches"),
        default="damage_anchors",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--reward-logit-clip", type=float, default=5.0)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--epsilon-return", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--resume-completed", action="store_true")
    return parser.parse_args()


def representative_launch_boundaries(
    rows: list[dict[str, Any]],
    *,
    batch_index: int,
    entity_types: dict[int, int],
) -> tuple[list[dict[str, Any]], int]:
    """Select one stratified single-entity launch boundary per seed.

    Every command-200 boundary in the frozen guidance traces is eligible.  A
    batch remains seed-independent by taking one boundary from each trace;
    ranks are spread over the full launch sequence and rotate across batches.
    """

    selected: list[dict[str, Any]] = []
    catalogue_count = 0
    row_count = len(rows)
    for row_index, source in enumerate(rows):
        trace = json.loads(Path(source["trace"]).read_text(encoding="utf-8"))
        boundaries: list[dict[str, int]] = []
        seen_entities: set[int] = set()
        for step_row in trace["steps"]:
            step = int(step_row["step"])
            for action in step_row.get("actions", ()):
                if int(action.get("commandType_id", -1)) != 200:
                    continue
                entity_id = int(action["executor_id"])
                if int(entity_types.get(entity_id, -1)) not in {21000, 21001}:
                    continue
                if entity_id in seen_entities:
                    continue
                seen_entities.add(entity_id)
                boundaries.append({
                    "timestep": step,
                    "executor_id": entity_id,
                    "command_type": 200,
                })
        if not boundaries:
            raise RuntimeError(f"seed {source['seed']} has no launch boundary")
        catalogue_count += len(boundaries)
        base_rank = min(
            len(boundaries) - 1,
            ((2 * row_index + 1) * len(boundaries)) // (2 * row_count),
        )
        rotation = ((int(batch_index) - 35) * 29) % len(boundaries)
        boundary = boundaries[(base_rank + rotation) % len(boundaries)]
        row = copy.deepcopy(source)
        target_id = int(row["assigned_target_id"])
        step = int(boundary["timestep"])
        entity_id = int(boundary["executor_id"])
        synthetic_anchor = {
            "decision_step": step,
            "snapshot_prefix_step": step - 1,
            "damage_step": step,
            "attacking_entity_id": entity_id,
            "decision_type": 200,
        }
        row["damage_anchors"] = {str(target_id): synthetic_anchor}
        row["damage_snapshot_prefix_step"] = step - 1
        row["b7_boundary"] = {
            **boundary,
            "catalogue_rank": (base_rank + rotation) % len(boundaries),
            "catalogue_size_for_seed": len(boundaries),
            "source": (
                "all_unique_goal_capable_launches_student_target_replacement"
            ),
        }
        selected.append(row)
    return selected, catalogue_count


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_weights_only_model(
    checkpoint_path: Path,
    device: str,
) -> tuple[dict[str, Any], HybridMAPPOConfig, HybridMAPPOModel]:
    payload: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("algorithm") != "target_conditioned_ctde_mappo_v11":
        raise ValueError(
            "C0a_goal requires a v11 Transformer target checkpoint; got "
            f"{payload.get('algorithm')!r}"
        )
    config_values = dict(payload["config"])
    config_values["device"] = device
    config = HybridMAPPOConfig(**config_values)
    if config.target_selector_arch != "transformer":
        raise ValueError("checkpoint target_selector_arch is not transformer")
    model = HybridMAPPOModel(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return payload, config, model


def make_frozen_batch_snapshot(
    source_checkpoint: Path,
    destination: Path,
    *,
    batch_index: int,
) -> dict[str, Any]:
    source: dict[str, Any] = torch.load(
        source_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    snapshot = {
        "algorithm": source["algorithm"],
        "config": dict(source["config"]),
        "model": source["model"],
        "update_count": 0,
        "transition_count": 0,
        "metrics": {
            "frozen_candidate_policy": 1.0,
            "old_optimizer_state_loaded": 0.0,
        },
        "continuation": {
            "stage": "C0a_goal",
            "role": "frozen_candidate_and_kl_anchor",
            "batch_index": int(batch_index),
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_checkpoint_sha256": file_sha256(source_checkpoint),
            "source_model_sha256": model_fingerprint(source["model"]),
            "optimizer_included": False,
            "scheduler_included": False,
            "ppo_state_included": False,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, destination)
    return snapshot["continuation"]


def reward_weighted_target(
    old_candidate_probabilities: torch.Tensor,
    returns: torch.Tensor,
    *,
    temperature: float,
    reward_logit_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    old_probabilities = old_candidate_probabilities / (
        old_candidate_probabilities.sum() + 1e-12
    )
    baseline = (old_probabilities * returns).sum()
    improvement = torch.clamp(
        (returns - baseline) / temperature,
        min=-reward_logit_clip,
        max=reward_logit_clip,
    )
    weights = old_probabilities * torch.exp(improvement)
    target = weights / (weights.sum() + 1e-12)
    return target, baseline


def target_logits(
    model: HybridMAPPOModel,
    observation: torch.Tensor,
    target_features: torch.Tensor,
    target_valid_mask: torch.Tensor,
) -> torch.Tensor:
    latent = model.encoder(observation)
    logits = model.transformer_target_logits(
        latent,
        target_features,
        target_valid_mask,
    )
    return logits.masked_fill(~target_valid_mask, float("-inf"))


def old_probabilities_for_item(
    model: HybridMAPPOModel,
    item: dict[str, Any],
    device: str,
) -> torch.Tensor:
    with torch.no_grad():
        logits = target_logits(
            model,
            item["observation"].to(device),
            item["target_features"].to(device),
            item["target_valid_mask"].to(device),
        )
        return torch.softmax(logits, dim=-1).cpu()


def candidate_item(
    result: dict[str, Any],
    old_model: HybridMAPPOModel,
    *,
    candidate_limit: int,
    device: str,
) -> dict[str, Any]:
    rollout_payload = torch.load(
        result["rollout"],
        map_location="cpu",
        weights_only=False,
    )
    rollout = rollout_payload["rollout"]
    active = rollout.action_mask.target.to(dtype=torch.bool).nonzero(
        as_tuple=False
    ).flatten()
    if int(active.numel()) != 1:
        raise RuntimeError(
            f"C0a_goal expected one target-active row, got {int(active.numel())}"
        )
    row_index = int(active.item())
    sampled_groups = list(result["sampled_target_rows"])
    if len(sampled_groups) != 1:
        raise RuntimeError(
            f"C0a_goal expected one candidate group, got {len(sampled_groups)}"
        )
    sampled = sampled_groups[0]
    factual_index = int(rollout.actions.target_index[row_index].item())
    factual_ids = list(map(int, result["student_target_ids"]))
    if len(factual_ids) != 1:
        raise RuntimeError(f"expected one factual target id, got {factual_ids}")
    candidate_indices = [factual_index]
    candidate_ids = [factual_ids[0]]
    official_returns = [float(result["score"]) / 100.0]
    for reference in sampled["samples"]:
        candidate_indices.append(int(reference["target_index"]))
        candidate_ids.append(int(reference["target_id"]))
        official_returns.append(float(reference["score"]) / 100.0)

    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError(f"duplicate candidate target ids: {candidate_ids}")
    if len(candidate_indices) != len(set(candidate_indices)):
        raise RuntimeError(f"duplicate candidate target slots: {candidate_indices}")
    if not 1 <= len(candidate_ids) <= candidate_limit:
        raise RuntimeError(
            f"candidate count {len(candidate_ids)} outside [1,{candidate_limit}]"
        )

    observation = rollout.observations[row_index : row_index + 1].detach().cpu()
    target_features = rollout.target_features[
        row_index : row_index + 1
    ].detach().cpu()
    valid_mask = rollout.target_valid_mask[
        row_index : row_index + 1
    ].detach().cpu().to(dtype=torch.bool)
    for index in candidate_indices:
        if index < 0 or index >= int(valid_mask.shape[1]):
            raise RuntimeError(f"candidate target slot out of range: {index}")
        if not bool(valid_mask[0, index].item()):
            raise RuntimeError(f"candidate target slot is invalid: {index}")

    item = {
        "seed": int(result["seed"]),
        "start_state_id": str(result["start_state_id"]),
        "observation": observation,
        "target_features": target_features,
        "target_valid_mask": valid_mask,
        "candidate_target_indices": torch.tensor(
            candidate_indices, dtype=torch.long
        ),
        "candidate_target_ids": torch.tensor(candidate_ids, dtype=torch.long),
        "official_joint_returns": torch.tensor(
            official_returns, dtype=torch.float32
        ),
        "return_source": (
            "official E01 total score / 100; the pre-decision prefix is common "
            "to every candidate, so reward-weighted advantages are identical "
            "to formal suffix-return advantages"
        ),
    }
    item["old_target_probabilities"] = old_probabilities_for_item(
        old_model,
        item,
        device,
    )
    return item


def update_target_transformer(
    snapshot: Path,
    batch: dict[str, Any],
    output_checkpoint: Path,
    *,
    learning_rate: float,
    temperature: float,
    reward_logit_clip: float,
    kl_coef: float,
    epsilon_return: float,
    max_grad_norm: float,
    optimizer_steps: int,
    target_kl_limit: float,
    backtrack_factor: float,
    device: str,
    seed: int,
) -> dict[str, Any]:
    if learning_rate <= 0.0:
        raise ValueError("goal learning rate must be positive")
    if optimizer_steps <= 0:
        raise ValueError("optimizer steps must be positive")
    if target_kl_limit <= 0.0:
        raise ValueError("target KL limit must be positive")
    if not 0.0 < backtrack_factor < 1.0:
        raise ValueError("backtrack factor must be in (0, 1)")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    parent, config, model = load_weights_only_model(snapshot, device)
    old_model = HybridMAPPOModel(config).to(device)
    old_model.load_state_dict(parent["model"], strict=True)
    old_model.eval()
    for parameter in old_model.parameters():
        parameter.requires_grad_(False)

    trainable_names: list[str] = []
    trainable_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(TARGET_TRANSFORMER_PREFIXES)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable_names.append(name)
            trainable_parameters.append(parameter)
    if not trainable_parameters:
        raise RuntimeError("no target Transformer parameters selected")
    before = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

    # This is intentionally a brand-new Adam. No prior state_dict is loaded.
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    model.eval()

    def objective() -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, list[float], int, int
    ]:
        ce_terms: list[torch.Tensor] = []
        kl_terms: list[torch.Tensor] = []
        baselines: list[float] = []
        skipped_ties = 0
        skipped_singletons = 0
        for item in batch["items"]:
            candidate_indices = item["candidate_target_indices"].to(device)
            returns = item["official_joint_returns"].to(device)
            if int(candidate_indices.numel()) < 2:
                skipped_singletons += 1
                continue
            if float((returns.max() - returns.min()).item()) <= epsilon_return:
                skipped_ties += 1
                continue
            observation = item["observation"].to(device)
            features = item["target_features"].to(device)
            valid = item["target_valid_mask"].to(device, dtype=torch.bool)
            old_full = item["old_target_probabilities"].to(device)
            current_logits = target_logits(model, observation, features, valid)[0]
            current_candidate_log = torch.log_softmax(
                current_logits[candidate_indices], dim=-1
            )
            old_candidate = old_full[0, candidate_indices]
            target, baseline = reward_weighted_target(
                old_candidate,
                returns,
                temperature=temperature,
                reward_logit_clip=reward_logit_clip,
            )
            ce_terms.append(-(target.detach() * current_candidate_log).sum())
            baselines.append(float(baseline.detach().cpu().item()))
            legal = valid[0]
            old_legal = old_full[0, legal]
            current_legal_log = torch.log_softmax(current_logits[legal], dim=-1)
            kl_terms.append((old_legal * (
                torch.log(old_legal.clamp_min(1e-12)) - current_legal_log
            )).sum())
        if not ce_terms:
            zero = torch.zeros((), device=device)
            return zero, zero, zero, baselines, skipped_ties, skipped_singletons
        ce_loss = torch.stack(ce_terms).mean()
        kl_loss = torch.stack(kl_terms).mean()
        return (
            ce_loss + kl_coef * kl_loss,
            ce_loss,
            kl_loss,
            baselines,
            skipped_ties,
            skipped_singletons,
        )

    initial_loss, initial_ce, initial_kl, baselines, skipped_ties, skipped_singletons = objective()
    informative_count = len(batch["items"]) - skipped_ties - skipped_singletons
    grad_norm = 0.0
    accepted_steps = 0
    rejected_steps = 0
    step_history: list[dict[str, float | int | bool]] = []
    named_parameters = dict(model.named_parameters())
    selected_step = 0
    best_loss_value = float(initial_loss.detach().cpu().item())
    best_parameters = {
        name: named_parameters[name].detach().clone()
        for name in trainable_names
    }
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    while informative_count and accepted_steps < optimizer_steps:
        retry_count = 0
        while True:
            parameter_backup = {
                name: named_parameters[name].detach().clone()
                for name in trainable_names
            }
            optimizer_backup = copy.deepcopy(optimizer.state_dict())
            loss, ce_loss, kl_loss, _, _, _ = objective()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                trainable_parameters, max_grad_norm
            ).item())
            attempted_lr = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            with torch.no_grad():
                post_loss, post_ce, post_kl, _, _, _ = objective()
            post_kl_value = float(post_kl.detach().cpu().item())
            accepted = bool(
                torch.isfinite(post_kl).item()
                and post_kl_value <= target_kl_limit
            )
            step_history.append({
                "attempt": len(step_history) + 1,
                "accepted": accepted,
                "learning_rate": attempted_lr,
                "loss": float(post_loss.detach().cpu().item()),
                "cross_entropy": float(post_ce.detach().cpu().item()),
                "kl_old_to_current": post_kl_value,
                "gradient_norm_before_clip": grad_norm,
            })
            if accepted:
                accepted_steps += 1
                post_loss_value = float(post_loss.detach().cpu().item())
                if post_loss_value < best_loss_value:
                    best_loss_value = post_loss_value
                    selected_step = accepted_steps
                    best_parameters = {
                        name: named_parameters[name].detach().clone()
                        for name in trainable_names
                    }
                    best_optimizer_state = copy.deepcopy(
                        optimizer.state_dict()
                    )
                break
            rejected_steps += 1
            for name, value in parameter_backup.items():
                named_parameters[name].data.copy_(value)
            optimizer.load_state_dict(optimizer_backup)
            for group in optimizer.param_groups:
                group["lr"] *= backtrack_factor
            retry_count += 1
            if retry_count >= 8:
                break
        if retry_count >= 8:
            break

    for name, value in best_parameters.items():
        named_parameters[name].data.copy_(value)
    optimizer.load_state_dict(best_optimizer_state)

    with torch.no_grad():
        loss, ce_loss, kl_loss, _, _, _ = objective()

    after = model.state_dict()
    frozen_changed: list[str] = []
    trainable_changed: list[str] = []
    for name, old_value in before.items():
        changed = not torch.equal(old_value, after[name].detach().cpu())
        if name.startswith(TARGET_TRANSFORMER_PREFIXES):
            if changed:
                trainable_changed.append(name)
        elif changed:
            frozen_changed.append(name)
    if frozen_changed:
        raise RuntimeError(f"frozen model state changed: {frozen_changed}")

    optimizer_step_values = sorted({
        int(state["step"].item())
        for state in optimizer.state.values()
        if "step" in state
    })
    expected_step_values = [selected_step] if selected_step else []
    if informative_count and optimizer_step_values != expected_step_values:
        raise RuntimeError(
            "fresh Adam step contract violated: "
            f"{optimizer_step_values} != {expected_step_values}"
        )

    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "cross_entropy": float(ce_loss.detach().cpu().item()),
        "kl_old_to_current": float(kl_loss.detach().cpu().item()),
        "initial_loss": float(initial_loss.detach().cpu().item()),
        "initial_cross_entropy": float(initial_ce.detach().cpu().item()),
        "initial_kl_old_to_current": float(initial_kl.detach().cpu().item()),
        "kl_coef": float(kl_coef),
        "temperature": float(temperature),
        "reward_logit_clip": float(reward_logit_clip),
        "epsilon_return": float(epsilon_return),
        "informative_state_count": informative_count,
        "skipped_tie_count": skipped_ties,
        "skipped_singleton_count": skipped_singletons,
        "batch_state_count": len(batch["items"]),
        "gradient_norm_before_clip": grad_norm,
        "trainable_parameter_count": sum(p.numel() for p in trainable_parameters),
        "trainable_tensor_count": len(trainable_parameters),
        "changed_trainable_tensors": trainable_changed,
        "frozen_changed_tensors": frozen_changed,
        "old_optimizer_state_loaded": 0,
        "requested_optimizer_steps": optimizer_steps,
        "accepted_optimizer_steps": accepted_steps,
        "selected_optimizer_step": selected_step,
        "rejected_optimizer_attempts": rejected_steps,
        "optimizer_step_values": optimizer_step_values,
        "target_kl_limit": target_kl_limit,
        "backtrack_factor": backtrack_factor,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "optimizer_step_history": step_history,
        "scheduler_used": 0,
        "ppo_used": 0,
        "gae_used": 0,
        "ppo_ratio_clipping_used": 0,
        "critic_updated": 0,
        "actor_trunk_updated": 0,
        "other_action_heads_updated": 0,
        "mean_old_weighted_baseline": (
            sum(baselines) / len(baselines) if baselines else 0.0
        ),
    }
    saved_config = asdict(replace(config, learning_rate=learning_rate))
    saved_config["device"] = device
    checkpoint = {
        "algorithm": parent["algorithm"],
        "config": saved_config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update_count": selected_step,
        "transition_count": len(batch["items"]),
        "metrics": metrics,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "continuation": {
            "stage": "C0a_goal",
            "method": "reward_weighted_candidate_distribution",
            "weights_loaded_strictly": True,
            "parent_model_sha256": model_fingerprint(parent["model"]),
            "trainable_parameter_prefixes": list(TARGET_TRANSFORMER_PREFIXES),
            "optimizer": "fresh Adam",
            "optimizer_state_inherited": False,
            "ppo_state_inherited": False,
            "scheduler_state_inherited": False,
            "clipping_state_inherited": False,
            "batch_snapshot": str(snapshot.resolve()),
        },
    }
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_checkpoint)
    metrics["output_checkpoint_sha256"] = file_sha256(output_checkpoint)
    metrics["output_model_sha256"] = model_fingerprint(checkpoint["model"])
    return metrics


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.candidate_limit < 2:
        raise ValueError("candidate-limit must be at least 2")
    if args.batch_size <= 0 or args.workers <= 0:
        raise ValueError("batch-size and workers must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entity_types = scenario_entity_types(Path(manifest["scenario"]))
    rows = ordered_trajectories(manifest)
    launch_catalogue_count = None
    if args.boundary_source == "all_launches":
        selected, launch_catalogue_count = representative_launch_boundaries(
            rows,
            batch_index=args.batch_index,
            entity_types=entity_types,
        )
        selected = selected[
            args.start_offset : args.start_offset + args.batch_size
        ]
    else:
        selected = rows[args.start_offset : args.start_offset + args.batch_size]
    if len(selected) != args.batch_size:
        raise ValueError(
            f"requested {args.batch_size} states but selected {len(selected)}"
        )
    seeds = [int(row["seed"]) for row in selected]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError(f"batch seeds are not distinct: {seeds}")

    snapshot = (
        args.output_dir
        / "batch_input"
        / f"batch_{args.batch_index:05d}_frozen_goal_policy.pt"
    )
    snapshot_meta = make_frozen_batch_snapshot(
        args.source_checkpoint,
        snapshot,
        batch_index=args.batch_index,
    )
    started = datetime.now(timezone.utc).isoformat()
    progress_path = args.output_dir / "training_progress.json"
    write_json(progress_path, {
        "status": "collecting_candidates",
        "started_at": started,
        "batch_index": args.batch_index,
        "seeds": seeds,
        "boundary_source": args.boundary_source,
        "eligible_launch_boundary_count": launch_catalogue_count,
        "selected_boundaries": [row.get("b7_boundary") for row in selected],
        "snapshot": str(snapshot.resolve()),
        "snapshot_metadata": snapshot_meta,
    })

    os.environ["RED_C0A_GOAL_CANDIDATES_WITHOUT_REPLACEMENT"] = "1"
    os.environ["RED_C0A_GOAL_EXCLUDE_FACTUAL_TARGET"] = "1"
    episode_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_episode,
                batch_index=args.batch_index,
                item_index=item_index,
                stage="C0",
                c0_component="goal",
                controlled_action_components="goal",
                anchor_group_id=(
                    f"c0a-goal-seed-{int(row['seed'])}-"
                    f"step-{int(row['damage_snapshot_prefix_step']) + 1}-"
                    f"entity-{int(row.get('b7_boundary', {}).get('executor_id', -1))}"
                ),
                c4_boundary=None,
                row=row,
                manifest=manifest,
                entity_types=entity_types,
                source_checkpoint=snapshot,
                output_dir=args.output_dir,
                attack_option_min_dwell_steps=60,
                option_control_handoff=False,
                c0_search_option_chaining=False,
                c0_search_option_dwell_steps=120,
                c0_search_option_reopen_distance_km=15.0,
                c0_search_option_emergency_reopen_distance_km=5.0,
                c0_search_reachable_mask=False,
                c0_search_leg_min_distance_km=50.0,
                c0_search_leg_max_distance_km=180.0,
                c0_deterministic_evasion=False,
                deterministic_actor=False,
                target_head_counterfactual=True,
                dense_target_counterfactual=False,
                target_counterfactual_samples=args.candidate_limit - 1,
                resume_completed=args.resume_completed,
            ): item_index
            for item_index, row in enumerate(selected)
        }
        for future in as_completed(futures):
            episode_results.append(future.result())
    episode_results.sort(key=lambda row: int(row["item"]))

    _, _, old_model = load_weights_only_model(snapshot, args.device)
    old_model.eval()
    for parameter in old_model.parameters():
        parameter.requires_grad_(False)
    items = [
        candidate_item(
            result,
            old_model,
            candidate_limit=args.candidate_limit,
            device=args.device,
        )
        for result in episode_results
    ]
    batch_payload = {
        "schema_version": 1,
        "stage": "C0a_goal",
        "batch_index": args.batch_index,
        "snapshot": str(snapshot.resolve()),
        "snapshot_model_sha256": snapshot_meta["source_model_sha256"],
        "candidate_sampling": "frozen_policy_without_replacement",
        "boundary_source": args.boundary_source,
        "eligible_launch_boundary_count": launch_catalogue_count,
        "selected_boundaries": [row.get("b7_boundary") for row in selected],
        "candidate_limit": args.candidate_limit,
        "items": items,
    }
    batch_path = (
        args.output_dir
        / "training_batches"
        / f"batch_{args.batch_index:05d}_goal_candidates.pt"
    )
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batch_payload, batch_path)

    output_checkpoint = args.output_dir / "checkpoints" / "working.pt"
    metrics = update_target_transformer(
        snapshot,
        batch_payload,
        output_checkpoint,
        learning_rate=args.goal_learning_rate,
        temperature=args.temperature,
        reward_logit_clip=args.reward_logit_clip,
        kl_coef=args.kl_coef,
        epsilon_return=args.epsilon_return,
        max_grad_norm=args.max_grad_norm,
        optimizer_steps=args.optimizer_steps,
        target_kl_limit=args.target_kl_limit,
        backtrack_factor=args.backtrack_factor,
        device=args.device,
        seed=args.seed + args.batch_index,
    )
    records = {
        "schema_version": 1,
        "stage": "C0a_goal",
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": file_sha256(args.source_checkpoint),
        "frozen_batch_snapshot": str(snapshot.resolve()),
        "candidate_batch": str(batch_path.resolve()),
        "output_checkpoint": str(output_checkpoint.resolve()),
        "seeds": seeds,
        "boundary_source": args.boundary_source,
        "eligible_launch_boundary_count": launch_catalogue_count,
        "selected_boundaries": [row.get("b7_boundary") for row in selected],
        "hyperparameters": {
            "candidate_limit": args.candidate_limit,
            "goal_learning_rate": args.goal_learning_rate,
            "temperature": args.temperature,
            "reward_logit_clip": args.reward_logit_clip,
            "kl_coef": args.kl_coef,
            "epsilon_return": args.epsilon_return,
            "max_grad_norm": args.max_grad_norm,
            "optimizer_steps": args.optimizer_steps,
            "target_kl_limit": args.target_kl_limit,
            "backtrack_factor": args.backtrack_factor,
        },
        "metrics": metrics,
        "episodes": episode_results,
    }
    write_json(args.output_dir / "training_records.json", records)
    write_json(progress_path, {
        "status": "complete",
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "batch_index": args.batch_index,
        "metrics": metrics,
        "output_checkpoint": str(output_checkpoint.resolve()),
    })
    report = f"""# C0a_goal reward-weighted continuation

- Source checkpoint: `{args.source_checkpoint.resolve()}`
- Source SHA-256: `{records['source_checkpoint_sha256']}`
- Frozen batch policy: `{snapshot.resolve()}`
- Output checkpoint: `{output_checkpoint.resolve()}`
- Distinct training seeds: {len(seeds)}
- Informative states: {metrics['informative_state_count']}
- Loss: {metrics['loss']:.8f}
- KL(old || current): {metrics['kl_old_to_current']:.8f}
- New Adam step values: {metrics['optimizer_step_values']}
- Frozen tensors changed: {metrics['frozen_changed_tensors']}
- PPO / GAE / critic updates: 0 / 0 / 0

Only the item encoder, query, Transformer encoder, and score layer were
trainable. Candidate generation and the KL anchor used the same per-batch
frozen weights-only snapshot. Official total E01 score is used for candidate
ranking; because every fork shares the complete pre-decision prefix, subtracting
the old-policy baseline makes it exactly equivalent to ranking by formal suffix
return.
"""
    (args.output_dir / "C0A_GOAL_CONTINUATION_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "output_checkpoint": str(output_checkpoint.resolve()),
        "metrics": metrics,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
