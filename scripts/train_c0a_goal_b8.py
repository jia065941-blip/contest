#!/usr/bin/env python3
"""Train a stronger C0a_goal target Transformer from formal-return candidates.

The optimizer never restores historical state.  It updates only the complete
target Transformer and selects the checkpoint by held-out candidate expected
official E01 return, not by teacher imitation or training loss.
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
    parser.add_argument(
        "--candidate-batch", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--holdout-stride", type=int, default=4)
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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_checkpoint(
    path: Path, device: str
) -> tuple[dict[str, Any], HybridMAPPOConfig, HybridMAPPOModel]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
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


def stack_items(
    paths: list[Path], device: str
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("stage") != "C0a_goal":
            raise ValueError(f"not a C0a_goal batch: {path}")
        for item in payload["items"]:
            if int(item["candidate_target_indices"].numel()) != 10:
                raise ValueError(f"b8 requires K=10 candidates: {path}")
            items.append(item)
            provenance.append({
                "batch": int(payload["batch_index"]),
                "seed": int(item["seed"]),
                "start_state_id": str(item["start_state_id"]),
            })
    data = {
        "observation": torch.cat(
            [item["observation"] for item in items], dim=0
        ).to(device),
        "features": torch.cat(
            [item["target_features"] for item in items], dim=0
        ).to(device),
        "valid": torch.cat(
            [item["target_valid_mask"] for item in items], dim=0
        ).to(device=device, dtype=torch.bool),
        "indices": torch.stack(
            [item["candidate_target_indices"] for item in items]
        ).to(device=device, dtype=torch.long),
        "returns": torch.stack(
            [item["official_joint_returns"] for item in items]
        ).to(device=device, dtype=torch.float32),
        "seeds": torch.tensor(
            [int(item["seed"]) for item in items],
            dtype=torch.long,
            device=device,
        ),
    }
    return data, provenance


def logits(
    model: HybridMAPPOModel,
    observation: torch.Tensor,
    features: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    latent = model.encoder(observation)
    output = model.transformer_target_logits(latent, features, valid)
    return output.masked_fill(~valid, float("-inf"))


def subset(data: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: value[mask] for name, value in data.items()}


def policy_metrics(
    model: HybridMAPPOModel,
    data: dict[str, torch.Tensor],
    anchor_probabilities: torch.Tensor,
) -> dict[str, float | int]:
    with torch.no_grad():
        current_logits = logits(
            model, data["observation"], data["features"], data["valid"]
        )
        probabilities = torch.softmax(current_logits, dim=-1)
        candidate_probabilities = probabilities.gather(1, data["indices"])
        candidate_probabilities = candidate_probabilities / candidate_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        expected_returns = (
            candidate_probabilities * data["returns"]
        ).sum(dim=-1)
        best = data["returns"] >= (
            data["returns"].max(dim=-1, keepdim=True).values - 1e-8
        )
        candidate_top = candidate_probabilities.argmax(dim=-1)
        candidate_top_is_best = best.gather(
            1, candidate_top.unsqueeze(-1)
        ).squeeze(-1)
        global_top = probabilities.argmax(dim=-1)
        global_in_candidates = (
            data["indices"] == global_top.unsqueeze(-1)
        )
        global_best = (global_in_candidates & best).any(dim=-1)
        tv = 0.5 * (probabilities - anchor_probabilities).abs().sum(dim=-1)
        legal_current = probabilities.clamp_min(1e-12)
        legal_anchor = anchor_probabilities.clamp_min(1e-12)
        kl = (
            anchor_probabilities
            * (legal_anchor.log() - legal_current.log())
        ).sum(dim=-1)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)
    return {
        "state_count": int(data["returns"].shape[0]),
        "candidate_ev": float(expected_returns.mean().item()),
        "candidate_top1_best": int(candidate_top_is_best.sum().item()),
        "global_top1_best": int(global_best.sum().item()),
        "mean_tv_from_source": float(tv.mean().item()),
        "max_tv_from_source": float(tv.max().item()),
        "mean_kl_from_source": float(kl.mean().item()),
        "max_kl_from_source": float(kl.max().item()),
        "mean_entropy": float(entropy.mean().item()),
    }


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.learning_rate <= 0.0:
        raise ValueError("steps and learning rate must be positive")
    if args.temperature <= 0.0 or args.holdout_stride < 2:
        raise ValueError("temperature must be positive and holdout stride >= 2")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source, config, model = load_checkpoint(args.source_checkpoint, args.device)
    _, _, anchor = load_checkpoint(args.source_checkpoint, args.device)
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)

    before = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    trainable: list[torch.nn.Parameter] = []
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(TARGET_PREFIXES)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
            trainable_names.append(name)
    if not trainable:
        raise RuntimeError("no target Transformer parameters selected")
    model.eval()

    all_data, provenance = stack_items(args.candidate_batch, args.device)
    unique_seeds = sorted({int(row["seed"]) for row in provenance})
    holdout_seeds = set(unique_seeds[:: args.holdout_stride])
    holdout_mask = torch.tensor(
        [int(row["seed"]) in holdout_seeds for row in provenance],
        dtype=torch.bool,
        device=args.device,
    )
    train_mask = ~holdout_mask
    if not bool(train_mask.any()) or not bool(holdout_mask.any()):
        raise RuntimeError("empty train or holdout split")
    train_data = subset(all_data, train_mask)
    holdout_data = subset(all_data, holdout_mask)

    with torch.no_grad():
        anchor_all = torch.softmax(logits(
            anchor,
            all_data["observation"],
            all_data["features"],
            all_data["valid"],
        ), dim=-1)
    anchor_train = anchor_all[train_mask]
    anchor_holdout = anchor_all[holdout_mask]

    baseline_train = policy_metrics(model, train_data, anchor_train)
    baseline_holdout = policy_metrics(model, holdout_data, anchor_holdout)
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    named_parameters = dict(model.named_parameters())
    best_parameters = {
        name: named_parameters[name].detach().clone()
        for name in trainable_names
    }
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    best_step = 0
    best_holdout_ev = float(baseline_holdout["candidate_ev"])
    best_holdout_top1 = int(baseline_holdout["global_top1_best"])
    history: list[dict[str, Any]] = []

    train_target = torch.softmax(
        (
            train_data["returns"]
            - train_data["returns"].max(dim=-1, keepdim=True).values
        ) / args.temperature,
        dim=-1,
    )

    for step in range(1, args.steps + 1):
        current_logits = logits(
            model,
            train_data["observation"],
            train_data["features"],
            train_data["valid"],
        )
        candidate_logits = current_logits.gather(1, train_data["indices"])
        candidate_log = torch.log_softmax(candidate_logits, dim=-1)
        cross_entropy = -(train_target * candidate_log).sum(dim=-1).mean()
        current_probabilities = torch.softmax(current_logits, dim=-1)
        kl = (
            anchor_train
            * (
                anchor_train.clamp_min(1e-12).log()
                - current_probabilities.clamp_min(1e-12).log()
            )
        ).sum(dim=-1).mean()
        loss = cross_entropy + args.kl_coef * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            trainable, args.max_grad_norm
        ).item())
        optimizer.step()

        train_metrics = policy_metrics(model, train_data, anchor_train)
        holdout_metrics = policy_metrics(model, holdout_data, anchor_holdout)
        holdout_ev = float(holdout_metrics["candidate_ev"])
        holdout_top1 = int(holdout_metrics["global_top1_best"])
        improved = (
            holdout_ev > best_holdout_ev + 1e-10
            or (
                abs(holdout_ev - best_holdout_ev) <= 1e-10
                and holdout_top1 > best_holdout_top1
            )
        )
        if improved:
            best_holdout_ev = holdout_ev
            best_holdout_top1 = holdout_top1
            best_step = step
            best_parameters = {
                name: named_parameters[name].detach().clone()
                for name in trainable_names
            }
            best_optimizer = copy.deepcopy(optimizer.state_dict())
        history.append({
            "step": step,
            "loss": float(loss.detach().item()),
            "cross_entropy": float(cross_entropy.detach().item()),
            "kl_penalty": float(kl.detach().item()),
            "gradient_norm_before_clip": gradient_norm,
            "train": train_metrics,
            "holdout": holdout_metrics,
            "selected_so_far": best_step,
        })
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(json.dumps(history[-1], ensure_ascii=False), flush=True)

    for name, value in best_parameters.items():
        named_parameters[name].data.copy_(value)
    optimizer.load_state_dict(best_optimizer)
    selected_train = policy_metrics(model, train_data, anchor_train)
    selected_holdout = policy_metrics(model, holdout_data, anchor_holdout)
    selected_all = policy_metrics(model, all_data, anchor_all)

    after = model.state_dict()
    changed_target: list[str] = []
    changed_frozen: list[str] = []
    for name, value in before.items():
        changed = not torch.equal(value, after[name].detach().cpu())
        if not changed:
            continue
        if name.startswith(TARGET_PREFIXES):
            changed_target.append(name)
        else:
            changed_frozen.append(name)
    if changed_frozen:
        raise RuntimeError(f"frozen tensors changed: {changed_frozen}")
    if best_step <= 0:
        raise RuntimeError("no held-out improvement; refusing to create b8")

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
        "update_count": best_step,
        "transition_count": int(train_data["returns"].shape[0]),
        "metrics": {
            "method": "return_only_listwise_holdout_selection",
            "selected_step": best_step,
            "baseline_train": baseline_train,
            "baseline_holdout": baseline_holdout,
            "selected_train": selected_train,
            "selected_holdout": selected_holdout,
            "selected_all": selected_all,
            "train_candidate_ev_delta_points": 100.0 * (
                float(selected_train["candidate_ev"])
                - float(baseline_train["candidate_ev"])
            ),
            "holdout_candidate_ev_delta_points": 100.0 * (
                float(selected_holdout["candidate_ev"])
                - float(baseline_holdout["candidate_ev"])
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
            "stage": "C0a_goal_b8",
            "source_checkpoint": str(args.source_checkpoint.resolve()),
            "source_checkpoint_sha256": file_sha256(args.source_checkpoint),
            "candidate_batches": [
                str(path.resolve()) for path in args.candidate_batch
            ],
            "train_seeds": sorted(set(unique_seeds) - holdout_seeds),
            "holdout_seeds": sorted(holdout_seeds),
            "target_temperature": args.temperature,
            "kl_regularizer_coef": args.kl_coef,
            "selection_metric": "heldout_candidate_expected_official_E01_return",
            "optimizer": "fresh Adam",
            "optimizer_state_inherited": False,
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
        "output_checkpoint": str(output_checkpoint.resolve()),
        "output_checkpoint_sha256": file_sha256(output_checkpoint),
        "output_model_sha256": model_sha256(checkpoint["model"]),
        "candidate_batches": [str(path.resolve()) for path in args.candidate_batch],
        "state_count": len(provenance),
        "train_state_count": int(train_mask.sum().item()),
        "holdout_state_count": int(holdout_mask.sum().item()),
        "selected_step": best_step,
        "baseline_train": baseline_train,
        "baseline_holdout": baseline_holdout,
        "selected_train": selected_train,
        "selected_holdout": selected_holdout,
        "selected_all": selected_all,
        "train_candidate_ev_delta_points": checkpoint["metrics"][
            "train_candidate_ev_delta_points"
        ],
        "holdout_candidate_ev_delta_points": checkpoint["metrics"][
            "holdout_candidate_ev_delta_points"
        ],
        "changed_target_transformer_tensors": changed_target,
        "changed_frozen_tensors": changed_frozen,
        "optimizer_step_values": optimizer_steps,
        "hyperparameters": {
            "requested_steps": args.steps,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "kl_coef": args.kl_coef,
            "max_grad_norm": args.max_grad_norm,
            "holdout_stride": args.holdout_stride,
            "seed": args.seed,
        },
    }
    write_json(args.output_dir / "training_summary.json", summary)
    write_json(args.output_dir / "training_history.json", history)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
