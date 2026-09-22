#!/usr/bin/env python3
"""Train the active target selector from exact counterfactual returns.

The rollout reward is the signed E01 score delta obtained by replaying the
same causal boundary with only the target restored to the teacher target.
The pairwise objective compares the sampled student target with its matched
teacher target.  The signed-policy objective instead uses every non-zero
student-target return as contextual-bandit credit.
"""

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
from tools.build_native_guidance_manifest import scenario_targets
from tools.train_start_state_option_curriculum import BASE_TARGET_SLOT_IDS


SHARED_ALLOCATOR_PREFIXES = (
    "target_item_encoder.",
    "target_query.",
    "target_score.",
)
TRANSFORMER_ALLOCATOR_PREFIXES = (
    "target_transformer_item_encoder.",
    "target_transformer_query.",
    "target_transformer_encoder.",
    "target_transformer_score.",
)


def trainable_prefixes(model, scope: str) -> tuple[str, ...]:
    """Return only the parameters used by the checkpoint's target selector."""

    if model.config.target_selector_arch == "transformer":
        return {
            "allocator": TRANSFORMER_ALLOCATOR_PREFIXES,
            "score": ("target_transformer_score.",),
            "score-last": ("target_transformer_score.1.",),
        }[scope]
    return {
        "allocator": SHARED_ALLOCATOR_PREFIXES,
        "score": ("target_score.",),
        "score-last": ("target_score.2.",),
    }[scope]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--kl-coef", type=float, default=0.5)
    parser.add_argument("--kl-limit", type=float, default=0.005)
    parser.add_argument(
        "--objective",
        choices=("pairwise", "signed-policy"),
        default="pairwise",
        help=(
            "Use legal student-vs-teacher ranking pairs, or all non-zero "
            "signed exact returns as a contextual-bandit policy objective."
        ),
    )
    parser.add_argument(
        "--trainable-scope",
        choices=("allocator", "score", "score-last"),
        default="allocator",
        help=(
            "Limit sparse counterfactual fine-tuning to all allocator weights, "
            "the score MLP, or only its final 64-to-1 layer."
        ),
    )
    parser.add_argument("--holdout-stride", type=int, default=5)
    parser.add_argument(
        "--split-mode",
        choices=("seed", "file"),
        default="seed",
        help=(
            "Use seed to keep repeated policy samples from the same scenario "
            "entirely in train or holdout; file preserves the legacy split."
        ),
    )
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--min-abs-delta", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def target_slots(manifest_path: Path) -> tuple[int, ...]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trace = json.loads(
        Path(manifest["trajectories"][0]["trace"]).read_text(encoding="utf-8")
    )
    objective_ids = {
        int(value) for value in trace["summary"]["score"]["objective_weights"]
    }
    targets = scenario_targets(Path(manifest["scenario"]), objective_ids)
    slots = list(BASE_TARGET_SLOT_IDS)
    slots.extend(sorted(int(value) for value in targets if int(value) not in slots))
    return tuple(slots)


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


def load_rows(
    paths: list[Path],
    *,
    slot_ids: tuple[int, ...],
    target_slots_count: int,
    target_feature_dim: int,
    holdout_stride: int,
    split_mode: str,
    min_abs_delta: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, int]]:
    slot_by_id = {target_id: index for index, target_id in enumerate(slot_ids)}
    if len(slot_ids) != target_slots_count:
        raise ValueError(
            f"manifest 目标槽位数 {len(slot_ids)} != 模型槽位数 {target_slots_count}"
        )
    fields = (
        "observations", "features", "valid", "student", "teacher",
        "teacher_legal", "delta",
    )
    partitions: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {field: [] for field in fields} for name in ("train", "holdout")
    }
    counts = {
        "train_files": 0,
        "holdout_files": 0,
        "skipped_same_target_rows": 0,
        "zero_delta_rows": 0,
        "teacher_illegal_rows": 0,
    }
    for file_index, path in enumerate(paths):
        mask_path = path.parent / "policy_control_mask.json"
        if not mask_path.exists():
            raise FileNotFoundError(f"rollout 缺少控制掩码: {mask_path}")
        specification = json.loads(mask_path.read_text(encoding="utf-8"))
        if not bool(specification.get("target_head_counterfactual", False)):
            raise ValueError(f"不是精确目标反事实 rollout: {path}")
        teacher_id = int(specification["target_id"])
        scenario_seed = int(specification["seed"])
        if teacher_id not in slot_by_id:
            raise ValueError(f"教师目标 {teacher_id} 不在固定槽位中: {path}")
        teacher_index = slot_by_id[teacher_id]

        payload = torch.load(path, map_location="cpu", weights_only=False)
        batch = payload["rollout"] if isinstance(payload, dict) else payload
        features = getattr(batch, "target_features", None)
        if features is None:
            raise ValueError(f"rollout 缺少合法 target_features: {path}")
        if features.shape[1:] != (target_slots_count, target_feature_dim):
            raise ValueError(f"target_features 形状不兼容: {path}")
        active = batch.action_mask.target.to(dtype=torch.bool)
        indices = active.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        split_value = scenario_seed if split_mode == "seed" else file_index
        partition = "holdout" if split_value % holdout_stride == 0 else "train"
        counts[f"{partition}_files"] += 1
        for index_tensor in indices:
            index = int(index_tensor.item())
            student_index = int(batch.actions.target_index[index].item())
            valid = batch.target_valid_mask[index].to(dtype=torch.bool)
            teacher_legal = bool(valid[teacher_index])
            if not teacher_legal:
                counts["teacher_illegal_rows"] += 1
            if not bool(valid[student_index]):
                raise ValueError(f"学生目标在因果边界不合法: {path}, row={index}")
            delta = float(batch.rewards[index].item())
            if student_index == teacher_index:
                counts["skipped_same_target_rows"] += 1
            if abs(delta) <= min_abs_delta:
                counts["zero_delta_rows"] += 1
            rows = partitions[partition]
            rows["observations"].append(batch.observations[index:index + 1].float())
            rows["features"].append(features[index:index + 1].float())
            rows["valid"].append(valid.unsqueeze(0))
            rows["student"].append(torch.tensor([student_index], dtype=torch.long))
            rows["teacher"].append(torch.tensor([teacher_index], dtype=torch.long))
            rows["teacher_legal"].append(
                torch.tensor([teacher_legal], dtype=torch.bool)
            )
            rows["delta"].append(torch.tensor([delta], dtype=torch.float32))

    packed: dict[str, dict[str, torch.Tensor]] = {}
    for name, values in partitions.items():
        if not values["observations"]:
            raise ValueError(f"{name} 分区没有目标边界样本")
        packed[name] = {
            field: torch.cat(tensors, dim=0) for field, tensors in values.items()
        }
        pair_mask = (
            packed[name]["delta"].abs().gt(min_abs_delta)
            & packed[name]["student"].ne(packed[name]["teacher"])
            & packed[name]["teacher_legal"]
        )
        pair_count = int(pair_mask.sum().item())
        if pair_count == 0:
            raise ValueError(f"{name} 分区没有非零异目标反事实配对")
        counts[f"{name}_rows"] = int(pair_mask.shape[0])
        counts[f"{name}_pair_rows"] = pair_count
        counts[f"{name}_credit_rows"] = int(
            packed[name]["delta"].abs().gt(min_abs_delta).sum().item()
        )
        counts[f"{name}_positive_pairs"] = int(
            (pair_mask & packed[name]["delta"].gt(0.0)).sum().item()
        )
        counts[f"{name}_negative_pairs"] = int(
            (pair_mask & packed[name]["delta"].lt(0.0)).sum().item()
        )
    return packed["train"], packed["holdout"], counts


def prepare(
    trainer: HybridMAPPOTrainer,
    rows: dict[str, torch.Tensor],
    *,
    device: torch.device,
    min_abs_delta: float,
) -> dict[str, torch.Tensor]:
    observations = rows["observations"].to(device)
    features = rows["features"].to(device)
    valid = rows["valid"].to(device)
    with torch.no_grad():
        latent = trainer.model.encoder(observations)
        parameters = trainer.model.distribution_parameters(
            observations, features, valid
        )
        old_logits = parameters["target_logits"].masked_fill(~valid, -1e9)
        old_prob = F.softmax(old_logits, dim=-1)
        old_log_prob = F.log_softmax(old_logits, dim=-1)
    delta = rows["delta"].to(device)
    student = rows["student"].to(device)
    teacher = rows["teacher"].to(device)
    teacher_legal = rows["teacher_legal"].to(device)
    return {
        "observations": observations,
        "latent": latent.detach(),
        "features": features,
        "valid": valid,
        "student": student,
        "teacher": teacher,
        "delta": delta,
        "pair_mask": (
            delta.abs().gt(min_abs_delta)
            & student.ne(teacher)
            & teacher_legal
        ),
        "credit_mask": delta.abs().gt(min_abs_delta),
        "old_prob": old_prob,
        "old_log_prob": old_log_prob,
    }


def current_logits(
    trainer: HybridMAPPOTrainer,
    rows: dict[str, torch.Tensor],
) -> torch.Tensor:
    model = trainer.model
    logits = model.distribution_parameters(
        rows["observations"], rows["features"], rows["valid"]
    )["target_logits"]
    return logits.masked_fill(~rows["valid"], -1e9)


def objective(
    trainer: HybridMAPPOTrainer,
    rows: dict[str, torch.Tensor],
    *,
    kl_coef: float,
    objective_kind: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = current_logits(trainer, rows)
    log_prob = F.log_softmax(logits, dim=-1)
    weights = rows["delta"].abs()
    if objective_kind == "pairwise":
        selected = rows["pair_mask"]
        student_logits = logits.gather(
            1, rows["student"].unsqueeze(1)
        ).squeeze(1)
        teacher_logits = logits.gather(
            1, rows["teacher"].unsqueeze(1)
        ).squeeze(1)
        direction = rows["delta"].sign()
        policy_loss = (
            F.softplus(
                -direction[selected]
                * (student_logits[selected] - teacher_logits[selected])
            ) * weights[selected]
        ).sum() / weights[selected].sum().clamp_min(1e-8)
    else:
        selected = rows["credit_mask"]
        selected_log_prob = log_prob.gather(
            1, rows["student"].unsqueeze(1)
        ).squeeze(1)
        policy_loss = -(
            rows["delta"][selected] * selected_log_prob[selected]
        ).sum() / weights[selected].sum().clamp_min(1e-8)
    kl = (rows["old_prob"] * (rows["old_log_prob"] - log_prob)).sum(-1).mean()
    return policy_loss + kl_coef * kl, policy_loss, kl


@torch.no_grad()
def evaluate(
    trainer: HybridMAPPOTrainer,
    rows: dict[str, torch.Tensor],
) -> dict[str, float]:
    logits = current_logits(trainer, rows)
    log_prob = F.log_softmax(logits, dim=-1)
    per_row_kl = (
        rows["old_prob"] * (rows["old_log_prob"] - log_prob)
    ).sum(-1)
    student_logits = logits.gather(1, rows["student"].unsqueeze(1)).squeeze(1)
    teacher_logits = logits.gather(1, rows["teacher"].unsqueeze(1)).squeeze(1)
    pair = rows["pair_mask"]
    direction = rows["delta"].sign()[pair]
    weights = rows["delta"].abs()[pair]
    signed_margin = direction * (student_logits[pair] - teacher_logits[pair])
    losses = F.softplus(-signed_margin)
    correct = signed_margin.gt(0.0).float()
    credit = rows["credit_mask"]
    selected_log_prob = log_prob.gather(
        1, rows["student"].unsqueeze(1)
    ).squeeze(1)
    old_selected_log_prob = rows["old_log_prob"].gather(
        1, rows["student"].unsqueeze(1)
    ).squeeze(1)
    credit_weights = rows["delta"].abs()[credit]
    credit_loss = -(
        rows["delta"][credit] * selected_log_prob[credit]
    ).sum() / credit_weights.sum().clamp_min(1e-8)
    probability_ratio = (
        selected_log_prob[credit] - old_selected_log_prob[credit]
    ).exp()
    credit_surrogate_gain = (
        rows["delta"][credit] * (probability_ratio - 1.0)
    ).sum() / credit_weights.sum().clamp_min(1e-8)
    signed_log_prob_shift = rows["delta"][credit].sign() * (
        selected_log_prob[credit] - old_selected_log_prob[credit]
    )
    return {
        "pair_loss": float((losses * weights).sum().div(weights.sum()).cpu()),
        "pair_accuracy": float(correct.mean().cpu()),
        "weighted_pair_accuracy": float(
            (correct * weights).sum().div(weights.sum()).cpu()
        ),
        "mean_signed_margin": float(signed_margin.mean().cpu()),
        "credit_loss": float(credit_loss.cpu()),
        "credit_surrogate_gain": float(credit_surrogate_gain.cpu()),
        "credit_direction_accuracy": float(
            signed_log_prob_shift.gt(0.0).float().mean().cpu()
        ),
        "mean_kl": float(per_row_kl.mean().cpu()),
        "max_kl": float(per_row_kl.max().cpu()),
    }


def allocator_state(
    trainer: HybridMAPPOTrainer,
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in trainer.model.state_dict().items()
        if name.startswith(prefixes)
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.learning_rate <= 0.0:
        raise ValueError("epochs 和 learning-rate 必须为正")
    if args.kl_coef < 0.0 or args.kl_limit <= 0.0:
        raise ValueError("KL 系数必须非负且 KL 上限必须为正")
    if args.holdout_stride < 2 or args.validation_interval <= 0:
        raise ValueError("holdout-stride 至少为 2，validation-interval 必须为正")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 CUDA 不可用")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = HybridMAPPOTrainer.load(args.checkpoint, device=args.device)
    model = trainer.model
    mix = float(model.target_allocator_mix.item())
    selector_arch = str(model.config.target_selector_arch)
    if selector_arch != "transformer" and mix <= 0.0:
        raise ValueError("共享评分器配对训练要求 target_allocator_mix > 0")
    paths = rollout_paths(args.rollout_root)
    train_raw, holdout_raw, counts = load_rows(
        paths,
        slot_ids=target_slots(args.manifest),
        target_slots_count=model.config.target_slots,
        target_feature_dim=model.config.target_feature_dim,
        holdout_stride=args.holdout_stride,
        split_mode=args.split_mode,
        min_abs_delta=args.min_abs_delta,
    )
    device = next(model.parameters()).device
    train = prepare(
        trainer, train_raw, device=device, min_abs_delta=args.min_abs_delta
    )
    holdout = prepare(
        trainer, holdout_raw, device=device, min_abs_delta=args.min_abs_delta
    )
    initial = {"train": evaluate(trainer, train), "holdout": evaluate(trainer, holdout)}

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    allocator_parameters = []
    selected_prefixes = trainable_prefixes(model, args.trainable_scope)
    for name, parameter in model.named_parameters():
        if name.startswith(selected_prefixes):
            parameter.requires_grad_(True)
            allocator_parameters.append(parameter)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in allocator_parameters
    )
    optimizer = torch.optim.Adam(allocator_parameters, lr=args.learning_rate)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    selection_metric = (
        "pair_loss" if args.objective == "pairwise" else "credit_loss"
    )
    best_holdout_loss = initial["holdout"][selection_metric]
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective(
            trainer,
            train,
            kl_coef=args.kl_coef,
            objective_kind=args.objective,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(allocator_parameters, 1.0)
        optimizer.step()
        if epoch % args.validation_interval != 0 and epoch != args.epochs:
            continue
        train_metrics = evaluate(trainer, train)
        holdout_metrics = evaluate(trainer, holdout)
        max_mean_kl = max(train_metrics["mean_kl"], holdout_metrics["mean_kl"])
        eligible = (
            train_metrics[selection_metric] < initial["train"][selection_metric]
            and holdout_metrics[selection_metric] < best_holdout_loss
            and max_mean_kl <= args.kl_limit
        )
        history.append({
            "epoch": epoch,
            "train_objective": train_metrics[selection_metric],
            "holdout_objective": holdout_metrics[selection_metric],
            "train_mean_kl": train_metrics["mean_kl"],
            "holdout_mean_kl": holdout_metrics["mean_kl"],
            "eligible": eligible,
        })
        if eligible:
            best_holdout_loss = holdout_metrics[selection_metric]
            best_epoch = epoch
            best_state = allocator_state(trainer, selected_prefixes)

    if best_state is None:
        terminal = {
            "train": evaluate(trainer, train),
            "holdout": evaluate(trainer, holdout),
        }
        result = {
            "schema_version": 1,
            "status": "rejected",
            "reason": "no_joint_train_holdout_improvement_within_kl_limit",
            "experiment": "TSA003_paired_target_counterfactual_ranking",
            "source_checkpoint": str(args.checkpoint.resolve()),
            "output_checkpoint": None,
            "rollout_roots": [
                str(path.resolve()) for path in args.rollout_root
            ],
            "target_allocator_mix": mix,
            "target_selector_arch": selector_arch,
            "trainable_prefixes": list(selected_prefixes),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "kl_coef": args.kl_coef,
            "kl_limit": args.kl_limit,
            "split_mode": args.split_mode,
            "objective": args.objective,
            "trainable_scope": args.trainable_scope,
            "trainable_parameter_count": trainable_parameter_count,
            "best_epoch": 0,
            "counts": counts,
            "initial": initial,
            "terminal": terminal,
            "history": history,
        }
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False))
        return
    model.load_state_dict(best_state, strict=False)
    final = {"train": evaluate(trainer, train), "holdout": evaluate(trainer, holdout)}
    trainer.last_metrics.update({
        "counterfactual_pair_best_epoch": float(best_epoch),
        "counterfactual_pair_train_loss": final["train"]["pair_loss"],
        "counterfactual_pair_holdout_loss": final["holdout"]["pair_loss"],
        "counterfactual_pair_holdout_accuracy": final["holdout"]["pair_accuracy"],
        "counterfactual_pair_holdout_kl": final["holdout"]["mean_kl"],
    })
    trainer.optimizer = torch.optim.Adam(
        trainer.model.parameters(), lr=trainer.config.learning_rate
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.output)
    result = {
        "schema_version": 1,
        "status": "accepted",
        "experiment": "TSA003_paired_target_counterfactual_ranking",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "output_checkpoint": str(args.output.resolve()),
        "rollout_roots": [str(path.resolve()) for path in args.rollout_root],
        "target_allocator_mix": mix,
        "target_selector_arch": selector_arch,
        "trainable_prefixes": list(selected_prefixes),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "kl_coef": args.kl_coef,
        "kl_limit": args.kl_limit,
        "split_mode": args.split_mode,
        "objective": args.objective,
        "trainable_scope": args.trainable_scope,
        "trainable_parameter_count": trainable_parameter_count,
        "best_epoch": best_epoch,
        "counts": counts,
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
