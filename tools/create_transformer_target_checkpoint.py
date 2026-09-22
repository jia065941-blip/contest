#!/usr/bin/env python3
"""Create a fresh strict-CTDE MAPPO checkpoint with a direct target Transformer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unified_mappo.model import HybridMAPPOConfig, HybridMAPPOTrainer
from policies.red.learning.red_policy import (
    GlobalStateEncoder,
    UNIFIED_LOCAL_OBSERVATION_DIM,
    UNIFIED_TARGET_SLOTS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-feature-dim", type=int, default=26)
    parser.add_argument("--target-model-dim", type=int, default=64)
    parser.add_argument("--target-heads", type=int, default=4)
    parser.add_argument("--target-layers", type=int, default=2)
    parser.add_argument("--target-ff-dim", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_feature_dim != 26:
        raise ValueError("当前主 Transformer 实验要求 target-feature-dim=26")
    config = HybridMAPPOConfig(
        observation_dim=UNIFIED_LOCAL_OBSERVATION_DIM,
        critic_state_dim=GlobalStateEncoder(
            max_steps=args.max_steps,
            target_slots=UNIFIED_TARGET_SLOTS,
        ).state_dim,
        critic_focal_observation_dim=UNIFIED_LOCAL_OBSERVATION_DIM,
        target_slots=UNIFIED_TARGET_SLOTS,
        target_feature_dim=args.target_feature_dim,
        target_allocator_hidden_dim=args.target_model_dim,
        target_selector_arch="transformer",
        target_transformer_heads=args.target_heads,
        target_transformer_layers=args.target_layers,
        target_transformer_ff_dim=args.target_ff_dim,
        target_allocator_mix=0.0,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device="cpu",
    )
    trainer = HybridMAPPOTrainer(config)
    if float(trainer.model.target_teacher_policy_scale.item()) != 0.0:
        raise RuntimeError("新模型不得启用教师残差")
    if float(trainer.model.target_cf_policy_scale.item()) != 0.0:
        raise RuntimeError("新模型不得启用反事实残差")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.output)
    restored = HybridMAPPOTrainer.load(args.output, device="cpu")
    if restored.config.target_selector_arch != "transformer":
        raise RuntimeError("checkpoint 回读后目标头不是 transformer")
    payload = torch.load(args.output, map_location="cpu", weights_only=False)
    if payload.get("algorithm") != "target_conditioned_ctde_mappo_v11":
        raise RuntimeError("Transformer checkpoint 算法版本不是 v11")

    transformer_parameters = sum(
        parameter.numel()
        for name, parameter in trainer.model.named_parameters()
        if name.startswith("target_transformer_")
    )
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "TST000_fresh_direct_target_transformer",
        "status": "created",
        "checkpoint": str(args.output.resolve()),
        "algorithm": payload["algorithm"],
        "target_logits_source": "direct_transformer_only",
        "teacher_residual_enabled": False,
        "counterfactual_residual_enabled": False,
        "total_parameter_count": sum(
            parameter.numel() for parameter in trainer.model.parameters()
        ),
        "target_transformer_parameter_count": transformer_parameters,
        "config": asdict(config),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
