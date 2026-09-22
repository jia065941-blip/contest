#!/usr/bin/env python3
"""Expand a target-feature checkpoint with function-preserving zero weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unified_mappo.model import HybridMAPPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-feature-dim", type=int, default=26)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    if args.target_feature_dim < 20:
        raise ValueError("target-feature-dim 不得小于历史基线的 20 维")
    trainer = HybridMAPPOTrainer.load(
        args.checkpoint,
        device="cpu",
        target_feature_dim=args.target_feature_dim,
    )
    trainer.last_metrics["target_feature_dim"] = float(args.target_feature_dim)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.output)
    result = {
        "schema_version": 1,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "output_checkpoint": str(args.output.resolve()),
        "target_feature_dim": int(trainer.config.target_feature_dim),
        "new_input_weight_abs_max": float(
            trainer.model.target_item_encoder[0].weight[
                :, 20:args.target_feature_dim
            ].abs().max().item()
        ),
    }
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
