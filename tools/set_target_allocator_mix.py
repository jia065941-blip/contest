#!/usr/bin/env python3
"""Create an immutable checkpoint with an explicit target-allocator gate."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    parser.add_argument("--mix", type=float, required=True)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.mix <= 1.0:
        raise ValueError("mix 必须位于 [0,1]")

    trainer = HybridMAPPOTrainer.load(args.checkpoint, device="cpu")
    updated_config = replace(trainer.config, target_allocator_mix=args.mix)
    trainer.config = updated_config
    trainer.model.config = updated_config
    trainer.model.set_target_allocator_mix(args.mix)
    trainer.last_metrics["target_allocator_mix"] = float(args.mix)
    trainer.save(args.output)

    result = {
        "schema_version": 1,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "output_checkpoint": str(args.output.resolve()),
        "target_allocator_mix": float(args.mix),
    }
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
