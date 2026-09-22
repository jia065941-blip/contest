#!/usr/bin/env python3
"""Set the additive policy gate for the teacher shared target residual."""

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
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    if args.scale < 0.0:
        raise ValueError("scale 必须非负")
    trainer = HybridMAPPOTrainer.load(args.checkpoint, device="cpu")
    trainer.model.set_target_teacher_policy_scale(args.scale)
    trainer.last_metrics["target_teacher_policy_scale"] = float(args.scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.output)
    result = {
        "schema_version": 1,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "output_checkpoint": str(args.output.resolve()),
        "target_teacher_policy_scale": float(args.scale),
    }
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
