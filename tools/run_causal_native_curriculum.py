"""Analyze native teacher causal boundaries, then start MAPPO from scratch."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/opt/conda/envs/competition/bin/python")
DEFAULT_MANIFEST = ROOT / "refine-logs" / "teacher_native_guidance_20260910" / "guidance_manifest.json"
DEFAULT_CAUSAL_DIR = ROOT / "refine-logs" / "teacher_native_guidance_20260910" / "causal_boundaries_joint_v1"
DEFAULT_OUTPUT = ROOT / "refine-logs" / "native_snapshot_mappo_20260910"
INITIAL_CHECKPOINT = ROOT / "refine-logs" / "trajectory_cf_mappo_20260909" / "runs" / "v5_detected_interceptor_satellite_20260910" / "checkpoints" / "initial.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--causal-dir", type=Path, default=DEFAULT_CAUSAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    causal_manifest = args.causal_dir / "guidance_manifest_causal.json"
    subprocess.run([
        str(PYTHON),
        str(ROOT / "tools" / "analyze_native_teacher_causal_boundaries.py"),
        "--manifest", str(args.manifest),
        "--output-dir", str(args.causal_dir),
        "--output-manifest", str(causal_manifest),
        "--workers", str(args.workers),
    ], cwd=ROOT, check=True)

    checkpoint = args.output_dir / "checkpoints" / "working.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INITIAL_CHECKPOINT, checkpoint)
    configuration = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "causal_manifest": str(causal_manifest.resolve()),
        "initial_checkpoint": str(INITIAL_CHECKPOINT.resolve()),
        "working_checkpoint": str(checkpoint.resolve()),
        "initial_update_count": 0,
        "initial_transition_count": 0,
        "stages": ["damage", "detection", "deployment", "t0"],
        "passes_per_stage": 1,
        "behavior_cloning": False,
        "teacher_shaping_reward": False,
        "training_reward": "weighted_damage_individual",
    }
    (args.output_dir / "configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.execv(str(PYTHON), [
        str(PYTHON),
        str(ROOT / "tools" / "train_native_snapshot_curriculum.py"),
        "--manifest", str(causal_manifest),
        "--checkpoint", str(checkpoint),
        "--output-dir", str(args.output_dir),
        "--passes-per-stage", "1",
    ])


if __name__ == "__main__":
    main()
