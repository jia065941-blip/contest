"""Start the native causal-handoff MAPPO curriculum from an initial model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/opt/conda/envs/competition/bin/python")
DEFAULT_MANIFEST = ROOT / "refine-logs" / "teacher_native_guidance_20260910" / "causal_boundaries_joint_v1" / "guidance_manifest_causal.json"
DEFAULT_OUTPUT = ROOT / "refine-logs" / "native_causal_handoff_mappo_20260910"
INITIAL_CHECKPOINT = ROOT / "refine-logs" / "trajectory_cf_mappo_20260909" / "runs" / "v5_detected_interceptor_satellite_20260910" / "checkpoints" / "initial.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.output_dir / "checkpoints" / "working.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INITIAL_CHECKPOINT, checkpoint)
    configuration = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "native_causal_handoff_mappo",
        "manifest": str(args.manifest.resolve()),
        "initial_checkpoint": str(INITIAL_CHECKPOINT.resolve()),
        "working_checkpoint": str(checkpoint.resolve()),
        "initial_update_count": 0,
        "initial_transition_count": 0,
        "curriculum_stages": [
            "single_unit",
            "same_step_group",
            "two_groups",
            "four_groups",
            "full_chain",
        ],
        "stage_advancement": "positive_COW_marginal_and_positive_formal_credit",
        "behavior_cloning": False,
        "teacher_shaping_reward": False,
        "training_reward": "weighted_damage_guided_cow",
        "teacher_steps_in_ppo_rollout": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.execv(str(PYTHON), [
        str(PYTHON),
        str(ROOT / "tools" / "train_native_causal_handoff_curriculum.py"),
        "--manifest", str(args.manifest),
        "--checkpoint", str(checkpoint),
        "--output-dir", str(args.output_dir),
    ])


if __name__ == "__main__":
    main()
