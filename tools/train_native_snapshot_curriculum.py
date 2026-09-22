"""Run the native teacher-snapshot MAPPO curriculum in strict stage order."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TARGETS = (2551, 2552, 169)
STAGES = ("damage", "detection", "deployment", "t0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--passes-per-stage", type=int, default=1)
    return parser.parse_args()


def ordered_trajectories(manifest: dict) -> list[dict]:
    required_rank = {target_id: rank for rank, target_id in enumerate(REQUIRED_TARGETS)}
    return sorted(
        manifest["trajectories"],
        key=lambda row: (
            required_rank.get(int(row["assigned_target_id"]), len(required_rank)),
            -float(row["teacher_score"]),
            int(row["seed"]),
        ),
    )


def run_episode(
    *,
    episode: int,
    stage: str,
    row: dict,
    manifest: dict,
    checkpoint: Path,
    output_dir: Path,
) -> dict:
    seed = int(row["seed"])
    prefix_step = {
        "damage": int(row["damage_snapshot_prefix_step"]),
        "detection": -1,
        "deployment": 0,
        "t0": 0,
    }[stage]
    episode_dir = output_dir / "episodes" / f"{episode:04d}_{stage}_s{seed:04d}"
    log_path = episode_dir / "run.log"
    episode_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "BLUE_POLICY": "b0_fixed_ratio_random",
        "BLUE_POLICY_SEED": str(seed),
        "SIMULATION_SEED": str(seed),
        "RED_POLICY": "r12_unified_mappo",
        "RED_POLICY_SEED": str(seed),
        "RED_MOTION_POLICY": "unified_mappo",
        "RED_LEARNING_TRAIN": "1",
        "RED_LEARNING_MODEL": str(checkpoint.resolve()),
        "RED_REWARD_MODE": "weighted_damage_individual",
        "RED_UNIFIED_DYNAMIC_LIFECYCLE": "1",
        "RED_NATIVE_SNAPSHOT_GUIDANCE": "1",
        "RED_NATIVE_GUIDANCE_TRACE": str(Path(row["trace"]).resolve()),
        "RED_NATIVE_GUIDANCE_STAGE": stage,
        "RED_NATIVE_GUIDANCE_PREFIX_STEP": str(prefix_step),
        "RED_NATIVE_ROLLOUT_DEVICE": "cuda",
        "RED_UNIFIED_UPDATE_DEVICE": "cuda",
    })
    libraries = [
        str(ROOT / "core" / "envengine" / "simulator" / "models" / "HXDMissileModel"),
        str(Path(sys.prefix) / "lib"),
    ]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    command = [
        sys.executable,
        "main.py",
        "--scenario", str(Path(manifest["scenario"]).resolve()),
        "--output-dir", str((episode_dir / "sim_results").resolve()),
        "--max-steps", "1200",
        "--total-rounds", "1",
        "--render-mode", "none",
        "--disable-log-color",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT / "core",
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"episode {episode} exited {completed.returncode}; log={log_path}")
    summaries = [
        json.loads(line.removeprefix("FINAL_SUMMARY "))
        for line in completed.stdout.splitlines()
        if line.startswith("FINAL_SUMMARY ")
    ]
    if len(summaries) != 1:
        raise RuntimeError(f"episode {episode} emitted {len(summaries)} summaries")
    summary = summaries[0]
    return {
        "episode": episode,
        "stage": stage,
        "seed": seed,
        "assigned_target_id": int(row["assigned_target_id"]),
        "snapshot_prefix_step": prefix_step,
        "score": float(summary["score"]["score"]),
        "destroyed_ids": list(map(int, summary["score"]["destroyed_ids"])),
        "official_reward_return": float(summary["red"]["official_reward_return"]),
        "ppo": dict(summary["red"]["dynamic_catalogue"]["unified_mappo"]["last_metrics"]),
        "log": str(log_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = ordered_trajectories(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "training_progress.json"
    progress: list[dict] = []
    episode = 0
    for stage in STAGES:
        for _ in range(args.passes_per_stage):
            for row in rows:
                episode += 1
                result = run_episode(
                    episode=episode,
                    stage=stage,
                    row=row,
                    manifest=manifest,
                    checkpoint=args.checkpoint,
                    output_dir=args.output_dir,
                )
                progress.append(result)
                progress_path.write_text(
                    json.dumps(progress, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
