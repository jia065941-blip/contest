"""Recompute a C0 search-head update with per-route-leg credit."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from tools.train_start_state_option_curriculum import (
    C0_SEARCH_PARAMETER_PREFIXES,
    apply_batch_update,
    search_segment_credits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--actor-learning-rate-scale", type=float, default=2.5)
    parser.add_argument("--actor-update-epochs", type=int, default=2)
    return parser.parse_args()


def final_summary(log_path: Path) -> dict:
    summaries = [
        json.loads(line.removeprefix("FINAL_SUMMARY "))
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("FINAL_SUMMARY ")
    ]
    if len(summaries) != 1:
        raise RuntimeError(f"{log_path} 包含 {len(summaries)} 条 FINAL_SUMMARY")
    return summaries[0]


def factual_result(probe_directory: Path) -> dict:
    candidates = [probe_directory / "result.json"]
    if not candidates[0].is_file():
        candidates = list(probe_directory.glob("*/result.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"事实 result.json 不唯一: probe={probe_directory}, candidates={candidates}"
        )
    return json.loads(candidates[0].read_text(encoding="utf-8"))["factual"]


def main() -> None:
    args = parse_args()
    episode_dirs = sorted(path.parent for path in args.episodes_dir.glob("b*/run.log"))
    if not episode_dirs:
        raise RuntimeError(f"没有找到已完成回合: {args.episodes_dir}")

    rollout_paths: list[Path] = []
    credits_by_episode: list[list[dict[str, float | int]]] = []
    for episode_dir in episode_dirs:
        rollout_path = episode_dir / "on_policy_rollout.pt"
        summary = final_summary(episode_dir / "run.log")
        handoff = summary["native_causal_handoff"]
        factual = factual_result(Path(handoff["probe_directory"]))
        rollout = torch.load(
            rollout_path, map_location="cpu", weights_only=False
        )["rollout"]
        active = rollout.action_mask.search_position.cpu().to(dtype=torch.bool)
        boundary_steps = rollout.steps.cpu()[active].tolist()
        credits = search_segment_credits(
            boundary_steps,
            factual.get("search_monitor_samples", ()),
            factual.get("search_discovery_steps", {}),
        )
        if len(credits) != len(set(map(int, boundary_steps))):
            raise RuntimeError(f"{episode_dir} 的 SEARCH 边界未完整生成信用")
        rollout_paths.append(rollout_path)
        credits_by_episode.append(credits)

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source_checkpoint, args.output_checkpoint)
    metrics = apply_batch_update(
        args.output_checkpoint,
        rollout_paths,
        anchor_group_ids=[path.parent.name for path in rollout_paths],
        normalize_actor_advantages=True,
        actor_update_epochs=args.actor_update_epochs,
        critic_update_epochs=4,
        actor_loss_scale=1.0,
        actor_learning_rate_scale=args.actor_learning_rate_scale,
        critic_learning_rate_scale=1.0,
        critic_beta1=0.0,
        actor_beta1=0.0,
        actor_position_kl_limit=0.01,
        actor_joint_kl_limit=0.02,
        actor_backtrack_factor=0.5,
        trainable_parameter_prefixes=C0_SEARCH_PARAMETER_PREFIXES,
        causal_advantage_projection=True,
        episode_search_segment_credits=credits_by_episode,
    )
    returns = [float(row["return"]) for rows in credits_by_episode for row in rows]
    result = {
        "episode_count": len(rollout_paths),
        "segment_count": len(returns),
        "segment_return_mean": sum(returns) / len(returns),
        "segment_positive_count": sum(value > 1e-12 for value in returns),
        "segment_zero_count": sum(abs(value) <= 1e-12 for value in returns),
        "source_checkpoint": str(args.source_checkpoint),
        "output_checkpoint": str(args.output_checkpoint),
        "ppo": metrics,
    }
    args.metrics_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
