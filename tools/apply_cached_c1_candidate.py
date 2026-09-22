"""Apply a KL-constrained C1 actor candidate to cached on-policy rollouts."""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.train_start_state_option_curriculum import (
    BASE_TARGET_SLOT_IDS,
    C1_TRAINABLE_PARAMETER_PREFIXES,
    apply_batch_update,
    causal_attack_groups,
    load_teacher_initial_supervision,
    scenario_targets,
    units_for_stage,
)


TRAINABLE_PREFIXES = {
    "target": ("target_head.",),
    "position": ("initial_mean_head.", "initial_log_std"),
    "c1": C1_TRAINABLE_PARAMETER_PREFIXES,
    "all": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--trainable-components",
        choices=tuple(TRAINABLE_PREFIXES),
        required=True,
    )
    parser.add_argument("--actor-update-epochs", type=int, default=32)
    parser.add_argument("--actor-learning-rate-scale", type=float, default=1.0)
    parser.add_argument("--target-supervision-coef", type=float, default=0.0)
    parser.add_argument("--initial-supervision-coef", type=float, default=0.0)
    parser.add_argument(
        "--teacher-decision-dataset-manifest",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--position-kl-limit", type=float, default=0.01)
    parser.add_argument("--joint-kl-limit", type=float, default=0.01)
    parser.add_argument(
        "--paired-score-advantages",
        action="store_true",
        help=(
            "Use leave-one-out E01 score advantages within repeated seeds "
            "instead of the cached target-return advantages."
        ),
    )
    parser.add_argument(
        "--teacher-score-advantages",
        action="store_true",
        help=(
            "Use the exact same-seed E01 student-minus-teacher difference "
            "as the C1 option-boundary advantage."
        ),
    )
    parser.add_argument(
        "--target-action-score-advantages",
        action="store_true",
        help=(
            "Estimate a target-head advantage by grouping repeated same-seed "
            "rollouts by their first sampled target action. The teacher score "
            "is included in the teacher-target group, identical actions are "
            "averaged, and only between-action score differences are used."
        ),
    )
    parser.add_argument(
        "--rollout-reward-advantages",
        action="store_true",
        help=(
            "Use each saved rollout row's reward directly as its actor "
            "advantage; intended for signed head-level counterfactual credit."
        ),
    )
    parser.add_argument("--metrics-output", type=Path)
    return parser.parse_args()


def target_slot_by_id(manifest: dict) -> dict[int, int]:
    first_trace = Path(manifest["trajectories"][0]["trace"])
    trace = json.loads(first_trace.read_text(encoding="utf-8"))
    objective_ids = {
        int(key) for key in trace["summary"]["score"]["objective_weights"]
    }
    targets = scenario_targets(Path(manifest["scenario"]), objective_ids)
    slot_ids = list(BASE_TARGET_SLOT_IDS)
    slot_ids.extend(sorted(
        int(target_id)
        for target_id in targets
        if int(target_id) not in slot_ids
    ))
    return {target_id: index for index, target_id in enumerate(slot_ids)}


def main() -> None:
    args = parse_args()
    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows_by_seed = {
        int(row["seed"]): row for row in manifest["trajectories"]
    }
    slots = target_slot_by_id(manifest)
    episodes = progress["episodes"]
    anchor_group_ids = None
    episode_actor_returns = None
    episode_actor_advantages = None
    score_advantage_modes = sum((
        bool(args.paired_score_advantages),
        bool(args.teacher_score_advantages),
        bool(args.target_action_score_advantages),
        bool(args.rollout_reward_advantages),
    ))
    if score_advantage_modes > 1:
        raise ValueError(
            "score advantage modes are mutually exclusive"
        )
    if args.paired_score_advantages:
        seed_counts: dict[int, int] = {}
        for episode in episodes:
            seed = int(episode["seed"])
            seed_counts[seed] = seed_counts.get(seed, 0) + 1
        singleton_seeds = sorted(
            seed for seed, count in seed_counts.items() if count < 2
        )
        if singleton_seeds:
            raise ValueError(
                "paired score advantages require at least two episodes per seed; "
                f"singleton seeds: {singleton_seeds}"
            )
        anchor_group_ids = [
            f"seed:{int(episode['seed'])}" for episode in episodes
        ]
        episode_actor_returns = [
            float(episode["score"]) for episode in episodes
        ]
    elif args.teacher_score_advantages:
        episode_actor_advantages = [
            (
                float(episode["score"])
                - float(rows_by_seed[int(episode["seed"])]["teacher_score"])
            ) / 100.0
            for episode in episodes
        ]
    elif args.target_action_score_advantages:
        first_target_indices: list[int] = []
        grouped_observations: dict[
            int, dict[int, list[float]]
        ] = collections.defaultdict(lambda: collections.defaultdict(list))
        for episode in episodes:
            rollout = torch.load(
                episode["rollout"], map_location="cpu", weights_only=False
            )["rollout"]
            target_rows = rollout.action_mask.target.cpu().to(dtype=torch.bool)
            active_indices = (
                rollout.actions.target_index.cpu()[target_rows].tolist()
            )
            if not active_indices:
                raise ValueError(
                    f"episode {episode['rollout']} has no active target action"
                )
            first_target_index = int(active_indices[0])
            first_target_indices.append(first_target_index)
            grouped_observations[int(episode["seed"])][
                first_target_index
            ].append(float(episode["score"]))
        for seed, action_scores in grouped_observations.items():
            row = rows_by_seed[seed]
            teacher_action = slots[int(row["assigned_target_id"])]
            action_scores[teacher_action].append(float(row["teacher_score"]))
        action_means = {
            seed: {
                action: sum(scores) / len(scores)
                for action, scores in action_scores.items()
            }
            for seed, action_scores in grouped_observations.items()
        }
        episode_actor_advantages = []
        for episode, action in zip(episodes, first_target_indices):
            means = action_means[int(episode["seed"])]
            other_means = [
                value
                for other_action, value in means.items()
                if other_action != action
            ]
            advantage = (
                means[action] - sum(other_means) / len(other_means)
                if other_means
                else 0.0
            )
            episode_actor_advantages.append(advantage / 100.0)
    target_supervision = None
    if args.target_supervision_coef > 0.0:
        target_supervision = []
        for episode in episodes:
            row = rows_by_seed[int(episode["seed"])]
            target_id = int(episode["assigned_target_id"])
            target_supervision.append((
                int(row["damage_anchors"][str(target_id)]["decision_step"]),
                slots[target_id],
            ))

    initial_supervision = None
    if args.initial_supervision_coef > 0.0:
        if not args.teacher_decision_dataset_manifest:
            raise ValueError(
                "initial supervision requires at least one teacher dataset manifest"
            )
        initial_lookup = {}
        for dataset_manifest in args.teacher_decision_dataset_manifest:
            for key, value in load_teacher_initial_supervision(
                dataset_manifest
            ).items():
                previous = initial_lookup.get(key)
                if previous is not None and previous != value:
                    raise ValueError(f"teacher initial-position key conflict: {key}")
                initial_lookup[key] = value
        initial_supervision = []
        for episode in episodes:
            row = rows_by_seed[int(episode["seed"])]
            target_id = int(episode["assigned_target_id"])
            specifications = []
            for unit in units_for_stage(causal_attack_groups(row), "C1"):
                if int(unit["command_type"]) != 200:
                    continue
                key = (
                    int(row["seed"]),
                    int(unit["executor_id"]),
                    int(unit["timestep"]),
                )
                if key not in initial_lookup:
                    raise KeyError(
                        f"teacher initial-position data lacks causal anchor: {key}"
                    )
                agent_id, coordinates = initial_lookup[key]
                specifications.append((
                    int(unit["timestep"]),
                    int(agent_id),
                    coordinates,
                    slots[target_id],
                ))
            initial_supervision.append(specifications)

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    source_payload = torch.load(
        args.source_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    source_rng_state = source_payload.get("torch_rng_state")
    source_cuda_rng_states = source_payload.get("cuda_rng_states")
    shutil.copy2(args.source_checkpoint, args.output_checkpoint)
    metrics = apply_batch_update(
        args.output_checkpoint,
        [Path(episode["rollout"]) for episode in episodes],
        anchor_group_ids=anchor_group_ids,
        normalize_actor_advantages=True,
        actor_update_epochs=args.actor_update_epochs,
        critic_update_epochs=0,
        actor_learning_rate_scale=args.actor_learning_rate_scale,
        actor_position_kl_limit=args.position_kl_limit,
        actor_joint_kl_limit=args.joint_kl_limit,
        actor_backtrack_factor=0.5,
        target_supervision=target_supervision,
        target_supervision_coef=args.target_supervision_coef,
        initial_supervision=initial_supervision,
        initial_supervision_coef=args.initial_supervision_coef,
        trainable_parameter_prefixes=TRAINABLE_PREFIXES[
            args.trainable_components
        ],
        episode_actor_returns=episode_actor_returns,
        episode_actor_advantages=episode_actor_advantages,
        rollout_rewards_as_actor_advantages=(
            args.rollout_reward_advantages
        ),
        causal_advantage_projection=not (
            args.paired_score_advantages
            or args.teacher_score_advantages
            or args.target_action_score_advantages
            or args.rollout_reward_advantages
        ),
    )
    output_payload = torch.load(
        args.output_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if source_rng_state is not None:
        output_payload["torch_rng_state"] = source_rng_state
    if source_cuda_rng_states is not None:
        output_payload["cuda_rng_states"] = source_cuda_rng_states
    torch.save(output_payload, args.output_checkpoint)
    source_rng_preserved = (
        source_rng_state is not None
        and source_cuda_rng_states is not None
    )
    result = {
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "source_progress": str(args.progress.resolve()),
        "trainable_components": args.trainable_components,
        "paired_score_advantages": args.paired_score_advantages,
        "teacher_score_advantages": args.teacher_score_advantages,
        "target_action_score_advantages": (
            args.target_action_score_advantages
        ),
        "rollout_reward_advantages": args.rollout_reward_advantages,
        "source_rng_preserved": source_rng_preserved,
        "metrics": metrics,
    }
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
