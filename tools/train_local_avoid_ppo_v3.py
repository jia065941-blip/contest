"""Fine-tune the local avoidance PPO on hard side-aspect encounters."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policies.red.skills.evasion_ppo.local_env import VectorLocalAvoidEnv
from policies.red.skills.evasion_ppo.local_ppo import (
    LocalAvoidPPO,
    collect_rollout,
    evaluate_policy,
    summarize_episodes,
)


PHASES = (
    ("side_45_100", 45.0, 100.0, (0.0, 0.1, 0.9)),
    ("all_aspect_180", 0.0, 180.0, (0.0, 0.1, 0.9)),
)


def parse_steps(value: str) -> tuple[int, int]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("phase steps must be integers") from error
    if len(result) != 2 or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("phase steps must contain two positive integers")
    return result


def phase_path(path: Path, name: str) -> Path:
    return path.with_name(f"{path.stem}_{name}{path.suffix}")


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["evade_success_rate"]
        - 0.20 * metrics["hit_rate"]
        - 0.01 * min(metrics["mean_extra_path_length_m"] / 100_000.0, 1.0)
        - 0.01 * metrics["oscillation_rate"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        type=Path,
        default=ROOT / "models/local_avoid_ppo_v2_stage2_best.pt",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/local_avoid_ppo_v3_stage2.pt",
    )
    parser.add_argument("--phase-steps", type=parse_steps, default=parse_steps("524288,524288"))
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "results/local_avoid_ppo/tensorboard_v3",
    )
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.phase_steps = (32, 32)
        args.num_envs = 4
        args.rollout_steps = 8
        args.minibatch_size = 16
        args.ppo_epochs = 1
        args.eval_episodes = 12
        args.eval_interval = 1
        args.disable_tensorboard = True

    policy, previous_environment_config, _payload = LocalAvoidPPO.load(
        args.resume, device=args.device
    )
    policy.config.num_parallel_envs = args.num_envs
    policy.config.rollout_steps = args.rollout_steps
    policy.config.minibatch_size = args.minibatch_size
    policy.config.ppo_epochs = args.ppo_epochs
    policy.config.learning_rate = args.learning_rate
    policy.config.clip_range = args.clip_range
    policy.config.entropy_coef = args.entropy_coef
    for group in policy.optimizer.param_groups:
        group["lr"] = args.learning_rate
    policy.discounted_reward = policy.discounted_reward[:0]
    policy.stage = 2

    v3_config = replace(
        previous_environment_config,
        integrated_turn_actions=True,
        dangerous_spawn_only=True,
        dangerous_max_cpa_m=500.0,
        progress_scale=0.20,
        hit_penalty=100.0,
        evade_bonus=20.0,
        deviation_penalty=0.002,
        oscillation_penalty=0.25,
        recovery_penalty=0.01,
        safety_improvement_scale=20.0,
        unsafe_straight_penalty=0.20,
        reward_clip=120.0,
    )
    full_evaluation_config = replace(
        v3_config,
        stage2_bearing_min_deg=0.0,
        stage2_bearing_limit_deg=180.0,
    )

    writer = None
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = args.log_dir / f"local_avoid_v3_{stamp}"
    if not args.disable_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "training_history.jsonl"
    history_file = history_path.open("a", encoding="utf-8")
    best_path = args.model.with_name(f"{args.model.stem}_best{args.model.suffix}")
    best_score = float("-inf")
    final_metrics: dict[str, float] = {}

    def evaluate_and_maybe_save(phase_name: str, phase_config, seed_offset: int) -> None:
        nonlocal best_score, final_metrics
        curriculum_metrics = evaluate_policy(
            policy,
            phase_config,
            stage=2,
            episodes=args.eval_episodes,
            seed=args.seed + 700_000 + seed_offset,
        )
        final_metrics = evaluate_policy(
            policy,
            full_evaluation_config,
            stage=2,
            episodes=args.eval_episodes,
            seed=args.seed + 900_000 + seed_offset,
        )
        score = selection_score(final_metrics)
        event = {
            "phase": phase_name,
            "total_steps": policy.total_steps,
            "curriculum": curriculum_metrics,
            "all_aspect": final_metrics,
            "selection_score": score,
        }
        print("LOCAL_AVOID_V3_EVAL " + json.dumps(event), flush=True)
        history_file.write(json.dumps({"evaluation": event}) + "\n")
        history_file.flush()
        if writer is not None:
            for key, value in curriculum_metrics.items():
                writer.add_scalar(f"eval_curriculum/{key}", value, policy.total_steps)
            for key, value in final_metrics.items():
                writer.add_scalar(f"eval_all_aspect/{key}", value, policy.total_steps)
            writer.add_scalar("eval_all_aspect/selection_score", score, policy.total_steps)
        if score > best_score:
            best_score = score
            policy.save(best_path, environment_config=full_evaluation_config, metrics=event)

    try:
        for phase_index, ((name, minimum, limit, distribution), budget) in enumerate(
            zip(PHASES, args.phase_steps)
        ):
            phase_config = replace(
                v3_config,
                stage2_bearing_min_deg=minimum,
                stage2_bearing_limit_deg=limit,
            )
            environment = VectorLocalAvoidEnv(
                args.num_envs,
                phase_config,
                stage=2,
                seed=args.seed + phase_index * 100_000,
                stage_distribution=distribution,
            )
            actor_observation, critic_state = environment.reset()
            phase_start = policy.total_steps
            phase_episodes: list[dict] = []
            while policy.total_steps - phase_start < budget:
                rollout, actor_observation, critic_state, completed = collect_rollout(
                    policy, environment, actor_observation, critic_state
                )
                phase_episodes.extend(completed)
                train_metrics = policy.update(rollout)
                record = {
                    "phase": name,
                    "bearing_band_deg": [minimum, limit],
                    "total_steps": policy.total_steps,
                    "phase_steps": policy.total_steps - phase_start,
                    "train": train_metrics,
                    "skill": summarize_episodes(phase_episodes[-5000:]),
                }
                print("LOCAL_AVOID_V3_TRAIN " + json.dumps(record), flush=True)
                history_file.write(json.dumps(record) + "\n")
                history_file.flush()
                if writer is not None:
                    for key, value in train_metrics.items():
                        writer.add_scalar(f"train/{key}", value, policy.total_steps)
                    for key, value in record["skill"].items():
                        writer.add_scalar(f"skill/{key}", value, policy.total_steps)
                if policy.update_count % args.eval_interval == 0:
                    evaluate_and_maybe_save(name, phase_config, policy.update_count)

            evaluate_and_maybe_save(name, phase_config, 50_000 + phase_index)
            policy.save(
                phase_path(args.model, name),
                environment_config=full_evaluation_config,
                metrics=final_metrics,
            )

        policy.save(args.model, environment_config=full_evaluation_config, metrics=final_metrics)
    finally:
        history_file.close()
        if writer is not None:
            writer.close()

    print(f"LOCAL_AVOID_V3_MODEL {args.model}")
    print(f"LOCAL_AVOID_V3_BEST {best_path}")
    print(f"LOCAL_AVOID_V3_HISTORY {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
