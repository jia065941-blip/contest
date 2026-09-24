"""Continue the local avoid skill with a staged all-aspect threat curriculum."""

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
    ("front_30", 30.0, (0.10, 0.30, 0.60)),
    ("front_side_90", 90.0, (0.10, 0.20, 0.70)),
    ("all_aspect_180", 180.0, (0.05, 0.15, 0.80)),
)


def parse_steps(value: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("phase steps must be integers") from error
    if len(result) != 3 or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("phase steps must contain three positive integers")
    return result


def phase_path(path: Path, name: str) -> Path:
    return path.with_name(f"{path.stem}_{name}{path.suffix}")


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["evade_success_rate"]
        - metrics["hit_rate"]
        - min(metrics["mean_extra_path_length_m"] / 100_000.0, 1.0)
        - 0.25 * metrics["oscillation_rate"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        type=Path,
        default=ROOT / "models/local_avoid_ppo_stage2_v1_stage1.pt",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/local_avoid_ppo_v2_stage2.pt",
    )
    parser.add_argument(
        "--phase-steps",
        type=parse_steps,
        default=parse_steps("262144,393216,786432"),
    )
    parser.add_argument(
        "--start-phase",
        choices=[name for name, _bearing, _distribution in PHASES],
        default=PHASES[0][0],
        help="Resume at this curriculum phase and skip earlier phases.",
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.015)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "results/local_avoid_ppo/tensorboard_v2",
    )
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.phase_steps = (32, 32, 32)
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

    v2_config = replace(
        previous_environment_config,
        integrated_turn_actions=True,
        straight_recovery_deg=15.0,
        dangerous_spawn_only=True,
        dangerous_max_cpa_m=500.0,
        hit_penalty=50.0,
        evade_bonus=10.0,
        deviation_penalty=0.01,
        oscillation_penalty=0.10,
        recovery_penalty=0.08,
        safety_improvement_scale=5.0,
        reward_clip=60.0,
    )
    full_evaluation_config = replace(v2_config, stage2_bearing_limit_deg=180.0)

    writer = None
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = args.log_dir / f"local_avoid_v2_{stamp}"
    if not args.disable_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "training_history.jsonl"
    history_file = history_path.open("a", encoding="utf-8")
    best_path = args.model.with_name(f"{args.model.stem}_best{args.model.suffix}")
    best_score = float("-inf")
    if best_path.exists():
        _best_policy, _best_environment, best_payload = LocalAvoidPPO.load(
            best_path, device="cpu"
        )
        best_metrics = best_payload.get("metrics", {})
        if "selection_score" in best_metrics:
            best_score = float(best_metrics["selection_score"])
        elif "evade_success_rate" in best_metrics:
            best_score = selection_score(best_metrics)
        del _best_policy
    final_metrics: dict[str, float] = {}

    def evaluate_and_maybe_save(
        phase_name: str,
        phase_config,
        *,
        seed_offset: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        nonlocal best_score
        curriculum_metrics = evaluate_policy(
            policy,
            phase_config,
            stage=2,
            episodes=args.eval_episodes,
            seed=args.seed + 700_000 + seed_offset,
        )
        full_metrics = evaluate_policy(
            policy,
            full_evaluation_config,
            stage=2,
            episodes=args.eval_episodes,
            seed=args.seed + 900_000 + seed_offset,
        )
        score = selection_score(full_metrics)
        event = {
            "phase": phase_name,
            "total_steps": policy.total_steps,
            "curriculum": curriculum_metrics,
            "all_aspect": full_metrics,
            "selection_score": score,
        }
        print("LOCAL_AVOID_V2_EVAL " + json.dumps(event, ensure_ascii=False), flush=True)
        history_file.write(json.dumps({"evaluation": event}, ensure_ascii=False) + "\n")
        history_file.flush()
        if writer is not None:
            for key, value in curriculum_metrics.items():
                writer.add_scalar(f"eval_curriculum/{key}", value, policy.total_steps)
            for key, value in full_metrics.items():
                writer.add_scalar(f"eval_all_aspect/{key}", value, policy.total_steps)
            writer.add_scalar("eval_all_aspect/selection_score", score, policy.total_steps)
        if score > best_score:
            best_score = score
            policy.save(
                best_path,
                environment_config=full_evaluation_config,
                metrics=event,
            )
        return curriculum_metrics, full_metrics

    try:
        start_phase_index = next(
            index for index, phase in enumerate(PHASES) if phase[0] == args.start_phase
        )
        indexed_phases = list(enumerate(zip(PHASES, args.phase_steps)))
        for phase_index, (
            (phase_name, bearing_limit, stage_distribution),
            budget,
        ) in indexed_phases[start_phase_index:]:
            phase_config = replace(v2_config, stage2_bearing_limit_deg=bearing_limit)
            environment = VectorLocalAvoidEnv(
                args.num_envs,
                phase_config,
                stage=2,
                seed=args.seed + phase_index * 100_000,
                stage_distribution=stage_distribution,
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
                skill_metrics = summarize_episodes(phase_episodes[-5000:])
                by_stage = {
                    str(stage): summarize_episodes(
                        [item for item in phase_episodes[-5000:] if item.get("stage") == stage]
                    )
                    for stage in range(3)
                }
                record = {
                    "phase": phase_name,
                    "bearing_limit_deg": bearing_limit,
                    "stage_distribution": stage_distribution,
                    "total_steps": policy.total_steps,
                    "phase_steps": policy.total_steps - phase_start,
                    "train": train_metrics,
                    "skill": skill_metrics,
                    "skill_by_stage": by_stage,
                }
                print("LOCAL_AVOID_V2_TRAIN " + json.dumps(record, ensure_ascii=False), flush=True)
                history_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                history_file.flush()
                if writer is not None:
                    for key, value in train_metrics.items():
                        writer.add_scalar(f"train/{key}", value, policy.total_steps)
                    for key, value in skill_metrics.items():
                        writer.add_scalar(f"skill/{key}", value, policy.total_steps)
                    writer.add_scalar("curriculum/bearing_limit_deg", bearing_limit, policy.total_steps)
                if policy.update_count % args.eval_interval == 0:
                    _curriculum, final_metrics = evaluate_and_maybe_save(
                        phase_name,
                        phase_config,
                        seed_offset=policy.update_count,
                    )

            _curriculum, final_metrics = evaluate_and_maybe_save(
                phase_name,
                phase_config,
                seed_offset=50_000 + phase_index,
            )
            policy.save(
                phase_path(args.model, phase_name),
                environment_config=full_evaluation_config,
                metrics=final_metrics,
            )

        policy.save(
            args.model,
            environment_config=full_evaluation_config,
            metrics=final_metrics,
        )
    finally:
        history_file.close()
        if writer is not None:
            writer.close()

    print(f"LOCAL_AVOID_V2_MODEL {args.model}")
    print(f"LOCAL_AVOID_V2_BEST {best_path}")
    print(f"LOCAL_AVOID_V2_HISTORY {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
