"""Train Stage-3 PPO avoidance against simultaneous and delayed double threats."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys

import torch


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
    ("simultaneous_pincer", 0.50, 0.50, 524_288),
    ("delayed_trail", 0.0, 0.0, 524_288),
    ("mixed_double", 0.34, 0.33, 786_432),
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


def initialize_second_threat_slot(policy: LocalAvoidPPO) -> None:
    first_slot = slice(6, 15)
    second_slot = slice(15, 24)
    first_extra = slice(33, 38)
    second_extra = slice(38, 43)
    with torch.no_grad():
        actor_input = policy.network.actor[0]
        critic_input = policy.network.critic[0]
        actor_input.weight[:, second_slot].copy_(actor_input.weight[:, first_slot])
        critic_input.weight[:, second_slot].copy_(critic_input.weight[:, first_slot])
        critic_input.weight[:, second_extra].copy_(critic_input.weight[:, first_extra])
    policy.actor_rms.mean[second_slot] = 0.0
    policy.actor_rms.var[second_slot] = policy.actor_rms.var[first_slot]
    policy.critic_rms.mean[second_slot] = 0.0
    policy.critic_rms.var[second_slot] = policy.critic_rms.var[first_slot]
    policy.critic_rms.mean[second_extra] = 0.0
    policy.critic_rms.var[second_extra] = policy.critic_rms.var[first_extra]


def selection_score(stage3: dict[str, float], stage2: dict[str, float]) -> float:
    return (
        stage3["evade_success_rate"]
        - 0.20 * stage3["hit_rate"]
        - 0.01 * min(stage3["mean_extra_path_length_m"] / 100_000.0, 1.0)
        - 0.01 * stage3["oscillation_rate"]
        + 0.20 * stage2["evade_success_rate"]
        - 3.0 * max(0.0, 0.98 - stage2["evade_success_rate"])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        type=Path,
        default=ROOT / "models/local_avoid_ppo_v3_stage2_best.pt",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/local_avoid_ppo_v4_stage3.pt",
    )
    parser.add_argument(
        "--phase-steps",
        type=parse_steps,
        default=parse_steps("524288,524288,786432"),
    )
    parser.add_argument(
        "--start-phase",
        choices=[phase[0] for phase in PHASES],
        default=PHASES[0][0],
    )
    parser.add_argument(
        "--end-phase",
        choices=[phase[0] for phase in PHASES],
        default=PHASES[-1][0],
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--symmetry-actor-coef", type=float, default=0.0)
    parser.add_argument("--symmetry-critic-coef", type=float, default=0.0)
    parser.add_argument("--permutation-actor-coef", type=float, default=0.0)
    parser.add_argument("--permutation-critic-coef", type=float, default=0.0)
    parser.add_argument("--phase-simultaneous-probability", type=float)
    parser.add_argument("--phase-pincer-probability", type=float)
    parser.add_argument("--phase-same-side-probability", type=float, default=0.0)
    parser.add_argument(
        "--normal-env-fraction",
        type=float,
        default=0.0,
        help="fraction of vector slots that keep the normal mixed Stage-3 config",
    )
    parser.add_argument("--stage2-replay-fraction", type=float, default=0.30)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-slot-init", action="store_true")
    parser.add_argument("--freeze-observation-normalizer", action="store_true")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "results/local_avoid_ppo/tensorboard_v4_s3",
    )
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.normal_env_fraction <= 1.0:
        parser.error("--normal-env-fraction must be between 0 and 1")
    if not 0.0 <= args.stage2_replay_fraction < 1.0:
        parser.error("--stage2-replay-fraction must be in [0, 1)")
    if args.smoke:
        args.phase_steps = (32, 32, 32)
        args.num_envs = 4
        args.rollout_steps = 8
        args.minibatch_size = 16
        args.ppo_epochs = 1
        args.eval_episodes = 8
        args.eval_interval = 1
        args.disable_tensorboard = True

    policy, previous_environment_config, _payload = LocalAvoidPPO.load(
        args.resume, device=args.device
    )
    if not args.skip_slot_init:
        initialize_second_threat_slot(policy)
    policy.config.num_parallel_envs = args.num_envs
    policy.config.rollout_steps = args.rollout_steps
    policy.config.minibatch_size = args.minibatch_size
    policy.config.ppo_epochs = args.ppo_epochs
    policy.config.learning_rate = args.learning_rate
    policy.config.clip_range = args.clip_range
    policy.config.entropy_coef = args.entropy_coef
    policy.config.symmetry_actor_coef = args.symmetry_actor_coef
    policy.config.symmetry_critic_coef = args.symmetry_critic_coef
    policy.config.permutation_actor_coef = args.permutation_actor_coef
    policy.config.permutation_critic_coef = args.permutation_critic_coef
    policy.config.freeze_observation_normalizer = args.freeze_observation_normalizer
    if args.reset_optimizer:
        policy.optimizer = torch.optim.Adam(
            policy.network.parameters(), lr=args.learning_rate
        )
    for group in policy.optimizer.param_groups:
        group["lr"] = args.learning_rate
    policy.discounted_reward = policy.discounted_reward[:0]
    policy.stage = 3

    base_config = replace(
        previous_environment_config,
        integrated_turn_actions=True,
        dangerous_spawn_only=True,
        dangerous_max_cpa_m=500.0,
        progress_scale=0.15,
        hit_penalty=120.0,
        evade_bonus=30.0,
        deviation_penalty=0.002,
        oscillation_penalty=0.30,
        recovery_penalty=0.01,
        safety_improvement_scale=25.0,
        unsafe_straight_penalty=0.25,
        reward_clip=150.0,
    )
    full_config = replace(
        base_config,
        stage3_simultaneous_probability=0.34,
        stage3_pincer_probability=0.33,
        stage3_same_side_probability=0.0,
    )

    writer = None
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = args.log_dir / f"local_avoid_v4_s3_{stamp}"
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
        curriculum = evaluate_policy(
            policy,
            phase_config,
            stage=3,
            episodes=args.eval_episodes,
            seed=args.seed + 700_000 + seed_offset,
        )
        stage3 = evaluate_policy(
            policy,
            full_config,
            stage=3,
            episodes=args.eval_episodes,
            seed=args.seed + 900_000 + seed_offset,
        )
        stage2 = evaluate_policy(
            policy,
            full_config,
            stage=2,
            episodes=args.eval_episodes,
            seed=args.seed + 1_100_000 + seed_offset,
        )
        score = selection_score(stage3, stage2)
        final_metrics = stage3
        event = {
            "phase": phase_name,
            "total_steps": policy.total_steps,
            "curriculum": curriculum,
            "stage3": stage3,
            "stage2_retention": stage2,
            "selection_score": score,
        }
        print("LOCAL_AVOID_V4_S3_EVAL " + json.dumps(event), flush=True)
        history_file.write(json.dumps({"evaluation": event}) + "\n")
        history_file.flush()
        if writer is not None:
            for key, value in stage3.items():
                writer.add_scalar(f"eval_stage3/{key}", value, policy.total_steps)
            for key, value in stage2.items():
                writer.add_scalar(f"eval_stage2/{key}", value, policy.total_steps)
            writer.add_scalar("eval_stage3/selection_score", score, policy.total_steps)
        if score > best_score:
            best_score = score
            policy.save(best_path, environment_config=full_config, metrics=event)

    try:
        evaluate_and_maybe_save("initial", full_config, 0)
        start_phase_index = next(
            index for index, phase in enumerate(PHASES) if phase[0] == args.start_phase
        )
        end_phase_index = next(
            index for index, phase in enumerate(PHASES) if phase[0] == args.end_phase
        )
        if end_phase_index < start_phase_index:
            raise ValueError("end phase must not precede start phase")
        indexed_phases = list(enumerate(zip(PHASES, args.phase_steps)))
        for phase_index, (
            (name, simultaneous, pincer, _default_budget),
            budget,
        ) in indexed_phases[start_phase_index:end_phase_index + 1]:
            phase_config = replace(
                base_config,
                stage3_simultaneous_probability=(
                    simultaneous
                    if args.phase_simultaneous_probability is None
                    else args.phase_simultaneous_probability
                ),
                stage3_pincer_probability=(
                    pincer
                    if args.phase_pincer_probability is None
                    else args.phase_pincer_probability
                ),
                stage3_same_side_probability=args.phase_same_side_probability,
            )
            normal_envs = int(round(args.num_envs * args.normal_env_fraction))
            environment_configs = (
                [full_config] * normal_envs
                + [phase_config] * (args.num_envs - normal_envs)
            )
            environment = VectorLocalAvoidEnv(
                args.num_envs,
                environment_configs,
                stage=3,
                seed=args.seed + phase_index * 100_000,
                stage_distribution=(
                    0.0,
                    0.0,
                    args.stage2_replay_fraction,
                    1.0 - args.stage2_replay_fraction,
                ),
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
                recent = phase_episodes[-5000:]
                record = {
                    "phase": name,
                    "total_steps": policy.total_steps,
                    "phase_steps": policy.total_steps - phase_start,
                    "train": train_metrics,
                    "skill": summarize_episodes(recent),
                    "by_pattern": {
                        pattern: summarize_episodes(
                            [item for item in recent if item.get("encounter_pattern") == pattern]
                        )
                        for pattern in ("simultaneous", "pincer", "delayed_trail")
                    },
                }
                print("LOCAL_AVOID_V4_S3_TRAIN " + json.dumps(record), flush=True)
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
                environment_config=full_config,
                metrics=final_metrics,
            )

        policy.save(args.model, environment_config=full_config, metrics=final_metrics)
    finally:
        history_file.close()
        if writer is not None:
            writer.close()

    print(f"LOCAL_AVOID_V4_S3_MODEL {args.model}")
    print(f"LOCAL_AVOID_V4_S3_BEST {best_path}")
    print(f"LOCAL_AVOID_V4_S3_HISTORY {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
