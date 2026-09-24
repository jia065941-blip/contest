"""Train Stage 0-2 of the context-conditioned residual PPO avoid skill."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policies.red.skills.evasion_ppo.local_env import LocalAvoidConfig, VectorLocalAvoidEnv
from policies.red.skills.evasion_ppo.local_ppo import (
    LocalAvoidPPO,
    LocalPPOConfig,
    collect_rollout,
    evaluate_policy,
    summarize_episodes,
)


def parse_steps(value: str) -> tuple[int, int, int]:
    parts = tuple(int(item.strip()) for item in value.split(","))
    if len(parts) != 3 or any(item <= 0 for item in parts):
        raise argparse.ArgumentTypeError("stage steps must be three positive integers")
    return parts


def stage_path(path: Path, stage: int) -> Path:
    return path.with_name(f"{path.stem}_stage{stage}{path.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-steps",
        type=parse_steps,
        default=parse_steps("131072,262144,524288"),
        help="transitions for Stage 0, 1, and 2",
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/local_avoid_ppo_stage2.pt",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "results/local_avoid_ppo/tensorboard",
    )
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.stage_steps = (32, 32, 32)
        args.num_envs = 4
        args.rollout_steps = 8
        args.minibatch_size = 16
        args.ppo_epochs = 1
        args.eval_episodes = 12
        args.eval_interval = 1
        args.disable_tensorboard = True
    if args.num_envs <= 0 or args.rollout_steps <= 0:
        raise ValueError("num-envs and rollout-steps must be positive")

    environment_config = LocalAvoidConfig()
    if args.resume:
        policy, environment_config, _payload = LocalAvoidPPO.load(
            args.resume, device=args.device
        )
        policy.config.num_parallel_envs = args.num_envs
        policy.config.rollout_steps = args.rollout_steps
        policy.config.minibatch_size = args.minibatch_size
        policy.config.ppo_epochs = args.ppo_epochs
        policy.discounted_reward = policy.discounted_reward[:0]
    else:
        policy = LocalAvoidPPO(
            LocalPPOConfig(
                actor_obs_dim=33,
                critic_state_dim=43,
                num_parallel_envs=args.num_envs,
                rollout_steps=args.rollout_steps,
                minibatch_size=args.minibatch_size,
                ppo_epochs=args.ppo_epochs,
                seed=args.seed,
                device=args.device,
            )
        )

    writer = None
    run_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = args.log_dir / f"local_avoid_{run_stamp}"
    if not args.disable_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard is required; install it or pass --disable-tensorboard"
            ) from error
        writer = SummaryWriter(str(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "training_history.jsonl"
    history_file = history_path.open("a", encoding="utf-8")
    best_score = float("-inf")
    best_path = args.model.with_name(f"{args.model.stem}_best{args.model.suffix}")
    final_metrics: dict = {}

    try:
        for stage, budget in enumerate(args.stage_steps):
            policy.stage = stage
            environment = VectorLocalAvoidEnv(
                args.num_envs,
                environment_config,
                stage=stage,
                seed=args.seed + stage * 100_000,
            )
            actor_observation, critic_state = environment.reset()
            stage_start = policy.total_steps
            stage_episodes: list[dict] = []
            while policy.total_steps - stage_start < budget:
                rollout, actor_observation, critic_state, completed = collect_rollout(
                    policy, environment, actor_observation, critic_state
                )
                stage_episodes.extend(completed)
                train_metrics = policy.update(rollout)
                skill_metrics = summarize_episodes(stage_episodes[-5000:])
                record = {
                    "stage": stage,
                    "total_steps": policy.total_steps,
                    "stage_steps": policy.total_steps - stage_start,
                    "train": train_metrics,
                    "skill": skill_metrics,
                }
                history_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                history_file.flush()
                print("LOCAL_AVOID_TRAIN " + json.dumps(record, ensure_ascii=False), flush=True)
                if writer is not None:
                    for key, value in train_metrics.items():
                        writer.add_scalar(f"train/{key}", value, policy.total_steps)
                    for key, value in skill_metrics.items():
                        writer.add_scalar(f"skill/{key}", value, policy.total_steps)
                    writer.add_scalar("curriculum/current_stage", stage, policy.total_steps)

                if policy.update_count % args.eval_interval == 0:
                    evaluation = evaluate_policy(
                        policy,
                        environment_config,
                        stage=stage,
                        episodes=args.eval_episodes,
                        seed=args.seed + 900_000 + policy.update_count,
                    )
                    print(
                        "LOCAL_AVOID_EVAL "
                        + json.dumps({"stage": stage, **evaluation}, ensure_ascii=False),
                        flush=True,
                    )
                    if writer is not None:
                        for key, value in evaluation.items():
                            writer.add_scalar(f"eval_stage_{stage}/{key}", value, policy.total_steps)
                    score = (
                        evaluation["straight_action_ratio"]
                        if stage == 0
                        else evaluation["evade_success_rate"]
                        - evaluation["hit_rate"]
                        - min(evaluation["mean_extra_path_length_m"] / 100_000.0, 1.0)
                    )
                    if stage == 2 and score > best_score:
                        best_score = score
                        policy.save(
                            best_path,
                            environment_config=environment_config,
                            metrics=evaluation,
                        )
            final_metrics = evaluate_policy(
                policy,
                environment_config,
                stage=stage,
                episodes=args.eval_episodes,
                seed=args.seed + 1_000_000 + stage,
            )
            policy.save(
                stage_path(args.model, stage),
                environment_config=environment_config,
                metrics=final_metrics,
            )
            print(
                "LOCAL_AVOID_STAGE_COMPLETE "
                + json.dumps({"stage": stage, **final_metrics}, ensure_ascii=False),
                flush=True,
            )
        policy.save(
            args.model,
            environment_config=environment_config,
            metrics=final_metrics,
        )
    finally:
        history_file.close()
        if writer is not None:
            writer.close()
    print(f"LOCAL_AVOID_MODEL {args.model}")
    print(f"LOCAL_AVOID_HISTORY {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
