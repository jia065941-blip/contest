"""单网络混合动作 MAPPO 的 UHM001、训练与冻结评估入口。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unified_mappo.model import HybridMAPPOConfig, HybridMAPPOTrainer
from policies.red.learning.red_policy import (
    GlobalStateEncoder,
    UNIFIED_LOCAL_OBSERVATION_DIM,
    UNIFIED_TARGET_SLOTS,
    UNIFIED_TEAM_OBSERVATION_DIM,
)

SCENARIO = ROOT / "scenarios" / "cases" / "easy" / "E01" / "scenario.json"
DEFAULT_VERSION = ROOT / "refine-logs" / "unified_hybrid_mappo_forward_20260907_214138"
OBJECTIVE_WEIGHTS = {
    "51": 5.0,
    "52": 5.0,
    "53": 5.0,
    "54": 2.0,
    "106": 2.0,
    "168": 1.0,
    "169": 1.0,
}
PUBLIC_OBJECTIVE_IDS = frozenset({51, 52, 53, 54, 106})
HIDDEN_OBJECTIVE_IDS = frozenset({168, 169})
INDIVIDUAL_REWARD_MODES = {
    "weighted_damage_individual",
    "weighted_damage_counterfactual",
    "weighted_damage_decision_anchored",
    "weighted_damage_trajectory_counterfactual",
}
LEGACY_COUNTERFACTUAL_REWARD_MODES = {
    "weighted_damage_counterfactual",
    "weighted_damage_decision_anchored",
}
TRAJECTORY_REWARD_MODE = "weighted_damage_trajectory_counterfactual"
COUNTERFACTUAL_REWARD_MODES = (
    LEGACY_COUNTERFACTUAL_REWARD_MODES | {TRAJECTORY_REWARD_MODE}
)
TRAJECTORY_ACTOR_OBSERVATION_DIM = UNIFIED_LOCAL_OBSERVATION_DIM
TARGET_SLOTS = UNIFIED_TARGET_SLOTS


def is_trajectory_counterfactual(reward_mode: str) -> bool:
    return reward_mode == TRAJECTORY_REWARD_MODE


def effective_dynamic_lifecycle(reward_mode: str, requested: bool) -> bool:
    """The trajectory method requires the per-step lifecycle action space."""

    return is_trajectory_counterfactual(reward_mode) or bool(requested)


def effective_retarget_interval(reward_mode: str, requested: int) -> int:
    """KEEP/RETARGET is evaluated every step under the trajectory method."""

    return 1 if is_trajectory_counterfactual(reward_mode) else max(1, int(requested))


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    scalar_keys = [
        key
        for key, value in records[0].items()
        if isinstance(value, (str, int, float, bool)) or value is None
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in scalar_keys})
    temporary.replace(path)


def json_serializable_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单网络混合动作 MAPPO 正向实验")
    parser.add_argument(
        "stage", choices=("sanity", "train", "evaluate", "curriculum", "all")
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--curriculum-manifest", type=Path)
    parser.add_argument("--curriculum-batch-start-states", type=int, default=8)
    parser.add_argument("--curriculum-workers", type=int, default=8)
    parser.add_argument("--curriculum-threshold", type=float, default=0.8)
    parser.add_argument("--curriculum-max-batches", type=int, default=0)
    parser.add_argument(
        "--curriculum-start-stage",
        choices=("C0", "C1", "C2", "C3", "C4"),
        default="C0",
    )
    parser.add_argument("--curriculum-start-offset", type=int, default=0)
    parser.add_argument("--training-seeds", type=int, default=24)
    parser.add_argument("--training-cycles", type=int, default=2)
    parser.add_argument("--evaluation-seeds", type=int, default=8)
    parser.add_argument("--sanity-steps", type=int, default=40)
    parser.add_argument("--sanity-blue-seed", type=int, default=41000000)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--red-seed", type=int, default=20262001)
    parser.add_argument("--simulation-seed", type=int, default=20262003)
    parser.add_argument("--parameter-seed", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--deployment-policy-share", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=32768)
    parser.add_argument("--initial-coordinate-log-std", type=float, default=-2.0)
    parser.add_argument("--counterfactual-value-coef", type=float, default=0.1)
    parser.add_argument("--reset-target-q", action="store_true")
    parser.add_argument("--dynamic-lifecycle", action="store_true")
    parser.add_argument("--retarget-interval", type=int, default=200)
    parser.add_argument("--episode-timeout", type=int, default=900)
    parser.add_argument(
        "--update-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="PPO optimizer device; trajectory rollout/replay remains on CPU",
    )
    parser.add_argument(
        "--reward-mode",
        choices=(
            "weighted_damage",
            "weighted_damage_individual",
            "weighted_damage_counterfactual",
            "weighted_damage_decision_anchored",
            TRAJECTORY_REWARD_MODE,
        ),
        default=TRAJECTORY_REWARD_MODE,
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> HybridMAPPOConfig:
    trajectory_counterfactual = is_trajectory_counterfactual(args.reward_mode)
    return HybridMAPPOConfig(
        observation_dim=(
            TRAJECTORY_ACTOR_OBSERVATION_DIM
            if trajectory_counterfactual else UNIFIED_TEAM_OBSERVATION_DIM
        ),
        critic_state_dim=(
            GlobalStateEncoder(
                max_steps=args.max_steps,
                target_slots=TARGET_SLOTS,
            ).state_dim
            if trajectory_counterfactual else None
        ),
        critic_focal_observation_dim=(
            TRAJECTORY_ACTOR_OBSERVATION_DIM
            if trajectory_counterfactual else None
        ),
        target_slots=TARGET_SLOTS,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        deployment_policy_share=args.deployment_policy_share,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        initial_coordinate_log_std=args.initial_coordinate_log_std,
        counterfactual_value_coef=args.counterfactual_value_coef,
        seed=args.parameter_seed,
        device="cpu" if trajectory_counterfactual else "cuda",
    )


def create_checkpoint(path: Path, args: argparse.Namespace) -> None:
    trainer = HybridMAPPOTrainer(config_from_args(args))
    trainer.save(path)


def reset_target_q(path: Path, seed: int) -> None:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    original_device = str(raw.get("config", {}).get("device", "auto"))
    trainer = HybridMAPPOTrainer.load(path, device="cpu")
    checkpoint_rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        first = trainer.model.target_q_head[0]
        final = trainer.model.target_q_head[2]
        first.reset_parameters()
        final.reset_parameters()
        with torch.no_grad():
            final.weight.zero_()
            final.bias.zero_()
    finally:
        torch.set_rng_state(checkpoint_rng_state)
    for parameter in trainer.model.target_q_head.parameters():
        trainer.optimizer.state.pop(parameter, None)
    trainer.config = replace(trainer.config, device=original_device)
    trainer.model.config = trainer.config
    trainer.last_metrics = dict(trainer.last_metrics)
    trainer.last_metrics["target_q_reset"] = 1.0
    trainer.save(path)


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("config", {}))
    return {
        "algorithm": checkpoint.get("algorithm"),
        "update_count": int(checkpoint.get("update_count", 0)),
        "transition_count": int(checkpoint.get("transition_count", 0)),
        "config": config,
        "metrics": checkpoint.get("metrics", {}),
        "actor_observation_dim": config.get("observation_dim"),
        "critic_state_dim": config.get("critic_state_dim"),
        "critic_focal_observation_dim": config.get("critic_focal_observation_dim"),
        "device": config.get("device"),
        "strict_ctde": config.get("critic_state_dim") is not None,
    }


def trajectory_checkpoint_contract(max_steps: int) -> dict[str, Any]:
    return {
        "algorithm": "target_conditioned_ctde_mappo_v7",
        "actor_observation_dim": TRAJECTORY_ACTOR_OBSERVATION_DIM,
        "critic_state_dim": GlobalStateEncoder(
            max_steps=max_steps,
            target_slots=TARGET_SLOTS,
        ).state_dim,
        "target_slots": TARGET_SLOTS,
        "critic_focal_observation_dim": TRAJECTORY_ACTOR_OBSERVATION_DIM,
        "device": "cpu",
        "strict_ctde": True,
    }


def validate_checkpoint_for_reward_mode(
    path: Path,
    *,
    reward_mode: str,
    max_steps: int,
) -> dict[str, Any]:
    metadata = checkpoint_metadata(path)
    if not is_trajectory_counterfactual(reward_mode):
        return metadata
    expected = trajectory_checkpoint_contract(max_steps)
    observed = {
        "algorithm": metadata["algorithm"],
        "actor_observation_dim": metadata["actor_observation_dim"],
        "critic_state_dim": metadata["critic_state_dim"],
        "target_slots": metadata["config"].get("target_slots"),
        "strict_ctde": metadata["strict_ctde"],
        "critic_focal_observation_dim": metadata["critic_focal_observation_dim"],
        "device": metadata["device"],
    }
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatches:
        raise ValueError(
            "trajectory-counterfactual checkpoint contract mismatch: "
            f"{mismatches}; create a fresh checkpoint for this reward mode"
        )
    return metadata


def run_episode(
    *,
    run_dir: Path,
    label: str,
    checkpoint: Path,
    blue_seed: int,
    red_seed: int,
    simulation_seed: int,
    max_steps: int,
    training: bool,
    train_initial_position: bool,
    timeout: int,
    reward_mode: str,
    counterfactual_value_coef: float,
    dynamic_lifecycle: bool,
    retarget_interval: int,
    update_device: str,
) -> dict[str, Any]:
    trajectory_counterfactual = is_trajectory_counterfactual(reward_mode)
    runtime_dynamic_lifecycle = effective_dynamic_lifecycle(
        reward_mode, dynamic_lifecycle
    )
    runtime_retarget_interval = effective_retarget_interval(
        reward_mode, retarget_interval
    )
    checkpoint_info = validate_checkpoint_for_reward_mode(
        checkpoint, reward_mode=reward_mode, max_steps=max_steps
    )
    config = checkpoint_info["config"]
    output_dir = run_dir / "sim_results" / f"{label}_{timestamp()}"
    log_path = run_dir / "logs" / f"{label}_{timestamp()}.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "main.py",
        "--scenario",
        str(SCENARIO),
        "--output-dir",
        str(output_dir),
        "--total-rounds",
        "1",
        "--max-steps",
        str(max_steps),
        "--render-mode",
        "none",
        "--disable-log-color",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "BLUE_POLICY": "b0_fixed_ratio_random",
            "BLUE_INTERCEPTOR_RATIO": "2",
            "RED_POLICY": "r12_unified_mappo",
            "RED_MOTION_POLICY": "unified_mappo",
            "RED_REWARD_MODE": reward_mode,
            "RED_LEARNING_MODEL": str(checkpoint.resolve()),
            "RED_LEARNING_TRAIN": "1" if training else "0",
            "BLUE_POLICY_SEED": str(blue_seed),
            "RED_POLICY_SEED": str(red_seed),
            "SIMULATION_SEED": str(simulation_seed),
            "BLUE_ASSET_VALUES": json.dumps(OBJECTIVE_WEIGHTS),
            # In trajectory mode initial coordinates are sampled by the
            # presence/LAUNCH decision, not by the staging deployment pass.
            "RED_UNIFIED_TRAIN_INITIAL": (
                "0" if trajectory_counterfactual
                else "1" if train_initial_position else "0"
            ),
            "RED_UNIFIED_DYNAMIC_LIFECYCLE": (
                "1" if runtime_dynamic_lifecycle else "0"
            ),
            "RED_UNIFIED_RETARGET_INTERVAL": str(runtime_retarget_interval),
        }
    )
    environment.update(
        {
            "RED_UNIFIED_HIDDEN_DIM": str(config["hidden_dim"]),
            "RED_UNIFIED_LEARNING_RATE": str(config["learning_rate"]),
            "RED_UNIFIED_GAMMA": str(config["gamma"]),
            "RED_UNIFIED_GAE_LAMBDA": str(config["gae_lambda"]),
            "RED_UNIFIED_CLIP_RATIO": str(config["clip_ratio"]),
            "RED_UNIFIED_VALUE_COEF": str(config["value_coef"]),
            "RED_UNIFIED_ENTROPY_COEF": str(config["entropy_coef"]),
            "RED_UNIFIED_DEPLOYMENT_POLICY_SHARE": str(
                config.get("deployment_policy_share", 0.5)
            ),
            "RED_UNIFIED_MAX_GRAD_NORM": str(config["max_grad_norm"]),
            "RED_UNIFIED_UPDATE_EPOCHS": str(config["update_epochs"]),
            "RED_UNIFIED_MINIBATCH_SIZE": str(config["minibatch_size"]),
            "RED_UNIFIED_INITIAL_LOG_STD": str(config["initial_coordinate_log_std"]),
            "RED_UNIFIED_COUNTERFACTUAL_VALUE_COEF": str(
                counterfactual_value_coef
            ),
            "RED_UNIFIED_DEVICE": "cpu" if trajectory_counterfactual else "cuda",
            "RED_UNIFIED_UPDATE_DEVICE": update_device,
        }
    )
    if trajectory_counterfactual:
        environment.update({
            "RED_UNIFIED_DEVICE": "cpu",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
    libraries = [
        str(CORE / "envengine" / "simulator" / "models" / "HXDMissileModel"),
        str(Path(sys.prefix) / "lib"),
    ]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=CORE,
            env=environment,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    completed_stdout = log_path.read_text(encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} exited with {completed.returncode}; log={log_path}"
        )
    summary_lines = [
        line[len("FINAL_SUMMARY ") :]
        for line in completed_stdout.splitlines()
        if line.startswith("FINAL_SUMMARY ")
    ]
    if len(summary_lines) != 1:
        raise RuntimeError(f"{label} did not emit exactly one FINAL_SUMMARY")
    summary = json.loads(summary_lines[0])
    official_score = float(summary["score"]["score"])
    reward_return = float(summary["red"]["official_reward_return"])
    if not math.isclose(reward_return, official_score / 100.0, abs_tol=1e-6, rel_tol=0.0):
        raise RuntimeError(
            f"weighted reward mismatch: return={reward_return}, score={official_score}"
        )
    commander_diagnostics = summary["red"]["dynamic_catalogue"]
    diagnostics = commander_diagnostics["unified_mappo"]
    return {
        "label": label,
        "blue_seed": blue_seed,
        "red_seed": red_seed,
        "simulation_seed": simulation_seed,
        "training": training,
        "max_steps": max_steps,
        "checkpoint_algorithm": checkpoint_info["algorithm"],
        "actor_observation_dim": checkpoint_info["actor_observation_dim"],
        "critic_state_dim": checkpoint_info["critic_state_dim"],
        "critic_focal_observation_dim": checkpoint_info["critic_focal_observation_dim"],
        "checkpoint_device": checkpoint_info["device"],
        "strict_ctde": checkpoint_info["strict_ctde"],
        "rollout_device": str(diagnostics.get("rollout_device", "unknown")),
        "ppo_update_device": str(
            diagnostics.get("ppo_update_device", "not_applicable")
        ),
        "dynamic_lifecycle": runtime_dynamic_lifecycle,
        "initial_coordinate_timing": (
            "launch" if trajectory_counterfactual
            else "deployment" if train_initial_position else "fixed"
        ),
        "retarget_interval": runtime_retarget_interval,
        "score": official_score,
        "raw_score": float(summary["score"]["raw_score"]),
        "official_reward_return": reward_return,
        "agent_reward_sum": float(summary["red"].get("agent_reward_sum", 0.0)),
        "credit_conservation_error": float(
            summary["red"].get("credit_conservation_error", 0.0)
        ),
        "rewarded_agent_count": int(summary["red"].get("rewarded_agent_count", 0)),
        "reward_mode": str(summary["red"].get("reward_mode", reward_mode)),
        "reward_identity_error": abs(reward_return - official_score / 100.0),
        "steps_executed": int(summary["steps_executed"]),
        "termination_reason": str(summary["termination_reason"]),
        "objectives": summary.get("objectives", []),
        "red_launched": int(summary["red"]["launched"]),
        "active_count": int(diagnostics["active_count"]),
        "assigned_count": int(commander_diagnostics["assigned_count"]),
        "commander_diagnostics": commander_diagnostics,
        "unified_diagnostics": diagnostics,
        "counterfactual_credit": summary.get("counterfactual_credit"),
        "elapsed_seconds": time.time() - started,
        "log": str(log_path.resolve()),
        "output_dir": str(output_dir.resolve()),
    }


def run_sanity(args: argparse.Namespace, run_dir: Path, checkpoint: Path) -> dict[str, Any]:
    trajectory_counterfactual = is_trajectory_counterfactual(args.reward_mode)
    runtime_dynamic_lifecycle = effective_dynamic_lifecycle(
        args.reward_mode, args.dynamic_lifecycle
    )
    if not checkpoint.is_file():
        create_checkpoint(checkpoint, args)
    before = checkpoint_metadata(checkpoint)
    record = run_episode(
        run_dir=run_dir,
        label="UHM001_sanity",
        checkpoint=checkpoint,
        blue_seed=args.sanity_blue_seed,
        red_seed=args.red_seed,
        simulation_seed=args.simulation_seed,
        max_steps=args.sanity_steps,
        training=True,
        train_initial_position=False,
        timeout=args.episode_timeout,
        reward_mode=args.reward_mode,
        counterfactual_value_coef=args.counterfactual_value_coef,
        dynamic_lifecycle=runtime_dynamic_lifecycle,
        retarget_interval=args.retarget_interval,
        update_device=args.update_device,
    )
    after = checkpoint_metadata(checkpoint)
    expected_algorithm = (
        "target_conditioned_ctde_mappo_v7"
        if trajectory_counterfactual else "unified_hybrid_mappo"
    )
    trajectory_contract = trajectory_checkpoint_contract(args.sanity_steps)
    credit_validation = record["unified_diagnostics"].get(
        "trajectory_credit_validation", {}
    )
    target_ids = set(record["unified_diagnostics"]["target_ids"])
    actor_runtime_target_ids = set(
        record["unified_diagnostics"].get("actor_runtime_target_ids", ())
    )
    discovered_target_ids = {
        int(target_id)
        for target_id in record["commander_diagnostics"].get(
            "first_discovery_step", {}
        )
    }
    provenance_target_ids = {
        int(target_id) for target_id in record["commander_diagnostics"].get("track_provenance", {})
    }
    criteria = {
        "algorithm": after["algorithm"] == expected_algorithm,
        "checkpoint_updated": after["update_count"] > before["update_count"],
        "transitions_recorded": after["transition_count"] > before["transition_count"],
        "active_entities": record["active_count"] > 0,
        "all_active_assigned": (
            record["assigned_count"]
            == record["commander_diagnostics"]["launched_count"]
            and record["assigned_count"] > 0
            if runtime_dynamic_lifecycle
            else record["assigned_count"] == record["active_count"]
        ),
        "all_active_launched": (
            record["red_launched"]
            == record["commander_diagnostics"]["launched_count"]
            and record["red_launched"] > 0
            if runtime_dynamic_lifecycle
            else (
                record["red_launched"] == record["active_count"]
                and record["commander_diagnostics"]["launched_count"]
                == record["active_count"]
            )
        ),
        "maneuver_transitions": (
            record["unified_diagnostics"]["last_maneuver_transition_count"] > 0
            and record["unified_diagnostics"]["maneuver_decision_count"] > 0
        ),
        "all_alive_entities_inferred": (
            record["unified_diagnostics"]["step_inference_count"]
            == record["unified_diagnostics"]["alive_inference_expected_count"]
        ),
        "waiting_entities_inferred": (
            record["unified_diagnostics"]["waiting_inference_count"] > 0
            and record["unified_diagnostics"]["last_waiting_transition_count"] > 0
        ),
        "discovery_gated_target_catalogue": (
            (
                PUBLIC_OBJECTIVE_IDS.issubset(target_ids)
                and target_ids <= PUBLIC_OBJECTIVE_IDS | HIDDEN_OBJECTIVE_IDS | {-100}
                and target_ids & HIDDEN_OBJECTIVE_IDS == discovered_target_ids
                and discovered_target_ids <= HIDDEN_OBJECTIVE_IDS
                and discovered_target_ids <= provenance_target_ids
                and -100 in target_ids
            )
            if runtime_dynamic_lifecycle
            else set(map(int, OBJECTIVE_WEIGHTS)).issubset(target_ids)
        ),
        "actor_runtime_state_discovery_gated": (
            record["unified_diagnostics"].get(
                "actor_target_state_discovery_gated", False
            )
            and actor_runtime_target_ids == target_ids - {-100}
            and not (
                (HIDDEN_OBJECTIVE_IDS - discovered_target_ids)
                & actor_runtime_target_ids
            )
        ),
        "dynamic_wait_and_launch": (
            not runtime_dynamic_lifecycle
            or (
                record["commander_diagnostics"]["wait_count"] > 0
                and record["commander_diagnostics"]["launch_count"] > 0
            )
        ),
        "dynamic_keep_or_retarget": (
            not runtime_dynamic_lifecycle
            or (
                record["commander_diagnostics"]["keep_count"] > 0
                or record["commander_diagnostics"]["retarget_count"] > 0
            )
        ),
        "legal_lifecycle": (
            not runtime_dynamic_lifecycle
            or record["commander_diagnostics"]["illegal_lifecycle_count"] == 0
        ),
        "legal_target_mask": (
            not runtime_dynamic_lifecycle
            or (
                record["unified_diagnostics"]["target_selection_legal_rate"]
                == 1.0
                and record["unified_diagnostics"]["low_invalid_target_count"]
                == 0
            )
        ),
        "low_performance_search": (
            not runtime_dynamic_lifecycle
            or record["commander_diagnostics"]["search_count"] > 0
        ),
        "legal_deployment": record["unified_diagnostics"]["deployment_legal_rate"] == 1.0,
        "legal_maneuver": record["unified_diagnostics"]["maneuver_legal_rate"] == 1.0,
        "actor_critic_contract": (
            (
                record["actor_observation_dim"]
                == trajectory_contract["actor_observation_dim"]
                and record["critic_state_dim"]
                == trajectory_contract["critic_state_dim"]
                and record["critic_focal_observation_dim"]
                == trajectory_contract["critic_focal_observation_dim"]
                and record["checkpoint_device"]
                == trajectory_contract["device"]
                and record["strict_ctde"]
                and record["unified_diagnostics"]["strict_ctde"]
                and not record["unified_diagnostics"]["actor_uses_team_context"]
            )
            if trajectory_counterfactual
            else (
                record["unified_diagnostics"]["team_context_dim"] == 15
                and record["unified_diagnostics"]["model_observation_dim"] == UNIFIED_TEAM_OBSERVATION_DIM
            )
        ),
        "trajectory_runtime_contract": (
            not trajectory_counterfactual
            or (
                record["dynamic_lifecycle"]
                and record["initial_coordinate_timing"] == "launch"
                and record["retarget_interval"] == 1
                and record["unified_diagnostics"]["retarget_interval"] == 1
            )
        ),
        "weighted_reward_identity": math.isclose(
            record["official_reward_return"],
            record["score"] / 100.0,
            abs_tol=1e-6,
            rel_tol=0.0,
        ),
        "individual_reward_conservation": (
            args.reward_mode not in INDIVIDUAL_REWARD_MODES
            or (
                trajectory_counterfactual
                and record["credit_conservation_error"] < 1e-6
                and bool(credit_validation.get("all_invariants_hold", False))
            )
            or (
                not trajectory_counterfactual
                and
                math.isclose(
                    record["agent_reward_sum"],
                    record["official_reward_return"],
                    abs_tol=1e-9,
                    rel_tol=0.0,
                )
                and record["credit_conservation_error"] < 1e-9
            )
        ),
        "individual_positive_reward": (
            args.reward_mode not in INDIVIDUAL_REWARD_MODES
            or (trajectory_counterfactual and (
                record["official_reward_return"] == 0.0
                or record["rewarded_agent_count"] > 0
            ))
            or (not trajectory_counterfactual and (
                record["official_reward_return"] > 0.0
                and record["rewarded_agent_count"] > 0
            ))
        ),
        "individual_reward_not_broadcast": (
            args.reward_mode not in INDIVIDUAL_REWARD_MODES
            or record["official_reward_return"] == 0.0
            or record["rewarded_agent_count"] < record["active_count"]
        ),
        "rollout_reward_conservation": (
            args.reward_mode not in INDIVIDUAL_REWARD_MODES
            or trajectory_counterfactual or math.isclose(
                float(after["metrics"].get("rollout_reward_sum", float("nan"))),
                record["official_reward_return"],
                abs_tol=1e-6,
                rel_tol=0.0,
            )
        ),
        "deployment_direct_reward_zero": (
            args.reward_mode not in INDIVIDUAL_REWARD_MODES or trajectory_counterfactual
            or abs(float(after["metrics"].get(
                "deployment_direct_reward_sum", float("nan")
            ))) < 1e-12
        ),
        "waiting_direct_reward_zero": (
            args.reward_mode not in INDIVIDUAL_REWARD_MODES or trajectory_counterfactual
            or abs(float(after["metrics"].get(
                "waiting_direct_reward_sum", float("nan")
            ))) < 1e-12
        ),
        "counterfactual_credit_conservation": (
            args.reward_mode not in COUNTERFACTUAL_REWARD_MODES
            or record["unified_diagnostics"].get(
                "max_counterfactual_conservation_error", float("inf")
            ) < 1e-6
        ),
        "counterfactual_allocator_observed": (
            args.reward_mode not in COUNTERFACTUAL_REWARD_MODES
            or record["official_reward_return"] == 0.0
            or (
                record["unified_diagnostics"].get(
                    "direct_fallback_step_count", 0
                )
                + record["unified_diagnostics"].get(
                    "counterfactual_step_count", 0
                )
                > 0
            )
        ),
        "target_q_loss_finite": (
            args.reward_mode not in LEGACY_COUNTERFACTUAL_REWARD_MODES
            or math.isfinite(float(after["metrics"].get(
                "target_q_loss", float("nan")
            )))
        ),
        "trajectory_credit_finalized": (
            not trajectory_counterfactual
            or (
                record["unified_diagnostics"].get("trajectory_credit_finalized", False)
                and bool(credit_validation.get("all_invariants_hold", False))
            )
        ),
        "decision_anchor_reward_conservation": (
            args.reward_mode != "weighted_damage_decision_anchored"
            or math.isclose(
                float(after["metrics"].get(
                    "credit_anchor_reward_sum", float("nan")
                )),
                record["official_reward_return"],
                abs_tol=1e-6,
                rel_tol=0.0,
            )
        ),
        "decision_anchor_non_anchor_zero": (
            args.reward_mode != "weighted_damage_decision_anchored"
            or abs(float(after["metrics"].get(
                "non_anchor_reward_sum", float("nan")
            ))) < 1e-9
        ),
        "decision_anchor_delayed": (
            args.reward_mode != "weighted_damage_decision_anchored"
            or (
                record["unified_diagnostics"].get(
                    "decision_anchored_assignment_count", 0
                ) > 0
                and record["unified_diagnostics"].get(
                    "decision_anchored_max_delay", 0
                ) > 0
            )
        ),
        "decision_anchor_routes_accounted": (
            args.reward_mode != "weighted_damage_decision_anchored"
            or record["unified_diagnostics"].get(
                "decision_anchored_assignment_count", 0
            )
            == record["unified_diagnostics"].get(
                "decision_anchor_target_match_count", 0
            ) + record["unified_diagnostics"].get(
                "decision_anchor_indirect_count", 0
            )
        ),
        "decision_anchor_indirect_candidates": (
            args.reward_mode != "weighted_damage_decision_anchored"
            or record["unified_diagnostics"].get(
                "decision_anchor_indirect_candidate_count", 0
            ) > 0
        ),
        "decision_anchor_advantage_signal": (
            args.reward_mode != "weighted_damage_decision_anchored"
            or (
                float(after["metrics"].get(
                    "credit_anchor_reward_nonzero_count", 0.0
                )) > 0.0
                and float(after["metrics"].get(
                    "credit_anchor_advantage_abs_mean", 0.0
                )) > 1e-12
                and float(after["metrics"].get(
                    "target_return_supervision_nonzero_count", 0.0
                )) > 0.0
                and math.isclose(
                    float(after["metrics"].get(
                        "target_return_supervision_sum", float("nan")
                    )),
                    record["official_reward_return"],
                    abs_tol=1e-6,
                    rel_tol=0.0,
                )
            )
        ),
        "decision_anchor_clean_q_fallback": (
            args.reward_mode != "weighted_damage_decision_anchored"
            or not args.reset_target_q
            or (
                record["unified_diagnostics"].get(
                    "direct_fallback_step_count", 0
                ) > 0
                and record["unified_diagnostics"].get(
                    "decision_anchor_indirect_count", 0
                ) == 0
            )
        ),
        "finite_metrics": bool(after["metrics"]) and all(
            math.isfinite(float(value))
            for value in after["metrics"].values()
            if isinstance(value, (int, float))
        ),
    }
    result = {
        "run_id": "UHM001",
        "status": "passed" if all(criteria.values()) else "failed",
        "before": before,
        "after": after,
        "record": record,
        "criteria": criteria,
    }
    write_json(run_dir / "UHM001_result.json", result)
    if result["status"] != "passed":
        raise RuntimeError(f"UHM001 criteria failed: {criteria}")
    return result


def run_training(args: argparse.Namespace, run_dir: Path, checkpoint: Path) -> list[dict[str, Any]]:
    if not checkpoint.is_file():
        create_checkpoint(checkpoint, args)
    records_path = run_dir / "training_records.json"
    records: list[dict[str, Any]] = (
        json.loads(records_path.read_text(encoding="utf-8"))
        if records_path.is_file() else []
    )
    for cycle in range(args.training_cycles):
        for offset in range(args.training_seeds):
            linear_index = cycle * args.training_seeds + offset
            blue_seed = 41000001 + offset
            label = f"UHM1{cycle + 1}_{offset + 1:02d}"
            if linear_index < len(records):
                continue
            record = run_episode(
                run_dir=run_dir,
                label=label,
                checkpoint=checkpoint,
                blue_seed=blue_seed,
                red_seed=args.red_seed,
                simulation_seed=args.simulation_seed,
                max_steps=args.max_steps,
                training=True,
                train_initial_position=cycle > 0,
                timeout=args.episode_timeout,
                reward_mode=args.reward_mode,
                counterfactual_value_coef=args.counterfactual_value_coef,
                dynamic_lifecycle=args.dynamic_lifecycle,
                retarget_interval=args.retarget_interval,
                update_device=args.update_device,
            )
            record["cycle"] = cycle + 1
            records.append(record)
            write_json(records_path, records)
            write_records_csv(run_dir / "training_records.csv", records)
            print(json.dumps({
                "stage": "train",
                "completed": len(records),
                "total": args.training_cycles * args.training_seeds,
                "score": record["score"],
            }, ensure_ascii=False), flush=True)
    return records


def paired_interval(differences: list[float]) -> dict[str, Any]:
    mean_value = statistics.mean(differences)
    if len(differences) < 2:
        return {"mean": mean_value, "ci95": [mean_value, mean_value]}
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    # Eight planned pairs use the exact two-sided t critical value for seven degrees of freedom.
    critical = 2.364624251 if len(differences) == 8 else 1.96
    return {
        "mean": mean_value,
        "ci95": [mean_value - critical * standard_error, mean_value + critical * standard_error],
    }


def run_evaluation(
    args: argparse.Namespace,
    run_dir: Path,
    initial_checkpoint: Path,
    final_checkpoint: Path,
) -> dict[str, Any]:
    records_path = run_dir / "evaluation_records.json"
    records = (
        json.loads(records_path.read_text(encoding="utf-8"))
        if records_path.is_file() else []
    )
    for offset in range(args.evaluation_seeds):
        blue_seed = 42000001 + offset
        for variant, checkpoint in (("initial", initial_checkpoint), ("final", final_checkpoint)):
            linear_index = offset * 2 + (0 if variant == "initial" else 1)
            if linear_index < len(records):
                continue
            records.append(
                {
                    "variant": variant,
                    **run_episode(
                        run_dir=run_dir,
                        label=f"UHM2_{variant}_{offset + 1:02d}",
                        checkpoint=checkpoint,
                        blue_seed=blue_seed,
                        red_seed=args.red_seed,
                        simulation_seed=args.simulation_seed,
                        max_steps=args.max_steps,
                        training=False,
                        train_initial_position=True,
                        timeout=args.episode_timeout,
                        reward_mode=args.reward_mode,
                        counterfactual_value_coef=args.counterfactual_value_coef,
                        dynamic_lifecycle=args.dynamic_lifecycle,
                        retarget_interval=args.retarget_interval,
                        update_device=args.update_device,
                    ),
                }
            )
            write_json(records_path, records)
            write_records_csv(run_dir / "evaluation_records.csv", records)
            print(json.dumps({
                "stage": "evaluate",
                "completed": len(records),
                "total": 2 * args.evaluation_seeds,
                "variant": variant,
                "score": records[-1]["score"],
            }, ensure_ascii=False), flush=True)
    by_key = {(row["variant"], row["blue_seed"]): row["score"] for row in records}
    differences = [
        by_key[("final", 42000001 + index)] - by_key[("initial", 42000001 + index)]
        for index in range(args.evaluation_seeds)
    ]
    paired = paired_interval(differences)
    initial_mean = statistics.mean(
        row["score"] for row in records if row["variant"] == "initial"
    )
    final_mean = statistics.mean(
        row["score"] for row in records if row["variant"] == "final"
    )
    result = {
        "run_id": "UHM201_UHM202",
        "status": "passed" if paired["mean"] > 0.0 else "not_passed",
        "stable_positive_gain": paired["ci95"][0] > 0.0,
        "records": records,
        "paired_difference": paired,
        "initial_mean": initial_mean,
        "final_mean": final_mean,
    }
    write_json(run_dir / "evaluation_result.json", result)
    return result


def main() -> int:
    args = parse_args()
    if args.stage == "curriculum":
        if args.curriculum_manifest is None:
            raise ValueError("curriculum stage requires --curriculum-manifest")
        run_dir = (
            args.run_dir
            or DEFAULT_VERSION / "runs" / f"curriculum_{timestamp()}"
        ).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = (
            args.checkpoint or run_dir / "checkpoints" / "working.pt"
        ).resolve()
        if not checkpoint.is_file():
            if args.source_checkpoint is None:
                raise ValueError(
                    "curriculum stage requires an existing --checkpoint or "
                    "--source-checkpoint"
                )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.source_checkpoint.resolve(), checkpoint)
        command = [
            sys.executable,
            str(ROOT / "tools" / "train_start_state_option_curriculum.py"),
            "--manifest", str(args.curriculum_manifest.resolve()),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(run_dir),
            "--batch-start-states", str(args.curriculum_batch_start_states),
            "--workers", str(args.curriculum_workers),
            "--threshold", str(args.curriculum_threshold),
            "--max-batches", str(args.curriculum_max_batches),
            "--start-stage", args.curriculum_start_stage,
            "--start-offset", str(args.curriculum_start_offset),
        ]
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    trajectory_counterfactual = is_trajectory_counterfactual(args.reward_mode)
    if trajectory_counterfactual and args.reset_target_q:
        raise ValueError("--reset-target-q only applies to legacy counterfactual modes")
    if (
        args.reward_mode == "weighted_damage_decision_anchored"
        and not args.dynamic_lifecycle
    ):
        raise ValueError(
            "decision-anchored reward requires --dynamic-lifecycle"
        )
    run_dir = (args.run_dir or DEFAULT_VERSION / "runs" / f"execution_{timestamp()}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (args.checkpoint or run_dir / "checkpoints" / "working.pt").resolve()
    initial_checkpoint = (
        args.initial_checkpoint or run_dir / "checkpoints" / "initial.pt"
    ).resolve()
    source_checkpoint = (
        args.source_checkpoint.resolve() if args.source_checkpoint is not None else None
    )
    if source_checkpoint is not None and not source_checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {source_checkpoint}")
    if source_checkpoint is not None and checkpoint == source_checkpoint:
        raise ValueError("source checkpoint and working checkpoint must be different files")
    if checkpoint == initial_checkpoint:
        raise ValueError("working checkpoint and initial checkpoint must be different files")
    if not checkpoint.is_file():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if source_checkpoint is not None:
            shutil.copy2(source_checkpoint, checkpoint)
        else:
            create_checkpoint(checkpoint, args)
        if args.reset_target_q:
            reset_target_q(checkpoint, args.parameter_seed)
    working_metadata = validate_checkpoint_for_reward_mode(
        checkpoint,
        reward_mode=args.reward_mode,
        max_steps=args.max_steps,
    )
    if not initial_checkpoint.is_file():
        initial_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, initial_checkpoint)
    initial_metadata = validate_checkpoint_for_reward_mode(
        initial_checkpoint,
        reward_mode=args.reward_mode,
        max_steps=args.max_steps,
    )
    configuration = json_serializable_arguments(args)
    configuration.update({
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "initial_checkpoint": str(initial_checkpoint),
        "effective_runtime": {
            "dynamic_lifecycle": effective_dynamic_lifecycle(
                args.reward_mode, args.dynamic_lifecycle
            ),
            "initial_coordinate_timing": (
                "launch" if trajectory_counterfactual
                else "deployment"
            ),
            "retarget_interval": effective_retarget_interval(
                args.reward_mode, args.retarget_interval
            ),
        },
        "working_checkpoint_metadata": working_metadata,
        "initial_checkpoint_metadata": initial_metadata,
    })
    write_json(run_dir / "configuration.json", configuration)
    if args.stage in {"sanity", "all"}:
        sanity_checkpoint = run_dir / "checkpoints" / "sanity.pt"
        shutil.copy2(initial_checkpoint, sanity_checkpoint)
        run_sanity(args, run_dir, sanity_checkpoint)
    if args.stage in {"train", "all"}:
        training_records = run_training(args, run_dir, checkpoint)
        first_12_mean = statistics.mean(row["score"] for row in training_records[:12])
        last_12_mean = statistics.mean(row["score"] for row in training_records[-12:])
        write_json(run_dir / "training_summary.json", {
            "run_id": "UHM101_UHM102",
            "status": "passed" if last_12_mean > first_12_mean else "not_passed",
            "count": len(training_records),
            "first_12_mean": first_12_mean,
            "last_12_mean": last_12_mean,
            "mean_gain": last_12_mean - first_12_mean,
            "max_reward_identity_error": max(row["reward_identity_error"] for row in training_records),
            "first_nonzero_record": next((row for row in training_records if row["score"] > 0.0), None),
        })
        if trajectory_counterfactual:
            contract = trajectory_checkpoint_contract(args.max_steps)
            positive_records = [
                row for row in training_records
                if row["official_reward_return"] > 0.0
            ]
            trajectory_criteria = {
                "strict_ctde_checkpoint": all(
                    row["checkpoint_algorithm"] == contract["algorithm"]
                    and row["actor_observation_dim"]
                    == contract["actor_observation_dim"]
                    and row["critic_state_dim"] == contract["critic_state_dim"]
                    and row["critic_focal_observation_dim"]
                    == contract["critic_focal_observation_dim"]
                    and row["checkpoint_device"]
                    == contract["device"]
                    and row["strict_ctde"]
                    for row in training_records
                ),
                "local_actor_global_critic": all(
                    row["unified_diagnostics"].get("strict_ctde", False)
                    and not row["unified_diagnostics"].get(
                        "actor_uses_team_context", True
                    )
                    for row in training_records
                ),
                "dynamic_lifecycle": all(
                    row["dynamic_lifecycle"]
                    and row["commander_diagnostics"].get(
                        "dynamic_lifecycle", False
                    )
                    for row in training_records
                ),
                "initial_coordinates_at_launch": all(
                    row["initial_coordinate_timing"] == "launch"
                    for row in training_records
                ),
                "retarget_every_step": all(
                    row["retarget_interval"] == 1
                    and row["unified_diagnostics"].get("retarget_interval") == 1
                    for row in training_records
                ),
                "trajectory_credit_finalized": all(
                    row["unified_diagnostics"].get(
                        "trajectory_credit_finalized", False
                    )
                    and row["unified_diagnostics"].get(
                        "trajectory_credit_validation", {}
                    ).get("all_invariants_hold", False)
                    for row in training_records
                ),
                "reward_conservation": all(
                    row["credit_conservation_error"] < 1e-6
                    for row in training_records
                ),
                "positive_events_replayed": all(
                    row["counterfactual_credit"] is not None
                    and row["counterfactual_credit"].get(
                        "single_replay_count", 0
                    ) > 0
                    for row in positive_records
                ),
            }
            trajectory_result = {
                "run_id": "TCM101",
                "reward_mode": TRAJECTORY_REWARD_MODE,
                "algorithm": contract["algorithm"],
                "architecture": contract,
                "runtime": {
                    "dynamic_lifecycle": True,
                    "initial_coordinate_timing": "launch",
                    "retarget_interval": 1,
                },
                "status": (
                    "passed" if all(trajectory_criteria.values()) else "failed"
                ),
                "criteria": trajectory_criteria,
                "episode_count": len(training_records),
                "positive_episode_count": len(positive_records),
            }
            write_json(run_dir / "TCM101_result.json", trajectory_result)
            if trajectory_result["status"] != "passed":
                raise RuntimeError(
                    f"TCM101 criteria failed: {trajectory_criteria}"
                )
        elif args.reward_mode in LEGACY_COUNTERFACTUAL_REWARD_MODES:
            criteria = {
                "reward_conservation": max(
                    row["credit_conservation_error"] for row in training_records
                ) < 1e-6,
                "counterfactual_conservation": max(
                    row["unified_diagnostics"].get(
                        "max_counterfactual_conservation_error", float("inf")
                    )
                    for row in training_records
                ) < 1e-6,
                "direct_fallback_observed": (
                    args.dynamic_lifecycle
                    or sum(
                        row["unified_diagnostics"].get(
                            "direct_fallback_step_count", 0
                        )
                        for row in training_records
                    ) > 0
                ),
                "counterfactual_credit_observed": sum(
                    row["unified_diagnostics"].get("counterfactual_step_count", 0)
                    for row in training_records
                ) > 0,
                "delayed_credit_observed": sum(
                    row["unified_diagnostics"].get("delayed_credit_count", 0)
                    for row in training_records
                ) > 0,
                "target_q_loss_finite": all(
                    math.isfinite(float(row["unified_diagnostics"]["last_metrics"].get(
                        "target_q_loss", float("nan")
                    )))
                    for row in training_records
                ),
                "decision_anchor_reward_conservation": (
                    args.reward_mode != "weighted_damage_decision_anchored"
                    or all(
                        math.isclose(
                            float(row["unified_diagnostics"]["last_metrics"].get(
                                "credit_anchor_reward_sum", float("nan")
                            )),
                            row["official_reward_return"],
                            abs_tol=1e-6,
                            rel_tol=0.0,
                        )
                        for row in training_records
                    )
                ),
                "decision_anchor_non_anchor_zero": (
                    args.reward_mode != "weighted_damage_decision_anchored"
                    or all(
                        abs(float(row["unified_diagnostics"]["last_metrics"].get(
                            "non_anchor_reward_sum", float("nan")
                        ))) < 1e-9
                        for row in training_records
                    )
                ),
                "decision_anchor_delayed": (
                    args.reward_mode != "weighted_damage_decision_anchored"
                    or sum(
                        row["unified_diagnostics"].get(
                            "decision_anchored_assignment_count", 0
                        )
                        for row in training_records
                    ) > 0
                    and max(
                        row["unified_diagnostics"].get(
                            "decision_anchored_max_delay", 0
                        )
                        for row in training_records
                    ) > 0
                ),
                "decision_anchor_routes_accounted": (
                    args.reward_mode != "weighted_damage_decision_anchored"
                    or all(
                        row["unified_diagnostics"].get(
                            "decision_anchored_assignment_count", 0
                        )
                        == row["unified_diagnostics"].get(
                            "decision_anchor_target_match_count", 0
                        ) + row["unified_diagnostics"].get(
                            "decision_anchor_indirect_count", 0
                        )
                        for row in training_records
                    )
                ),
                "decision_anchor_indirect_candidates": (
                    args.reward_mode != "weighted_damage_decision_anchored"
                    or sum(
                        row["unified_diagnostics"].get(
                            "decision_anchor_indirect_candidate_count", 0
                        )
                        for row in training_records
                    ) > 0
                ),
                "decision_anchor_advantage_signal": (
                    args.reward_mode != "weighted_damage_decision_anchored"
                    or all(
                        row["official_reward_return"] <= 0.0
                        or (
                            float(row["unified_diagnostics"]["last_metrics"].get(
                                "credit_anchor_reward_nonzero_count", 0.0
                            )) > 0.0
                            and float(row["unified_diagnostics"]["last_metrics"].get(
                                "credit_anchor_advantage_abs_mean", 0.0
                            )) > 1e-12
                            and float(row["unified_diagnostics"]["last_metrics"].get(
                                "target_return_supervision_nonzero_count", 0.0
                            )) > 0.0
                            and math.isclose(
                                float(row["unified_diagnostics"]["last_metrics"].get(
                                    "target_return_supervision_sum", float("nan")
                                )),
                                row["official_reward_return"],
                                abs_tol=1e-6,
                                rel_tol=0.0,
                            )
                        )
                        for row in training_records
                    )
                ),
                "decision_anchor_indirect_attribution": (
                    args.reward_mode != "weighted_damage_decision_anchored"
                    or sum(
                        row["unified_diagnostics"].get(
                            "decision_anchor_indirect_count", 0
                        )
                        for row in training_records
                    ) > 0
                ),
            }
            result_run_id = (
                "ICM401"
                if args.reward_mode == "weighted_damage_decision_anchored"
                else "ICM201"
            )
            icm201_result = {
                "run_id": result_run_id,
                "status": "passed" if all(criteria.values()) else "failed",
                "criteria": criteria,
                "scores": [row["score"] for row in training_records],
                "presence_rates": [
                    row["unified_diagnostics"]["presence_rate"]
                    for row in training_records
                ],
                "counterfactual_steps": [
                    row["unified_diagnostics"].get("counterfactual_step_count", 0)
                    for row in training_records
                ],
                "direct_fallback_steps": [
                    row["unified_diagnostics"].get("direct_fallback_step_count", 0)
                    for row in training_records
                ],
                "delayed_credit_counts": [
                    row["unified_diagnostics"].get("delayed_credit_count", 0)
                    for row in training_records
                ],
                "decision_anchor_assignment_counts": [
                    row["unified_diagnostics"].get(
                        "decision_anchored_assignment_count", 0
                    )
                    for row in training_records
                ],
                "decision_anchor_mean_delays": [
                    row["unified_diagnostics"].get(
                        "decision_anchored_mean_delay", 0.0
                    )
                    for row in training_records
                ],
                "decision_anchor_max_delays": [
                    row["unified_diagnostics"].get(
                        "decision_anchored_max_delay", 0
                    )
                    for row in training_records
                ],
            }
            write_json(
                run_dir / f"{result_run_id}_result.json", icm201_result
            )
            if icm201_result["status"] != "passed":
                raise RuntimeError(f"{result_run_id} criteria failed: {criteria}")
            if args.dynamic_lifecycle:
                lifecycle_criteria = {
                    "lifecycle_enabled": all(
                        row["commander_diagnostics"].get(
                            "dynamic_lifecycle", False
                        )
                        for row in training_records
                    ),
                    "wait_observed": sum(
                        row["commander_diagnostics"].get("wait_count", 0)
                        for row in training_records
                    ) > 0,
                    "launch_observed": sum(
                        row["commander_diagnostics"].get("launch_count", 0)
                        for row in training_records
                    ) > 0,
                    "retarget_observed": sum(
                        row["commander_diagnostics"].get(
                            "retarget_count", 0
                        )
                        for row in training_records
                    ) > 0,
                    "search_observed": sum(
                        row["commander_diagnostics"].get("search_count", 0)
                        for row in training_records
                    ) > 0,
                    "legal_lifecycle": all(
                        row["commander_diagnostics"].get(
                            "illegal_lifecycle_count", -1
                        ) == 0
                        for row in training_records
                    ),
                    "legal_low_performance_targets": all(
                        row["unified_diagnostics"].get(
                            "target_selection_legal_rate", 0.0
                        ) == 1.0
                        and row["unified_diagnostics"].get(
                            "low_invalid_target_count", -1
                        ) == 0
                        for row in training_records
                    ),
                }
                icm301_result = {
                    "run_id": "ICM301",
                    "status": (
                        "passed"
                        if all(lifecycle_criteria.values()) else "failed"
                    ),
                    "criteria": lifecycle_criteria,
                    "scores": [row["score"] for row in training_records],
                    "participation_rates": [
                        row["unified_diagnostics"]["presence_rate"]
                        for row in training_records
                    ],
                    "lifecycle_counts": [
                        {
                            key: row["commander_diagnostics"].get(key, 0)
                            for key in (
                                "wait_count",
                                "launch_count",
                                "keep_count",
                                "retarget_count",
                                "search_count",
                            )
                        }
                        for row in training_records
                    ],
                }
                write_json(run_dir / "ICM301_result.json", icm301_result)
                if icm301_result["status"] != "passed":
                    raise RuntimeError(
                        f"ICM301 criteria failed: {lifecycle_criteria}"
                    )
    if args.stage in {"evaluate", "all"}:
        run_evaluation(args, run_dir, initial_checkpoint, checkpoint)
    print(json.dumps({"status": "complete", "run_dir": str(run_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
