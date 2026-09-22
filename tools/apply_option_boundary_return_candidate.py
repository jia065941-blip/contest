"""Apply a C1 MAPPO candidate using option-boundary return attribution.

The actor and its legal action masks are left unchanged.  A small, training-only
ensemble learns the E01 score residual attributable to the legal option-boundary
tokens already stored in C1 rollouts.  Labels are centred within each scenario
seed so that the model cannot improve its loss merely by memorising map
difficulty.  Ensemble disagreement shrinks uncertain advantages before the
usual KL-constrained PPO update.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.unified_mappo.model import HybridMAPPOModel, HybridMAPPOTrainer
from tools.train_start_state_option_curriculum import apply_batch_update


C1_ACTOR_PREFIXES = (
    "presence_head.",
    "initial_mean_head.",
    "initial_log_std",
    "target_head.",
)


@dataclass(frozen=True)
class BoundaryEpisode:
    rollout_path: Path
    progress_path: Path
    seed: int
    score: float
    suffix_return: float
    tokens: torch.Tensor


class OptionBoundaryReturnModel(nn.Module):
    """Small recurrent return model over legal option-boundary tokens."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.token_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.sequence = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        padded_tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if padded_tokens.ndim != 3:
            raise ValueError("padded_tokens must be [batch,time,feature]")
        if lengths.shape != (padded_tokens.shape[0],):
            raise ValueError("lengths must be [batch]")
        if bool((lengths <= 0).any().item()):
            raise ValueError("every option-boundary sequence must be non-empty")
        encoded = self.token_encoder(padded_tokens)
        sequence, _ = self.sequence(encoded)
        row_indices = torch.arange(lengths.numel(), device=lengths.device)
        final = sequence[row_indices, lengths - 1]
        return self.return_head(final).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--actor-progress", type=Path, required=True)
    parser.add_argument(
        "--replay-root",
        type=Path,
        action="append",
        required=True,
        help="Root whose immediate children may contain training_progress.json.",
    )
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--return-model-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--huber-beta", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--actor-update-epochs", type=int, default=32)
    parser.add_argument("--actor-learning-rate-scale", type=float, default=1.0)
    parser.add_argument("--position-kl-limit", type=float, default=0.005)
    parser.add_argument("--joint-kl-limit", type=float, default=0.01)
    return parser.parse_args()


def _resolved_rollout_path(progress_path: Path, episode: dict) -> Path:
    recorded = Path(episode["rollout"])
    if recorded.is_file():
        return recorded.resolve()
    moved = (
        progress_path.parent
        / "episodes"
        / recorded.parent.name
        / recorded.name
    )
    return moved.resolve()


def discover_progress_files(roots: Sequence[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        if root.name == "training_progress.json" and root.is_file():
            paths.add(root.resolve())
            continue
        paths.update(path.resolve() for path in root.glob("*/training_progress.json"))
    return sorted(paths)


@torch.no_grad()
def boundary_tokens(
    policy: HybridMAPPOModel,
    rollout: object,
) -> torch.Tensor:
    """Build training-only CTDE tokens without changing actor observations."""

    observations = rollout.observations.to(dtype=torch.float32)
    actor_latent = policy.encoder(observations)
    if not policy.strict_ctde:
        critic_latent = actor_latent
    else:
        if rollout.critic_states is None:
            raise ValueError("strict CTDE rollout lacks critic_states")
        agent_ids = rollout.agent_ids.to(dtype=torch.long)
        identity = policy.agent_embedding(agent_ids)
        critic_latent = policy.critic_encoder(torch.cat(
            (
                rollout.critic_states.to(dtype=torch.float32),
                observations,
                identity,
            ),
            dim=-1,
        ))
    action_semantics = rollout.actions.semantic_tensor().to(dtype=torch.float32)
    target_one_hot = torch.nn.functional.one_hot(
        rollout.actions.target_index.to(dtype=torch.long),
        num_classes=policy.config.target_slots,
    ).to(dtype=torch.float32)
    action_masks = torch.stack(
        (
            rollout.action_mask.presence,
            rollout.action_mask.initial_position,
            rollout.action_mask.search_position,
            rollout.action_mask.retarget,
            rollout.action_mask.target,
            rollout.action_mask.maneuver,
            rollout.action_mask.satellite,
        ),
        dim=-1,
    ).to(dtype=torch.float32)
    normalized_steps = rollout.steps.to(dtype=torch.float32).unsqueeze(-1) / 100.0
    return torch.cat(
        (
            actor_latent,
            critic_latent,
            action_semantics,
            target_one_hot,
            action_masks,
            normalized_steps,
        ),
        dim=-1,
    ).cpu()


def load_boundary_episodes(
    progress_paths: Iterable[Path],
    policy: HybridMAPPOModel,
) -> list[BoundaryEpisode]:
    episodes: list[BoundaryEpisode] = []
    seen_rollouts: set[Path] = set()
    policy.eval()
    for progress_path in progress_paths:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        for episode in progress.get("episodes", []):
            if episode.get("stage") != "C1":
                continue
            if bool(episode.get("deterministic_actor", False)):
                continue
            rollout_path = _resolved_rollout_path(progress_path, episode)
            if not rollout_path.is_file() or rollout_path in seen_rollouts:
                continue
            score = float(episode["score"])
            if not math.isfinite(score):
                continue
            stored = torch.load(
                rollout_path,
                map_location="cpu",
                weights_only=False,
            )
            rollout = stored["rollout"]
            if rollout.batch_size <= 0:
                continue
            seen_rollouts.add(rollout_path)
            episodes.append(BoundaryEpisode(
                rollout_path=rollout_path,
                progress_path=progress_path.resolve(),
                seed=int(episode["seed"]),
                score=score,
                suffix_return=float(episode["student_suffix_return"]),
                tokens=boundary_tokens(policy, rollout),
            ))
    return episodes


def pad_tokens(
    episodes: Sequence[BoundaryEpisode],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not episodes:
        raise ValueError("option-boundary dataset is empty")
    lengths = torch.tensor(
        [episode.tokens.shape[0] for episode in episodes],
        dtype=torch.long,
    )
    input_dim = int(episodes[0].tokens.shape[1])
    if any(episode.tokens.shape[1] != input_dim for episode in episodes):
        raise ValueError("option-boundary feature dimensions are inconsistent")
    padded = torch.zeros(
        len(episodes),
        int(lengths.max().item()),
        input_dim,
        dtype=torch.float32,
    )
    for index, episode in enumerate(episodes):
        padded[index, : lengths[index]] = episode.tokens
    return padded, lengths


def seed_centered_labels(episodes: Sequence[BoundaryEpisode]) -> torch.Tensor:
    """Remove scenario difficulty and retain within-seed action effects."""

    labels = torch.tensor(
        [episode.score / 100.0 for episode in episodes],
        dtype=torch.float32,
    )
    seed_indices: dict[int, list[int]] = {}
    for index, episode in enumerate(episodes):
        seed_indices.setdefault(episode.seed, []).append(index)
    for indices in seed_indices.values():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        labels[index_tensor] -= labels[index_tensor].mean()
    return labels


def train_return_ensemble(
    episodes: Sequence[BoundaryEpisode],
    *,
    ensemble_size: int,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    huber_beta: float,
    batch_size: int,
) -> tuple[list[OptionBoundaryReturnModel], dict[str, float]]:
    if ensemble_size <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("ensemble_size, epochs and batch_size must be positive")
    inputs, lengths = pad_tokens(episodes)
    labels = seed_centered_labels(episodes)
    models: list[OptionBoundaryReturnModel] = []
    losses: list[float] = []
    for member in range(ensemble_size):
        seed = 7 + 31 * member
        torch.manual_seed(seed)
        model = OptionBoundaryReturnModel(
            int(inputs.shape[-1]),
            hidden_dim=hidden_dim,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        generator = torch.Generator().manual_seed(seed)
        final_loss = float("nan")
        for _ in range(epochs):
            model.train()
            for indices in torch.randperm(
                len(episodes), generator=generator
            ).split(batch_size):
                prediction = model(inputs[indices], lengths[indices])
                loss = torch.nn.functional.smooth_l1_loss(
                    prediction,
                    labels[indices],
                    beta=huber_beta,
                ) + 0.02 * prediction.mean().square()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                final_loss = float(loss.item())
        model.eval()
        models.append(model)
        losses.append(final_loss)
    return models, {
        "ensemble_size": float(ensemble_size),
        "training_episode_count": float(len(episodes)),
        "training_boundary_count": float(sum(
            episode.tokens.shape[0] for episode in episodes
        )),
        "final_member_loss_mean": sum(losses) / len(losses),
        "final_member_loss_max": max(losses),
    }


@torch.no_grad()
def predict_ensemble(
    models: Sequence[OptionBoundaryReturnModel],
    episodes: Sequence[BoundaryEpisode],
) -> torch.Tensor:
    inputs, lengths = pad_tokens(episodes)
    return torch.stack([model(inputs, lengths) for model in models])


def uncertainty_shrunk_returns(
    predictions: torch.Tensor,
    seeds: Sequence[int],
) -> tuple[list[float], dict[str, float]]:
    """Create dense, zero-centred returns and shrink ensemble disagreements."""

    if predictions.ndim != 2 or predictions.shape[1] != len(seeds):
        raise ValueError("predictions must be [ensemble,episode]")
    grouped: dict[int, list[int]] = {}
    for index, seed in enumerate(seeds):
        grouped.setdefault(int(seed), []).append(index)
    singletons = sorted(seed for seed, indices in grouped.items() if len(indices) < 2)
    if singletons:
        raise ValueError(
            "option-boundary actor batch requires repeated seeds; "
            f"singletons={singletons}"
        )
    actor_returns = torch.zeros(predictions.shape[1], dtype=torch.float32)
    reliabilities: list[float] = []
    raw_effects: list[float] = []
    shrunk_effects: list[float] = []
    for indices in grouped.values():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        centered = predictions[:, index_tensor]
        centered = centered - centered.mean(dim=1, keepdim=True)
        mean = centered.mean(dim=0)
        disagreement = centered.std(dim=0, unbiased=False)
        reliability = mean.abs() / (mean.abs() + disagreement + 1e-6)
        shrunk = mean * reliability
        # merge_rollouts applies a leave-one-out baseline.  This scale makes
        # its resulting advantage equal to the desired centred effect.
        shrunk = shrunk - shrunk.mean()
        actor_returns[index_tensor] = shrunk * (
            (len(indices) - 1) / len(indices)
        )
        reliabilities.extend(reliability.tolist())
        raw_effects.extend(mean.tolist())
        shrunk_effects.extend(shrunk.tolist())
    return actor_returns.tolist(), {
        "actor_episode_count": float(len(seeds)),
        "actor_seed_group_count": float(len(grouped)),
        "raw_effect_abs_mean": float(torch.tensor(raw_effects).abs().mean()),
        "shrunk_effect_abs_mean": float(
            torch.tensor(shrunk_effects).abs().mean()
        ),
        "ensemble_reliability_mean": sum(reliabilities) / len(reliabilities),
        "ensemble_reliability_min": min(reliabilities),
    }


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    x = torch.tensor(left, dtype=torch.float32)
    y = torch.tensor(right, dtype=torch.float32)
    x -= x.mean()
    y -= y.mean()
    denominator = (x.square().sum() * y.square().sum()).sqrt()
    if float(denominator.item()) <= 1e-12:
        return float("nan")
    return float((x * y).sum().div(denominator).item())


def paired_diagnostics(
    episodes: Sequence[BoundaryEpisode],
    actor_returns: Sequence[float],
) -> dict[str, float]:
    grouped: dict[int, list[int]] = {}
    for index, episode in enumerate(episodes):
        grouped.setdefault(episode.seed, []).append(index)
    predicted_deltas: list[float] = []
    score_deltas: list[float] = []
    suffix_deltas: list[float] = []
    for indices in grouped.values():
        if len(indices) != 2:
            continue
        left, right = indices
        predicted_deltas.append(actor_returns[left] - actor_returns[right])
        score_deltas.append(
            (episodes[left].score - episodes[right].score) / 100.0
        )
        suffix_deltas.append(
            episodes[left].suffix_return - episodes[right].suffix_return
        )
    non_ties = [
        index for index, delta in enumerate(score_deltas)
        if abs(delta) > 1e-12
    ]
    sign_accuracy = (
        sum(
            (predicted_deltas[index] > 0.0) == (score_deltas[index] > 0.0)
            for index in non_ties
        ) / len(non_ties)
        if non_ties else float("nan")
    )
    return {
        "paired_seed_count": float(len(predicted_deltas)),
        "predicted_score_delta_correlation": _correlation(
            predicted_deltas, score_deltas
        ),
        "suffix_score_delta_correlation": _correlation(
            suffix_deltas, score_deltas
        ),
        "predicted_delta_sign_accuracy": sign_accuracy,
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    trainer = HybridMAPPOTrainer.load(args.source_checkpoint, device="cpu")
    actor_progress = args.actor_progress.resolve()
    replay_progress = [
        path for path in discover_progress_files(args.replay_root)
        if path != actor_progress
    ]
    replay_episodes = load_boundary_episodes(replay_progress, trainer.model)
    actor_episodes = load_boundary_episodes((actor_progress,), trainer.model)
    if not replay_episodes or not actor_episodes:
        raise ValueError("both replay and actor datasets must be non-empty")
    replay_paths = {episode.rollout_path for episode in replay_episodes}
    overlap = replay_paths.intersection(
        episode.rollout_path for episode in actor_episodes
    )
    if overlap:
        raise ValueError("actor rollouts leaked into return-model replay")

    models, training_metrics = train_return_ensemble(
        replay_episodes,
        ensemble_size=args.ensemble_size,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        huber_beta=args.huber_beta,
        batch_size=args.batch_size,
    )
    predictions = predict_ensemble(models, actor_episodes)
    actor_returns, attribution_metrics = uncertainty_shrunk_returns(
        predictions,
        [episode.seed for episode in actor_episodes],
    )
    diagnostics = paired_diagnostics(actor_episodes, actor_returns)

    args.return_model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "method": "option_boundary_seed_centered_return_ensemble",
            "input_dim": int(replay_episodes[0].tokens.shape[-1]),
            "hidden_dim": args.hidden_dim,
            "models": [model.state_dict() for model in models],
            "training_progress": [str(path) for path in replay_progress],
            "actor_progress_excluded": str(actor_progress),
            "training_metrics": training_metrics,
            "attribution_metrics": attribution_metrics,
            "paired_diagnostics": diagnostics,
        },
        args.return_model_output,
    )

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source_checkpoint, args.output_checkpoint)
    ppo_metrics = apply_batch_update(
        args.output_checkpoint,
        [episode.rollout_path for episode in actor_episodes],
        anchor_group_ids=[
            f"seed:{episode.seed}" for episode in actor_episodes
        ],
        normalize_actor_advantages=True,
        actor_update_epochs=args.actor_update_epochs,
        critic_update_epochs=0,
        actor_learning_rate_scale=args.actor_learning_rate_scale,
        actor_position_kl_limit=args.position_kl_limit,
        actor_joint_kl_limit=args.joint_kl_limit,
        actor_backtrack_factor=0.5,
        trainable_parameter_prefixes=C1_ACTOR_PREFIXES,
        episode_actor_returns=actor_returns,
        causal_advantage_projection=False,
    )
    result = {
        "method": "option_boundary_return_redistribution",
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "actor_progress": str(actor_progress),
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "return_model_output": str(args.return_model_output.resolve()),
        "training": training_metrics,
        "attribution": attribution_metrics,
        "held_out_actor_diagnostics": diagnostics,
        "ppo": ppo_metrics,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
