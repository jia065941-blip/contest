#!/usr/bin/env python3
"""Train the existing C0 search head from full-grid counterfactual returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unified_mappo.model import HybridMAPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate-scale", type=float, default=5.0)
    parser.add_argument("--backtrack-factor", type=float, default=0.5)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--kl-limit", type=float, default=0.02)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument(
        "--direct-weight",
        type=float,
        default=1.0,
        help="Weight direct detection relative to the proximity component.",
    )
    return parser.parse_args()


@torch.no_grad()
def search_metrics(
    trainer: HybridMAPPOTrainer,
    observations: torch.Tensor,
    target_features: torch.Tensor,
    target_valid_mask: torch.Tensor,
    returns: torch.Tensor,
    direct_returns: torch.Tensor,
) -> dict[str, float]:
    logits = trainer.model.distribution_parameters(
        observations, target_features, target_valid_mask
    )["search_logits"]
    probabilities = torch.softmax(logits, dim=-1)
    direct_mask = direct_returns.gt(0.0)
    top_indices = probabilities.argmax(dim=-1)
    oracle_indices = returns.argmax(dim=-1)
    return {
        "expected_counterfactual_return": float(
            (probabilities * returns).sum(dim=-1).mean().item()
        ),
        "expected_direct_detection_probability": float(
            (probabilities * direct_mask).sum(dim=-1).mean().item()
        ),
        "top1_direct_detection_fraction": float(
            direct_mask.gather(1, top_indices.unsqueeze(-1))
            .float().mean().item()
        ),
        "oracle_return_mean": float(
            returns.gather(1, oracle_indices.unsqueeze(-1)).mean().item()
        ),
        "oracle_direct_reachable_fraction": float(
            direct_mask.any(dim=-1).float().mean().item()
        ),
        "entropy": float(
            torch.distributions.Categorical(logits=logits)
            .entropy().mean().item()
        ),
        "top_probability_mean": float(
            probabilities.max(dim=-1).values.mean().item()
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = torch.load(
        args.dataset, map_location="cpu", weights_only=False
    )
    observations = dataset["observations"].to(device)
    target_features = dataset["target_features"].to(device)
    target_valid_mask = dataset["target_valid_mask"].to(
        device, dtype=torch.bool
    )
    returns = dataset["returns"].to(device)
    direct_returns = dataset["direct_returns"].to(device)
    training_returns = returns + (
        float(args.direct_weight) - 1.0
    ) * direct_returns
    centered_returns = training_returns - training_returns.mean(
        dim=-1, keepdim=True
    )
    return_scale = centered_returns.std(
        dim=-1, unbiased=False, keepdim=True
    ).clamp_min(1e-8)
    normalized_returns = centered_returns / return_scale

    baseline = HybridMAPPOTrainer.load(args.checkpoint, device=args.device)
    baseline.model.eval()
    with torch.no_grad():
        old_logits = baseline.model.distribution_parameters(
            observations, target_features, target_valid_mask
        )["search_logits"].detach()
        old_distribution = torch.distributions.Categorical(logits=old_logits)
    before = search_metrics(
        baseline,
        observations,
        target_features,
        target_valid_mask,
        returns,
        direct_returns,
    )

    accepted: HybridMAPPOTrainer | None = None
    accepted_scale = 0.0
    accepted_kl = 0.0
    accepted_objective = 0.0
    attempts: list[dict[str, float]] = []
    scale = float(args.learning_rate_scale)
    for attempt in range(max(1, int(args.max_attempts))):
        trainer = HybridMAPPOTrainer.load(
            args.checkpoint, device=args.device
        )
        trainer.model.train()
        for parameter in trainer.model.parameters():
            parameter.requires_grad_(False)
        search_parameters = list(trainer.model.search_head.parameters())
        for parameter in search_parameters:
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(
            search_parameters,
            lr=float(trainer.config.learning_rate) * scale,
            betas=(0.0, 0.999),
        )
        last_objective = 0.0
        for _ in range(max(1, int(args.epochs))):
            logits = trainer.model.distribution_parameters(
                observations, target_features, target_valid_mask
            )["search_logits"]
            distribution = torch.distributions.Categorical(logits=logits)
            probabilities = distribution.probs
            objective = (
                (probabilities * normalized_returns).sum(dim=-1)
                + float(args.entropy_coef) * distribution.entropy()
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            (-objective).backward()
            torch.nn.utils.clip_grad_norm_(search_parameters, 0.5)
            optimizer.step()
            last_objective = float(objective.detach().item())
        trainer.model.eval()
        with torch.no_grad():
            new_logits = trainer.model.distribution_parameters(
                observations, target_features, target_valid_mask
            )["search_logits"]
            exact_kl = float(torch.distributions.kl_divergence(
                old_distribution,
                torch.distributions.Categorical(logits=new_logits),
            ).mean().item())
        attempts.append({
            "attempt": float(attempt + 1),
            "learning_rate_scale": scale,
            "exact_search_kl": exact_kl,
            "normalized_objective": last_objective,
        })
        if exact_kl <= float(args.kl_limit):
            accepted = trainer
            accepted_scale = scale
            accepted_kl = exact_kl
            accepted_objective = last_objective
            break
        scale *= float(args.backtrack_factor)
    if accepted is None:
        raise RuntimeError(
            "no full-grid candidate satisfied KL limit: "
            + json.dumps(attempts)
        )

    after = search_metrics(
        accepted,
        observations,
        target_features,
        target_valid_mask,
        returns,
        direct_returns,
    )
    if (
        after["expected_counterfactual_return"]
        <= before["expected_counterfactual_return"]
    ):
        raise RuntimeError(
            "full-grid update did not improve expected return: "
            + json.dumps({"before": before, "after": after})
        )
    # The next on-policy PPO batch must not inherit stale search-head moments.
    for parameter in accepted.model.search_head.parameters():
        accepted.optimizer.state.pop(parameter, None)
        parameter.requires_grad_(True)
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    accepted.save(args.output_checkpoint)
    metrics = {
        "schema_version": 1,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "dataset_validation": dataset["validation"],
        "samples": int(observations.shape[0]),
        "actions_per_sample": int(returns.shape[1]),
        "epochs": int(args.epochs),
        "entropy_coef": float(args.entropy_coef),
        "direct_weight": float(args.direct_weight),
        "kl_limit": float(args.kl_limit),
        "accepted_learning_rate_scale": accepted_scale,
        "exact_search_kl": accepted_kl,
        "normalized_objective": accepted_objective,
        "attempts": attempts,
        "before": before,
        "after": after,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
