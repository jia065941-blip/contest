from __future__ import annotations

from dataclasses import asdict
import json

import pytest
import torch

from experiments.unified_mappo.model import (
    HybridActionMask,
    HybridMAPPOConfig,
    HybridMAPPOTrainer,
    HybridRolloutBatch,
)
from tools.train_start_state_option_curriculum import (
    C0_GOAL_PARAMETER_PREFIXES,
    formal_joint_suffix_return,
    merge_rollouts,
)
from tools.train_c0a_goal_reward_weighted import (
    TARGET_TRANSFORMER_PREFIXES,
    reward_weighted_target,
    update_target_transformer,
)


def test_c0a_goal_updates_full_transformer_only():
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=5,
            critic_state_dim=7,
            critic_focal_observation_dim=5,
            target_slots=3,
            target_feature_dim=4,
            target_selector_arch="transformer",
            hidden_dim=16,
            device="cpu",
        )
    )
    trainable = {
        name
        for name, _ in trainer.model.named_parameters()
        if name.startswith(C0_GOAL_PARAMETER_PREFIXES)
    }
    assert any(name.startswith("target_transformer_item_encoder.") for name in trainable)
    assert any(name.startswith("target_transformer_query.") for name in trainable)
    assert any(name.startswith("target_transformer_encoder.") for name in trainable)
    assert any(name.startswith("target_transformer_score.") for name in trainable)
    assert not any(name.startswith("agent_embedding.") for name in trainable)
    assert not any(name.startswith("critic_encoder.") for name in trainable)
    assert not any(name.startswith("value_head.") for name in trainable)
    assert not any(name.startswith("encoder.") for name in trainable)
    assert not any(name.startswith("presence_head.") for name in trainable)
    assert not any(name.startswith("search_head.") for name in trainable)
    assert not any(name.startswith("maneuver_head.") for name in trainable)


def test_reward_weighted_target_uses_old_policy_baseline():
    old = torch.tensor([0.5, 0.3, 0.2])
    returns = torch.tensor([0.0, 1.0, 2.0])
    target, baseline = reward_weighted_target(
        old,
        returns,
        temperature=0.1,
        reward_logit_clip=5.0,
    )
    assert baseline.item() == pytest.approx(0.7)
    assert target.sum().item() == pytest.approx(1.0)
    assert target[2] > target[1] > target[0]


def test_reward_weighted_update_is_fresh_and_transformer_only(tmp_path):
    config = HybridMAPPOConfig(
        observation_dim=5,
        critic_state_dim=7,
        critic_focal_observation_dim=5,
        target_slots=3,
        target_feature_dim=4,
        target_allocator_hidden_dim=16,
        target_transformer_heads=4,
        target_transformer_layers=1,
        target_transformer_ff_dim=32,
        target_selector_arch="transformer",
        hidden_dim=16,
        device="cpu",
    )
    trainer = HybridMAPPOTrainer(config)
    source = tmp_path / "source.pt"
    torch.save({
        "algorithm": "target_conditioned_ctde_mappo_v11",
        "config": asdict(config),
        "model": trainer.model.state_dict(),
        "optimizer": {"legacy": "must not be loaded"},
        "update_count": 99,
        "transition_count": 999,
    }, source)
    observation = torch.randn(1, 5)
    features = torch.randn(1, 3, 4)
    valid = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        latent = trainer.model.encoder(observation)
        logits = trainer.model.transformer_target_logits(
            latent,
            features,
            valid,
        )
        old_probabilities = torch.softmax(logits, dim=-1)
    batch = {"items": [{
        "observation": observation,
        "target_features": features,
        "target_valid_mask": valid,
        "candidate_target_indices": torch.tensor([0, 1]),
        "candidate_target_ids": torch.tensor([51, 52]),
        "official_joint_returns": torch.tensor([0.0, 1.0]),
        "old_target_probabilities": old_probabilities,
    }]}
    output = tmp_path / "output.pt"
    metrics = update_target_transformer(
        source,
        batch,
        output,
        learning_rate=1e-3,
        temperature=0.1,
        reward_logit_clip=5.0,
        kl_coef=0.1,
        epsilon_return=1e-6,
        max_grad_norm=0.5,
        optimizer_steps=1,
        target_kl_limit=0.02,
        backtrack_factor=0.5,
        device="cpu",
        seed=7,
    )
    result = torch.load(output, map_location="cpu", weights_only=False)
    source_state = trainer.model.state_dict()
    assert metrics["old_optimizer_state_loaded"] == 0
    assert metrics["optimizer_step_values"] == [1]
    assert metrics["ppo_used"] == 0
    assert metrics["critic_updated"] == 0
    assert metrics["frozen_changed_tensors"] == []
    assert result["update_count"] == 1
    assert result["config"]["learning_rate"] == pytest.approx(1e-3)
    assert any(
        not torch.equal(source_state[name], value)
        for name, value in result["model"].items()
        if name.startswith(TARGET_TRANSFORMER_PREFIXES)
    )
    assert all(
        torch.equal(source_state[name], value)
        for name, value in result["model"].items()
        if not name.startswith(TARGET_TRANSFORMER_PREFIXES)
    )


def test_formal_joint_suffix_return_uses_dynamic_total_weight(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps({
            "steps": [{
                "step": 2,
                "causal_events": [{
                    "target_entity_id": 1,
                    "target_health_after": 8.0,
                }],
            }],
        }),
        encoding="utf-8",
    )
    row = {"trace": str(trace_path)}
    summary = {
        "score": {
            "objective_weights": {"1": 2.0, "2": 3.0},
            "total_objective_weight": 5.0,
        },
        "objectives": [
            {"id": 1, "initial_health": 10.0, "final_health": 0.0},
            {"id": 2, "initial_health": 20.0, "final_health": 10.0},
        ],
    }
    # At boundary step 2, target 1 has 8/10 health. Suffix damage is 0.8;
    # target 2 suffix damage is 0.5. Official weighted return is 0.62.
    assert formal_joint_suffix_return(row, 2, summary) == pytest.approx(0.62)


def test_signed_actor_advantage_is_separate_from_factual_value_target(tmp_path):
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=5,
            target_slots=3,
            hidden_dim=16,
            device="cpu",
        )
    )
    observations = torch.zeros(1, 5)
    coordinates = torch.zeros(1, 3, 2)
    valid = torch.ones(1, 3, dtype=torch.bool)
    inactive = torch.zeros(1, dtype=torch.bool)
    action_mask = HybridActionMask(
        presence=inactive,
        initial_position=inactive,
        search_position=inactive,
        retarget=inactive,
        target=torch.ones(1, dtype=torch.bool),
        maneuver=inactive,
        satellite=inactive,
    )
    output = trainer.model.act(observations, coordinates, valid, action_mask)
    rollout = HybridRolloutBatch(
        observations=observations,
        target_coordinates=coordinates,
        target_valid_mask=valid,
        action_mask=output.action_mask,
        actions=output.action,
        old_log_probs=output.log_prob.detach(),
        old_values=torch.zeros(1),
        rewards=torch.tensor([-0.25]),
        dones=torch.ones(1),
        next_observations=torch.zeros_like(observations),
        agent_ids=torch.zeros(1, dtype=torch.long),
        steps=torch.ones(1, dtype=torch.long),
        search_valid_mask=torch.ones(1, 16 * 12, dtype=torch.bool),
        critic_states=observations,
        next_critic_states=torch.zeros_like(observations),
        target_credits=torch.zeros(1, 3),
        credit_anchor_mask=torch.ones(1, dtype=torch.bool),
    )
    rollout_path = tmp_path / "rollout.pt"
    torch.save({"rollout": rollout}, rollout_path)

    merged = merge_rollouts(
        [rollout_path],
        anchor_group_ids=["seed-1"],
        normalize_actor_advantages=False,
        rollout_rewards_as_actor_advantages=True,
        episode_value_targets=[0.75],
    )

    torch.testing.assert_close(merged.actor_advantages, torch.tensor([-0.25]))
    torch.testing.assert_close(merged.value_targets, torch.tensor([0.75]))
