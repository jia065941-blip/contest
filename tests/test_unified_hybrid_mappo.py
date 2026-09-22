from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from experiments.unified_mappo.model import (
    HybridActionMask,
    HybridMAPPOConfig,
    HybridMAPPOTrainer,
    HybridRolloutBatch,
)


def _hybrid_mask(*, presence, initial_position, target, maneuver):
    inactive = torch.zeros_like(presence, dtype=torch.bool)
    return HybridActionMask(
        presence=presence,
        initial_position=initial_position,
        search_position=inactive,
        retarget=inactive,
        target=target,
        maneuver=maneuver,
        satellite=inactive,
    )


def test_hybrid_action_has_ten_semantic_dimensions_and_respects_target_mask():
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=16,
            target_slots=4,
            hidden_dim=32,
            device="cpu",
        )
    )
    observations = torch.zeros(5, 16)
    coordinates = torch.tensor(
        [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]] * 5
    )
    target_mask = torch.tensor([[True, False, True, False]] * 5)
    action_mask = _hybrid_mask(
        presence=torch.ones(5, dtype=torch.bool),
        initial_position=torch.ones(5, dtype=torch.bool),
        target=torch.ones(5, dtype=torch.bool),
        maneuver=torch.ones(5, dtype=torch.bool),
    )
    output = trainer.model.act(observations, coordinates, target_mask, action_mask)
    assert output.action.semantic_tensor().shape == (5, 10)
    assert set(output.action.target_index.tolist()).issubset({0, 2})
    assert torch.isfinite(output.log_prob).all()


def test_inactive_factors_do_not_change_joint_log_probability():
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=8,
            target_slots=2,
            hidden_dim=16,
            device="cpu",
        )
    )
    observations = torch.randn(3, 8)
    coordinates = torch.zeros(3, 2, 2)
    target_mask = torch.ones(3, 2, dtype=torch.bool)
    action_mask = _hybrid_mask(
        presence=torch.zeros(3, dtype=torch.bool),
        initial_position=torch.zeros(3, dtype=torch.bool),
        target=torch.zeros(3, dtype=torch.bool),
        maneuver=torch.ones(3, dtype=torch.bool),
    )
    output = trainer.model.act(observations, coordinates, target_mask, action_mask)
    expected = torch.distributions.Categorical(
        logits=trainer.model.distribution_parameters(observations)["maneuver_logits"]
    ).log_prob(output.action.maneuver_index)
    assert torch.allclose(output.log_prob, expected, atol=1e-6)


def test_checkpoint_round_trip_preserves_all_parameters(tmp_path):
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=8,
            target_slots=2,
            hidden_dim=16,
            device="cpu",
        )
    )
    checkpoint = tmp_path / "model.pt"
    trainer.save(checkpoint)
    restored = HybridMAPPOTrainer.load(checkpoint, device="cpu")
    for first, second in zip(trainer.model.parameters(), restored.model.parameters()):
        assert torch.equal(first, second)


def test_waiting_entity_zeroes_launch_and_option_factors():
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=8,
            target_slots=2,
            hidden_dim=16,
            device="cpu",
        )
    )
    with torch.no_grad():
        trainer.model.presence_head.bias.fill_(-100.0)
    output = trainer.model.act(
        torch.zeros(1, 8),
        torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]),
        torch.ones(1, 2, dtype=torch.bool),
        _hybrid_mask(
            presence=torch.ones(1, dtype=torch.bool),
            initial_position=torch.ones(1, dtype=torch.bool),
            target=torch.ones(1, dtype=torch.bool),
            maneuver=torch.zeros(1, dtype=torch.bool),
        ),
        deterministic=True,
    )
    assert output.action.presence.item() == 0
    assert not output.action_mask.initial_position.item()
    assert not output.action_mask.target.item()
    assert not output.action_mask.maneuver.item()
    assert torch.count_nonzero(output.action.target_xy).item() == 0
    recomputed, _, _ = trainer.model.evaluate_actions(
        torch.zeros(1, 8),
        torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]),
        torch.ones(1, 2, dtype=torch.bool),
        output.action_mask,
        output.action,
    )
    assert torch.allclose(output.log_prob, recomputed, atol=1e-6)



def test_lifecycle_policy_weights_balance_deployment_and_maneuver_mass():
    deployment_count = 2
    maneuver_count = 998
    mask = _hybrid_mask(
        presence=torch.tensor([True] * deployment_count + [False] * maneuver_count),
        initial_position=torch.zeros(1000, dtype=torch.bool),
        target=torch.zeros(1000, dtype=torch.bool),
        maneuver=torch.tensor([False] * deployment_count + [True] * maneuver_count),
    )
    weights = HybridMAPPOTrainer._lifecycle_policy_weights(mask, 0.5)
    assert torch.allclose(weights[:deployment_count].sum(), torch.tensor(500.0))
    assert torch.allclose(weights[deployment_count:].sum(), torch.tensor(500.0))


def test_value_only_waiting_rows_do_not_dilute_policy_weight_mass():
    mask = _hybrid_mask(
        presence=torch.tensor([True, False, False, False]),
        initial_position=torch.zeros(4, dtype=torch.bool),
        target=torch.zeros(4, dtype=torch.bool),
        maneuver=torch.tensor([False, True, False, False]),
    )
    weights = HybridMAPPOTrainer._lifecycle_policy_weights(mask, 0.5)
    assert torch.allclose(weights, torch.tensor([2.0, 2.0, 0.0, 0.0]))
    assert torch.allclose(weights.sum(), torch.tensor(4.0))



def test_on_field_presence_semantic_is_one_but_factor_stays_masked():
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=8,
            target_slots=2,
            hidden_dim=16,
            device="cpu",
        )
    )
    observations = torch.zeros(1, 8)
    coordinates = torch.zeros(1, 2, 2)
    target_mask = torch.ones(1, 2, dtype=torch.bool)
    on_field_mask = _hybrid_mask(
        presence=torch.zeros(1, dtype=torch.bool),
        initial_position=torch.zeros(1, dtype=torch.bool),
        target=torch.zeros(1, dtype=torch.bool),
        maneuver=torch.ones(1, dtype=torch.bool),
    )
    output = trainer.model.act(
        observations,
        coordinates,
        target_mask,
        on_field_mask,
        deterministic=True,
    )
    assert output.action.presence.item() == 1
    expected = torch.distributions.Categorical(
        logits=trainer.model.distribution_parameters(observations)["maneuver_logits"]
    ).log_prob(output.action.maneuver_index)
    assert torch.allclose(output.log_prob, expected, atol=1e-6)

    invalid = replace(
        output.action,
        presence=torch.zeros_like(output.action.presence),
    )
    with pytest.raises(ValueError, match="场上存活实体"):
        trainer.model.evaluate_actions(
            observations,
            coordinates,
            target_mask,
            output.action_mask,
            invalid,
        )


def test_per_agent_advantage_normalization_never_mixes_agent_statistics():
    advantages = torch.tensor([1.0, 3.0, 100.0, 104.0, 7.0])
    agent_ids = torch.tensor([0, 0, 1, 1, 2])
    normalized = HybridMAPPOTrainer._normalize_advantages(
        advantages,
        agent_ids,
        per_agent=True,
    )
    for indices in (torch.tensor([0, 1]), torch.tensor([2, 3])):
        values = normalized[indices]
        assert torch.allclose(values.mean(), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(values.std(unbiased=False), torch.tensor(1.0), atol=1e-6)
    assert normalized[4].item() == advantages[4].item()


def test_empirical_advantage_normalization_centers_batch_returns():
    returns = torch.tensor([0.0, 1.0, 3.0, 4.0])
    normalized = HybridMAPPOTrainer._normalize_empirical_advantages(returns)
    assert torch.allclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(
        normalized.std(unbiased=False), torch.tensor(1.0), atol=1e-6
    )
    assert normalized[:2].lt(0.0).all()
    assert normalized[2:].gt(0.0).all()


def test_constant_empirical_returns_have_no_actor_signal():
    returns = torch.full((8,), 0.25)
    normalized = HybridMAPPOTrainer._normalize_empirical_advantages(returns)
    assert torch.equal(normalized, torch.zeros_like(returns))


def test_strict_ctde_critic_concatenates_global_focal_and_identity_inputs():
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=5,
            critic_state_dim=7,
            critic_focal_observation_dim=5,
            max_agents=4,
            agent_embedding_dim=3,
            target_slots=2,
            hidden_dim=16,
            device="cpu",
        )
    )
    first_layer = trainer.model.critic_encoder[0]
    assert first_layer.in_features == 7 + 5 + 3
    states = torch.arange(7, dtype=torch.float32).reshape(1, 7)
    focal = torch.arange(5, dtype=torch.float32).reshape(1, 5)
    agent_ids = torch.tensor([2])
    actual = trainer.model.critic_value(
        states,
        agent_ids,
        focal,
    )
    identity = trainer.model.agent_embedding(agent_ids)
    concatenated = torch.cat((states, focal, identity), dim=-1)
    manual = trainer.model.value_head(trainer.model.critic_encoder(concatenated)).squeeze(-1)
    assert torch.allclose(actual, manual)
    with pytest.raises(ValueError, match="focal_observations"):
        trainer.model.critic_value(states, agent_ids)


def test_strict_ctde_v4_checkpoint_preserves_focal_contract(tmp_path):
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=5,
            critic_state_dim=7,
            critic_focal_observation_dim=5,
            max_agents=4,
            agent_embedding_dim=3,
            target_slots=2,
            hidden_dim=16,
            device="cpu",
        )
    )
    checkpoint = tmp_path / "strict.pt"
    trainer.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["algorithm"] == "target_conditioned_ctde_mappo_v10"
    restored = HybridMAPPOTrainer.load(checkpoint, device="cpu")
    assert (
        restored.config.critic_focal_observation_dim
        == 5
    )
    assert restored.model.critic_encoder[0].in_features == 7 + 5 + 3


def test_target_feature_expansion_zero_pads_weights_and_preserves_logits(tmp_path):
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=9,
            critic_state_dim=7,
            critic_focal_observation_dim=9,
            target_slots=4,
            target_feature_dim=20,
            hidden_dim=16,
            device="cpu",
        )
    )
    trainer.model.set_target_allocator_mix(0.25)
    observations = torch.randn(3, 9)
    old_features = torch.randn(3, 4, 20)
    valid = torch.ones(3, 4, dtype=torch.bool)
    old_logits = trainer.model.distribution_parameters(
        observations,
        old_features,
        valid,
    )["target_logits"]
    checkpoint = tmp_path / "target20.pt"
    trainer.save(checkpoint)

    restored = HybridMAPPOTrainer.load(
        checkpoint,
        device="cpu",
        target_feature_dim=26,
    )
    expanded = torch.cat((old_features, torch.randn(3, 4, 6)), dim=-1)
    new_logits = restored.model.distribution_parameters(
        observations,
        expanded,
        valid,
    )["target_logits"]
    assert restored.config.target_feature_dim == 26
    assert torch.equal(
        restored.model.target_item_encoder[0].weight[:, 20:],
        torch.zeros_like(restored.model.target_item_encoder[0].weight[:, 20:]),
    )
    assert torch.equal(old_logits, new_logits)

def test_strict_ctde_rollout_update_uses_focal_rows():
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=5,
            critic_state_dim=7,
            critic_focal_observation_dim=5,
            max_agents=4,
            agent_embedding_dim=3,
            target_slots=2,
            hidden_dim=16,
            update_epochs=1,
            minibatch_size=8,
            device="cpu",
        )
    )
    observations = torch.randn(4, 5)
    coordinates = torch.zeros(4, 2, 2)
    target_valid = torch.ones(4, 2, dtype=torch.bool)
    action_mask = _hybrid_mask(
        presence=torch.zeros(4, dtype=torch.bool),
        initial_position=torch.zeros(4, dtype=torch.bool),
        target=torch.zeros(4, dtype=torch.bool),
        maneuver=torch.ones(4, dtype=torch.bool),
    )
    output = trainer.model.act(observations, coordinates, target_valid, action_mask)
    agent_ids = torch.tensor([0, 0, 1, 1])
    critic_states = torch.randn(4, 7)
    old_values = trainer.model.critic_value(
        critic_states,
        agent_ids,
        observations,
    ).detach()
    rollout = HybridRolloutBatch(
        observations=observations,
        target_coordinates=coordinates,
        target_valid_mask=target_valid,
        action_mask=output.action_mask,
        actions=output.action,
        old_log_probs=output.log_prob.detach(),
        old_values=old_values,
        rewards=torch.tensor([1.0, 0.0, 10.0, 0.0]),
        dones=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        next_observations=torch.randn(4, 5),
        agent_ids=agent_ids,
        steps=torch.tensor([1, 2, 1, 2]),
        critic_states=critic_states,
        next_critic_states=torch.randn(4, 7),
        per_agent_advantage_normalization=True,
    )
    metrics = trainer.update(rollout)
    assert metrics["samples"] == 4.0
    assert metrics["per_agent_advantage_normalization"] == 1.0
    assert all(torch.isfinite(parameter).all() for parameter in trainer.model.parameters())


def test_target_allocator_mix_zero_is_exactly_function_preserving():
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=9,
            target_slots=4,
            target_feature_dim=6,
            hidden_dim=16,
            device="cpu",
        )
    )
    observations = torch.randn(7, 9)
    target_valid = torch.ones(7, 4, dtype=torch.bool)
    baseline = trainer.model.distribution_parameters(observations)
    features = torch.randn(7, 4, 6)
    with_features = trainer.model.distribution_parameters(observations, features, target_valid)
    for key in baseline:
        assert torch.equal(baseline[key], with_features[key])


def test_shared_target_allocator_is_permutation_equivariant():
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=9,
            target_slots=4,
            target_feature_dim=6,
            hidden_dim=16,
            device="cpu",
        )
    )
    trainer.model.set_target_allocator_mix(1.0)
    with torch.no_grad():
        trainer.model.target_score[-1].weight.normal_(mean=0.0, std=0.2)
    observations = torch.randn(2, 9)
    target_features = torch.randn(2, 4, 6)
    target_valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
    permutation = torch.tensor([2, 0, 3, 1])
    original = trainer.model.distribution_parameters(
        observations, target_features, target_valid
    )["target_logits"]
    permuted = trainer.model.distribution_parameters(
        observations,
        target_features[:, permutation],
        target_valid[:, permutation],
    )["target_logits"]
    assert torch.allclose(
        permuted,
        original[:, permutation],
        atol=1e-6,
        rtol=0.0,
    )


def test_counterfactual_value_scale_zero_is_exactly_function_preserving():
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=9,
            target_slots=4,
            target_feature_dim=6,
            hidden_dim=16,
            device="cpu",
        )
    )
    observations = torch.randn(5, 9)
    features = torch.randn(5, 4, 6)
    valid = torch.ones(5, 4, dtype=torch.bool)
    baseline = trainer.model.distribution_parameters(observations)
    with_features = trainer.model.distribution_parameters(observations, features, valid)
    assert torch.equal(baseline["target_logits"], with_features["target_logits"])


def test_shared_counterfactual_values_are_permutation_equivariant():
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=9,
            target_slots=4,
            target_feature_dim=6,
            hidden_dim=16,
            device="cpu",
        )
    )
    with torch.no_grad():
        trainer.model.target_cf_value_head.weight.normal_(mean=0.0, std=0.2)
    observations = torch.randn(2, 9)
    latent = trainer.model.encoder(observations)
    features = torch.randn(2, 4, 6)
    valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
    permutation = torch.tensor([2, 0, 3, 1])
    original = trainer.model.shared_target_cf_values(latent, features, valid)
    permuted = trainer.model.shared_target_cf_values(
        latent, features[:, permutation], valid[:, permutation]
    )
    assert torch.allclose(permuted, original[:, permutation], atol=1e-6, rtol=0.0)


def test_counterfactual_values_can_use_teacher_residual_context():
    trainer = HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=9,
            target_slots=4,
            target_feature_dim=6,
            hidden_dim=16,
            device="cpu",
        )
    )
    trainer.model.set_target_cf_context("teacher")
    with torch.no_grad():
        trainer.model.target_cf_value_head.weight.normal_(mean=0.0, std=0.2)
    observations = torch.randn(2, 9)
    latent = trainer.model.encoder(observations)
    features = torch.randn(2, 4, 6)
    valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
    permutation = torch.tensor([2, 0, 3, 1])
    original = trainer.model.shared_target_cf_values(latent, features, valid)
    permuted = trainer.model.shared_target_cf_values(
        latent, features[:, permutation], valid[:, permutation]
    )
    assert torch.allclose(permuted, original[:, permutation], atol=1e-6, rtol=0.0)


def _transformer_target_trainer() -> HybridMAPPOTrainer:
    return HybridMAPPOTrainer(
        HybridMAPPOConfig(
            observation_dim=9,
            critic_state_dim=7,
            critic_focal_observation_dim=9,
            target_slots=4,
            target_feature_dim=6,
            target_allocator_hidden_dim=16,
            target_selector_arch="transformer",
            target_transformer_heads=4,
            target_transformer_layers=2,
            target_transformer_ff_dim=32,
            hidden_dim=16,
            device="cpu",
        )
    )


def test_transformer_target_head_is_permutation_equivariant():
    trainer = _transformer_target_trainer()
    observations = torch.randn(3, 9)
    features = torch.randn(3, 4, 6)
    valid = torch.tensor([
        [True, True, False, True],
        [True, False, True, True],
        [False, True, True, True],
    ])
    permutation = torch.tensor([2, 0, 3, 1])
    original = trainer.model.distribution_parameters(
        observations, features, valid
    )["target_logits"]
    permuted = trainer.model.distribution_parameters(
        observations,
        features[:, permutation],
        valid[:, permutation],
    )["target_logits"]
    assert torch.allclose(
        permuted,
        original[:, permutation],
        atol=1e-6,
        rtol=0.0,
    )


def test_transformer_invalid_target_cannot_change_legal_logits():
    trainer = _transformer_target_trainer()
    observations = torch.randn(2, 9)
    features = torch.randn(2, 4, 6)
    valid = torch.tensor([
        [True, False, True, False],
        [False, True, True, False],
    ])
    baseline = trainer.model.distribution_parameters(
        observations, features, valid
    )["target_logits"]
    changed = features.clone()
    changed[~valid] = torch.randn_like(changed[~valid]) * 1e6
    modified = trainer.model.distribution_parameters(
        observations, changed, valid
    )["target_logits"]
    assert torch.allclose(
        modified[valid], baseline[valid], atol=1e-6, rtol=0.0
    )


def test_transformer_target_logits_ignore_all_legacy_and_residual_gates():
    trainer = _transformer_target_trainer()
    observations = torch.randn(2, 9)
    features = torch.randn(2, 4, 6)
    valid = torch.ones(2, 4, dtype=torch.bool)
    baseline = trainer.model.distribution_parameters(
        observations, features, valid
    )["target_logits"]
    trainer.model.set_target_allocator_mix(1.0)
    trainer.model.set_target_cf_policy_scale(100.0)
    trainer.model.set_target_teacher_policy_scale(100.0)
    with torch.no_grad():
        trainer.model.target_head.weight.normal_(std=10.0)
        trainer.model.target_score[-1].weight.normal_(std=10.0)
        trainer.model.target_cf_value_head.weight.normal_(std=10.0)
        trainer.model.target_teacher_score[-1].weight.normal_(std=10.0)
    actual = trainer.model.distribution_parameters(
        observations, features, valid
    )["target_logits"]
    assert torch.equal(actual, baseline)


def test_transformer_target_checkpoint_is_v11_and_round_trips(tmp_path):
    trainer = _transformer_target_trainer()
    observations = torch.randn(2, 9)
    features = torch.randn(2, 4, 6)
    valid = torch.ones(2, 4, dtype=torch.bool)
    expected = trainer.model.distribution_parameters(
        observations, features, valid
    )["target_logits"]
    checkpoint = tmp_path / "transformer.pt"
    trainer.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["algorithm"] == "target_conditioned_ctde_mappo_v11"
    restored = HybridMAPPOTrainer.load(checkpoint, device="cpu")
    assert restored.config.target_selector_arch == "transformer"
    actual = restored.model.distribution_parameters(
        observations, features, valid
    )["target_logits"]
    assert torch.equal(actual, expected)
