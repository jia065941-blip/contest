"""单一参数集合上的混合动作 MAPPO 分布与更新规则。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal


@dataclass(frozen=True)
class HybridMAPPOConfig:
    observation_dim: int = 128
    # None preserves the historical shared-observation model. A positive
    # value enables strict CTDE: actor input stays local, while the critic gets
    # global state, focal local observation, and stable agent identity.
    critic_state_dim: int | None = None
    # Strict CTDE also receives the focal agent's legal local actor observation.
    # Keeping this dimension explicit makes the critic/checkpoint contract
    # self-describing.
    critic_focal_observation_dim: int | None = None
    max_agents: int = 512
    agent_embedding_dim: int = 8
    target_slots: int = 8
    target_feature_dim: int = 26
    target_allocator_hidden_dim: int = 64
    target_allocator_mix: float = 0.0
    # "legacy" keeps historical checkpoints bit-compatible.  Fresh models
    # can select "transformer" to make the legal 26-D target set the sole
    # source of target logits, without a fixed-slot or residual path.
    target_selector_arch: str = "legacy"
    target_transformer_heads: int = 4
    target_transformer_layers: int = 2
    target_transformer_ff_dim: int = 128
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    deployment_policy_share: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 32768
    initial_coordinate_log_std: float = -2.0
    search_grid_width: int = 16
    search_grid_height: int = 12
    search_target_index: int = 7
    counterfactual_value_coef: float = 0.1
    seed: int = 7
    device: str = "auto"


@dataclass(frozen=True)
class HybridActionMask:
    """每个语义因子是否在当前时刻参与联合动作概率。"""

    presence: torch.Tensor
    initial_position: torch.Tensor
    search_position: torch.Tensor
    retarget: torch.Tensor
    target: torch.Tensor
    maneuver: torch.Tensor
    satellite: torch.Tensor

    def to(self, device: torch.device) -> "HybridActionMask":
        return HybridActionMask(
            presence=self.presence.to(device=device, dtype=torch.bool),
            initial_position=self.initial_position.to(device=device, dtype=torch.bool),
            search_position=self.search_position.to(device=device, dtype=torch.bool),
            retarget=self.retarget.to(device=device, dtype=torch.bool),
            target=self.target.to(device=device, dtype=torch.bool),
            maneuver=self.maneuver.to(device=device, dtype=torch.bool),
            satellite=self.satellite.to(device=device, dtype=torch.bool),
        )

    def validate(self, batch_size: int) -> None:
        for name, value in asdict(self).items():
            if value.shape != (batch_size,):
                raise ValueError(f"{name} mask shape {tuple(value.shape)} != {(batch_size,)}")


@dataclass(frozen=True)
class HybridAction:
    """混合外部动作及其概率重计算所需的内部离散变量。"""

    presence: torch.Tensor
    retarget: torch.Tensor
    initial_xy: torch.Tensor
    search_xy: torch.Tensor
    target_xy: torch.Tensor
    maneuver: torch.Tensor
    initial_raw: torch.Tensor
    search_index: torch.Tensor
    target_index: torch.Tensor
    maneuver_index: torch.Tensor
    satellite: torch.Tensor

    def semantic_tensor(self) -> torch.Tensor:
        return torch.cat(
            (
                self.presence.to(dtype=self.initial_xy.dtype).unsqueeze(-1),
                self.retarget.to(dtype=self.initial_xy.dtype).unsqueeze(-1),
                self.initial_xy,
                self.search_xy,
                self.target_xy,
                self.maneuver.to(dtype=self.initial_xy.dtype).unsqueeze(-1),
                self.satellite.to(dtype=self.initial_xy.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )


@dataclass(frozen=True)
class HybridPolicyOutput:
    action: HybridAction
    action_mask: HybridActionMask
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class JointTargetReturn:
    observations: torch.Tensor
    action_semantics: torch.Tensor
    target_index: int
    target_return: float


@dataclass(frozen=True)
class HybridRolloutBatch:
    observations: torch.Tensor
    target_coordinates: torch.Tensor
    target_valid_mask: torch.Tensor
    action_mask: HybridActionMask
    actions: HybridAction
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_observations: torch.Tensor
    agent_ids: torch.Tensor
    steps: torch.Tensor
    search_valid_mask: torch.Tensor | None = None
    target_features: torch.Tensor | None = None
    critic_states: torch.Tensor | None = None
    next_critic_states: torch.Tensor | None = None
    target_credits: torch.Tensor | None = None
    credit_anchor_mask: torch.Tensor | None = None
    joint_target_returns: tuple[JointTargetReturn, ...] = ()
    joint_target_mode: bool = False
    per_agent_advantage_normalization: bool = False
    actor_advantages: torch.Tensor | None = None
    actor_advantage_mask: torch.Tensor | None = None
    value_targets: torch.Tensor | None = None
    normalize_actor_advantages: bool = True
    actor_update_epochs: int | None = None
    critic_update_epochs: int | None = None
    actor_loss_scale: float = 1.0
    actor_learning_rate_scale: float = 1.0
    critic_learning_rate_scale: float = 1.0
    critic_beta1: float | None = None
    actor_beta1: float | None = None
    causal_advantage_projection: bool = False
    actor_position_kl_limit: float | None = None
    actor_joint_kl_limit: float | None = None
    actor_backtrack_factor: float = 0.5
    target_supervision_indices: torch.Tensor | None = None
    target_supervision_mask: torch.Tensor | None = None
    target_supervision_coef: float = 0.0
    initial_supervision_xy: torch.Tensor | None = None
    initial_supervision_mask: torch.Tensor | None = None
    initial_supervision_coef: float = 0.0

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[0])


class HybridMAPPOModel(nn.Module):
    """局部 actor 与可选独立 centralized critic 的混合动作模型。"""

    def __init__(self, config: HybridMAPPOConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.observation_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.presence_head = nn.Linear(config.hidden_dim, 1)
        self.retarget_head = nn.Linear(config.hidden_dim, 1)
        self.initial_mean_head = nn.Linear(config.hidden_dim, 2)
        self.search_head = nn.Linear(
            config.hidden_dim,
            config.search_grid_width * config.search_grid_height,
        )
        self.target_head = nn.Linear(config.hidden_dim, config.target_slots)
        if config.target_feature_dim <= 0 or config.target_allocator_hidden_dim <= 0:
            raise ValueError("target_feature_dim 和 target_allocator_hidden_dim 必须为正数")
        allocator_dim = int(config.target_allocator_hidden_dim)
        self.target_item_encoder = nn.Sequential(
            nn.Linear(config.target_feature_dim, allocator_dim),
            nn.Tanh(),
            nn.Linear(allocator_dim, allocator_dim),
            nn.Tanh(),
        )
        self.target_query = nn.Linear(config.hidden_dim, allocator_dim)
        self.target_score = nn.Sequential(
            nn.Linear(allocator_dim * 3, allocator_dim),
            nn.Tanh(),
            nn.Linear(allocator_dim, 1),
        )
        if config.target_selector_arch not in {"legacy", "transformer"}:
            raise ValueError(
                "target_selector_arch 必须为 legacy 或 transformer"
            )
        if config.target_transformer_heads <= 0:
            raise ValueError("target_transformer_heads 必须为正数")
        if allocator_dim % config.target_transformer_heads != 0:
            raise ValueError(
                "target_allocator_hidden_dim 必须能被 "
                "target_transformer_heads 整除"
            )
        if config.target_transformer_layers <= 0:
            raise ValueError("target_transformer_layers 必须为正数")
        if config.target_transformer_ff_dim <= 0:
            raise ValueError("target_transformer_ff_dim 必须为正数")
        self.target_transformer_item_encoder = nn.Sequential(
            nn.Linear(config.target_feature_dim, allocator_dim),
            nn.GELU(),
            nn.LayerNorm(allocator_dim),
        )
        self.target_transformer_query = nn.Sequential(
            nn.Linear(config.hidden_dim, allocator_dim),
            nn.Tanh(),
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=allocator_dim,
            nhead=int(config.target_transformer_heads),
            dim_feedforward=int(config.target_transformer_ff_dim),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.target_transformer_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=int(config.target_transformer_layers),
            norm=nn.LayerNorm(allocator_dim),
            enable_nested_tensor=False,
        )
        self.target_transformer_score = nn.Sequential(
            nn.LayerNorm(allocator_dim),
            nn.Linear(allocator_dim, 1),
        )
        # Independently gated shared scorer for teacher-allocation residuals.
        # Historical policy logits are exact while the scale remains zero.
        self.target_teacher_item_encoder = nn.Sequential(
            nn.Linear(config.target_feature_dim, allocator_dim),
            nn.Tanh(),
            nn.Linear(allocator_dim, allocator_dim),
            nn.Tanh(),
        )
        self.target_teacher_query = nn.Linear(config.hidden_dim, allocator_dim)
        self.target_teacher_score = nn.Sequential(
            nn.Linear(allocator_dim * 3, allocator_dim),
            nn.Tanh(),
            nn.Linear(allocator_dim, 1),
        )
        # A separate, low-capacity shared head predicts normalized exact
        # counterfactual return for every legal target.  Its policy scale is
        # zero by default, so adding the head is function preserving for all
        # historical checkpoints.
        self.target_cf_value_head = nn.Linear(allocator_dim * 3, 1)
        self.register_buffer(
            "target_allocator_mix",
            torch.tensor(float(config.target_allocator_mix), dtype=torch.float32),
        )
        self.register_buffer(
            "target_cf_policy_scale",
            torch.tensor(0.0, dtype=torch.float32),
        )
        self.register_buffer(
            "target_cf_context_mode",
            torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "target_teacher_policy_scale",
            torch.tensor(0.0, dtype=torch.float32),
        )
        self.maneuver_head = nn.Linear(config.hidden_dim, 3)
        self.satellite_head = nn.Linear(config.hidden_dim, 1)
        self.strict_ctde = config.critic_state_dim is not None
        if self.strict_ctde:
            if int(config.critic_state_dim or 0) <= 0:
                raise ValueError("critic_state_dim 必须为正数")
            if int(config.critic_focal_observation_dim or 0) <= 0:
                raise ValueError("critic_focal_observation_dim 必须为正数")
            if config.critic_focal_observation_dim != config.observation_dim:
                raise ValueError(
                    "critic_focal_observation_dim 必须与 actor observation_dim 一致"
                )
            if config.max_agents <= 0 or config.agent_embedding_dim <= 0:
                raise ValueError("max_agents 和 agent_embedding_dim 必须为正数")
            self.agent_embedding: nn.Embedding | None = nn.Embedding(
                config.max_agents,
                config.agent_embedding_dim,
            )
            self.critic_encoder: nn.Sequential | None = nn.Sequential(
                nn.Linear(
                    int(config.critic_state_dim)
                    + int(config.critic_focal_observation_dim)
                    + config.agent_embedding_dim,
                    config.hidden_dim,
                ),
                nn.Tanh(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.Tanh(),
            )
        else:
            self.agent_embedding = None
            self.critic_encoder = None
        self.value_head = nn.Linear(config.hidden_dim, 1)
        self.target_q_head = nn.Sequential(
            nn.Linear(config.hidden_dim + 10, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.target_slots),
        )
        self.initial_log_std = nn.Parameter(
            torch.full((2,), float(config.initial_coordinate_log_std))
        )
        self.reset_action_priors()
        nn.init.zeros_(self.target_q_head[-1].weight)
        nn.init.zeros_(self.target_q_head[-1].bias)
        nn.init.zeros_(self.target_score[-1].weight)
        nn.init.zeros_(self.target_score[-1].bias)
        nn.init.normal_(self.target_transformer_score[-1].weight, std=0.02)
        nn.init.zeros_(self.target_transformer_score[-1].bias)
        nn.init.zeros_(self.target_cf_value_head.weight)
        nn.init.zeros_(self.target_cf_value_head.bias)

    def reset_action_priors(self) -> None:
        """以高参与率、区域中心、均匀目标和直行建立有效初始策略。"""

        nn.init.constant_(self.presence_head.bias, 4.0)
        nn.init.constant_(self.retarget_head.bias, -4.0)
        nn.init.zeros_(self.initial_mean_head.bias)
        nn.init.zeros_(self.search_head.bias)
        nn.init.zeros_(self.target_head.bias)
        nn.init.zeros_(self.maneuver_head.bias)
        nn.init.zeros_(self.satellite_head.bias)
        nn.init.zeros_(self.presence_head.weight)
        nn.init.zeros_(self.retarget_head.weight)
        nn.init.zeros_(self.initial_mean_head.weight)
        nn.init.zeros_(self.search_head.weight)
        nn.init.zeros_(self.target_head.weight)
        nn.init.zeros_(self.maneuver_head.weight)
        nn.init.zeros_(self.satellite_head.weight)
        with torch.no_grad():
            self.maneuver_head.bias[1] = 1.0

    def set_target_allocator_mix(self, value: float) -> None:
        """Set the explicit legacy/shared-target interpolation coefficient."""

        mix = float(value)
        if not 0.0 <= mix <= 1.0:
            raise ValueError("target_allocator_mix 必须位于 [0,1]")
        self.target_allocator_mix.fill_(mix)

    def set_target_cf_policy_scale(self, value: float) -> None:
        """Set the logit bonus applied to normalized counterfactual values."""

        scale = float(value)
        if scale < 0.0:
            raise ValueError("target_cf_policy_scale 必须非负")
        self.target_cf_policy_scale.fill_(scale)

    def set_target_cf_context(self, mode: str) -> None:
        """Choose the frozen representation used by the exact-return head."""

        modes = {"shared": 0, "teacher": 1}
        if mode not in modes:
            raise ValueError("target counterfactual context 必须为 shared 或 teacher")
        self.target_cf_context_mode.fill_(modes[mode])

    def set_target_teacher_policy_scale(self, value: float) -> None:
        """Set the additive gate for the teacher-allocation residual scorer."""

        scale = float(value)
        if scale < 0.0:
            raise ValueError("target_teacher_policy_scale 必须非负")
        self.target_teacher_policy_scale.fill_(scale)

    def shared_target_context(
        self,
        latent: torch.Tensor,
        target_features: torch.Tensor,
        target_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build permutation-equivariant item/query/set representations."""

        expected = (
            latent.shape[0],
            self.config.target_slots,
            self.config.target_feature_dim,
        )
        if target_features.shape != expected:
            raise ValueError(
                f"target_features shape {tuple(target_features.shape)} != {expected}"
            )
        target_features = target_features.to(
            device=latent.device,
            dtype=latent.dtype,
        )
        if target_valid_mask is None:
            target_valid_mask = torch.ones(
                expected[:2], dtype=torch.bool, device=latent.device
            )
        else:
            target_valid_mask = target_valid_mask.to(
                device=latent.device, dtype=torch.bool
            )
        if target_valid_mask.shape != expected[:2]:
            raise ValueError("target_valid_mask 必须为 [batch,target_slots]")
        item_embeddings = self.target_item_encoder(target_features)
        valid_weights = target_valid_mask.to(dtype=latent.dtype).unsqueeze(-1)
        pooled = (item_embeddings * valid_weights).sum(dim=1) / valid_weights.sum(
            dim=1
        ).clamp_min(1.0)
        query = self.target_query(latent)
        context = pooled.unsqueeze(1).expand_as(item_embeddings)
        query = query.unsqueeze(1).expand_as(item_embeddings)
        return torch.cat((item_embeddings, query, context), dim=-1)

    def shared_target_logits(
        self,
        latent: torch.Tensor,
        target_features: torch.Tensor,
        target_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score an unordered legal target set with parameters shared by slots."""

        return self.target_score(
            self.shared_target_context(latent, target_features, target_valid_mask)
        ).squeeze(-1)

    def transformer_target_logits(
        self,
        latent: torch.Tensor,
        target_features: torch.Tensor,
        target_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Directly score an unordered legal target set with self-attention.

        The actor latent is a single state/query token.  Target slots receive
        no positional encoding, so permuting target tokens and their mask
        permutes the resulting logits.  Invalid targets are excluded as keys
        and values; they therefore cannot influence any legal target score.
        """

        expected = (
            latent.shape[0],
            self.config.target_slots,
            self.config.target_feature_dim,
        )
        if target_features.shape != expected:
            raise ValueError(
                f"target_features shape {tuple(target_features.shape)} != {expected}"
            )
        target_features = target_features.to(
            device=latent.device,
            dtype=latent.dtype,
        )
        if target_valid_mask is None:
            target_valid_mask = torch.ones(
                expected[:2], dtype=torch.bool, device=latent.device
            )
        else:
            target_valid_mask = target_valid_mask.to(
                device=latent.device, dtype=torch.bool
            )
        if target_valid_mask.shape != expected[:2]:
            raise ValueError("target_valid_mask 必须为 [batch,target_slots]")

        query = self.target_transformer_query(latent).unsqueeze(1)
        items = self.target_transformer_item_encoder(target_features)
        tokens = torch.cat((query, items), dim=1)
        query_is_valid = torch.ones(
            (latent.shape[0], 1), dtype=torch.bool, device=latent.device
        )
        token_is_valid = torch.cat((query_is_valid, target_valid_mask), dim=1)
        encoded = self.target_transformer_encoder(
            tokens,
            src_key_padding_mask=~token_is_valid,
        )
        return self.target_transformer_score(encoded[:, 1:]).squeeze(-1)

    def teacher_target_context(
        self,
        latent: torch.Tensor,
        target_features: torch.Tensor,
        target_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the independent 26-D residual item/query/set context."""

        expected = (
            latent.shape[0],
            self.config.target_slots,
            self.config.target_feature_dim,
        )
        if target_features.shape != expected:
            raise ValueError(
                f"target_features shape {tuple(target_features.shape)} != {expected}"
            )
        target_features = target_features.to(
            device=latent.device,
            dtype=latent.dtype,
        )
        if target_valid_mask is None:
            target_valid_mask = torch.ones(
                expected[:2], dtype=torch.bool, device=latent.device
            )
        else:
            target_valid_mask = target_valid_mask.to(
                device=latent.device, dtype=torch.bool
            )
        if target_valid_mask.shape != expected[:2]:
            raise ValueError("target_valid_mask 必须为 [batch,target_slots]")
        items = self.target_teacher_item_encoder(target_features)
        weights = target_valid_mask.to(dtype=latent.dtype).unsqueeze(-1)
        pooled = (items * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        query = self.target_teacher_query(latent)
        context = torch.cat((
            items,
            query.unsqueeze(1).expand_as(items),
            pooled.unsqueeze(1).expand_as(items),
        ), dim=-1)
        return context

    def teacher_target_logits(
        self,
        latent: torch.Tensor,
        target_features: torch.Tensor,
        target_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score targets with the independently gated teacher residual."""

        return self.target_teacher_score(self.teacher_target_context(
            latent, target_features, target_valid_mask
        )).squeeze(-1)

    def shared_target_cf_values(
        self,
        latent: torch.Tensor,
        target_features: torch.Tensor,
        target_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict normalized exact counterfactual return for each target."""

        return self.target_cf_value_head(
            self.target_cf_context(
                latent, target_features, target_valid_mask
            )
        ).squeeze(-1)

    def target_cf_context(
        self,
        latent: torch.Tensor,
        target_features: torch.Tensor,
        target_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the configured permutation-equivariant return context."""

        mode = int(self.target_cf_context_mode.item())
        if mode == 0:
            context = self.shared_target_context(
                latent, target_features, target_valid_mask
            )
        elif mode == 1:
            context = self.teacher_target_context(
                latent, target_features, target_valid_mask
            )
        else:
            raise RuntimeError(f"未知 target counterfactual context mode: {mode}")
        item, query, pooled = context.chunk(3, dim=-1)
        return torch.cat((item, item * query, item * pooled), dim=-1)

    def shared_target_cf_context(
        self,
        latent: torch.Tensor,
        target_features: torch.Tensor,
        target_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create low-rank state-conditioned features for target value."""

        context = self.shared_target_context(
            latent, target_features, target_valid_mask
        )
        item, query, pooled = context.chunk(3, dim=-1)
        return torch.cat((item, item * query, item * pooled), dim=-1)

    def distribution_parameters(
        self,
        observations: torch.Tensor,
        target_features: torch.Tensor | None = None,
        target_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return actor parameters without accepting centralized information.

        In strict-CTDE mode value is a zero placeholder. Values must be
        requested through critic_value; keeping global state out of this
        signature makes actor-side information leakage impossible.
        """

        latent = self.encoder(observations)
        legacy_target_logits = self.target_head(latent)
        if self.config.target_selector_arch == "transformer":
            if target_features is None:
                raise ValueError(
                    "Transformer 目标选择头必须提供 target_features"
                )
            target_logits = self.transformer_target_logits(
                latent, target_features, target_valid_mask
            )
            return {
                "presence_logits": self.presence_head(latent).squeeze(-1),
                "retarget_logits": self.retarget_head(latent).squeeze(-1),
                "initial_mean": self.initial_mean_head(latent),
                "search_logits": self.search_head(latent),
                "target_logits": target_logits,
                "legacy_target_logits": legacy_target_logits,
                "maneuver_logits": self.maneuver_head(latent),
                "satellite_logits": self.satellite_head(latent).squeeze(-1),
                "value": (
                    torch.zeros(
                        observations.shape[0],
                        dtype=observations.dtype,
                        device=observations.device,
                    )
                    if self.strict_ctde
                    else self.value_head(latent).squeeze(-1)
                ),
            }
        mix = float(self.target_allocator_mix.item())
        cf_policy_scale = float(self.target_cf_policy_scale.item())
        teacher_policy_scale = float(self.target_teacher_policy_scale.item())
        if mix <= 0.0 and cf_policy_scale <= 0.0 and teacher_policy_scale <= 0.0:
            target_logits = legacy_target_logits
        else:
            if target_features is None:
                raise ValueError(
                    "共享目标评分或反事实价值启用时必须提供 target_features"
                )
            if mix > 0.0:
                shared_logits = self.shared_target_logits(
                    latent, target_features, target_valid_mask
                )
                target_logits = shared_logits if mix >= 1.0 else legacy_target_logits + mix * (shared_logits - legacy_target_logits)
            else:
                target_logits = legacy_target_logits
            if cf_policy_scale > 0.0:
                target_logits = target_logits + cf_policy_scale * self.shared_target_cf_values(
                    latent, target_features, target_valid_mask
                )
            if teacher_policy_scale > 0.0:
                target_logits = target_logits + teacher_policy_scale * self.teacher_target_logits(
                    latent, target_features, target_valid_mask
                )
        return {
            "presence_logits": self.presence_head(latent).squeeze(-1),
            "retarget_logits": self.retarget_head(latent).squeeze(-1),
            "initial_mean": self.initial_mean_head(latent),
            "search_logits": self.search_head(latent),
            "target_logits": target_logits,
            "legacy_target_logits": legacy_target_logits,
            "maneuver_logits": self.maneuver_head(latent),
            "satellite_logits": self.satellite_head(latent).squeeze(-1),
            "value": (
                torch.zeros(
                    observations.shape[0],
                    dtype=observations.dtype,
                    device=observations.device,
                )
                if self.strict_ctde
                else self.value_head(latent).squeeze(-1)
            ),
        }

    def critic_value(
        self,
        critic_states: torch.Tensor,
        agent_ids: torch.Tensor,
        focal_observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one centralized value for every focal agent/sample row."""

        if (
            not self.strict_ctde
            or self.critic_encoder is None
            or self.agent_embedding is None
        ):
            if critic_states.shape[-1] != self.config.observation_dim:
                raise ValueError("共享 critic 输入维度与 observation_dim 不一致")
            return self.value_head(self.encoder(critic_states)).squeeze(-1)
        if (
            critic_states.ndim != 2
            or critic_states.shape[1] != self.config.critic_state_dim
        ):
            raise ValueError(
                "critic_states 必须为 "
                f"[batch,{self.config.critic_state_dim}]"
            )
        if agent_ids.shape != (critic_states.shape[0],):
            raise ValueError("agent_ids 必须为 [batch]")
        agent_ids = agent_ids.to(device=critic_states.device, dtype=torch.long)
        if focal_observations is None or focal_observations.shape != (
            critic_states.shape[0],
            int(self.config.critic_focal_observation_dim or 0),
        ):
            raise ValueError(
                "strict CTDE focal_observations 必须为 "
                f"[batch,{self.config.critic_focal_observation_dim}]"
            )
        focal_observations = focal_observations.to(
            device=critic_states.device,
            dtype=critic_states.dtype,
        )
        if torch.any(agent_ids < 0) or torch.any(
            agent_ids >= self.config.max_agents
        ):
            raise ValueError("agent_ids 超出 critic embedding 范围")
        identity = self.agent_embedding(agent_ids)
        latent = self.critic_encoder(
            torch.cat((critic_states, focal_observations, identity), dim=-1)
        )
        return self.value_head(latent).squeeze(-1)

    def target_action_values(
        self,
        observations: torch.Tensor,
        action_semantics: torch.Tensor,
    ) -> torch.Tensor:
        if action_semantics.shape != (observations.shape[0], 10):
            raise ValueError("action_semantics 必须为 [batch,10]")
        latent = self.encoder(observations)
        return self.target_q_head(torch.cat((latent, action_semantics), dim=-1))

    @staticmethod
    def _masked_logits(
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.shape != valid_mask.shape:
            raise ValueError(
                f"target mask shape {tuple(valid_mask.shape)} != logits shape {tuple(logits.shape)}"
            )
        if active_mask.shape != (logits.shape[0],):
            raise ValueError("target active mask 必须为 [batch]")
        if torch.any(active_mask & ~valid_mask.any(dim=-1)):
            raise ValueError("启用目标动作的样本至少需要一个合法目标槽位")
        effective_mask = valid_mask.clone()
        effective_mask[~active_mask, 0] = True
        masked = logits.masked_fill(
            ~effective_mask,
            torch.finfo(logits.dtype).min,
        )
        return masked, effective_mask

    def _masked_search_logits(
        self,
        logits: torch.Tensor,
        valid_mask: torch.Tensor | None,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_size = self.config.search_grid_width * self.config.search_grid_height
        if valid_mask is None:
            valid_mask = torch.ones(
                logits.shape[0], grid_size,
                dtype=torch.bool, device=logits.device,
            )
        else:
            valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
        if logits.shape != valid_mask.shape:
            raise ValueError(
                f"search mask shape {tuple(valid_mask.shape)} != logits shape {tuple(logits.shape)}"
            )
        if active_mask.shape != (logits.shape[0],):
            raise ValueError("search active mask 必须为 [batch]")
        if torch.any(active_mask & ~valid_mask.any(dim=-1)):
            raise ValueError("启用搜索动作的样本至少需要一个合法格点")
        effective_mask = valid_mask.clone()
        effective_mask[~active_mask, 0] = True
        masked = logits.masked_fill(
            ~effective_mask,
            torch.finfo(logits.dtype).min,
        )
        return masked, effective_mask

    @staticmethod
    def _squashed_normal_log_prob(
        distribution: Normal,
        raw_action: torch.Tensor,
    ) -> torch.Tensor:
        log_two = torch.log(torch.tensor(2.0, device=raw_action.device))
        log_jacobian = 2.0 * (
            log_two - raw_action - torch.nn.functional.softplus(-2.0 * raw_action)
        )
        return (distribution.log_prob(raw_action) - log_jacobian).sum(dim=-1)

    @torch.no_grad()
    def act(
        self,
        observations: torch.Tensor,
        target_coordinates: torch.Tensor,
        target_valid_mask: torch.Tensor,
        action_mask: HybridActionMask,
        *,
        target_features: torch.Tensor | None = None,
        search_valid_mask: torch.Tensor | None = None,
        deterministic: bool = False,
        independent_target: bool = False,
    ) -> HybridPolicyOutput:
        batch_size = int(observations.shape[0])
        action_mask = action_mask.to(observations.device)
        action_mask.validate(batch_size)
        if target_coordinates.shape != (batch_size, self.config.target_slots, 2):
            raise ValueError("target_coordinates 必须为 [batch,target_slots,2]")
        target_coordinates = target_coordinates.to(
            device=observations.device,
            dtype=observations.dtype,
        )
        target_valid_mask = target_valid_mask.to(
            device=observations.device,
            dtype=torch.bool,
        )
        parameters = self.distribution_parameters(
            observations, target_features, target_valid_mask
        )

        presence_distribution = Bernoulli(logits=parameters["presence_logits"])
        retarget_distribution = Bernoulli(logits=parameters["retarget_logits"])
        initial_std = self.initial_log_std.exp().expand_as(parameters["initial_mean"])
        initial_distribution = Normal(parameters["initial_mean"], initial_std)
        masked_search_logits, effective_search_valid_mask = (
            self._masked_search_logits(
                parameters["search_logits"],
                search_valid_mask,
                action_mask.search_position,
            )
        )
        search_distribution = Categorical(logits=masked_search_logits)
        masked_target_logits, _ = self._masked_logits(
            parameters["target_logits"],
            target_valid_mask,
            action_mask.target,
        )
        target_distribution = Categorical(logits=masked_target_logits)
        maneuver_distribution = Categorical(logits=parameters["maneuver_logits"])
        satellite_distribution = Bernoulli(logits=parameters["satellite_logits"])

        if deterministic:
            presence = (parameters["presence_logits"] >= 0).to(dtype=torch.long)
            retarget = (parameters["retarget_logits"] >= 0).to(dtype=torch.long)
            initial_raw = parameters["initial_mean"]
            search_index = torch.argmax(masked_search_logits, dim=-1)
            target_index = torch.argmax(target_distribution.logits, dim=-1)
            maneuver_index = torch.argmax(parameters["maneuver_logits"], dim=-1)
            satellite = (parameters["satellite_logits"] >= 0).to(dtype=torch.long)
        else:
            presence = presence_distribution.sample().to(dtype=torch.long)
            retarget = retarget_distribution.sample().to(dtype=torch.long)
            initial_raw = initial_distribution.sample()
            search_index = search_distribution.sample()
            target_index = target_distribution.sample()
            maneuver_index = maneuver_distribution.sample()
            satellite = satellite_distribution.sample().to(dtype=torch.long)

        # z is a semantic state/action component: an already-deployed living
        # entity is on field (z=1), while pre-launch WAIT remains z=0.
        presence = torch.where(action_mask.presence, presence, torch.ones_like(presence))
        entity_present = presence.to(dtype=torch.bool)
        retarget = torch.where(
            action_mask.retarget & entity_present,
            retarget,
            torch.zeros_like(retarget),
        )
        legacy_target = ~action_mask.presence & ~action_mask.retarget
        target_trigger = (
            torch.ones_like(action_mask.target)
            if independent_target
            else (
                legacy_target
                | (action_mask.presence & entity_present)
                | (action_mask.retarget & retarget.to(dtype=torch.bool))
            )
        )
        effective_action_mask = HybridActionMask(
            presence=action_mask.presence,
            initial_position=action_mask.initial_position & entity_present,
            search_position=(
                action_mask.search_position
                & entity_present
                & target_trigger
                & target_index.eq(int(self.config.search_target_index))
            ),
            retarget=action_mask.retarget & entity_present,
            target=action_mask.target & target_trigger,
            maneuver=action_mask.maneuver & entity_present,
            satellite=action_mask.satellite & entity_present,
        )
        initial_raw = torch.where(
            effective_action_mask.initial_position.unsqueeze(-1),
            initial_raw,
            torch.zeros_like(initial_raw),
        )
        initial_xy = torch.where(
            effective_action_mask.initial_position.unsqueeze(-1),
            torch.tanh(initial_raw),
            torch.zeros_like(initial_raw),
        )
        search_index = torch.where(
            effective_action_mask.search_position,
            search_index,
            torch.zeros_like(search_index),
        )
        search_x = (
            -1.0
            + (search_index.remainder(self.config.search_grid_width).to(
                dtype=observations.dtype
            ) + 0.5)
            * (2.0 / self.config.search_grid_width)
        )
        search_y = (
            -1.0
            + (search_index.div(
                self.config.search_grid_width,
                rounding_mode="floor",
            ).to(dtype=observations.dtype) + 0.5)
            * (2.0 / self.config.search_grid_height)
        )
        search_xy = torch.where(
            effective_action_mask.search_position.unsqueeze(-1),
            torch.stack((search_x, search_y), dim=-1),
            torch.zeros(batch_size, 2, dtype=observations.dtype, device=observations.device),
        )
        target_index = torch.where(
            effective_action_mask.target,
            target_index,
            torch.zeros_like(target_index),
        )
        maneuver_index = torch.where(
            effective_action_mask.maneuver,
            maneuver_index,
            torch.ones_like(maneuver_index),
        )
        satellite = torch.where(
            effective_action_mask.satellite,
            satellite,
            torch.zeros_like(satellite),
        )
        batch_indices = torch.arange(batch_size, device=observations.device)
        target_xy = torch.where(
            effective_action_mask.target.unsqueeze(-1),
            target_coordinates[batch_indices, target_index],
            torch.zeros(batch_size, 2, dtype=observations.dtype, device=observations.device),
        )
        maneuver = maneuver_index - 1
        action = HybridAction(
            presence=presence,
            retarget=retarget,
            initial_xy=initial_xy,
            search_xy=search_xy,
            target_xy=target_xy,
            maneuver=maneuver,
            initial_raw=initial_raw,
            search_index=search_index,
            target_index=target_index,
            maneuver_index=maneuver_index,
            satellite=satellite,
        )
        log_prob, entropy, value = self.evaluate_actions(
            observations,
            target_coordinates,
            target_valid_mask,
            effective_action_mask,
            action,
            target_features=target_features,
            search_valid_mask=effective_search_valid_mask,
        )
        return HybridPolicyOutput(
            action=action,
            action_mask=effective_action_mask,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
        )

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        target_coordinates: torch.Tensor,
        target_valid_mask: torch.Tensor,
        action_mask: HybridActionMask,
        actions: HybridAction,
        *,
        target_features: torch.Tensor | None = None,
        search_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(observations.shape[0])
        action_mask = action_mask.to(observations.device)
        action_mask.validate(batch_size)
        target_coordinates = target_coordinates.to(
            device=observations.device,
            dtype=observations.dtype,
        )
        target_valid_mask = target_valid_mask.to(
            device=observations.device,
            dtype=torch.bool,
        )
        actions = HybridAction(
            presence=actions.presence.detach().to(observations.device),
            retarget=actions.retarget.detach().to(observations.device),
            initial_xy=actions.initial_xy.detach().to(observations.device),
            search_xy=actions.search_xy.detach().to(observations.device),
            target_xy=actions.target_xy.detach().to(observations.device),
            maneuver=actions.maneuver.detach().to(observations.device),
            initial_raw=actions.initial_raw.detach().to(observations.device),
            search_index=actions.search_index.detach().to(observations.device),
            target_index=actions.target_index.detach().to(observations.device),
            maneuver_index=actions.maneuver_index.detach().to(observations.device),
            satellite=actions.satellite.detach().to(observations.device),
        )
        if torch.any((actions.target_index < 0) | (actions.target_index >= self.config.target_slots)):
            raise ValueError("target_index 超出目标槽位范围")
        grid_size = self.config.search_grid_width * self.config.search_grid_height
        if torch.any((actions.search_index < 0) | (actions.search_index >= grid_size)):
            raise ValueError("search_index 超出搜索格点范围")
        _, effective_search_valid_mask = self._masked_search_logits(
            torch.zeros(
                batch_size, grid_size,
                dtype=observations.dtype, device=observations.device,
            ),
            search_valid_mask,
            action_mask.search_position,
        )
        chosen_search_legal = effective_search_valid_mask.gather(
            1, actions.search_index.unsqueeze(-1)
        ).squeeze(-1)
        if torch.any(action_mask.search_position & ~chosen_search_legal):
            raise ValueError("search_index 不在当前合法可达格点掩码中")
        batch_indices = torch.arange(batch_size, device=observations.device)
        expected_initial_xy = torch.where(
            action_mask.initial_position.unsqueeze(-1),
            torch.tanh(actions.initial_raw),
            torch.zeros_like(actions.initial_xy),
        )
        expected_search_x = (
            -1.0
            + (actions.search_index.remainder(self.config.search_grid_width).to(
                dtype=observations.dtype
            ) + 0.5)
            * (2.0 / self.config.search_grid_width)
        )
        expected_search_y = (
            -1.0
            + (actions.search_index.div(
                self.config.search_grid_width,
                rounding_mode="floor",
            ).to(dtype=observations.dtype) + 0.5)
            * (2.0 / self.config.search_grid_height)
        )
        expected_search_xy = torch.where(
            action_mask.search_position.unsqueeze(-1),
            torch.stack((expected_search_x, expected_search_y), dim=-1),
            torch.zeros_like(actions.search_xy),
        )
        expected_target_xy = torch.where(
            action_mask.target.unsqueeze(-1),
            target_coordinates[batch_indices, actions.target_index],
            torch.zeros_like(actions.target_xy),
        )
        expected_maneuver = torch.where(
            action_mask.maneuver,
            actions.maneuver_index - 1,
            torch.zeros_like(actions.maneuver),
        )
        if not torch.allclose(actions.initial_xy, expected_initial_xy, atol=1e-6, rtol=0.0):
            raise ValueError("initial_xy 与 initial_raw 或生命周期掩码不一致")
        if not torch.allclose(actions.search_xy, expected_search_xy, atol=1e-6, rtol=0.0):
            raise ValueError("search_xy 与 search_index 或搜索掩码不一致")
        if not torch.allclose(actions.target_xy, expected_target_xy, atol=1e-6, rtol=0.0):
            raise ValueError("target_xy 与 target_index 或生命周期掩码不一致")
        if not torch.equal(actions.maneuver, expected_maneuver):
            raise ValueError("maneuver 与 maneuver_index 或生命周期掩码不一致")
        if torch.any((actions.presence != 0) & (actions.presence != 1)):
            raise ValueError("presence 语义值必须属于 {0,1}")
        if torch.any((actions.retarget != 0) & (actions.retarget != 1)):
            raise ValueError("retarget 语义值必须属于 {0,1}")
        if torch.any((~action_mask.retarget) & (actions.retarget != 0)):
            raise ValueError("非改目标决策行的 retarget 语义值必须为零")
        if torch.any((~action_mask.presence) & (actions.presence != 1)):
            raise ValueError("场上存活实体的 presence 语义值必须为一")
        parameters = self.distribution_parameters(
            observations, target_features, target_valid_mask
        )
        presence_distribution = Bernoulli(logits=parameters["presence_logits"])
        retarget_distribution = Bernoulli(logits=parameters["retarget_logits"])
        initial_std = self.initial_log_std.exp().expand_as(parameters["initial_mean"])
        initial_distribution = Normal(parameters["initial_mean"], initial_std)
        masked_search_logits, _ = self._masked_search_logits(
            parameters["search_logits"],
            effective_search_valid_mask,
            action_mask.search_position,
        )
        search_distribution = Categorical(logits=masked_search_logits)
        masked_target_logits, _ = self._masked_logits(
            parameters["target_logits"],
            target_valid_mask,
            action_mask.target,
        )
        target_distribution = Categorical(logits=masked_target_logits)
        maneuver_distribution = Categorical(logits=parameters["maneuver_logits"])
        satellite_distribution = Bernoulli(logits=parameters["satellite_logits"])

        presence_log_prob = presence_distribution.log_prob(actions.presence.to(dtype=torch.float32))
        retarget_log_prob = retarget_distribution.log_prob(actions.retarget.to(dtype=torch.float32))
        coordinate_log_prob = self._squashed_normal_log_prob(
            initial_distribution,
            actions.initial_raw,
        )
        search_log_prob = search_distribution.log_prob(actions.search_index)
        target_log_prob = target_distribution.log_prob(actions.target_index)
        maneuver_log_prob = maneuver_distribution.log_prob(actions.maneuver_index)
        satellite_log_prob = satellite_distribution.log_prob(
            actions.satellite.to(dtype=torch.float32)
        )

        presence_entropy = presence_distribution.entropy()
        retarget_entropy = retarget_distribution.entropy()
        entropy_raw = initial_distribution.rsample()
        coordinate_entropy = -self._squashed_normal_log_prob(
            initial_distribution,
            entropy_raw,
        )
        search_entropy = search_distribution.entropy()
        target_entropy = target_distribution.entropy()
        maneuver_entropy = maneuver_distribution.entropy()
        satellite_entropy = satellite_distribution.entropy()

        log_prob = (
            action_mask.presence * presence_log_prob
            + action_mask.retarget * retarget_log_prob
            + action_mask.initial_position * coordinate_log_prob
            + action_mask.search_position * search_log_prob
            + action_mask.target * target_log_prob
            + action_mask.maneuver * maneuver_log_prob
            + action_mask.satellite * satellite_log_prob
        )
        entropy = (
            action_mask.presence * presence_entropy
            + action_mask.retarget * retarget_entropy
            + action_mask.initial_position * coordinate_entropy
            + action_mask.search_position * search_entropy
            + action_mask.target * target_entropy
            + action_mask.maneuver * maneuver_entropy
            + action_mask.satellite * satellite_entropy
        )
        return log_prob, entropy, parameters["value"]


class HybridMAPPOTrainer:
    """对混合动作联合对数概率执行 PPO 裁剪更新。"""

    def __init__(self, config: HybridMAPPOConfig):
        self.config = config
        torch.manual_seed(config.seed)
        device_name = config.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.model = HybridMAPPOModel(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.update_count = 0
        self.transition_count = 0
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def resolve_runtime_device(device: str | torch.device) -> torch.device:
        """Resolve an update device without changing the checkpoint contract."""

        device_name = str(device)
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(device_name)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA PPO 更新，但当前 PyTorch 无法使用 CUDA")
        return resolved

    def move_runtime_device(self, device: str | torch.device) -> torch.device:
        """Move model/Adam moments while leaving ``config.device`` unchanged."""

        destination = self.resolve_runtime_device(device)
        if destination == self.device:
            return destination
        self.model.to(destination)
        for state in self.optimizer.state.values():
            for name, value in tuple(state.items()):
                # Adam keeps its scalar step counter on CPU unless the optimizer
                # is capturable/fused; moment tensors follow the parameters.
                if torch.is_tensor(value) and name != "step":
                    state[name] = value.to(destination)
        self.device = destination
        return destination

    @staticmethod
    def _lifecycle_policy_weights(
        action_mask: HybridActionMask,
        deployment_share: float,
    ) -> torch.Tensor:
        """使低频部署决策与高频机动决策具有指定的策略目标质量。"""

        deployment = (
            action_mask.presence
            | action_mask.initial_position
            | action_mask.search_position
            | action_mask.retarget
            | action_mask.target
        )
        # LAUNCH/RETARGET rows may also contain a simultaneous maneuver.
        # Treat the joint row as a lifecycle sample so the high-frequency
        # maneuver population cannot overwrite its sampling weight.
        maneuver = (action_mask.maneuver | action_mask.satellite) & ~deployment
        weights = torch.zeros_like(deployment, dtype=torch.float32)
        deployment_count = int(deployment.sum().item())
        maneuver_count = int(maneuver.sum().item())
        # Value-only rows for entities waiting to launch must not dilute the
        # policy objective averaged over the whole rollout batch.
        sample_count = max(1, int(deployment.numel()))
        if deployment_count and maneuver_count:
            share = min(1.0, max(0.0, float(deployment_share)))
            weights[deployment] = share * sample_count / deployment_count
            weights[maneuver] = (1.0 - share) * sample_count / maneuver_count
        elif deployment_count:
            weights[deployment] = sample_count / deployment_count
        elif maneuver_count:
            weights[maneuver] = sample_count / maneuver_count
        return weights

    def _advantages_and_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        agent_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        for agent_id in torch.unique(agent_ids).tolist():
            indices = torch.nonzero(agent_ids == agent_id, as_tuple=False).flatten()
            gae = torch.zeros((), device=self.device)
            for index in reversed(indices.tolist()):
                nonterminal = 1.0 - dones[index]
                delta = rewards[index] + self.config.gamma * next_values[index] * nonterminal - values[index]
                gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * gae
                advantages[index] = gae
        return advantages, advantages + values

    @staticmethod
    def _normalize_advantages(
        advantages: torch.Tensor,
        agent_ids: torch.Tensor,
        *,
        per_agent: bool,
    ) -> torch.Tensor:
        """Normalize without mixing one agent's return scale into another's."""

        if advantages.shape != agent_ids.shape:
            raise ValueError("advantages 与 agent_ids 必须具有相同的一维形状")
        normalized = advantages.clone()
        if not per_agent:
            if advantages.numel() > 1:
                normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return normalized
        for agent_id in torch.unique(agent_ids).tolist():
            indices = torch.nonzero(
                agent_ids == agent_id,
                as_tuple=False,
            ).flatten()
            # A singleton has no temporal variance. Preserve its unscaled
            # advantage rather than borrowing statistics from other agents.
            if indices.numel() < 2:
                continue
            values = advantages[indices]
            std = values.std(unbiased=False)
            if not bool(torch.isfinite(std).item()) or float(std.item()) <= 1e-8:
                continue
            normalized[indices] = (values - values.mean()) / (std + 1e-8)
        return normalized

    @staticmethod
    def _normalize_empirical_advantages(
        advantages: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Center batch returns before using them as policy advantages.

        An action-independent batch baseline preserves the policy-gradient
        expectation while ensuring below-average empirical returns receive a
        negative learning signal. Constant batches contain no comparative
        actor signal and therefore normalize to zero.
        """

        active = (
            torch.ones_like(advantages, dtype=torch.bool)
            if mask is None else mask.to(dtype=torch.bool)
        )
        if active.shape != advantages.shape:
            raise ValueError("经验优势掩码必须与 advantages 同形")
        normalized = torch.zeros_like(advantages)
        values = advantages[active]
        if values.numel() == 0:
            return normalized
        centered = values - values.mean()
        scale = centered.std(unbiased=False)
        if not bool(torch.isfinite(scale).item()) or float(scale.item()) <= 1e-8:
            return normalized
        normalized[active] = centered / (scale + 1e-8)
        return normalized

    @torch.no_grad()
    def _actor_exact_kls(
        self,
        old_parameters: dict[str, torch.Tensor],
        old_initial_std: torch.Tensor,
        observations: torch.Tensor,
        target_valid_mask: torch.Tensor,
        search_valid_mask: torch.Tensor,
        action_mask: HybridActionMask,
        target_features: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compute masked analytic KLs from the frozen behavior policy."""

        new_parameters = self.model.distribution_parameters(
            observations, target_features, target_valid_mask
        )
        old_target_logits, _ = self.model._masked_logits(
            old_parameters["target_logits"],
            target_valid_mask,
            action_mask.target,
        )
        new_target_logits, _ = self.model._masked_logits(
            new_parameters["target_logits"],
            target_valid_mask,
            action_mask.target,
        )
        old_search_logits, _ = self.model._masked_search_logits(
            old_parameters["search_logits"],
            search_valid_mask,
            action_mask.search_position,
        )
        new_search_logits, _ = self.model._masked_search_logits(
            new_parameters["search_logits"],
            search_valid_mask,
            action_mask.search_position,
        )
        new_initial_std = self.model.initial_log_std.exp().expand_as(
            new_parameters["initial_mean"]
        )
        per_sample = {
            "presence": torch.distributions.kl_divergence(
                Bernoulli(logits=old_parameters["presence_logits"]),
                Bernoulli(logits=new_parameters["presence_logits"]),
            ),
            "retarget": torch.distributions.kl_divergence(
                Bernoulli(logits=old_parameters["retarget_logits"]),
                Bernoulli(logits=new_parameters["retarget_logits"]),
            ),
            "position": torch.distributions.kl_divergence(
                Normal(old_parameters["initial_mean"], old_initial_std),
                Normal(new_parameters["initial_mean"], new_initial_std),
            ).sum(dim=-1),
            "search": torch.distributions.kl_divergence(
                Categorical(logits=old_search_logits),
                Categorical(logits=new_search_logits),
            ),
            "target": torch.distributions.kl_divergence(
                Categorical(logits=old_target_logits),
                Categorical(logits=new_target_logits),
            ),
            "maneuver": torch.distributions.kl_divergence(
                Categorical(logits=old_parameters["maneuver_logits"]),
                Categorical(logits=new_parameters["maneuver_logits"]),
            ),
            "satellite": torch.distributions.kl_divergence(
                Bernoulli(logits=old_parameters["satellite_logits"]),
                Bernoulli(logits=new_parameters["satellite_logits"]),
            ),
        }
        masks = {
            "presence": action_mask.presence,
            "retarget": action_mask.retarget,
            "position": action_mask.initial_position,
            "search": action_mask.search_position,
            "target": action_mask.target,
            "maneuver": action_mask.maneuver,
            "satellite": action_mask.satellite,
        }
        metrics: dict[str, float] = {}
        joint_kl = torch.zeros_like(per_sample["presence"])
        for name, values in per_sample.items():
            mask = masks[name]
            joint_kl = joint_kl + mask * values
            denominator = mask.sum().clamp_min(1)
            metrics[f"{name}_head_exact_kl"] = float(
                (mask * values).sum().div(denominator).item()
            )
        metrics["joint_exact_kl"] = float(joint_kl.mean().item())
        metrics["position_active_samples"] = float(
            action_mask.initial_position.sum().item()
        )
        return metrics

    def update(self, rollout: HybridRolloutBatch) -> dict[str, float]:
        observations = rollout.observations.to(self.device)
        next_observations = rollout.next_observations.to(self.device)
        target_coordinates = rollout.target_coordinates.to(
            self.device,
            dtype=torch.float32,
        )
        raw_target_features = getattr(rollout, "target_features", None)
        target_features = (
            None if raw_target_features is None
            else raw_target_features.to(self.device, dtype=observations.dtype)
        )
        if target_features is None and (
            float(self.model.target_allocator_mix.item()) > 0.0
            or float(self.model.target_cf_policy_scale.item()) > 0.0
        ):
            raise ValueError("target allocator PPO rollout 缺少 target_features")
        expected_target_feature_shape = (rollout.batch_size, self.config.target_slots, self.config.target_feature_dim)
        target_valid_mask = rollout.target_valid_mask.to(self.device, dtype=torch.bool)
        raw_search_valid_mask = getattr(rollout, "search_valid_mask", None)
        search_grid_size = (
            self.config.search_grid_width * self.config.search_grid_height
        )
        search_valid_mask = (
            torch.ones(
                rollout.batch_size,
                search_grid_size,
                dtype=torch.bool,
                device=self.device,
            )
            if raw_search_valid_mask is None
            else raw_search_valid_mask.to(self.device, dtype=torch.bool)
        )
        if search_valid_mask.shape != (rollout.batch_size, search_grid_size):
            raise ValueError(
                "search_valid_mask 必须为 "
                f"[batch,{search_grid_size}]，实际为 {tuple(search_valid_mask.shape)}"
            )
        action_mask = rollout.action_mask.to(self.device)
        actions = HybridAction(
            presence=rollout.actions.presence.detach().to(self.device),
            retarget=rollout.actions.retarget.detach().to(self.device),
            initial_xy=rollout.actions.initial_xy.detach().to(self.device),
            search_xy=rollout.actions.search_xy.detach().to(self.device),
            target_xy=rollout.actions.target_xy.detach().to(self.device),
            maneuver=rollout.actions.maneuver.detach().to(self.device),
            initial_raw=rollout.actions.initial_raw.detach().to(self.device),
            search_index=rollout.actions.search_index.detach().to(self.device),
            target_index=rollout.actions.target_index.detach().to(self.device),
            maneuver_index=rollout.actions.maneuver_index.detach().to(self.device),
            satellite=rollout.actions.satellite.detach().to(self.device),
        )
        old_log_probs = rollout.old_log_probs.detach().to(self.device)
        old_values = rollout.old_values.detach().to(self.device)
        rewards = rollout.rewards.to(self.device)
        target_credits = (
            rollout.target_credits.to(self.device)
            if rollout.target_credits is not None
            else torch.zeros(
                rollout.batch_size,
                self.config.target_slots,
                dtype=rewards.dtype,
                device=self.device,
            )
        )
        credit_anchor_mask = (
            rollout.credit_anchor_mask.to(self.device, dtype=torch.bool)
            if rollout.credit_anchor_mask is not None
            else action_mask.presence
        )
        if credit_anchor_mask.shape != (rollout.batch_size,):
            raise ValueError("credit_anchor_mask 必须为 [batch]")
        raw_target_supervision_indices = getattr(
            rollout, "target_supervision_indices", None
        )
        target_supervision_indices = (
            raw_target_supervision_indices.to(self.device, dtype=torch.long)
            if raw_target_supervision_indices is not None
            else torch.full(
                (rollout.batch_size,), -1, dtype=torch.long, device=self.device
            )
        )
        raw_target_supervision_mask = getattr(
            rollout, "target_supervision_mask", None
        )
        target_supervision_mask = (
            raw_target_supervision_mask.to(self.device, dtype=torch.bool)
            if raw_target_supervision_mask is not None
            else torch.zeros(
                rollout.batch_size, dtype=torch.bool, device=self.device
            )
        )
        if target_supervision_indices.shape != (rollout.batch_size,):
            raise ValueError("target_supervision_indices 必须为 [batch]")
        if target_supervision_mask.shape != (rollout.batch_size,):
            raise ValueError("target_supervision_mask 必须为 [batch]")
        supervised_indices = target_supervision_indices[target_supervision_mask]
        if bool((supervised_indices < 0).any().item()) or bool(
            (supervised_indices >= self.config.target_slots).any().item()
        ):
            raise ValueError("target supervision 槽位超出范围")
        safe_supervision_indices = target_supervision_indices.clamp(min=0)
        supervised_legal = target_valid_mask.gather(
            1, safe_supervision_indices.unsqueeze(-1)
        ).squeeze(-1)
        if bool((target_supervision_mask & ~action_mask.target).any().item()) or bool(
            (target_supervision_mask & ~supervised_legal).any().item()
        ):
            raise ValueError("target supervision 只能作用于合法目标边界")
        target_supervision_coef = float(
            getattr(rollout, "target_supervision_coef", 0.0)
        )
        if target_supervision_coef < 0.0:
            raise ValueError("target_supervision_coef 必须非负")
        raw_initial_supervision_xy = getattr(
            rollout, "initial_supervision_xy", None
        )
        initial_supervision_xy = (
            raw_initial_supervision_xy.to(self.device, dtype=observations.dtype)
            if raw_initial_supervision_xy is not None
            else torch.zeros(
                (rollout.batch_size, 2),
                dtype=observations.dtype,
                device=self.device,
            )
        )
        raw_initial_supervision_mask = getattr(
            rollout, "initial_supervision_mask", None
        )
        initial_supervision_mask = (
            raw_initial_supervision_mask.to(self.device, dtype=torch.bool)
            if raw_initial_supervision_mask is not None
            else torch.zeros(
                rollout.batch_size, dtype=torch.bool, device=self.device
            )
        )
        if initial_supervision_xy.shape != (rollout.batch_size, 2):
            raise ValueError("initial_supervision_xy 必须为 [batch,2]")
        if initial_supervision_mask.shape != (rollout.batch_size,):
            raise ValueError("initial_supervision_mask 必须为 [batch]")
        if bool((initial_supervision_mask & ~action_mask.initial_position).any().item()):
            raise ValueError("initial supervision 只能作用于合法 LAUNCH 位置边界")
        supervised_initial_xy = initial_supervision_xy[initial_supervision_mask]
        if bool(((supervised_initial_xy < -1.0) | (supervised_initial_xy > 1.0)).any().item()):
            raise ValueError("initial supervision 坐标必须位于 [-1,1]")
        initial_supervision_coef = float(
            getattr(rollout, "initial_supervision_coef", 0.0)
        )
        if initial_supervision_coef < 0.0:
            raise ValueError("initial_supervision_coef 必须非负")
        dones = rollout.dones.to(self.device)
        if rollout.agent_ids.shape != (rollout.batch_size,):
            raise ValueError("agent_ids 必须为 [batch]，且每条样本必须显式标注实体")
        if rollout.steps.shape != (rollout.batch_size,):
            raise ValueError("steps 必须为 [batch]，且每条样本必须显式标注时刻")
        agent_ids = rollout.agent_ids.to(self.device, dtype=torch.long)
        rollout_steps = rollout.steps.to(self.device)
        if self.model.strict_ctde:
            if rollout.critic_states is None or rollout.next_critic_states is None:
                raise ValueError("严格 CTDE 必须为每条样本提供 critic_states")
            critic_states = rollout.critic_states.to(
                self.device,
                dtype=torch.float32,
            )
            next_critic_states = rollout.next_critic_states.to(
                self.device,
                dtype=torch.float32,
            )
            expected_shape = (
                rollout.batch_size,
                int(self.config.critic_state_dim or 0),
            )
            if critic_states.shape != expected_shape:
                raise ValueError(
                    f"critic_states shape {tuple(critic_states.shape)} != {expected_shape}"
                )
            if next_critic_states.shape != expected_shape:
                raise ValueError(
                    "next_critic_states shape "
                    f"{tuple(next_critic_states.shape)} != {expected_shape}"
                )
            old_values = self.model.critic_value(
                critic_states, agent_ids, observations
            ).detach()
        else:
            critic_states = observations
            next_critic_states = next_observations
        lifecycle_weights = self._lifecycle_policy_weights(
            action_mask,
            self.config.deployment_policy_share,
        )

        with torch.no_grad():
            next_values = self.model.critic_value(
                next_critic_states, agent_ids, next_observations
            )
            critic_advantages, critic_returns = self._advantages_and_returns(
                rewards, dones, old_values, next_values, agent_ids
            )
            returns = (
                rollout.value_targets.to(self.device, dtype=rewards.dtype)
                if rollout.value_targets is not None
                else critic_returns
            )
            if returns.shape != (rollout.batch_size,):
                raise ValueError("value_targets 必须为 [batch]")
            empirical_actor_advantage = rollout.actor_advantages is not None
            advantages = (
                rollout.actor_advantages.to(self.device, dtype=rewards.dtype)
                if empirical_actor_advantage
                else critic_advantages
            )
            if advantages.shape != (rollout.batch_size,):
                raise ValueError("actor_advantages 必须为 [batch]")
            actor_advantage_mask = (
                rollout.actor_advantage_mask.to(self.device, dtype=torch.bool)
                if rollout.actor_advantage_mask is not None
                else torch.ones(
                    rollout.batch_size, dtype=torch.bool, device=self.device
                )
            )
            if actor_advantage_mask.shape != (rollout.batch_size,):
                raise ValueError("actor_advantage_mask 必须为 [batch]")
            raw_advantages = advantages.clone()
            raw_active_advantages = raw_advantages[actor_advantage_mask]
            raw_advantage_mean = float(
                raw_active_advantages.mean().item()
                if raw_active_advantages.numel() else 0.0
            )
            raw_advantage_std = float(
                raw_active_advantages.std(unbiased=False).item()
                if raw_active_advantages.numel() else 0.0
            )
            critic_advantage_mean = float(critic_advantages.mean().item())
            pre_actor_parameters = self.model.distribution_parameters(
                observations, target_features, target_valid_mask
            )
            pre_initial_mean = pre_actor_parameters["initial_mean"].detach()
            pre_initial_std = self.model.initial_log_std.exp().expand_as(
                pre_initial_mean
            ).detach()
            if empirical_actor_advantage:
                if rollout.normalize_actor_advantages:
                    advantages = self._normalize_empirical_advantages(
                        advantages, actor_advantage_mask
                    )
                else:
                    advantages = torch.where(
                        actor_advantage_mask,
                        advantages,
                        torch.zeros_like(advantages),
                    )
            else:
                advantages = self._normalize_advantages(
                    advantages,
                    agent_ids,
                    per_agent=rollout.per_agent_advantage_normalization,
                )
            causal_projected_mask = (
                actor_advantage_mask
                & credit_anchor_mask
                & rewards.eq(0.0)
                & advantages.gt(0.0)
                if rollout.causal_advantage_projection
                else torch.zeros_like(credit_anchor_mask)
            )
            causal_projected_positive_mass = float(
                advantages[causal_projected_mask].sum().item()
            )
            advantages = torch.where(
                causal_projected_mask,
                torch.zeros_like(advantages),
                advantages,
            )
            effective_active_advantages = advantages[actor_advantage_mask]
            effective_advantage_mean = float(
                effective_active_advantages.mean().item()
                if effective_active_advantages.numel() else 0.0
            )
            effective_advantage_std = float(
                effective_active_advantages.std(unbiased=False).item()
                if effective_active_advantages.numel() else 0.0
            )
            effective_advantage_positive_count = int(
                effective_active_advantages.gt(1e-12).sum().item()
            )
            effective_advantage_negative_count = int(
                effective_active_advantages.lt(-1e-12).sum().item()
            )
            actor_signal_present = (
                not (
                    empirical_actor_advantage
                    or rollout.causal_advantage_projection
                )
                or bool(effective_active_advantages.ne(0.0).any().item())
            )
            critic_target_mae = float((old_values - returns).abs().mean().item())

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "target_q_loss": 0.0,
            "target_supervision_loss": 0.0,
            "initial_supervision_loss": 0.0,
        }
        steps = 0
        actor_steps = 0
        actor_candidate_attempts = 0
        actor_trust_region_backtrack_count = 0
        accepted_actor_learning_rate_scale = 0.0
        sample_count = rollout.batch_size
        actor_update_epochs = (
            self.config.update_epochs
            if rollout.actor_update_epochs is None
            else int(rollout.actor_update_epochs)
        )
        target_supervision_active = (
            target_supervision_coef > 0.0
            and bool(target_supervision_mask.any().item())
        )
        initial_supervision_active = (
            initial_supervision_coef > 0.0
            and bool(initial_supervision_mask.any().item())
        )
        if not actor_signal_present and not (
            target_supervision_active or initial_supervision_active
        ):
            actor_update_epochs = 0
        critic_update_epochs = (
            self.config.update_epochs
            if rollout.critic_update_epochs is None
            else int(rollout.critic_update_epochs)
        )
        if self.model.strict_ctde:
            critic_prefixes = (
                "agent_embedding.",
                "critic_encoder.",
                "value_head.",
            )
            actor_parameters = [
                parameter
                for name, parameter in self.model.named_parameters()
                if not name.startswith(critic_prefixes)
            ]
            critic_parameters = [
                parameter
                for name, parameter in self.model.named_parameters()
                if name.startswith(critic_prefixes)
            ]
        critic_steps = 0
        for update_epoch in range(max(
            actor_update_epochs,
            critic_update_epochs,
        )):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                sliced_mask = HybridActionMask(
                    presence=action_mask.presence[indices],
                    initial_position=action_mask.initial_position[indices],
                    search_position=action_mask.search_position[indices],
                    retarget=action_mask.retarget[indices],
                    target=action_mask.target[indices],
                    maneuver=action_mask.maneuver[indices],
                    satellite=action_mask.satellite[indices],
                )
                sliced_actions = HybridAction(
                    presence=actions.presence[indices],
                    retarget=actions.retarget[indices],
                    initial_xy=actions.initial_xy[indices],
                    search_xy=actions.search_xy[indices],
                    target_xy=actions.target_xy[indices],
                    maneuver=actions.maneuver[indices],
                    initial_raw=actions.initial_raw[indices],
                    search_index=actions.search_index[indices],
                    target_index=actions.target_index[indices],
                    maneuver_index=actions.maneuver_index[indices],
                    satellite=actions.satellite[indices],
                )
                new_log_probs, entropy, actor_value_placeholder = self.model.evaluate_actions(
                    observations[indices],
                    target_coordinates[indices],
                    target_valid_mask[indices],
                    sliced_mask,
                    sliced_actions,
                    target_features=None if target_features is None else target_features[indices],
                    search_valid_mask=search_valid_mask[indices],
                )
                predicted_values = (
                    self.model.critic_value(
                        critic_states[indices],
                        agent_ids[indices],
                        observations[indices],
                    )
                    if self.model.strict_ctde
                    else actor_value_placeholder
                )
                if self.model.strict_ctde or rollout.joint_target_mode:
                    target_q_loss = predicted_values.sum() * 0.0
                else:
                    action_semantics = sliced_actions.semantic_tensor().to(
                        self.device,
                        dtype=observations.dtype,
                    )
                    predicted_target_q = self.model.target_action_values(
                        observations[indices],
                        action_semantics,
                    )
                    null_target_q = self.model.target_action_values(
                        observations[indices],
                        torch.zeros_like(action_semantics),
                    )
                    credit_anchor_rows = credit_anchor_mask[indices]
                    if bool(credit_anchor_rows.any().item()):
                        target_q_loss = (
                            torch.nn.functional.mse_loss(
                                predicted_target_q[credit_anchor_rows],
                                target_credits[indices][credit_anchor_rows],
                            )
                            + 0.1 * null_target_q[credit_anchor_rows].square().mean()
                        )
                    else:
                        target_q_loss = predicted_target_q.sum() * 0.0
                supervision_rows = target_supervision_mask[indices]
                if (
                    target_supervision_coef > 0.0
                    and bool(supervision_rows.any().item())
                ):
                    target_parameters = self.model.distribution_parameters(
                        observations[indices],
                        None if target_features is None else target_features[indices],
                        target_valid_mask[indices],
                    )
                    supervised_target_logits, _ = self.model._masked_logits(
                        target_parameters["target_logits"],
                        target_valid_mask[indices],
                        sliced_mask.target,
                    )
                    target_supervision_loss = torch.nn.functional.cross_entropy(
                        supervised_target_logits[supervision_rows],
                        target_supervision_indices[indices][supervision_rows],
                    )
                else:
                    target_supervision_loss = predicted_values.sum() * 0.0
                initial_supervision_rows = initial_supervision_mask[indices]
                if (
                    initial_supervision_coef > 0.0
                    and bool(initial_supervision_rows.any().item())
                ):
                    initial_parameters = self.model.distribution_parameters(
                        observations[indices],
                        None if target_features is None else target_features[indices],
                        target_valid_mask[indices],
                    )
                    supervised_initial_raw = torch.atanh(
                        initial_supervision_xy[indices][initial_supervision_rows]
                        .clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
                    )
                    initial_supervision_loss = torch.nn.functional.mse_loss(
                        initial_parameters["initial_mean"][initial_supervision_rows],
                        supervised_initial_raw,
                    )
                else:
                    initial_supervision_loss = predicted_values.sum() * 0.0
                log_ratio = new_log_probs - old_log_probs[indices]
                ratio = torch.exp(log_ratio)
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantages[indices]
                policy_loss = -(
                    torch.minimum(unclipped, clipped) * lifecycle_weights[indices]
                ).mean()
                value_loss = torch.nn.functional.mse_loss(predicted_values, returns[indices])
                entropy_mean = (entropy * lifecycle_weights[indices]).mean()
                actor_loss = (
                    float(rollout.actor_loss_scale)
                    * (
                        policy_loss
                        + target_supervision_coef * target_supervision_loss
                        + initial_supervision_coef * initial_supervision_loss
                        - self.config.entropy_coef * entropy_mean
                    )
                    if update_epoch < actor_update_epochs
                    else policy_loss * 0.0
                )
                critic_loss = (
                    self.config.value_coef * value_loss
                    + self.config.counterfactual_value_coef * target_q_loss
                    if update_epoch < critic_update_epochs
                    else value_loss * 0.0
                )
                loss = (
                    actor_loss
                    + critic_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.model.strict_ctde:
                    if update_epoch < actor_update_epochs:
                        nn.utils.clip_grad_norm_(
                            actor_parameters,
                            self.config.max_grad_norm,
                        )
                    if update_epoch < critic_update_epochs:
                        nn.utils.clip_grad_norm_(
                            critic_parameters,
                            self.config.max_grad_norm,
                        )
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                if self.model.strict_ctde:
                    actor_gradients = [
                        None
                        if parameter.grad is None
                        else parameter.grad.detach().clone()
                        for parameter in actor_parameters
                    ]
                    critic_gradients = [
                        None
                        if parameter.grad is None
                        else parameter.grad.detach().clone()
                        for parameter in critic_parameters
                    ]
                    original_learning_rates = [
                        float(group["lr"])
                        for group in self.optimizer.param_groups
                    ]
                    original_betas = [
                        tuple(group["betas"])
                        for group in self.optimizer.param_groups
                    ]
                    if update_epoch < actor_update_epochs:
                        actor_parameter_values = [
                            parameter.detach().clone()
                            for parameter in actor_parameters
                        ]
                        actor_optimizer_states = [
                            copy.deepcopy(self.optimizer.state[parameter])
                            if parameter in self.optimizer.state
                            else None
                            for parameter in actor_parameters
                        ]
                        candidate_scale = float(
                            rollout.actor_learning_rate_scale
                        )
                        while True:
                            with torch.no_grad():
                                for parameter, value in zip(
                                    actor_parameters,
                                    actor_parameter_values,
                                ):
                                    parameter.copy_(value)
                            for parameter, state in zip(
                                actor_parameters,
                                actor_optimizer_states,
                            ):
                                if state is None:
                                    self.optimizer.state.pop(parameter, None)
                                else:
                                    self.optimizer.state[parameter] = (
                                        copy.deepcopy(state)
                                    )
                            for parameter, gradient in zip(
                                actor_parameters,
                                actor_gradients,
                            ):
                                parameter.grad = gradient
                            for parameter in critic_parameters:
                                parameter.grad = None
                            for group, learning_rate in zip(
                                self.optimizer.param_groups,
                                original_learning_rates,
                            ):
                                group["lr"] = (
                                    learning_rate * candidate_scale
                                )
                            if rollout.actor_beta1 is not None:
                                for group, betas in zip(
                                    self.optimizer.param_groups,
                                    original_betas,
                                ):
                                    group["betas"] = (
                                        float(rollout.actor_beta1),
                                        float(betas[1]),
                                    )
                            self.optimizer.step()
                            for group, learning_rate, betas in zip(
                                self.optimizer.param_groups,
                                original_learning_rates,
                                original_betas,
                            ):
                                group["lr"] = learning_rate
                                group["betas"] = betas
                            actor_candidate_attempts += 1
                            candidate_kls = self._actor_exact_kls(
                                pre_actor_parameters,
                                pre_initial_std,
                                observations,
                                target_valid_mask,
                                search_valid_mask,
                                action_mask,
                                target_features,
                            )
                            position_within_limit = (
                                candidate_kls["position_active_samples"] == 0.0
                                or rollout.actor_position_kl_limit is None
                                or candidate_kls["position_head_exact_kl"]
                                <= float(rollout.actor_position_kl_limit)
                            )
                            joint_within_limit = (
                                rollout.actor_joint_kl_limit is None
                                or candidate_kls["joint_exact_kl"]
                                <= float(rollout.actor_joint_kl_limit)
                            )
                            if position_within_limit and joint_within_limit:
                                accepted_actor_learning_rate_scale = (
                                    candidate_scale
                                    if accepted_actor_learning_rate_scale == 0.0
                                    else min(
                                        accepted_actor_learning_rate_scale,
                                        candidate_scale,
                                    )
                                )
                                break
                            actor_trust_region_backtrack_count += 1
                            candidate_scale *= float(
                                rollout.actor_backtrack_factor
                            )
                    if update_epoch < critic_update_epochs:
                        for parameter in actor_parameters:
                            parameter.grad = None
                        for parameter, gradient in zip(
                            critic_parameters,
                            critic_gradients,
                        ):
                            parameter.grad = gradient
                        for group, learning_rate in zip(
                            self.optimizer.param_groups,
                            original_learning_rates,
                        ):
                            group["lr"] = (
                                learning_rate
                                * float(rollout.critic_learning_rate_scale)
                            )
                        for group, betas in zip(
                            self.optimizer.param_groups,
                            original_betas,
                        ):
                            group["betas"] = (
                                (
                                    float(rollout.critic_beta1)
                                    if rollout.critic_beta1 is not None
                                    else float(betas[0])
                                ),
                                float(betas[1]),
                            )
                        self.optimizer.step()
                    for group, learning_rate, betas in zip(
                        self.optimizer.param_groups,
                        original_learning_rates,
                        original_betas,
                    ):
                        group["lr"] = learning_rate
                        group["betas"] = betas
                    for parameter, gradient in zip(
                        actor_parameters,
                        actor_gradients,
                    ):
                        parameter.grad = gradient
                else:
                    self.optimizer.step()

                if update_epoch < critic_update_epochs:
                    totals["value_loss"] += float(value_loss.item())
                    critic_steps += 1
                if update_epoch < actor_update_epochs:
                    totals["target_supervision_loss"] += float(
                        target_supervision_loss.item()
                    )
                    totals["initial_supervision_loss"] += float(
                        initial_supervision_loss.item()
                    )
                    totals["policy_loss"] += float(policy_loss.item())
                    totals["entropy"] += float(entropy_mean.item())
                    totals["approx_kl"] += float(
                        (ratio - 1.0 - log_ratio).mean().item()
                    )
                    totals["clip_fraction"] += float(
                        (
                            torch.abs(ratio - 1.0) > self.config.clip_ratio
                        ).float().mean().item()
                    )
                    actor_steps += 1
                if not self.model.strict_ctde and not rollout.joint_target_mode:
                    totals["target_q_loss"] += float(target_q_loss.item())
                steps += 1

        joint_target_steps = 0
        for _ in range(
            self.config.update_epochs
            if rollout.joint_target_returns and not self.model.strict_ctde
            else 0
        ):
            group_losses: list[torch.Tensor] = []
            for group in rollout.joint_target_returns:
                group_observations = group.observations.to(self.device)
                group_semantics = group.action_semantics.to(
                    self.device, dtype=group_observations.dtype
                )
                factor_values = self.model.target_action_values(
                    group_observations, group_semantics
                )
                joint_value = factor_values[:, int(group.target_index)].mean()
                target_return = torch.as_tensor(
                    group.target_return,
                    dtype=joint_value.dtype,
                    device=self.device,
                )
                group_losses.append((joint_value - target_return).square())
            joint_target_q_loss = torch.stack(group_losses).mean()
            self.optimizer.zero_grad(set_to_none=True)
            (
                self.config.counterfactual_value_coef * joint_target_q_loss
            ).backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            totals["target_q_loss"] += float(joint_target_q_loss.item())
            joint_target_steps += 1

        with torch.no_grad():
            post_log_probs, _, actor_value_placeholder = self.model.evaluate_actions(
                observations,
                target_coordinates,
                target_valid_mask,
                action_mask,
                actions,
                target_features=target_features,
                search_valid_mask=search_valid_mask,
            )
            post_log_ratio = post_log_probs - old_log_probs
            post_ratio = torch.exp(post_log_ratio)
            post_values = (
                self.model.critic_value(
                    critic_states,
                    agent_ids,
                    observations,
                )
                if self.model.strict_ctde
                else actor_value_placeholder
            )
            post_target_parameters = self.model.distribution_parameters(
                observations, target_features, target_valid_mask
            )
            post_target_logits, _ = self.model._masked_logits(
                post_target_parameters["target_logits"],
                target_valid_mask,
                action_mask.target,
            )
            post_target_supervision_accuracy = (
                float(
                    (
                        post_target_logits[target_supervision_mask].argmax(dim=-1)
                        == target_supervision_indices[target_supervision_mask]
                    )
                    .float()
                    .mean()
                    .item()
                )
                if bool(target_supervision_mask.any().item())
                else 0.0
            )
            post_update_initial_supervision_rmse = (
                float(
                    torch.nn.functional.mse_loss(
                        torch.tanh(post_target_parameters["initial_mean"])[initial_supervision_mask],
                        initial_supervision_xy[initial_supervision_mask],
                    ).sqrt().item()
                )
                if bool(initial_supervision_mask.any().item())
                else 0.0
            )
            post_update_sampled_forward_kl = float(
                (-post_log_ratio).mean().item()
            )
            post_update_approx_kl = float(
                (post_ratio - 1.0 - post_log_ratio).mean().item()
            )
            post_update_clip_fraction = float(
                (
                    torch.abs(post_ratio - 1.0) > self.config.clip_ratio
                ).float().mean().item()
            )
            post_update_surrogate_gain = float(
                (
                    (post_ratio - 1.0)
                    * advantages
                    * lifecycle_weights
                ).mean().item()
            )
            post_update_actor_exact_kls = self._actor_exact_kls(
                pre_actor_parameters,
                pre_initial_std,
                observations,
                target_valid_mask,
                search_valid_mask,
                action_mask,
                target_features,
            )
            post_update_target_head_exact_kl = (
                post_update_actor_exact_kls["target_head_exact_kl"]
            )
            post_update_position_head_exact_kl = (
                post_update_actor_exact_kls["position_head_exact_kl"]
            )
            post_update_critic_target_mae = float(
                (post_values - returns).abs().mean().item()
            )

        self.update_count += 1
        self.transition_count += sample_count
        self.last_metrics = {
            name: value / max(1, steps)
            for name, value in totals.items()
            if name != "target_q_loss"
        }
        for name in (
            "policy_loss", "entropy", "approx_kl", "clip_fraction",
            "target_supervision_loss",
            "initial_supervision_loss",
        ):
            self.last_metrics[name] = totals[name] / max(1, actor_steps)
        self.last_metrics["value_loss"] = (
            totals["value_loss"] / max(1, critic_steps)
        )
        self.last_metrics["target_q_loss"] = (
            totals["target_q_loss"]
            / max(1, joint_target_steps if rollout.joint_target_mode else steps)
        )
        self.last_metrics["samples"] = float(sample_count)
        self.last_metrics["target_supervision_count"] = float(
            target_supervision_mask.sum().item()
        )
        self.last_metrics["target_supervision_coef"] = target_supervision_coef
        self.last_metrics["post_update_target_supervision_accuracy"] = (
            post_target_supervision_accuracy
        )
        self.last_metrics["initial_supervision_count"] = float(
            initial_supervision_mask.sum().item()
        )
        self.last_metrics["initial_supervision_coef"] = initial_supervision_coef
        self.last_metrics["post_update_initial_supervision_rmse"] = (
            post_update_initial_supervision_rmse
        )
        self.last_metrics["deployment_policy_share"] = float(
            self.config.deployment_policy_share
        )
        self.last_metrics["per_agent_advantage_normalization"] = float(
            rollout.per_agent_advantage_normalization
        )
        self.last_metrics["raw_advantage_mean"] = raw_advantage_mean
        self.last_metrics["raw_advantage_std"] = raw_advantage_std
        self.last_metrics["empirical_actor_advantage"] = float(
            empirical_actor_advantage
        )
        self.last_metrics["normalized_empirical_actor_advantage"] = float(
            empirical_actor_advantage
            and rollout.normalize_actor_advantages
        )
        self.last_metrics["effective_advantage_mean"] = (
            effective_advantage_mean
        )
        self.last_metrics["effective_advantage_std"] = (
            effective_advantage_std
        )
        self.last_metrics["effective_advantage_positive_count"] = float(
            effective_advantage_positive_count
        )
        self.last_metrics["effective_advantage_negative_count"] = float(
            effective_advantage_negative_count
        )
        self.last_metrics["actor_signal_present"] = float(
            actor_signal_present
        )
        self.last_metrics["causal_advantage_projection"] = float(
            rollout.causal_advantage_projection
        )
        self.last_metrics["causal_projected_positive_count"] = float(
            causal_projected_mask.sum().item()
        )
        self.last_metrics["causal_projected_positive_mass"] = (
            causal_projected_positive_mass
        )
        self.last_metrics["actor_update_epochs"] = float(actor_update_epochs)
        self.last_metrics["critic_update_epochs"] = float(critic_update_epochs)
        self.last_metrics["actor_loss_scale"] = float(
            rollout.actor_loss_scale
        )
        self.last_metrics["actor_learning_rate_scale"] = float(
            rollout.actor_learning_rate_scale
        )
        self.last_metrics["actor_position_kl_limit"] = (
            -1.0
            if rollout.actor_position_kl_limit is None
            else float(rollout.actor_position_kl_limit)
        )
        self.last_metrics["actor_joint_kl_limit"] = (
            -1.0
            if rollout.actor_joint_kl_limit is None
            else float(rollout.actor_joint_kl_limit)
        )
        self.last_metrics["actor_backtrack_factor"] = float(
            rollout.actor_backtrack_factor
        )
        self.last_metrics["actor_candidate_attempts"] = float(
            actor_candidate_attempts
        )
        self.last_metrics["actor_trust_region_backtrack_count"] = float(
            actor_trust_region_backtrack_count
        )
        self.last_metrics["accepted_actor_learning_rate_scale"] = float(
            accepted_actor_learning_rate_scale
        )
        self.last_metrics["critic_learning_rate_scale"] = float(
            rollout.critic_learning_rate_scale
        )
        self.last_metrics["critic_beta1"] = (
            -1.0
            if rollout.critic_beta1 is None
            else float(rollout.critic_beta1)
        )
        self.last_metrics["actor_beta1"] = (
            -1.0
            if rollout.actor_beta1 is None
            else float(rollout.actor_beta1)
        )
        self.last_metrics["critic_bootstrap_advantage_mean"] = (
            critic_advantage_mean
        )
        self.last_metrics["critic_prediction_mean"] = float(
            old_values.mean().item()
        )
        self.last_metrics["critic_target_mean"] = float(returns.mean().item())
        self.last_metrics["critic_target_mae"] = critic_target_mae
        self.last_metrics["post_update_sampled_forward_kl"] = (
            post_update_sampled_forward_kl
        )
        self.last_metrics["post_update_approx_kl"] = post_update_approx_kl
        self.last_metrics["post_update_clip_fraction"] = (
            post_update_clip_fraction
        )
        self.last_metrics["post_update_surrogate_gain"] = (
            post_update_surrogate_gain
        )
        for name, value in post_update_actor_exact_kls.items():
            self.last_metrics[f"post_update_{name}"] = float(value)
        self.last_metrics["post_update_target_head_exact_kl"] = (
            post_update_target_head_exact_kl
        )
        self.last_metrics["post_update_position_head_exact_kl"] = (
            post_update_position_head_exact_kl
        )
        self.last_metrics["post_update_critic_prediction_mean"] = float(
            post_values.mean().item()
        )
        self.last_metrics["post_update_critic_target_mae"] = (
            post_update_critic_target_mae
        )
        self.last_metrics["reward_nonzero_fraction"] = float(
            (rewards.abs() > 1e-12).float().mean().item()
        )
        if bool(credit_anchor_mask.any().item()):
            anchor_advantages = raw_advantages[credit_anchor_mask]
            self.last_metrics["credit_anchor_advantage_abs_mean"] = float(
                anchor_advantages.abs().mean().item()
            )
        else:
            self.last_metrics["credit_anchor_advantage_abs_mean"] = 0.0
        rewarded_anchor_mask = (
            credit_anchor_mask & (rewards.abs() > 1e-12)
        )
        if bool(rewarded_anchor_mask.any().item()):
            self.last_metrics["rewarded_anchor_advantage_mean"] = float(
                raw_advantages[rewarded_anchor_mask].mean().item()
            )
        else:
            self.last_metrics["rewarded_anchor_advantage_mean"] = 0.0
        deployment_mask = rollout_steps < 0
        waiting_mask = (
            (rollout_steps >= 0)
            & action_mask.presence
            & (actions.presence == 0)
        )
        self.last_metrics["rollout_reward_sum"] = float(rewards.sum().item())
        self.last_metrics["credit_anchor_reward_sum"] = float(
            rewards[credit_anchor_mask].sum().item()
        )
        self.last_metrics["non_anchor_reward_sum"] = float(
            rewards[~credit_anchor_mask].sum().item()
        )
        self.last_metrics["credit_anchor_reward_nonzero_count"] = float(
            (
                credit_anchor_mask
                & (rewards.abs() > 1e-12)
            ).sum().item()
        )
        self.last_metrics["target_return_supervision_sum"] = float(
            sum(group.target_return for group in rollout.joint_target_returns)
            if rollout.joint_target_returns
            else target_credits.sum().item()
        )
        self.last_metrics["target_return_supervision_nonzero_count"] = float(
            sum(
                abs(group.target_return) > 1e-12
                for group in rollout.joint_target_returns
            )
            if rollout.joint_target_returns
            else (target_credits.abs() > 1e-12).sum().item()
        )
        self.last_metrics["joint_target_return_group_count"] = float(
            len(rollout.joint_target_returns)
        )
        self.last_metrics["deployment_direct_reward_sum"] = float(
            rewards[deployment_mask].sum().item()
        )
        self.last_metrics["waiting_direct_reward_sum"] = float(
            rewards[waiting_mask].sum().item()
        )
        return dict(self.last_metrics)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": (
                    (
                        "target_conditioned_ctde_mappo_v11"
                        if self.config.target_selector_arch == "transformer"
                        else "target_conditioned_ctde_mappo_v10"
                    )
                    if self.model.strict_ctde
                    else "unified_hybrid_mappo"
                ),
                "config": asdict(self.config),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_count": self.update_count,
                "transition_count": self.transition_count,
                "metrics": self.last_metrics,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            },
            destination,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        target_feature_dim: int | None = None,
    ) -> "HybridMAPPOTrainer":
        checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        algorithm = checkpoint.get("algorithm")
        supported = {
            "unified_hybrid_mappo",
            "target_conditioned_ctde_mappo_v7",
            "target_conditioned_ctde_mappo_v8",
            "target_conditioned_ctde_mappo_v9",
            "target_conditioned_ctde_mappo_v10",
            "target_conditioned_ctde_mappo_v11",
        }
        if algorithm not in supported:
            raise ValueError(f"不支持的 checkpoint 算法标识: {algorithm!r}")
        values = dict(checkpoint["config"])
        if "target_selector_arch" not in values:
            values["target_selector_arch"] = "legacy"
        stored_target_feature_dim = int(values.get("target_feature_dim", 20))
        if target_feature_dim is not None:
            values["target_feature_dim"] = int(target_feature_dim)
        if device is not None:
            values["device"] = device
        trainer = cls(HybridMAPPOConfig(**values))
        expected_algorithms = (
            {
                "target_conditioned_ctde_mappo_v7",
                "target_conditioned_ctde_mappo_v8",
                "target_conditioned_ctde_mappo_v9",
                "target_conditioned_ctde_mappo_v10",
                "target_conditioned_ctde_mappo_v11",
            }
            if trainer.model.strict_ctde
            else {"unified_hybrid_mappo"}
        )
        if algorithm not in expected_algorithms:
            raise ValueError(
                f"checkpoint 算法/config 不一致: {algorithm!r} not in {expected_algorithms!r}"
            )
        model_state = dict(checkpoint["model"])
        migrated_v7 = False
        migrated_target_features = False
        requested_target_feature_dim = int(trainer.config.target_feature_dim)
        if stored_target_feature_dim != requested_target_feature_dim:
            encoder_keys = (
                "target_item_encoder.0.weight",
                "target_teacher_item_encoder.0.weight",
                "target_transformer_item_encoder.0.weight",
            )
            expected_state = trainer.model.state_dict()
            for key in encoder_keys:
                old_weight = model_state.get(key)
                if old_weight is None:
                    if key == encoder_keys[0]:
                        raise ValueError(
                            "checkpoint 缺少共享目标特征编码器，无法安全迁移"
                        )
                    continue
                expected_weight = expected_state[key]
                if (
                    old_weight.ndim != 2
                    or old_weight.shape[0] != expected_weight.shape[0]
                    or old_weight.shape[1] > expected_weight.shape[1]
                ):
                    raise ValueError(
                        "checkpoint 目标特征维度无法安全迁移: "
                        f"{stored_target_feature_dim} -> {requested_target_feature_dim}"
                    )
                model_state[key] = torch.cat(
                    (
                        old_weight,
                        old_weight.new_zeros(
                            old_weight.shape[0],
                            expected_weight.shape[1] - old_weight.shape[1],
                        ),
                    ),
                    dim=1,
                )
            migrated_target_features = True
        if algorithm == "target_conditioned_ctde_mappo_v7":
            # v8 appends the independent satellite request to the joint action
            # semantics. Preserve all learned v7 columns and initialize the
            # new satellite column to zero so restored Q values are identical.
            key = "target_q_head.0.weight"
            old_weight = model_state.get(key)
            expected_weight = trainer.model.state_dict()[key]
            if (
                old_weight is not None
                and old_weight.ndim == 2
                and old_weight.shape[0] == expected_weight.shape[0]
                and old_weight.shape[1] + 1 == expected_weight.shape[1]
            ):
                model_state[key] = torch.cat(
                    (old_weight, torch.zeros_like(old_weight[:, :1])), dim=1
                )
                migrated_v7 = True
        incompatibility = trainer.model.load_state_dict(model_state, strict=False)
        allowed_missing = set() if trainer.model.strict_ctde else {
            "target_q_head.0.weight",
            "target_q_head.0.bias",
            "target_q_head.2.weight",
            "target_q_head.2.bias",
        }
        allowed_missing.update(
            key for key in trainer.model.state_dict()
            if key == "target_cf_policy_scale"
            or key == "target_cf_context_mode"
            or key.startswith("target_cf_value_head.")
            or key == "target_teacher_policy_scale"
            or key.startswith("target_teacher_item_encoder.")
            or key.startswith("target_teacher_query.")
            or key.startswith("target_teacher_score.")
        )
        if algorithm != "target_conditioned_ctde_mappo_v11":
            allowed_missing.update(
                key for key in trainer.model.state_dict()
                if key.startswith("target_transformer_item_encoder.")
                or key.startswith("target_transformer_query.")
                or key.startswith("target_transformer_encoder.")
                or key.startswith("target_transformer_score.")
            )
        if algorithm not in {
            "target_conditioned_ctde_mappo_v9",
            "target_conditioned_ctde_mappo_v10",
        }:
            allowed_missing.update(
                key for key in trainer.model.state_dict()
                if key == "target_allocator_mix"
                or key.startswith("target_item_encoder.")
                or key.startswith("target_query.")
                or key.startswith("target_score.")
            )
        if set(incompatibility.missing_keys) - allowed_missing or incompatibility.unexpected_keys:
            raise ValueError(
                "checkpoint 参数不兼容: "
                f"missing={incompatibility.missing_keys}, "
                f"unexpected={incompatibility.unexpected_keys}"
            )
        if (
            "optimizer" in checkpoint
            and not migrated_v7
            and not migrated_target_features
        ):
            try:
                trainer.optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError:
                trainer.optimizer = torch.optim.Adam(
                    trainer.model.parameters(),
                    lr=trainer.config.learning_rate,
                )
        trainer.update_count = int(checkpoint.get("update_count", 0))
        trainer.transition_count = int(checkpoint.get("transition_count", 0))
        trainer.last_metrics = dict(checkpoint.get("metrics", {}))
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        cuda_states = checkpoint.get("cuda_rng_states", [])
        if cuda_states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)
        return trainer
