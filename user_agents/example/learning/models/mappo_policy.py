"""集中训练、分散执行的参数共享 MAPPO 机动示例。

Actor消费本机编码；训练Critic额外消费runner在步进前后传入的全局编码。
经验按实体编号分别计算GAE，再合并为共享Actor和Critic的优化批次。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .red_policy import ACTION_DIM, GlobalStateEncoder, PolicyTransition, SharedPolicy


@dataclass
class MAPPOConfig:
    """本机与全局编码尺寸、网络结构、GAE及PPO优化参数。"""

    # 默认本机85维、全局70维；R9本机输入由构造入口改为90维。
    observation_dim: int = 85
    global_state_dim: int = 70
    action_dim: int = ACTION_DIM
    # Actor与Critic使用独立网络；Critic输入为全局编码和本机编码的拼接。
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    learning_rate: float = 3e-4
    # gamma折扣未来价值，gae_lambda控制同一实体轨迹上的优势递推。
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    # 缓冲规模按所有实体经验条数计算；回合边界只在达到min_update_size时更新。
    rollout_size: int = 8192
    min_update_size: int = 512
    seed: int = 0
    device: str = "cpu"
    max_steps: int = 1000  # 全局编码中时间进度的归一化分母。


class MAPPOActorCritic(nn.Module):
    """独立的本机Actor与集中式Critic，均使用两层Tanh隐藏网络。"""
    def __init__(self, config: MAPPOConfig):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(config.observation_dim, config.actor_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.actor_hidden_dim, config.actor_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.actor_hidden_dim, config.action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(
                config.global_state_dim + config.observation_dim,
                config.critic_hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(config.critic_hidden_dim, config.critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.critic_hidden_dim, 1),
        )

    def actor_forward(self, local_observation: torch.Tensor) -> torch.Tensor:
        """把[N, observation_dim]本机批次映射为动作logits。"""
        return self.actor(local_observation)

    def critic_forward(
        self, global_state: torch.Tensor, local_observation: torch.Tensor
    ) -> torch.Tensor:
        """拼接同一样本的全局和本机编码，返回[N]状态价值。"""
        critic_input = torch.cat((global_state, local_observation), dim=-1)
        return self.critic(critic_input).squeeze(-1)


class MAPPOSharedPolicy(SharedPolicy):
    """所有人族单位共享 Actor，训练时由集中式 Critic 读取全局状态。"""

    def __init__(self, config: MAPPOConfig | None = None):
        """建立共享模型、独立全局编码器、动作待配对队列及经验缓冲。"""
        self.config = config or MAPPOConfig()
        # 全局编码尺寸由固定实体分类和目标槽数决定，构造时与网络输入核对。
        self.global_encoder = GlobalStateEncoder(max_steps=self.config.max_steps)
        if self.global_encoder.state_dim != self.config.global_state_dim:
            raise ValueError(
                f"global_state_dim={self.config.global_state_dim}，"
                f"但编码器输出维度为{self.global_encoder.state_dim}"
            )
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.device = torch.device(self.config.device)
        self.network = MAPPOActorCritic(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        # pending按动作选择顺序暂存旧概率/价值；buffer保留配对完成的本机经验。
        self._buffer: list[tuple[PolicyTransition, float, float]] = []
        self._pending: list[tuple[float, float]] = []
        # 当前/下一全局上下文由显式训练入口设置，供本步所有实体经验复用。
        self._global_state: np.ndarray | None = None
        self._next_global_state: np.ndarray | None = None
        self.update_count = 0
        self.transition_count = 0
        self.last_metrics: dict[str, float] = {}
        self.training = True

    def begin_environment_step(self, full_observation) -> None:
        """训练runner在联合动作生成前提供当前完整态势。"""
        if not self.training:
            return
        self._global_state = self.global_encoder.encode(full_observation)
        self._next_global_state = None

    def end_environment_step(self, full_observation) -> None:
        """训练runner在环境推进后提供下一完整态势。"""
        if not self.training:
            return
        self._next_global_state = self.global_encoder.encode(full_observation)

    def get_global_state_context(self):
        """供示例经验装配层取出本步前后全局编码。"""
        return self._global_state, self._next_global_state

    def finish_environment_step(self) -> None:
        """等本步所有实体反馈完成后检查共享缓冲是否需要更新。"""
        if self.training and len(self._buffer) >= self.config.rollout_size:
            self.update()

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """由本机Actor选动作；训练时同时计算用于优势估计的旧价值。"""
        if self.training and self._global_state is None:
            raise RuntimeError("MAPPO选择动作前未设置全局状态")
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        # 推理只经过Actor；训练Critic额外读取runner设置的本步全局编码。
        with torch.no_grad():
            logits = self.network.actor_forward(observation_tensor)
            value = None
            if self.training:
                state_tensor = torch.as_tensor(
                    self._global_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                value = self.network.critic_forward(state_tensor, observation_tensor)
            # 掩码在采样前生效：训练使用分类采样，推理选择最大logit。
            logits = logits.masked_fill(~mask_tensor, torch.finfo(logits.dtype).min)
            distribution = Categorical(logits=logits)
            action = distribution.sample() if self.training else torch.argmax(logits, dim=-1)
            log_prob = distribution.log_prob(action)
        if self.training:
            assert value is not None
            self._pending.append((float(log_prob.item()), float(value.item())))
        return int(action.item())

    def observe(self, transition: PolicyTransition) -> None:
        """FIFO匹配动作采样与本机反馈，保留带前后全局状态的经验。"""
        if not self.training:
            return
        if not self._pending:
            raise RuntimeError("MAPPO收到transition前没有对应动作")
        if transition.global_state is None or transition.next_global_state is None:
            raise RuntimeError("MAPPO transition缺少全局状态")
        log_prob, value = self._pending.pop(0)
        self._buffer.append((transition, log_prob, value))
        self.transition_count += 1

    def reset_episode(self) -> None:
        """在回合边界更新足量经验，并清除已结束回合的全局上下文。"""
        if not self.training:
            return
        if self._pending:
            raise RuntimeError("回合结束时仍有未匹配的MAPPO动作")
        if len(self._buffer) >= self.config.min_update_size:
            self.update()
        self._global_state = None
        self._next_global_state = None

    def set_training(self, training: bool) -> None:
        """切换模型模式；进入推理时清空经验、待配对采样和Critic上下文。"""
        self.training = bool(training)
        self.network.train(self.training)
        if not self.training:
            self._buffer.clear()
            self._pending.clear()
            self._global_state = None
            self._next_global_state = None

    def _advantages_and_returns(self, rewards, dones, values, next_values, entity_ids):
        """按实体分别逆序递推GAE，done样本切断跨终止状态的累计。"""
        advantages = torch.zeros_like(rewards)
        # 共享缓冲中各实体经验交错存放；先抽出同一实体的有序下标再递推。
        for entity_id in torch.unique(entity_ids).tolist():
            indices = torch.nonzero(entity_ids == entity_id, as_tuple=False).flatten()
            gae = torch.zeros((), dtype=torch.float32, device=self.device)
            for index in reversed(indices.tolist()):
                nonterminal = 1.0 - dones[index]
                delta = rewards[index] + self.config.gamma * next_values[index] * nonterminal - values[index]
                gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * gae
                advantages[index] = gae
        return advantages, advantages + values

    def update(self) -> dict[str, float]:
        """用集中式价值估计计算GAE，再对共享Actor/Critic做PPO优化。"""
        if len(self._buffer) < self.config.min_update_size:
            return self.last_metrics

        # 同一经验的本机观测、全局观测、掩码和旧采样统计保持相同批次顺序。
        transitions = [item[0] for item in self._buffer]
        observations = torch.as_tensor(np.stack([x.observation for x in transitions]), dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(
            np.stack([x.next_observation for x in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        global_states = torch.as_tensor(np.stack([x.global_state for x in transitions]), dtype=torch.float32, device=self.device)
        next_global_states = torch.as_tensor(np.stack([x.next_global_state for x in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([x.action for x in transitions], dtype=torch.long, device=self.device)
        action_masks = torch.as_tensor(np.stack([x.action_mask for x in transitions]), dtype=torch.bool, device=self.device)
        rewards = torch.as_tensor([x.reward for x in transitions], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([x.done for x in transitions], dtype=torch.float32, device=self.device)
        entity_ids = torch.as_tensor([x.entity_id for x in transitions], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor([item[1] for item in self._buffer], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor([item[2] for item in self._buffer], dtype=torch.float32, device=self.device)

        # Critic以下一全局/本机状态自举；GAE按agent分轨迹后整体标准化。
        with torch.no_grad():
            next_values = self.network.critic_forward(next_global_states, next_observations)
            advantages, returns = self._advantages_and_returns(
                rewards, dones, old_values, next_values, entity_ids
            )
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0
        sample_count = observations.shape[0]
        # 每轮打乱样本并分成小批次，所有实体共同更新一套模型参数。
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.config.minibatch_size):
                indices = permutation[start:start + self.config.minibatch_size]
                logits = self.network.actor_forward(observations[indices])
                # 复用采样时掩码，保持新旧动作概率的可选集合一致。
                logits = logits.masked_fill(~action_masks[indices], torch.finfo(logits.dtype).min)
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(actions[indices])
                entropy = distribution.entropy().mean()
                ratio = torch.exp(new_log_probs - old_log_probs[indices])
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(
                    ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                ) * advantages[indices]
                # 策略损失采用裁剪概率比；Critic拟合GAE回报，熵项鼓励探索。
                policy_loss = -torch.min(unclipped, clipped).mean()
                predicted_values = self.network.critic_forward(
                    global_states[indices], observations[indices]
                )
                value_loss = torch.nn.functional.mse_loss(predicted_values, returns[indices])
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                # 两个网络使用同一个优化器，反传后统一裁剪梯度范数。
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                updates += 1

        # 本批训练结束后保留平均指标与累计计数，释放已消费的经验。
        self.update_count += 1
        self.last_metrics = {
            key: value / max(updates, 1) for key, value in totals.items()
        }
        self.last_metrics["samples"] = float(sample_count)
        self._buffer.clear()
        return self.last_metrics

    def save(self, path: str) -> None:
        """保存算法标识、构造配置、网络/优化器状态及训练计数。"""
        torch.save({
            "algorithm": "mappo",
            "config": asdict(self.config),
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "transition_count": self.transition_count,
        }, path)

    def load(self, path: str) -> None:
        """核对MAPPO标识后，将参数恢复到已创建的网络和优化器。"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("algorithm") != "mappo":
            raise ValueError("该checkpoint不是MAPPO模型")
        self.network.load_state_dict(checkpoint["network"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_count = int(checkpoint.get("update_count", 0))
        self.transition_count = int(checkpoint.get("transition_count", 0))
