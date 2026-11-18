"""红方示例的策略配置与学习模式检查。

配置描述任务分配层、机动层和模型路径；运行入口通过 options 传入示例。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .baselines import RED_POLICY_CHOICES


# 机动层名称：straight 保持直飞，reactive_evasion 使用规则规避，其余走离散 Actor。
LEARNING_MOTION_CHOICES = ("random_masked", "ppo", "mappo")
RED_MOTION_POLICY_CHOICES = ("straight", "reactive_evasion", *LEARNING_MOTION_CHOICES)


@dataclass(frozen=True)
class RedAgentConfig:
    """一组示例算法选项，使用冻结数据类在同局各实例间传递。"""

    # 任务层选择目标或搜索航点，机动层选择横向机动；两层在执行组件中组合。
    policy_name: str = "r0_random"
    motion_policy: str = "reactive_evasion"
    # seed 控制示例内部随机源；max_steps 为特征编码使用的步数尺度。
    seed: int | None = 1
    max_steps: int = 1000
    # 模型路径供 PPO/MAPPO 加载或保存；training 决定是否收集样本和更新参数。
    model_path: str | None = None
    training: bool = False

    def __post_init__(self) -> None:
        """检查任务层、机动层和学习模式的组合。"""
        if self.policy_name not in RED_POLICY_CHOICES:
            raise ValueError(f"Unsupported red baseline '{self.policy_name}'")
        if self.motion_policy not in RED_MOTION_POLICY_CHOICES:
            raise ValueError(f"Unsupported red motion policy: {self.motion_policy}")
        if self.policy_name == "r9_hierarchical_learning" and self.motion_policy not in LEARNING_MOTION_CHOICES:
            raise ValueError("r9_hierarchical_learning requires random_masked, ppo or mappo motion")
        if self.training and self.motion_policy not in {"ppo", "mappo"}:
            raise ValueError("Training requires ppo or mappo motion")
        if self.model_path is not None and self.motion_policy not in {"ppo", "mappo"}:
            raise ValueError("model_path requires ppo or mappo motion")
        # 推理加载已有模型；训练对象可先创建，保存路径由训练入口检查。
        if self.motion_policy in {"ppo", "mappo"} and not self.training:
            if not self.model_path:
                raise ValueError("model_path is required for PPO/MAPPO inference")
            if not Path(self.model_path).is_file():
                raise ValueError(f"Learning checkpoint does not exist: {self.model_path}")
