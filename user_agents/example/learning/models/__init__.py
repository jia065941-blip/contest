"""红方学习策略的稳定接口。

此处导出的编码器、动作适配器和随机策略使用 NumPy。训练算法实现
:class:`SharedPolicy`，由红方示例内部装配共享参数与本机推理接口。
集中训练由 ``example.learning.train`` 的可信入口显式提供 Critic 全态；
普通单机推理仍传入本机编码及动作掩码。
"""

# 数据接口和编码工具：可在不实例化神经网络的情况下使用。
from .red_policy import (
    ACTION_DIM,
    DEFAULT_TARGET_SLOTS,
    HighLevelAction,
    GlobalStateEncoder,
    LearningActionAdapter,
    ObservationEncoder,
    PolicyTransition,
    RandomMaskedPolicy,
    SharedPolicy,
)
# 模型构造入口：算法模块在实际选择 PPO/MAPPO 时加载。
from .runtime import LEARNING_MOTION_CHOICES, build_learning_motion_policy

__all__ = [
    "ACTION_DIM",
    "DEFAULT_TARGET_SLOTS",
    "HighLevelAction",
    "GlobalStateEncoder",
    "LearningActionAdapter",
    "LEARNING_MOTION_CHOICES",
    "ObservationEncoder",
    "PolicyTransition",
    "RandomMaskedPolicy",
    "SharedPolicy",
    "build_learning_motion_policy",
]
