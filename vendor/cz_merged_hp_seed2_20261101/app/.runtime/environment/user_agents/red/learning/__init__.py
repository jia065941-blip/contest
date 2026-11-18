"""红方学习策略的稳定接口。

该包只依赖 NumPy。训练算法可以实现 :class:`SharedPolicy`，由红方示例
工厂装配共享参数与本机推理接口。集中训练生命周期由 ``red.train`` 的
可信训练入口驱动，完整 Critic 状态不作为普通 Agent 的输入。
"""

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
