"""红方学习策略的稳定接口。

该包只依赖 NumPy。训练算法可以实现 :class:`SharedPolicy`，然后将同一个
策略实例注入所有 ``AttackMissileAgent``，从而实现参数共享。
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

__all__ = [
    "ACTION_DIM",
    "DEFAULT_TARGET_SLOTS",
    "HighLevelAction",
    "GlobalStateEncoder",
    "LearningActionAdapter",
    "ObservationEncoder",
    "PolicyTransition",
    "RandomMaskedPolicy",
    "SharedPolicy",
]
