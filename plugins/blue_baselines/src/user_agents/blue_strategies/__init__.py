# -*- coding: utf-8 -*-
from __future__ import annotations

from .base import (
    BlueObservation,
    BluePolicy,
    DefendedAsset,
    InterceptorAssignment,
    InterceptorState,
    ThreatTrack,
    Vector3,
)
from .min_cost_assignment import MinCostAssignmentPolicy
from .nearest_interceptor import NearestInterceptorPolicy
from .random_salvo import FixedRatioRandomPolicy
from .threat_priority import ThreatPriorityPolicy


_POLICIES = {
    FixedRatioRandomPolicy.name: FixedRatioRandomPolicy,
    "random_salvo": FixedRatioRandomPolicy,
    NearestInterceptorPolicy.name: NearestInterceptorPolicy,
    ThreatPriorityPolicy.name: ThreatPriorityPolicy,
    MinCostAssignmentPolicy.name: MinCostAssignmentPolicy,
}


def build_blue_policy(
    name: str | None = None,
    max_shots_per_target: int = 5,
    seed: int | None = None,
) -> BluePolicy:
    policy_name = (name or FixedRatioRandomPolicy.name).strip().lower().replace("-", "_")
    policy_cls = _POLICIES.get(policy_name)
    if policy_cls is None:
        valid_names = ", ".join(sorted(_POLICIES))
        raise ValueError(f"Unknown blue policy '{name}'. Valid policies: {valid_names}")

    if policy_cls is FixedRatioRandomPolicy:
        return policy_cls(max_shots_per_target=max_shots_per_target, seed=seed)
    return policy_cls(max_shots_per_target=max_shots_per_target)


__all__ = [
    "BlueObservation",
    "BluePolicy",
    "DefendedAsset",
    "FixedRatioRandomPolicy",
    "InterceptorAssignment",
    "InterceptorState",
    "MinCostAssignmentPolicy",
    "NearestInterceptorPolicy",
    "ThreatPriorityPolicy",
    "ThreatTrack",
    "Vector3",
    "build_blue_policy",
]
