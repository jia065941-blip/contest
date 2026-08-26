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
from .b0_fixed_ratio_random import B0FixedRatioRandomPolicy
from .b1_nearest_interceptor import B1NearestInterceptorPolicy
from .b2_threat_priority import B2ThreatPriorityPolicy
from .b3_min_cost_assignment import B3MinCostAssignmentPolicy


_POLICIES = {
    B0FixedRatioRandomPolicy.name: B0FixedRatioRandomPolicy,
    B1NearestInterceptorPolicy.name: B1NearestInterceptorPolicy,
    B2ThreatPriorityPolicy.name: B2ThreatPriorityPolicy,
    B3MinCostAssignmentPolicy.name: B3MinCostAssignmentPolicy,
}


def build_blue_policy(
    name: str | None = None,
    max_shots_per_target: int = 5,
    seed: int | None = None,
) -> BluePolicy:
    policy_name = (name or B0FixedRatioRandomPolicy.name).strip().lower().replace("-", "_")
    policy_cls = _POLICIES.get(policy_name)
    if policy_cls is None:
        valid_names = ", ".join(sorted(_POLICIES))
        raise ValueError(f"Unknown blue policy '{name}'. Valid policies: {valid_names}")

    if policy_cls is B0FixedRatioRandomPolicy:
        return policy_cls(max_shots_per_target=max_shots_per_target, seed=seed)
    return policy_cls(max_shots_per_target=max_shots_per_target)


__all__ = [
    "BlueObservation",
    "BluePolicy",
    "DefendedAsset",
    "B0FixedRatioRandomPolicy",
    "InterceptorAssignment",
    "InterceptorState",
    "B1NearestInterceptorPolicy",
    "B2ThreatPriorityPolicy",
    "B3MinCostAssignmentPolicy",
    "ThreatTrack",
    "Vector3",
    "build_blue_policy",
]
