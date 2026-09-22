"""Target-conditioned counterfactual credit and reward backfilling.

This module deliberately has no simulator or tensor-library dependency.  It
operates on completed target-reward events whose factual and counterfactual
target-window returns have already been measured by the simulator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any


CAUSAL_CATEGORIES = frozenset({"damage", "search", "intercept", "satellite"})
ALL_ZERO_DELTA_REASON = "all_single_counterfactual_deltas_zero"


class CreditStrategy(str, Enum):
    """Policy used to turn candidate evidence into normalized credit."""

    SINGLE_COUNTERFACTUAL = "single_counterfactual_normalization"
    DIRECT_DAMAGE_REDUNDANCY = "direct_damage_redundancy_actual_damage"
    JOINT_REMOVAL_EQUAL_SPLIT = "joint_removal_redundancy_equal_split"
    NO_CAUSAL_CREDIT = "no_causal_effect_zero_credit"


class DuplicateEventError(ValueError):
    """Raised when a target reward event would be recorded more than once."""


class CreditInvariantError(RuntimeError):
    """Raised when reward-conservation or temporal-discount invariants fail."""


@dataclass(frozen=True, slots=True)
class CreditCandidate:
    """One entity eligible for credit from a target reward event.

    ``categories`` is any non-empty subset of ``damage``, ``search``,
    ``intercept`` and ``satellite``.  A direct-damage-only redundancy fallback
    is allowed only when every candidate has exactly the ``damage`` category.
    """

    agent_id: Hashable
    entity_id: Hashable
    tau: int
    decision_type: str
    categories: frozenset[str]
    direct_damage: float = 0.0

    def __post_init__(self) -> None:
        _require_hashable(self.agent_id, "agent_id")
        _require_hashable(self.entity_id, "entity_id")
        if isinstance(self.tau, bool) or not isinstance(self.tau, int):
            raise TypeError("tau must be an integer timestep")
        if self.tau < 0:
            raise ValueError("tau must be non-negative")
        if not isinstance(self.decision_type, str) or not self.decision_type.strip():
            raise ValueError("decision_type must be a non-empty string")

        try:
            categories = frozenset(
                str(category).strip().lower() for category in self.categories
            )
        except TypeError as exc:
            raise TypeError("categories must be an iterable of strings") from exc
        if not categories:
            raise ValueError("categories must not be empty")
        unknown = categories - CAUSAL_CATEGORIES
        if unknown:
            raise ValueError(f"unknown causal categories: {sorted(unknown)!r}")

        direct_damage = _finite_float(self.direct_damage, "direct_damage")
        if direct_damage < 0.0:
            raise ValueError("direct_damage must be non-negative")
        if "damage" not in categories and direct_damage > 0.0:
            raise ValueError("direct_damage requires the 'damage' category")

        object.__setattr__(self, "decision_type", self.decision_type.strip())
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "direct_damage", direct_damage)


@dataclass(frozen=True, slots=True)
class TargetRewardEvent:
    """A positive target reward and its measured causal window returns.

    ``counterfactual_returns`` is keyed by candidate ``entity_id`` and must
    contain exactly one measured return for every candidate.  Non-candidates
    are intentionally absent and therefore have implicit alpha and credit 0.
    ``joint_counterfactual_return`` denotes removal of the complete candidate
    set and is consulted only for an all-zero single-removal anomaly involving
    indirect or mixed causal candidates.
    """

    event_id: Hashable
    target_id: Hashable
    timestep: int
    reward: float
    factual_return: float
    candidates: tuple[CreditCandidate, ...]
    counterfactual_returns: Mapping[Hashable, float]
    joint_counterfactual_return: float | None = None

    def __post_init__(self) -> None:
        _require_hashable(self.event_id, "event_id")
        _require_hashable(self.target_id, "target_id")
        if isinstance(self.timestep, bool) or not isinstance(self.timestep, int):
            raise TypeError("timestep must be an integer")
        if self.timestep < 0:
            raise ValueError("timestep must be non-negative")

        reward = _finite_float(self.reward, "reward")
        if reward <= 0.0:
            raise ValueError("a TargetRewardEvent must have a positive reward")
        factual_return = _finite_float(self.factual_return, "factual_return")
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, CreditCandidate) for candidate in candidates):
            raise TypeError("candidates must contain CreditCandidate instances")
        if any(candidate.tau > self.timestep for candidate in candidates):
            raise ValueError("candidate tau cannot be later than the reward timestep")

        entity_ids = [candidate.entity_id for candidate in candidates]
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("candidate entity_id values must be unique within an event")
        if not isinstance(self.counterfactual_returns, Mapping):
            raise TypeError("counterfactual_returns must be a mapping by entity_id")
        supplied_ids = set(self.counterfactual_returns)
        expected_ids = set(entity_ids)
        missing = expected_ids - supplied_ids
        extra = supplied_ids - expected_ids
        if missing or extra:
            raise ValueError(
                "counterfactual_returns must match candidate entity IDs exactly; "
                f"missing={_stable_repr(missing)}, extra={_stable_repr(extra)}"
            )
        counterfactual_returns = {
            entity_id: _finite_float(value, f"counterfactual return for {entity_id!r}")
            for entity_id, value in self.counterfactual_returns.items()
        }
        joint_return = self.joint_counterfactual_return
        if joint_return is not None:
            joint_return = _finite_float(
                joint_return, "joint_counterfactual_return"
            )

        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "factual_return", factual_return)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "counterfactual_returns",
            MappingProxyType(counterfactual_returns),
        )
        object.__setattr__(self, "joint_counterfactual_return", joint_return)


@dataclass(frozen=True, slots=True)
class CreditAnomaly:
    """Audit record for a positive reward with no single-removal difference."""

    event_id: Hashable
    target_id: Hashable
    timestep: int
    reason: str
    candidate_entity_ids: tuple[Hashable, ...]
    joint_delta: float | None
    resolution_strategy: CreditStrategy | None


class UnresolvedCreditError(RuntimeError):
    """Raised when an all-zero anomaly has no authorized redundancy fallback."""

    def __init__(self, anomaly: CreditAnomaly) -> None:
        self.anomaly = anomaly
        super().__init__(
            "unresolved target-credit anomaly for "
            f"event {anomaly.event_id!r}: {anomaly.reason}"
        )


@dataclass(frozen=True, slots=True)
class CandidateCredit:
    """Credit and temporal backfill for one causal candidate."""

    candidate: CreditCandidate
    counterfactual_return: float
    delta: float
    alpha: float
    credit: float
    backfill_multiplier: float
    backfilled_reward: float


@dataclass(frozen=True, slots=True)
class EventCreditAssignment:
    """Complete, auditable credit assignment for one target reward event."""

    event: TargetRewardEvent
    strategy: CreditStrategy
    candidate_credits: tuple[CandidateCredit, ...]
    anomaly: CreditAnomaly | None = None

    def allocation_for_entity(self, entity_id: Hashable) -> CandidateCredit | None:
        """Return a candidate allocation, or ``None`` for a non-candidate."""

        for allocation in self.candidate_credits:
            if allocation.candidate.entity_id == entity_id:
                return allocation
        return None

    def alpha_for_entity(self, entity_id: Hashable) -> float:
        """Return target-conditioned alpha; non-candidates are implicitly zero."""

        allocation = self.allocation_for_entity(entity_id)
        return 0.0 if allocation is None else allocation.alpha

    def credit_for_entity(self, entity_id: Hashable) -> float:
        """Return allocated target reward; non-candidates are implicitly zero."""

        allocation = self.allocation_for_entity(entity_id)
        return 0.0 if allocation is None else allocation.credit


@dataclass(frozen=True, slots=True)
class CreditValidationReport:
    """Numerical evidence that allocation and temporal relocation are valid."""

    gamma: float
    event_count: int
    unique_event_count: int
    original_reward_total: float
    allocated_credit_total: float
    unattributed_event_count: int
    unattributed_reward_total: float
    max_event_conservation_error: float
    discounted_original_return: float
    discounted_backfilled_return: float
    discounted_unattributed_return: float
    discounted_return_error: float
    gamma_one_original_return: float
    gamma_one_backfilled_return: float
    gamma_one_unattributed_return: float
    gamma_one_conservation_error: float
    event_ids_unique: bool
    every_reward_recorded_once: bool
    all_invariants_hold: bool


@dataclass(frozen=True, slots=True)
class TrajectoryCreditResult:
    """All event assignments and rewards ready to add to a rollout buffer."""

    assignments: tuple[EventCreditAssignment, ...]
    backfilled_rewards: Mapping[tuple[Hashable, int], float]
    anomalies: tuple[CreditAnomaly, ...]
    validation: CreditValidationReport

    def reward_for(self, agent_id: Hashable, timestep: int) -> float:
        """Return the aggregated backfill for one agent at one decision step."""

        return self.backfilled_rewards.get((agent_id, timestep), 0.0)


def assign_target_reward_event(
    event: TargetRewardEvent,
    *,
    gamma: float,
    epsilon: float = 1e-12,
) -> EventCreditAssignment:
    """Assign one factual reward using measured target-window counterfactuals.

    Normal events use ``max(G_F - G_CF_i, 0)`` normalized over candidates.  An
    all-zero event is always recorded as an anomaly and follows exactly one of
    two explicit redundancy policies:

    * direct-damage-only candidates: normalize their measured damage;
    * indirect/mixed candidates with an effective joint removal: equal split.

    If neither single nor joint removal changes target damage, no MAPPO action
    receives this event reward and training continues with zero credit.
    """

    if not isinstance(event, TargetRewardEvent):
        raise TypeError("event must be a TargetRewardEvent")
    gamma = _validate_gamma(gamma)
    epsilon = _validate_tolerance(epsilon, "epsilon")

    deltas = tuple(
        max(
            event.factual_return
            - event.counterfactual_returns[candidate.entity_id],
            0.0,
        )
        for candidate in event.candidates
    )
    delta_total = math.fsum(deltas)
    anomaly: CreditAnomaly | None = None

    if delta_total > epsilon:
        strategy = CreditStrategy.SINGLE_COUNTERFACTUAL
        alphas = tuple(delta / delta_total for delta in deltas)
    else:
        joint_delta = (
            None
            if event.joint_counterfactual_return is None
            else max(
                event.factual_return - event.joint_counterfactual_return,
                0.0,
            )
        )
        direct_damage_only = bool(event.candidates) and all(
            candidate.categories == frozenset({"damage"})
            for candidate in event.candidates
        )
        direct_damage_total = math.fsum(
            candidate.direct_damage for candidate in event.candidates
        )

        if direct_damage_only and direct_damage_total > epsilon:
            strategy = CreditStrategy.DIRECT_DAMAGE_REDUNDANCY
            alphas = tuple(
                candidate.direct_damage / direct_damage_total
                for candidate in event.candidates
            )
        elif (
            not direct_damage_only
            and bool(event.candidates)
            and joint_delta is not None
            and joint_delta > epsilon
        ):
            # This is deliberately equal rather than damage-proportional: the
            # joint experiment establishes set-level necessity but contains no
            # evidence that identifies unequal individual shares.
            strategy = CreditStrategy.JOINT_REMOVAL_EQUAL_SPLIT
            equal_share = 1.0 / len(event.candidates)
            alphas = (equal_share,) * len(event.candidates)
        else:
            strategy = CreditStrategy.NO_CAUSAL_CREDIT
            alphas = (0.0,) * len(event.candidates)

        anomaly = CreditAnomaly(
            event_id=event.event_id,
            target_id=event.target_id,
            timestep=event.timestep,
            reason=ALL_ZERO_DELTA_REASON,
            candidate_entity_ids=tuple(
                candidate.entity_id for candidate in event.candidates
            ),
            joint_delta=joint_delta,
            resolution_strategy=strategy,
        )

    allocations = tuple(
        _candidate_credit(
            event=event,
            candidate=candidate,
            counterfactual_return=event.counterfactual_returns[candidate.entity_id],
            delta=delta,
            alpha=alpha,
            gamma=gamma,
        )
        for candidate, delta, alpha in zip(event.candidates, deltas, alphas)
    )
    assignment = EventCreditAssignment(
        event=event,
        strategy=strategy,
        candidate_credits=allocations,
        anomaly=anomaly,
    )
    # Validate immediately so a malformed assignment can never enter a buffer.
    validate_credit_assignments((assignment,), gamma=gamma)
    return assignment


def assign_trajectory_credit(
    events: Iterable[TargetRewardEvent],
    *,
    gamma: float,
    epsilon: float = 1e-12,
    atol: float = 1e-9,
) -> TrajectoryCreditResult:
    """Assign a collection of events exactly once and aggregate buffer updates."""

    gamma = _validate_gamma(gamma)
    epsilon = _validate_tolerance(epsilon, "epsilon")
    atol = _validate_tolerance(atol, "atol")
    event_tuple = tuple(events)
    _ensure_unique_event_ids(event.event_id for event in event_tuple)

    assignments = tuple(
        assign_target_reward_event(event, gamma=gamma, epsilon=epsilon)
        for event in event_tuple
    )
    validation = validate_credit_assignments(assignments, gamma=gamma, atol=atol)

    aggregated: defaultdict[tuple[Hashable, int], float] = defaultdict(float)
    for assignment in assignments:
        for allocation in assignment.candidate_credits:
            key = (allocation.candidate.agent_id, allocation.candidate.tau)
            aggregated[key] += allocation.backfilled_reward

    anomalies = tuple(
        assignment.anomaly
        for assignment in assignments
        if assignment.anomaly is not None
    )
    return TrajectoryCreditResult(
        assignments=assignments,
        backfilled_rewards=MappingProxyType(dict(aggregated)),
        anomalies=anomalies,
        validation=validation,
    )


def validate_credit_assignments(
    assignments: Iterable[EventCreditAssignment],
    *,
    gamma: float,
    atol: float = 1e-9,
) -> CreditValidationReport:
    """Validate conservation, one-time recording, and discounted equivalence.

    Discount weights use the formulation in the proposal, ``gamma**(step-1)``.
    Moving ``c`` from ``t`` to ``tau`` as ``gamma**(t-tau) * c`` therefore
    leaves the discounted trajectory return unchanged.  The check at gamma=1
    is reported separately to make undiscounted episode conservation explicit.
    """

    gamma = _validate_gamma(gamma)
    atol = _validate_tolerance(atol, "atol")
    assignment_tuple = tuple(assignments)
    if any(
        not isinstance(assignment, EventCreditAssignment)
        for assignment in assignment_tuple
    ):
        raise TypeError("assignments must contain EventCreditAssignment instances")

    event_ids = [assignment.event.event_id for assignment in assignment_tuple]
    _ensure_unique_event_ids(event_ids)
    event_errors: list[float] = []

    for assignment in assignment_tuple:
        event = assignment.event
        expected_entities = tuple(
            candidate.entity_id for candidate in event.candidates
        )
        allocated_entities = tuple(
            allocation.candidate.entity_id
            for allocation in assignment.candidate_credits
        )
        if allocated_entities != expected_entities:
            raise CreditInvariantError(
                f"event {event.event_id!r} does not allocate each candidate once"
            )

        alpha_total = math.fsum(
            allocation.alpha for allocation in assignment.candidate_credits
        )
        expected_alpha_total = (
            0.0
            if assignment.strategy is CreditStrategy.NO_CAUSAL_CREDIT
            else 1.0
        )
        if not _close(alpha_total, expected_alpha_total, atol):
            raise CreditInvariantError(
                f"event {event.event_id!r} alpha values sum to {alpha_total}, "
                f"not {expected_alpha_total}"
            )

        for allocation in assignment.candidate_credits:
            if not all(
                math.isfinite(value) and value >= 0.0
                for value in (
                    allocation.delta,
                    allocation.alpha,
                    allocation.credit,
                    allocation.backfill_multiplier,
                    allocation.backfilled_reward,
                )
            ):
                raise CreditInvariantError(
                    f"event {event.event_id!r} contains invalid allocation values"
                )
            expected_delta = max(
                event.factual_return - allocation.counterfactual_return,
                0.0,
            )
            if not _close(allocation.delta, expected_delta, atol):
                raise CreditInvariantError(
                    f"event {event.event_id!r} contains an incorrect delta"
                )
            expected_credit = allocation.alpha * event.reward
            if not _close(allocation.credit, expected_credit, atol):
                raise CreditInvariantError(
                    f"event {event.event_id!r} violates c = alpha * reward"
                )
            expected_multiplier = gamma ** (
                event.timestep - allocation.candidate.tau
            )
            expected_backfill = expected_multiplier * allocation.credit
            if not _close(allocation.backfill_multiplier, expected_multiplier, atol):
                raise CreditInvariantError(
                    f"event {event.event_id!r} has an incorrect backfill multiplier"
                )
            if not _close(allocation.backfilled_reward, expected_backfill, atol):
                raise CreditInvariantError(
                    f"event {event.event_id!r} has an incorrect backfilled reward"
                )

        allocated = math.fsum(
            allocation.credit for allocation in assignment.candidate_credits
        )
        unattributed = (
            event.reward
            if assignment.strategy is CreditStrategy.NO_CAUSAL_CREDIT
            else 0.0
        )
        event_error = abs(allocated + unattributed - event.reward)
        event_errors.append(event_error)
        if not _close(allocated + unattributed, event.reward, atol):
            raise CreditInvariantError(
                f"event {event.event_id!r} does not conserve its target reward"
            )

    original_total = math.fsum(
        assignment.event.reward for assignment in assignment_tuple
    )
    allocated_total = math.fsum(
        allocation.credit
        for assignment in assignment_tuple
        for allocation in assignment.candidate_credits
    )
    unattributed_assignments = tuple(
        assignment
        for assignment in assignment_tuple
        if assignment.strategy is CreditStrategy.NO_CAUSAL_CREDIT
    )
    unattributed_total = math.fsum(
        assignment.event.reward for assignment in unattributed_assignments
    )
    discounted_original = math.fsum(
        gamma ** (assignment.event.timestep - 1) * assignment.event.reward
        for assignment in assignment_tuple
    )
    discounted_backfilled = math.fsum(
        gamma ** (allocation.candidate.tau - 1)
        * allocation.backfilled_reward
        for assignment in assignment_tuple
        for allocation in assignment.candidate_credits
    )
    discounted_unattributed = math.fsum(
        gamma ** (assignment.event.timestep - 1) * assignment.event.reward
        for assignment in unattributed_assignments
    )
    discounted_error = abs(
        discounted_backfilled + discounted_unattributed - discounted_original
    )
    if not _close(
        discounted_backfilled + discounted_unattributed,
        discounted_original,
        atol,
    ):
        raise CreditInvariantError(
            "backfilled rewards do not preserve the discounted trajectory return"
        )

    # At gamma=1 the prescribed backfill multiplier is exactly one, so these
    # totals are the undiscounted original and relocated episode returns.
    gamma_one_original = original_total
    gamma_one_backfilled = allocated_total
    gamma_one_unattributed = unattributed_total
    gamma_one_error = abs(
        gamma_one_backfilled + gamma_one_unattributed - gamma_one_original
    )
    if not _close(
        gamma_one_backfilled + gamma_one_unattributed,
        gamma_one_original,
        atol,
    ):
        raise CreditInvariantError(
            "credit assignments do not preserve the gamma=1 episode return"
        )

    return CreditValidationReport(
        gamma=gamma,
        event_count=len(assignment_tuple),
        unique_event_count=len(set(event_ids)),
        original_reward_total=original_total,
        allocated_credit_total=allocated_total,
        unattributed_event_count=len(unattributed_assignments),
        unattributed_reward_total=unattributed_total,
        max_event_conservation_error=max(event_errors, default=0.0),
        discounted_original_return=discounted_original,
        discounted_backfilled_return=discounted_backfilled,
        discounted_unattributed_return=discounted_unattributed,
        discounted_return_error=discounted_error,
        gamma_one_original_return=gamma_one_original,
        gamma_one_backfilled_return=gamma_one_backfilled,
        gamma_one_unattributed_return=gamma_one_unattributed,
        gamma_one_conservation_error=gamma_one_error,
        event_ids_unique=True,
        every_reward_recorded_once=True,
        all_invariants_hold=True,
    )


def _candidate_credit(
    *,
    event: TargetRewardEvent,
    candidate: CreditCandidate,
    counterfactual_return: float,
    delta: float,
    alpha: float,
    gamma: float,
) -> CandidateCredit:
    credit = alpha * event.reward
    multiplier = gamma ** (event.timestep - candidate.tau)
    return CandidateCredit(
        candidate=candidate,
        counterfactual_return=counterfactual_return,
        delta=delta,
        alpha=alpha,
        credit=credit,
        backfill_multiplier=multiplier,
        backfilled_reward=multiplier * credit,
    )


def _ensure_unique_event_ids(event_ids: Iterable[Hashable]) -> None:
    seen: set[Hashable] = set()
    for event_id in event_ids:
        _require_hashable(event_id, "event_id")
        if event_id in seen:
            raise DuplicateEventError(
                f"target reward event {event_id!r} was supplied more than once"
            )
        seen.add(event_id)


def _require_hashable(value: Any, name: str) -> None:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be hashable") from exc


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _validate_gamma(gamma: Any) -> float:
    converted = _finite_float(gamma, "gamma")
    if not 0.0 < converted <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    return converted


def _validate_tolerance(value: Any, name: str) -> float:
    converted = _finite_float(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _close(first: float, second: float, atol: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-9, abs_tol=atol)


def _stable_repr(values: set[Hashable]) -> str:
    return repr(sorted((repr(value) for value in values)))


__all__ = [
    "ALL_ZERO_DELTA_REASON",
    "CAUSAL_CATEGORIES",
    "CandidateCredit",
    "CreditAnomaly",
    "CreditCandidate",
    "CreditInvariantError",
    "CreditStrategy",
    "CreditValidationReport",
    "DuplicateEventError",
    "EventCreditAssignment",
    "TargetRewardEvent",
    "TrajectoryCreditResult",
    "UnresolvedCreditError",
    "assign_target_reward_event",
    "assign_trajectory_credit",
    "validate_credit_assignments",
]
