"""Destruction probability from explicitly supplied, legal evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import math


# Conditional applied damage is base damage times an adjusted rate. These
# bounds are deterministic diagnostics; they are not probability parameters.
_DAMAGE_INTERVALS = {
    (21000, 9400): (9.6, 16.0, 19.2),
    (21000, 9600): (12.0, 12.0, 12.0),
    (21001, 9400): (2.4, 4.0, 4.8),
    (21001, 9600): (3.0, 3.0, 3.0),
    (21002, 9500): (0.8, 0.8, 0.8),
}
_TARGET_TYPES = frozenset(target_type for _, target_type in _DAMAGE_INTERVALS)
_COMMITMENT_TYPES = frozenset(("core", "flexible"))


def _integer(name: str, value: object, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(name: str, value: object, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def _probability(name: str, value: object) -> None:
    _number(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _event_id(value: object) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("event_id must be a non-empty string or None")


@dataclass(frozen=True)
class DamageInterval:
    """Deterministic conditional-damage bounds, not hit probabilities."""

    lower: float = 0.0
    nominal: float = 0.0
    upper: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("lower", self.lower),
            ("nominal", self.nominal),
            ("upper", self.upper),
        ):
            _number(name, value)
            if value < 0:
                raise ValueError(f"{name} damage cannot be negative")
        if not self.lower <= self.nominal <= self.upper:
            raise ValueError("damage interval must satisfy lower <= nominal <= upper")


def _damage_interval(
    missile_type: int,
    target_type: int,
    *,
    satellite_active: bool = False,
    adjusted_damage_override: float | None = None,
) -> DamageInterval:
    try:
        values = _DAMAGE_INTERVALS[(missile_type, target_type)]
    except KeyError as error:
        raise ValueError(
            f"unsupported missile/target damage pair: {missile_type}/{target_type}"
        ) from error

    if adjusted_damage_override is not None:
        return DamageInterval(
            adjusted_damage_override,
            adjusted_damage_override,
            adjusted_damage_override,
        )
    if satellite_active and (missile_type, target_type) == (21000, 9400):
        return DamageInterval(20.0, 20.0, 20.0)
    return DamageInterval(*values)


def _damage(missile_type: int, target_type: int) -> float:
    return _damage_interval(missile_type, target_type).nominal


@dataclass(frozen=True)
class TargetPriorEvidence:
    target_id: int
    target_type: int
    initial_health: float
    discovered_step: int = 0
    source: str = "initial_catalogue"

    def __post_init__(self) -> None:
        _integer("target_id", self.target_id)
        _integer("target_type", self.target_type)
        if self.target_type not in _TARGET_TYPES:
            raise ValueError(f"unsupported target type: {self.target_type}")
        _number("initial_health", self.initial_health, positive=True)
        _integer("discovered_step", self.discovered_step)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")


@dataclass(frozen=True)
class CommitmentEvidence:
    missile_id: int
    missile_type: int
    target_id: int
    commitment_type: str = "core"
    event_id: str | None = None

    def __post_init__(self) -> None:
        _integer("missile_id", self.missile_id)
        _integer("missile_type", self.missile_type)
        _integer("target_id", self.target_id)
        if self.commitment_type not in _COMMITMENT_TYPES:
            raise ValueError("commitment_type must be 'core' or 'flexible'")
        _event_id(self.event_id)


@dataclass(frozen=True)
class DeliveryEvidence:
    missile_id: int
    missile_type: int
    target_id: int
    step: int
    last_stage: int
    minimum_target_distance_metres: float | None
    disappeared: bool = True
    event_id: str | None = None
    probability: float | None = None
    satellite_active: bool = False
    adjusted_damage_override: float | None = None

    def __post_init__(self) -> None:
        _integer("missile_id", self.missile_id)
        _integer("missile_type", self.missile_type)
        _integer("target_id", self.target_id)
        _integer("step", self.step)
        _integer("last_stage", self.last_stage)
        if self.minimum_target_distance_metres is not None:
            _number(
                "minimum_target_distance_metres",
                self.minimum_target_distance_metres,
            )
            if self.minimum_target_distance_metres < 0:
                raise ValueError("minimum_target_distance_metres cannot be negative")
        if type(self.disappeared) is not bool:
            raise ValueError("disappeared must be a bool")
        _event_id(self.event_id)
        if self.probability is not None:
            _probability("probability", self.probability)
        if type(self.satellite_active) is not bool:
            raise ValueError("satellite_active must be a bool")
        if self.adjusted_damage_override is not None:
            _number("adjusted_damage_override", self.adjusted_damage_override)
            if self.adjusted_damage_override < 0:
                raise ValueError("adjusted_damage_override cannot be negative")


@dataclass(frozen=True)
class HitReceipt:
    """Optional exact damage receipt supplied by a future legal interface."""

    missile_id: int
    missile_type: int
    target_id: int
    step: int
    applied_damage: float
    target_destroyed: bool = False
    event_id: str | None = None

    def __post_init__(self) -> None:
        _integer("missile_id", self.missile_id)
        _integer("missile_type", self.missile_type)
        _integer("target_id", self.target_id)
        _integer("step", self.step)
        _number("applied_damage", self.applied_damage)
        if self.applied_damage < 0:
            raise ValueError("applied_damage cannot be negative")
        if type(self.target_destroyed) is not bool:
            raise ValueError("target_destroyed must be a bool")
        _event_id(self.event_id)


@dataclass(frozen=True)
class TargetDamageProbabilityBelief:
    target_id: int
    target_type: int
    initial_health: float
    destruction_probability: float
    projected_destruction_probability: float
    confirmed_destroyed: bool
    exact_received_damage: float
    delivered_expected_damage: float
    projected_expected_damage: float
    committed_missiles: tuple[int, ...]
    core_committed_missiles: tuple[int, ...]
    flexible_committed_missiles: tuple[int, ...]
    delivery_missiles: tuple[int, ...]
    receipt_missiles: tuple[int, ...]
    delivery_probabilities: tuple[tuple[int, float], ...]
    commitment_probabilities: tuple[tuple[int, float], ...]
    lower_expected_damage: float = 0.0
    nominal_expected_damage: float = 0.0
    upper_expected_damage: float = 0.0
    projected_lower_expected_damage: float = 0.0
    projected_nominal_expected_damage: float = 0.0
    projected_upper_expected_damage: float = 0.0


@dataclass(frozen=True)
class EvidenceEvent:
    """Immutable idempotency record retained across evidence export/replay."""

    event_id: str
    operation: str
    payload: tuple[object, ...]

    def __post_init__(self) -> None:
        _event_id(self.event_id)
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("operation must be a non-empty string")
        if type(self.payload) is not tuple:
            raise ValueError("event payload must be a tuple")


@dataclass(frozen=True)
class EvidenceSnapshot:
    """Immutable, portable evidence state for one communication owner/domain."""

    target_priors: tuple[TargetPriorEvidence, ...] = ()
    commitments: tuple[CommitmentEvidence, ...] = ()
    deliveries: tuple[DeliveryEvidence, ...] = ()
    receipts: tuple[HitReceipt, ...] = ()
    confirmed_target_ids: tuple[int, ...] = ()
    processed_events: tuple[EvidenceEvent, ...] = ()
    core_commitment_probability: float = 1.0
    flexible_commitment_probability: float = 0.5

    def __post_init__(self) -> None:
        collections = (
            ("target_priors", self.target_priors, TargetPriorEvidence),
            ("commitments", self.commitments, CommitmentEvidence),
            ("deliveries", self.deliveries, DeliveryEvidence),
            ("receipts", self.receipts, HitReceipt),
            ("processed_events", self.processed_events, EvidenceEvent),
        )
        for name, values, expected_type in collections:
            if type(values) is not tuple or not all(
                isinstance(value, expected_type) for value in values
            ):
                raise ValueError(f"{name} must be a tuple of {expected_type.__name__}")
        if type(self.confirmed_target_ids) is not tuple:
            raise ValueError("confirmed_target_ids must be a tuple")
        for target_id in self.confirmed_target_ids:
            _integer("confirmed target ID", target_id)
        if len(set(self.confirmed_target_ids)) != len(self.confirmed_target_ids):
            raise ValueError("confirmed_target_ids cannot contain duplicates")
        _probability(
            "core_commitment_probability", self.core_commitment_probability
        )
        _probability(
            "flexible_commitment_probability", self.flexible_commitment_probability
        )


class TargetDamageProbabilitySkill:
    """Maintain delivered and projected destruction probabilities."""

    def __init__(
        self,
        *,
        core_commitment_probability: float = 1.0,
        flexible_commitment_probability: float = 0.5,
    ) -> None:
        _probability("core_commitment_probability", core_commitment_probability)
        _probability(
            "flexible_commitment_probability", flexible_commitment_probability
        )
        self._core_commitment_probability = float(core_commitment_probability)
        self._flexible_commitment_probability = float(
            flexible_commitment_probability
        )
        self._targets: dict[int, TargetPriorEvidence] = {}
        self._commitments: dict[int, CommitmentEvidence] = {}
        self._deliveries: dict[int, DeliveryEvidence] = {}
        self._receipts: dict[int, HitReceipt] = {}
        self._confirmed_target_ids: set[int] = set()
        self._processed_events: dict[str, EvidenceEvent] = {}

    def reset(self) -> None:
        self._targets.clear()
        self._commitments.clear()
        self._deliveries.clear()
        self._receipts.clear()
        self._confirmed_target_ids.clear()
        self._processed_events.clear()

    def register(self, evidence: TargetPriorEvidence) -> None:
        if not isinstance(evidence, TargetPriorEvidence):
            raise TypeError("evidence must be TargetPriorEvidence")
        previous = self._targets.get(evidence.target_id)
        if previous is not None and previous != evidence:
            raise ValueError("target prior changed after registration")
        self._targets[evidence.target_id] = evidence

    def commit(self, evidence: CommitmentEvidence) -> None:
        if not isinstance(evidence, CommitmentEvidence):
            raise TypeError("evidence must be CommitmentEvidence")
        event = self._make_event(evidence.event_id, "commit", (evidence,))
        if self._is_duplicate(event):
            return
        target = self._require_target(evidence.target_id)
        _damage(evidence.missile_type, target.target_type)
        previous = self._commitments.get(evidence.missile_id)
        if previous is not None and previous != evidence:
            raise ValueError(
                f"missile {evidence.missile_id} already has a different commitment"
            )
        self._commitments[evidence.missile_id] = evidence
        self._record_event(event)

    def transfer(
        self, missile_id: int, target_id: int, *, event_id: str | None = None
    ) -> None:
        _integer("missile_id", missile_id)
        _integer("target_id", target_id)
        _event_id(event_id)
        event = self._make_event(event_id, "transfer", (missile_id, target_id))
        if self._is_duplicate(event):
            return
        previous = self._commitments.get(missile_id)
        if previous is None:
            raise ValueError(f"missile {missile_id} has no active commitment")
        target = self._require_target(target_id)
        _damage(previous.missile_type, target.target_type)
        if previous.target_id != target_id:
            self._commitments[missile_id] = replace(
                previous, target_id=target_id, event_id=None
            )
        self._record_event(event)

    def release(self, missile_id: int, *, event_id: str | None = None) -> None:
        _integer("missile_id", missile_id)
        _event_id(event_id)
        event = self._make_event(event_id, "release", (missile_id,))
        if self._is_duplicate(event):
            return
        self._commitments.pop(missile_id, None)
        self._record_event(event)

    def report_delivery(self, evidence: DeliveryEvidence) -> None:
        if not isinstance(evidence, DeliveryEvidence):
            raise TypeError("evidence must be DeliveryEvidence")
        event = self._make_event(evidence.event_id, "delivery", (evidence,))
        if self._is_duplicate(event):
            return
        target = self._require_target(evidence.target_id)
        _damage(evidence.missile_type, target.target_type)
        self._validate_missile_assignment(
            evidence.missile_id, evidence.missile_type, evidence.target_id
        )
        previous = self._deliveries.get(evidence.missile_id)
        if previous is not None and previous != evidence:
            raise ValueError(
                f"missile {evidence.missile_id} already has different delivery evidence"
            )
        receipt = self._receipts.get(evidence.missile_id)
        if receipt is not None:
            self._validate_same_missile_evidence(evidence, receipt)
        self._commitments.pop(evidence.missile_id, None)
        self._deliveries[evidence.missile_id] = evidence
        self._record_event(event)

    def report_receipt(self, receipt: HitReceipt) -> None:
        if not isinstance(receipt, HitReceipt):
            raise TypeError("receipt must be HitReceipt")
        event = self._make_event(receipt.event_id, "receipt", (receipt,))
        if self._is_duplicate(event):
            return
        target = self._require_target(receipt.target_id)
        _damage(receipt.missile_type, target.target_type)
        self._validate_missile_assignment(
            receipt.missile_id, receipt.missile_type, receipt.target_id
        )
        delivery = self._deliveries.get(receipt.missile_id)
        if delivery is not None:
            self._validate_same_missile_evidence(delivery, receipt)
        previous = self._receipts.get(receipt.missile_id)
        if previous is not None and previous != receipt:
            raise ValueError(
                f"missile {receipt.missile_id} already has a different receipt"
            )
        self._commitments.pop(receipt.missile_id, None)
        self._receipts[receipt.missile_id] = receipt
        if receipt.target_destroyed:
            self._confirmed_target_ids.add(receipt.target_id)
        self._record_event(event)

    def confirm(self, target_id: int, *, event_id: str | None = None) -> None:
        _integer("target_id", target_id)
        _event_id(event_id)
        event = self._make_event(event_id, "confirm", (target_id,))
        if self._is_duplicate(event):
            return
        self._require_target(target_id)
        self._confirmed_target_ids.add(target_id)
        self._record_event(event)

    def estimate(self, target_id: int) -> TargetDamageProbabilityBelief:
        target = self._require_target(target_id)
        commitments = tuple(
            sorted(
                (
                    item
                    for item in self._commitments.values()
                    if item.target_id == target_id
                ),
                key=lambda item: item.missile_id,
            )
        )
        deliveries = tuple(
            sorted(
                (
                    item
                    for item in self._deliveries.values()
                    if item.target_id == target_id
                ),
                key=lambda item: item.missile_id,
            )
        )
        receipts = tuple(
            sorted(
                (
                    item
                    for item in self._receipts.values()
                    if item.target_id == target_id
                ),
                key=lambda item: item.missile_id,
            )
        )

        exact_damage = sum(item.applied_damage for item in receipts)
        receipt_missile_ids = {item.missile_id for item in receipts}
        delivered_variables = tuple(
            (
                _damage_interval(
                    item.missile_type,
                    target.target_type,
                    satellite_active=item.satellite_active,
                    adjusted_damage_override=item.adjusted_damage_override,
                ),
                self._delivery_probability(item),
            )
            for item in deliveries
            if item.missile_id not in receipt_missile_ids
        )
        commitment_variables = tuple(
            (
                _damage_interval(item.missile_type, target.target_type),
                self._commitment_probability(item),
            )
            for item in commitments
        )
        confirmed = target_id in self._confirmed_target_ids
        destruction_probability = self._destruction_probability(
            target.initial_health,
            exact_damage,
            tuple(
                (damage.nominal, probability)
                for damage, probability in delivered_variables
            ),
        )
        projected_probability = self._destruction_probability(
            target.initial_health,
            exact_damage,
            tuple(
                (damage.nominal, probability)
                for damage, probability in delivered_variables + commitment_variables
            ),
        )
        if confirmed:
            destruction_probability = 1.0
            projected_probability = 1.0

        delivered_damage_interval = self._expected_damage_interval(
            exact_damage, delivered_variables
        )
        projected_damage_interval = self._expected_damage_interval(
            exact_damage, delivered_variables + commitment_variables
        )
        return TargetDamageProbabilityBelief(
            target_id=target.target_id,
            target_type=target.target_type,
            initial_health=target.initial_health,
            destruction_probability=destruction_probability,
            projected_destruction_probability=projected_probability,
            confirmed_destroyed=confirmed,
            exact_received_damage=exact_damage,
            delivered_expected_damage=delivered_damage_interval.nominal,
            projected_expected_damage=projected_damage_interval.nominal,
            committed_missiles=tuple(item.missile_id for item in commitments),
            core_committed_missiles=tuple(
                item.missile_id
                for item in commitments
                if item.commitment_type == "core"
            ),
            flexible_committed_missiles=tuple(
                item.missile_id
                for item in commitments
                if item.commitment_type == "flexible"
            ),
            delivery_missiles=tuple(item.missile_id for item in deliveries),
            receipt_missiles=tuple(item.missile_id for item in receipts),
            delivery_probabilities=tuple(
                (item.missile_id, self._delivery_probability(item))
                for item in deliveries
                if item.missile_id not in receipt_missile_ids
            ),
            commitment_probabilities=tuple(
                (item.missile_id, self._commitment_probability(item))
                for item in commitments
            ),
            lower_expected_damage=delivered_damage_interval.lower,
            nominal_expected_damage=delivered_damage_interval.nominal,
            upper_expected_damage=delivered_damage_interval.upper,
            projected_lower_expected_damage=projected_damage_interval.lower,
            projected_nominal_expected_damage=projected_damage_interval.nominal,
            projected_upper_expected_damage=projected_damage_interval.upper,
        )

    def snapshot(
        self, target_id: int | None = None
    ) -> TargetDamageProbabilityBelief | tuple[TargetDamageProbabilityBelief, ...]:
        if target_id is not None:
            return self.estimate(target_id)
        return tuple(self.estimate(item) for item in sorted(self._targets))

    def export(self) -> EvidenceSnapshot:
        return EvidenceSnapshot(
            target_priors=tuple(self._targets[item] for item in sorted(self._targets)),
            commitments=tuple(
                self._commitments[item] for item in sorted(self._commitments)
            ),
            deliveries=tuple(
                self._deliveries[item] for item in sorted(self._deliveries)
            ),
            receipts=tuple(self._receipts[item] for item in sorted(self._receipts)),
            confirmed_target_ids=tuple(sorted(self._confirmed_target_ids)),
            processed_events=tuple(
                self._processed_events[item] for item in sorted(self._processed_events)
            ),
            core_commitment_probability=self._core_commitment_probability,
            flexible_commitment_probability=self._flexible_commitment_probability,
        )

    def replay(self, snapshot: EvidenceSnapshot) -> None:
        if not isinstance(snapshot, EvidenceSnapshot):
            raise TypeError("snapshot must be EvidenceSnapshot")
        rebuilt = type(self).from_snapshot(snapshot)
        self._core_commitment_probability = rebuilt._core_commitment_probability
        self._flexible_commitment_probability = (
            rebuilt._flexible_commitment_probability
        )
        self._targets = rebuilt._targets
        self._commitments = rebuilt._commitments
        self._deliveries = rebuilt._deliveries
        self._receipts = rebuilt._receipts
        self._confirmed_target_ids = rebuilt._confirmed_target_ids
        self._processed_events = rebuilt._processed_events

    @classmethod
    def from_snapshot(
        cls, snapshot: EvidenceSnapshot
    ) -> TargetDamageProbabilitySkill:
        if not isinstance(snapshot, EvidenceSnapshot):
            raise TypeError("snapshot must be EvidenceSnapshot")
        rebuilt = cls(
            core_commitment_probability=snapshot.core_commitment_probability,
            flexible_commitment_probability=snapshot.flexible_commitment_probability,
        )
        for evidence in snapshot.target_priors:
            rebuilt.register(evidence)
        for evidence in snapshot.deliveries:
            rebuilt.report_delivery(evidence)
        for receipt in snapshot.receipts:
            rebuilt.report_receipt(receipt)
        for evidence in snapshot.commitments:
            rebuilt.commit(evidence)
        for target_id in snapshot.confirmed_target_ids:
            rebuilt.confirm(target_id)
        for event in snapshot.processed_events:
            previous = rebuilt._processed_events.get(event.event_id)
            if previous is not None and previous != event:
                raise ValueError(
                    f"event_id {event.event_id!r} has conflicting replay payloads"
                )
            rebuilt._processed_events[event.event_id] = event
        return rebuilt

    def _require_target(self, target_id: int) -> TargetPriorEvidence:
        _integer("target_id", target_id)
        target = self._targets.get(target_id)
        if target is None:
            raise ValueError(f"unknown target ID {target_id}")
        return target

    def _validate_missile_assignment(
        self, missile_id: int, missile_type: int, target_id: int
    ) -> None:
        commitment = self._commitments.get(missile_id)
        if commitment is not None and (
            commitment.missile_type != missile_type
            or commitment.target_id != target_id
        ):
            raise ValueError(
                f"missile {missile_id} evidence conflicts with its commitment"
            )

    @staticmethod
    def _validate_same_missile_evidence(
        first: DeliveryEvidence | HitReceipt,
        second: DeliveryEvidence | HitReceipt,
    ) -> None:
        if (
            first.missile_id != second.missile_id
            or first.missile_type != second.missile_type
            or first.target_id != second.target_id
        ):
            raise ValueError("delivery and receipt identify different missiles or targets")

    def _commitment_probability(self, evidence: CommitmentEvidence) -> float:
        if evidence.commitment_type == "core":
            return self._core_commitment_probability
        return self._flexible_commitment_probability

    @staticmethod
    def _delivery_probability(evidence: DeliveryEvidence) -> float:
        if evidence.probability is not None:
            return float(evidence.probability)
        distance = evidence.minimum_target_distance_metres
        # Observations are second-granular while missiles can cover several
        # hundred metres per step; a final sample within 500 m is terminal-close.
        if distance is not None and distance <= 500.0:
            return 0.98
        if evidence.last_stage >= 4:
            return 0.85
        if distance is not None and distance <= 1200.0:
            return 0.65
        if evidence.disappeared:
            return 0.15
        return 0.0

    @staticmethod
    def _destruction_probability(
        health: float,
        exact_damage: float,
        variables: tuple[tuple[float, float], ...],
    ) -> float:
        distribution: dict[Decimal, float] = {Decimal(str(exact_damage)): 1.0}
        for damage, probability in variables:
            damage_value = Decimal(str(damage))
            next_distribution: dict[Decimal, float] = {}
            for total_damage, mass in distribution.items():
                nondelivery_mass = mass * (1.0 - probability)
                delivery_mass = mass * probability
                next_distribution[total_damage] = (
                    next_distribution.get(total_damage, 0.0) + nondelivery_mass
                )
                delivered_total = total_damage + damage_value
                next_distribution[delivered_total] = (
                    next_distribution.get(delivered_total, 0.0) + delivery_mass
                )
            distribution = next_distribution
        threshold = Decimal(str(health))
        result = sum(
            mass for total_damage, mass in distribution.items() if total_damage >= threshold
        )
        return round(min(1.0, max(0.0, result)), 15)

    @staticmethod
    def _expected_damage_interval(
        exact_damage: float,
        variables: tuple[tuple[DamageInterval, float], ...],
    ) -> DamageInterval:
        return DamageInterval(
            math.fsum(
                (exact_damage,)
                + tuple(
                    damage.lower * probability
                    for damage, probability in variables
                )
            ),
            math.fsum(
                (exact_damage,)
                + tuple(
                    damage.nominal * probability
                    for damage, probability in variables
                )
            ),
            math.fsum(
                (exact_damage,)
                + tuple(
                    damage.upper * probability
                    for damage, probability in variables
                )
            ),
        )

    @staticmethod
    def _make_event(
        event_id: str | None, operation: str, payload: tuple[object, ...]
    ) -> EvidenceEvent | None:
        if event_id is None:
            return None
        return EvidenceEvent(event_id, operation, payload)

    def _is_duplicate(self, event: EvidenceEvent | None) -> bool:
        if event is None:
            return False
        previous = self._processed_events.get(event.event_id)
        if previous is None:
            return False
        if previous != event:
            raise ValueError(f"event_id {event.event_id!r} was reused with new payload")
        return True

    def _record_event(self, event: EvidenceEvent | None) -> None:
        if event is not None:
            self._processed_events[event.event_id] = event


__all__ = [
    "CommitmentEvidence",
    "DamageInterval",
    "DeliveryEvidence",
    "EvidenceEvent",
    "EvidenceSnapshot",
    "HitReceipt",
    "TargetDamageProbabilityBelief",
    "TargetDamageProbabilitySkill",
    "TargetPriorEvidence",
]
