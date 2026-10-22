"""Probabilistic damage accounting without engine-global truth access."""

from __future__ import annotations

from dataclasses import dataclass


_BASE_DAMAGE = {
    21000: {9400: 16.0, 9600: 12.0, 9500: 0.0},
    21001: {9400: 4.0, 9600: 3.0, 9500: 0.0},
    21002: {9400: 0.0, 9600: 0.0, 9500: 0.8},
}
_MIN_DAMAGE = {
    21000: {9400: 9.6, 9600: 12.0, 9500: 0.0},
    21001: {9400: 2.4, 9600: 3.0, 9500: 0.0},
    21002: {9400: 0.0, 9600: 0.0, 9500: 0.8},
}
_MAX_DAMAGE = {
    21000: {9400: 20.0, 9600: 20.0, 9500: 0.0},
    21001: {9400: 4.8, 9600: 3.0, 9500: 0.0},
    21002: {9400: 0.0, 9600: 0.0, 9500: 0.8},
}


@dataclass(frozen=True)
class TargetPriorEvidence:
    target_id: int
    target_type: int
    initial_health: float
    discovered_step: int = 0
    source: str = "initial_catalogue"


@dataclass(frozen=True)
class DeliveryEvidence:
    missile_id: int
    missile_type: int
    target_id: int
    step: int
    last_stage: int
    minimum_target_distance_metres: float | None
    disappeared: bool = True
    satellite_active: bool = False
    event_id: str | None = None


@dataclass(frozen=True)
class HitReceipt:
    """Optional future legal receipt supplied by the environment/controller."""

    missile_id: int
    missile_type: int
    target_id: int
    step: int
    applied_damage: float
    target_destroyed: bool = False
    event_id: str | None = None


@dataclass(frozen=True)
class TargetHealthBelief:
    target_id: int
    target_type: int
    initial_health: float
    lower_hp: float
    expected_hp: float
    upper_hp: float
    projected_expected_hp: float
    confidence: float
    confirmed_destroyed: bool
    committed_missiles: tuple[int, ...]
    delivery_missiles: tuple[int, ...]
    receipt_missiles: tuple[int, ...]
    delivered_expected_damage: float = 0.0
    confidence_source: str = "heuristic_delivery"


class TargetDamageBeliefSkill:
    """Maintain target-health intervals from explicitly supplied legal facts."""

    def __init__(self) -> None:
        self._targets: dict[int, TargetPriorEvidence] = {}
        self._commitments: dict[int, dict[int, tuple[int, bool]]] = {}
        self._deliveries: dict[int, dict[int, DeliveryEvidence]] = {}
        self._receipts: dict[int, dict[int, HitReceipt]] = {}
        self._confirmed_destroyed: set[int] = set()
        self._commitment_target: dict[int, int] = {}
        self._processed_event_ids: set[str] = set()

    def reset(self) -> None:
        self._targets.clear()
        self._commitments.clear()
        self._deliveries.clear()
        self._receipts.clear()
        self._confirmed_destroyed.clear()
        self._commitment_target.clear()
        self._processed_event_ids.clear()

    def register_target(self, evidence: TargetPriorEvidence) -> None:
        if evidence.target_type not in (9400, 9500, 9600):
            raise ValueError("unsupported score target type")
        if evidence.initial_health <= 0.0:
            raise ValueError("initial target health must be positive")
        previous = self._targets.get(int(evidence.target_id))
        if previous is not None and previous != evidence:
            raise ValueError("target prior changed after registration")
        self._targets[int(evidence.target_id)] = evidence

    def commit_missile(
        self,
        *,
        missile_id: int,
        missile_type: int,
        target_id: int,
        satellite_expected: bool = False,
    ) -> None:
        self._require_target(target_id)
        self._nominal_damage(missile_type, target_id, satellite_expected)
        previous_target = self._commitment_target.get(int(missile_id))
        if previous_target is not None and previous_target != int(target_id):
            self._commitments.get(previous_target, {}).pop(int(missile_id), None)
        self._commitments.setdefault(int(target_id), {})[int(missile_id)] = (
            int(missile_type),
            bool(satellite_expected),
        )
        self._commitment_target[int(missile_id)] = int(target_id)

    def transfer_missile(self, missile_id: int, target_id: int) -> None:
        missile_id = int(missile_id)
        old_target = self._commitment_target.get(missile_id)
        if old_target is None:
            raise ValueError(f"missile {missile_id} has no active commitment")
        if old_target == int(target_id):
            return
        missile_type, satellite = self._commitments[old_target][missile_id]
        self.commit_missile(
            missile_id=missile_id,
            missile_type=missile_type,
            target_id=int(target_id),
            satellite_expected=satellite,
        )

    def release_missile(self, missile_id: int) -> None:
        missile_id = int(missile_id)
        target_id = self._commitment_target.pop(missile_id, None)
        if target_id is not None:
            self._commitments.get(target_id, {}).pop(missile_id, None)

    def report_delivery(self, evidence: DeliveryEvidence) -> None:
        if evidence.event_id is not None and evidence.event_id in self._processed_event_ids:
            return
        self._require_target(evidence.target_id)
        self._nominal_damage(
            evidence.missile_type,
            evidence.target_id,
            evidence.satellite_active,
        )
        self.release_missile(evidence.missile_id)
        self._deliveries.setdefault(int(evidence.target_id), {})[
            int(evidence.missile_id)
        ] = evidence
        if evidence.event_id is not None:
            self._processed_event_ids.add(evidence.event_id)

    def report_hit_receipt(self, receipt: HitReceipt) -> None:
        if receipt.event_id is not None and receipt.event_id in self._processed_event_ids:
            return
        self._require_target(receipt.target_id)
        if receipt.applied_damage < 0.0:
            raise ValueError("applied damage must be non-negative")
        self.release_missile(receipt.missile_id)
        self._receipts.setdefault(int(receipt.target_id), {})[
            int(receipt.missile_id)
        ] = receipt
        if receipt.target_destroyed:
            self._confirmed_destroyed.add(int(receipt.target_id))
        if receipt.event_id is not None:
            self._processed_event_ids.add(receipt.event_id)

    def confirm_destroyed(self, target_id: int) -> None:
        self._require_target(target_id)
        self._confirmed_destroyed.add(int(target_id))

    def estimate(self, target_id: int) -> TargetHealthBelief:
        target = self._require_target(target_id)
        deliveries = self._deliveries.get(int(target_id), {})
        receipts = self._receipts.get(int(target_id), {})
        exact_damage = sum(item.applied_damage for item in receipts.values())
        lower_damage = exact_damage
        expected_damage = exact_damage
        upper_damage = exact_damage
        heuristic_count = 0
        for missile_id, evidence in deliveries.items():
            if missile_id in receipts:
                continue
            probability = self._delivery_probability(evidence)
            nominal = self._nominal_damage(
                evidence.missile_type,
                target_id,
                evidence.satellite_active,
            )
            maximum = self._damage_bound(
                _MAX_DAMAGE, evidence.missile_type, target.target_type
            )
            expected_damage += nominal * probability
            upper_damage += maximum * min(1.0, probability + 0.25)
            heuristic_count += 1

        committed = self._commitments.get(int(target_id), {})
        committed_damage = sum(
            self._nominal_damage(missile_type, target_id, satellite)
            for missile_type, satellite in committed.values()
        )
        destroyed = int(target_id) in self._confirmed_destroyed
        if destroyed:
            lower_hp = expected_hp = upper_hp = projected_hp = 0.0
            confidence = 1.0
        else:
            lower_hp = max(0.0, target.initial_health - upper_damage)
            expected_hp = max(0.0, target.initial_health - expected_damage)
            upper_hp = max(0.0, target.initial_health - lower_damage)
            projected_hp = max(0.0, expected_hp - committed_damage)
            receipt_fraction = min(1.0, exact_damage / target.initial_health)
            confidence = min(0.95, receipt_fraction + 0.15 * heuristic_count)
        return TargetHealthBelief(
            target_id=target.target_id,
            target_type=target.target_type,
            initial_health=target.initial_health,
            lower_hp=lower_hp,
            expected_hp=expected_hp,
            upper_hp=upper_hp,
            projected_expected_hp=projected_hp,
            confidence=confidence,
            confirmed_destroyed=destroyed,
            committed_missiles=tuple(sorted(committed)),
            delivery_missiles=tuple(sorted(deliveries)),
            receipt_missiles=tuple(sorted(receipts)),
            delivered_expected_damage=expected_damage,
            confidence_source=(
                "confirmed_receipt"
                if receipts
                else ("heuristic_delivery" if deliveries else "prior_only")
            ),
        )

    def snapshot(self) -> tuple[TargetHealthBelief, ...]:
        return tuple(self.estimate(target_id) for target_id in sorted(self._targets))

    def _require_target(self, target_id: int) -> TargetPriorEvidence:
        target = self._targets.get(int(target_id))
        if target is None:
            raise ValueError(f"unknown target ID {target_id}")
        return target

    def _nominal_damage(
        self, missile_type: int, target_id: int, satellite_active: bool
    ) -> float:
        target = self._require_target(target_id)
        if satellite_active and int(missile_type) == 21000 and target.target_type in (9400, 9600):
            return 20.0
        return self._damage_bound(_BASE_DAMAGE, missile_type, target.target_type)

    @staticmethod
    def _damage_bound(table: dict, missile_type: int, target_type: int) -> float:
        try:
            return float(table[int(missile_type)][int(target_type)])
        except KeyError as error:
            raise ValueError("unsupported missile/target damage pair") from error

    @staticmethod
    def _delivery_probability(evidence: DeliveryEvidence) -> float:
        distance = evidence.minimum_target_distance_metres
        if distance is not None and distance <= 100.0:
            return 0.98
        if evidence.last_stage >= 4:
            return 0.85
        if distance is not None and distance <= 1200.0:
            return 0.65
        return 0.15 if evidence.disappeared else 0.0


__all__ = [
    "DeliveryEvidence",
    "HitReceipt",
    "TargetDamageBeliefSkill",
    "TargetHealthBelief",
    "TargetPriorEvidence",
]
