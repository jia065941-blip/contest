"""Pure L response rules driven by target-health belief and local baselines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from target_damage_belief_skill import TargetHealthBelief


class LResponseRole(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    BACKUP = "backup"
    RESERVE = "reserve"
    LOST = "lost"


@dataclass(frozen=True)
class ResponseMemberSnapshot:
    entity_id: int
    target_id: int
    alive: bool
    launched: bool
    delivered: bool
    launch_step: int
    estimated_eta_seconds: float | None
    baseline_action: int = 1


@dataclass(frozen=True)
class StopCommitmentRequest:
    target_id: int
    release_member_ids: tuple[int, ...]
    reason: str
    confidence: float


@dataclass(frozen=True)
class SupplementRequest:
    target_id: int
    requested_members: int
    reason: str


@dataclass(frozen=True)
class RetargetRequest:
    from_target_id: int
    to_target_id: int
    member_ids: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class ResponsePackageStatus:
    target_id: int
    roles: tuple[tuple[int, LResponseRole], ...]
    live_member_ids: tuple[int, ...]
    launched_member_ids: tuple[int, ...]
    delivered_member_ids: tuple[int, ...]
    expected_hp: float
    projected_expected_hp: float
    eta_spread_seconds: float | None


@dataclass(frozen=True)
class ResponsePackageDecision:
    actions: tuple[tuple[int, int], ...]
    status: ResponsePackageStatus
    stop_commitment_request: StopCommitmentRequest | None
    supplement_request: SupplementRequest | None
    retarget_request: RetargetRequest | None


class LResponsePackageSkill:
    def __init__(
        self,
        *,
        min_members: int = 2,
        max_members: int = 4,
        damage_per_hit: float = 0.8,
        belief_confidence_threshold: float = 0.75,
    ) -> None:
        self.min_members = int(min_members)
        self.max_members = int(max_members)
        self.damage_per_hit = float(damage_per_hit)
        self.belief_confidence_threshold = float(belief_confidence_threshold)

    def reset(self) -> None:
        return None

    def decide(
        self,
        *,
        members: tuple[ResponseMemberSnapshot, ...],
        belief: TargetHealthBelief,
        authorized_fallback_target_ids: tuple[int, ...] = (),
    ) -> ResponsePackageDecision:
        if not self.min_members <= len(members) <= self.max_members:
            raise ValueError("L response package requires 2-4 members")
        ordered = tuple(sorted(members, key=lambda item: (item.launch_step, item.entity_id)))
        if len({item.entity_id for item in ordered}) != len(ordered):
            raise ValueError("response member IDs must be unique")
        if len({item.target_id for item in ordered}) != 1:
            raise ValueError("response members must share target ID")
        if ordered[0].target_id != belief.target_id or belief.target_type != 9500:
            raise ValueError("response belief must describe the same hidden 9500")
        if any(item.baseline_action not in (0, 1, 2) for item in ordered):
            raise ValueError("baseline actions must be 0, 1, or 2")

        roles = self._roles(ordered)
        live = tuple(item.entity_id for item in ordered if item.alive)
        launched = tuple(item.entity_id for item in ordered if item.alive and item.launched)
        delivered = tuple(item.entity_id for item in ordered if item.delivered)
        unlaunched = tuple(item.entity_id for item in ordered if item.alive and not item.launched)
        etas = [
            item.estimated_eta_seconds
            for item in ordered
            if item.alive and item.launched and item.estimated_eta_seconds is not None
        ]
        eta_spread = max(etas) - min(etas) if etas else None

        confident_dead = belief.confirmed_destroyed or (
            belief.expected_hp <= 0.0
            and belief.confidence >= self.belief_confidence_threshold
        )
        stop_request = None
        retarget_request = None
        if confident_dead and unlaunched:
            stop_request = StopCommitmentRequest(
                target_id=belief.target_id,
                release_member_ids=unlaunched,
                reason="target_damage_belief_satisfied",
                confidence=1.0 if belief.confirmed_destroyed else belief.confidence,
            )
        if confident_dead and authorized_fallback_target_ids and live:
            retarget_request = RetargetRequest(
                from_target_id=belief.target_id,
                to_target_id=int(authorized_fallback_target_ids[0]),
                member_ids=live,
                reason="target_confirmed_or_expected_destroyed",
            )

        supplement = None
        if not confident_dead and belief.projected_expected_hp > 0.0:
            required_hits = math.ceil(belief.projected_expected_hp / self.damage_per_hit)
            available_attempts = sum(
                item.alive and not item.delivered for item in ordered
            )
            if required_hits > available_attempts:
                supplement = SupplementRequest(
                    target_id=belief.target_id,
                    requested_members=min(
                        self.max_members - len(ordered),
                        required_hits - available_attempts,
                    ),
                    reason="projected_damage_shortfall",
                )
                if supplement.requested_members <= 0:
                    supplement = None

        return ResponsePackageDecision(
            actions=tuple(
                (item.entity_id, item.baseline_action)
                for item in ordered
                if item.alive and item.launched
            ),
            status=ResponsePackageStatus(
                target_id=belief.target_id,
                roles=tuple((item.entity_id, roles[item.entity_id]) for item in ordered),
                live_member_ids=live,
                launched_member_ids=launched,
                delivered_member_ids=delivered,
                expected_hp=belief.expected_hp,
                projected_expected_hp=belief.projected_expected_hp,
                eta_spread_seconds=eta_spread,
            ),
            stop_commitment_request=stop_request,
            supplement_request=supplement,
            retarget_request=retarget_request,
        )

    @staticmethod
    def _roles(members: tuple[ResponseMemberSnapshot, ...]) -> dict[int, LResponseRole]:
        roles = {item.entity_id: LResponseRole.LOST for item in members}
        live = [item for item in members if item.alive]
        labels = (
            (LResponseRole.PRIMARY, LResponseRole.BACKUP)
            if len(live) == 2
            else (
                LResponseRole.PRIMARY,
                LResponseRole.SECONDARY,
                LResponseRole.BACKUP,
                LResponseRole.RESERVE,
            )
        )
        for item, role in zip(live, labels, strict=False):
            roles[item.entity_id] = role
        return roles
