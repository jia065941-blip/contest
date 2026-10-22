"""Pure M package rules; the selected local network owns all maneuver actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from target_damage_belief_skill import TargetHealthBelief


class MPackageRole(str, Enum):
    PRIMARY = "primary"
    SUPPORT = "support"
    FOLLOWUP_PRIMARY = "followup_primary"
    FOLLOWUP_SUPPORT = "followup_support"
    BACKUP = "backup"
    RESERVE = "reserve"
    LOST = "lost"


@dataclass(frozen=True)
class MMemberSnapshot:
    entity_id: int
    missile_type: int
    target_id: int
    target_type: int
    planned_release_step: int
    alive: bool
    launched: bool
    delivered: bool
    baseline_action: int


@dataclass(frozen=True)
class MStrikeMissionContext:
    belief: TargetHealthBelief
    current_step: int
    last_accepted_release_step: int | None
    delivery_timeout_steps: int
    authorized_fallback_target_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class MStopRequest:
    target_id: int
    member_ids: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class MSupplementRequest:
    target_id: int
    full_unmet_count: int
    recommended_grant_count: int
    reason: str


@dataclass(frozen=True)
class MRetargetRequest:
    from_target_id: int
    authorized_target_ids: tuple[int, ...]
    member_ids: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class MStrikePackageStatus:
    target_id: int
    target_type: int
    roles: tuple[tuple[int, MPackageRole], ...]
    live_member_ids: tuple[int, ...]
    launched_member_ids: tuple[int, ...]
    delivered_member_ids: tuple[int, ...]
    expected_hp: float
    projected_expected_hp: float


@dataclass(frozen=True)
class MStrikePackageDecision:
    status: MStrikePackageStatus
    motion_overrides: tuple[tuple[int, int], ...]
    stop_request: MStopRequest | None
    supplement_request: MSupplementRequest | None
    retarget_request: MRetargetRequest | None


class MStrikePackageSkill:
    def __init__(
        self,
        *,
        min_members: int = 2,
        max_members: int = 6,
        maximum_release_span_steps: int = 10,
        belief_confidence_threshold: float = 0.75,
    ) -> None:
        self.min_members = min_members
        self.max_members = max_members
        self.maximum_release_span_steps = maximum_release_span_steps
        self.belief_confidence_threshold = belief_confidence_threshold

    def reset(self) -> None:
        return None

    def decide(
        self,
        *,
        members: tuple[MMemberSnapshot, ...],
        context: MStrikeMissionContext,
    ) -> MStrikePackageDecision:
        if not self.min_members <= len(members) <= self.max_members:
            raise ValueError("M package requires 2-6 members")
        ordered = tuple(
            sorted(members, key=lambda item: (item.planned_release_step, item.entity_id))
        )
        if len({item.entity_id for item in ordered}) != len(ordered):
            raise ValueError("M member IDs must be unique")
        if any(item.missile_type != 21001 for item in ordered):
            raise ValueError("M package accepts only type 21001")
        if len({item.target_id for item in ordered}) != 1:
            raise ValueError("M package members must share target ID")
        if len({item.target_type for item in ordered}) != 1:
            raise ValueError("M package members must share target type")
        if ordered[0].target_type not in (9400, 9600):
            raise ValueError("M package target must be public 9400/9600")
        if (
            ordered[-1].planned_release_step - ordered[0].planned_release_step
            > self.maximum_release_span_steps
        ):
            raise ValueError("M package release span exceeds limit")
        if context.belief.target_id != ordered[0].target_id:
            raise ValueError("M belief target mismatch")
        if any(item.baseline_action not in (0, 1, 2) for item in ordered):
            raise ValueError("baseline actions must be 0, 1, or 2")

        roles = self._roles(ordered)
        live = tuple(item.entity_id for item in ordered if item.alive)
        launched = tuple(item.entity_id for item in ordered if item.alive and item.launched)
        delivered = tuple(item.entity_id for item in ordered if item.delivered)
        unlaunched = tuple(
            item.entity_id
            for item in ordered
            if item.alive and not item.launched and not item.delivered
            and roles[item.entity_id] not in (
                MPackageRole.PRIMARY,
                MPackageRole.SUPPORT,
            )
        )
        launched_undelivered = tuple(
            item.entity_id
            for item in ordered
            if item.alive and item.launched and not item.delivered
        )
        belief = context.belief
        confirmed_or_delivered_satisfied = belief.confirmed_destroyed or (
            belief.expected_hp <= 0.0
            and belief.confidence >= self.belief_confidence_threshold
        )
        projected_satisfied = (
            belief.projected_expected_hp <= 0.0
            and belief.confidence >= self.belief_confidence_threshold
        )

        stop = None
        if unlaunched and (belief.confirmed_destroyed or projected_satisfied):
            stop = MStopRequest(
                target_id=belief.target_id,
                member_ids=unlaunched,
                reason="confirmed_or_projected_commitment_satisfied",
            )
        retarget = None
        if (
            confirmed_or_delivered_satisfied
            and launched_undelivered
            and context.authorized_fallback_target_ids
        ):
            retarget = MRetargetRequest(
                from_target_id=belief.target_id,
                authorized_target_ids=tuple(context.authorized_fallback_target_ids),
                member_ids=launched_undelivered,
                reason="delivered_damage_satisfied",
            )

        supplement = None
        timeout_elapsed = (
            context.last_accepted_release_step is not None
            and context.current_step - context.last_accepted_release_step
            >= context.delivery_timeout_steps
        )
        if (
            not confirmed_or_delivered_satisfied
            and timeout_elapsed
            and belief.projected_expected_hp > 0.0
        ):
            lower_damage = 2.4 if ordered[0].target_type == 9400 else 3.0
            full_count = math.ceil(belief.projected_expected_hp / lower_damage)
            supplement = MSupplementRequest(
                target_id=belief.target_id,
                full_unmet_count=full_count,
                recommended_grant_count=min(2, full_count),
                reason="conservative_damage_shortfall_after_timeout",
            )

        return MStrikePackageDecision(
            status=MStrikePackageStatus(
                target_id=ordered[0].target_id,
                target_type=ordered[0].target_type,
                roles=tuple((item.entity_id, roles[item.entity_id]) for item in ordered),
                live_member_ids=live,
                launched_member_ids=launched,
                delivered_member_ids=delivered,
                expected_hp=belief.expected_hp,
                projected_expected_hp=belief.projected_expected_hp,
            ),
            motion_overrides=(),
            stop_request=stop,
            supplement_request=supplement,
            retarget_request=retarget,
        )

    @staticmethod
    def _roles(members: tuple[MMemberSnapshot, ...]) -> dict[int, MPackageRole]:
        templates = {
            2: (MPackageRole.PRIMARY, MPackageRole.SUPPORT),
            3: (MPackageRole.PRIMARY, MPackageRole.SUPPORT, MPackageRole.BACKUP),
            4: (
                MPackageRole.PRIMARY,
                MPackageRole.SUPPORT,
                MPackageRole.FOLLOWUP_PRIMARY,
                MPackageRole.FOLLOWUP_SUPPORT,
            ),
            5: (
                MPackageRole.PRIMARY,
                MPackageRole.SUPPORT,
                MPackageRole.FOLLOWUP_PRIMARY,
                MPackageRole.FOLLOWUP_SUPPORT,
                MPackageRole.BACKUP,
            ),
            6: (
                MPackageRole.PRIMARY,
                MPackageRole.SUPPORT,
                MPackageRole.FOLLOWUP_PRIMARY,
                MPackageRole.FOLLOWUP_SUPPORT,
                MPackageRole.BACKUP,
                MPackageRole.RESERVE,
            ),
        }
        roles = {}
        for item, role in zip(members, templates[len(members)], strict=True):
            roles[item.entity_id] = role if item.alive else MPackageRole.LOST
        return roles
