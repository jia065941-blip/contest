"""Build strict target-conditioned causal candidate sets from event ledgers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class CausalCandidateError(RuntimeError):
    """Raised when a positive target reward lacks auditable causal evidence."""


def _event_type(event: Mapping[str, Any]) -> str:
    return str(event.get("event_type", "")).strip().upper()


def _step(event: Mapping[str, Any]) -> int:
    return int(event.get("step", event.get("timestep", -1)))


def _target_id(event: Mapping[str, Any]) -> int | None:
    value = event.get("target_id", event.get("target_entity_id"))
    return None if value is None else int(value)


def _latest(
    events: Sequence[Mapping[str, Any]],
    *,
    entity_id: int,
    at_or_before: int,
    event_types: set[str],
    target_id: int | None = None,
    after_or_at: int | None = None,
    nonzero_action: bool = False,
) -> Mapping[str, Any] | None:
    matches = []
    for event in events:
        if int(event.get("entity_id", -1)) != entity_id:
            continue
        event_step = _step(event)
        if event_step > at_or_before:
            continue
        if after_or_at is not None and event_step < after_or_at:
            continue
        if _event_type(event) not in event_types:
            continue
        if target_id is not None and _target_id(event) != target_id:
            continue
        if nonzero_action:
            try:
                if int(event.get("action", 0)) == 0:
                    continue
            except (TypeError, ValueError):
                continue
        matches.append(event)
    if not matches:
        return None
    return max(matches, key=lambda row: (_step(row), int(row.get("event_id", 0))))


def build_factual_target_events(
    *,
    episode_id: int | str,
    timestep: int,
    target_rewards: Mapping[int | str, float],
    decision_events: Sequence[Mapping[str, Any]],
    simulator_events: Sequence[Mapping[str, Any]],
    entity_to_agent: Mapping[int, int],
    epsilon: float = 1e-12,
) -> tuple[dict[str, Any], ...]:
    """Return one strict causal-candidate record per positive target reward.

    Damage candidates require an actual positive-damage ledger event at this
    physics step. Search candidates are admitted only when the damaging
    attack's LAUNCH/RETARGET decision explicitly records that search source.
    Intercept candidates require an actual interceptor-launch event and a
    causal option targeting the rewarded objective. Satellite candidates must
    be cited by an effect-bearing hit, attack decision, or intercept maneuver
    and resolve to a recorded SATELLITE_REQUEST. Everyone else is absent, which
    makes their target-conditioned alpha exactly zero by construction.
    """

    timestep = int(timestep)
    decisions = tuple(dict(row) for row in decision_events)
    simulator = tuple(dict(row) for row in simulator_events)
    events: list[dict[str, Any]] = []
    for target_key, raw_reward in target_rewards.items():
        reward = float(raw_reward)
        if reward <= epsilon:
            continue
        target_id = int(target_key)
        candidates: dict[int, dict[str, Any]] = {}

        def add_candidate(
            *,
            entity_id: int,
            anchor: Mapping[str, Any],
            category: str,
            direct_damage: float = 0.0,
        ) -> None:
            agent_id = entity_to_agent.get(entity_id)
            if agent_id is None:
                raw_agent = anchor.get("agent_id")
                if raw_agent is None:
                    raise CausalCandidateError(
                        f"entity {entity_id} has causal evidence but no agent mapping"
                    )
                agent_id = int(raw_agent)
            tau = _step(anchor)
            if tau < 0 or tau > timestep:
                raise CausalCandidateError(
                    f"entity {entity_id} has invalid causal tau={tau} for reward t={timestep}"
                )
            existing = candidates.get(entity_id)
            if existing is None:
                candidates[entity_id] = {
                    "agent_id": int(agent_id),
                    "entity_id": entity_id,
                    "tau": tau,
                    "decision_type": _event_type(anchor),
                    "categories": {category},
                    "direct_damage": max(0.0, float(direct_damage)),
                }
                return
            existing["categories"].add(category)
            existing["direct_damage"] += max(0.0, float(direct_damage))
            # A same-step satellite request is the causal action component to
            # null when it is at least as recent as an existing anchor.
            if tau > int(existing["tau"]) or (
                category == "satellite" and tau == int(existing["tau"])
            ):
                existing["tau"] = tau
                existing["decision_type"] = _event_type(anchor)

        def add_satellite_candidate(
            raw_entity_id: Any,
            *,
            at_or_before: int,
            evidence: str,
        ) -> None:
            if raw_entity_id is None:
                return
            satellite_entity_id = int(raw_entity_id)
            satellite_anchor = _latest(
                decisions,
                entity_id=satellite_entity_id,
                at_or_before=at_or_before,
                event_types={"SATELLITE_REQUEST"},
            )
            if satellite_anchor is None:
                raise CausalCandidateError(
                    f"{evidence} cites satellite source {satellite_entity_id} "
                    "without a SATELLITE_REQUEST anchor"
                )
            add_candidate(
                entity_id=satellite_entity_id,
                anchor=satellite_anchor,
                category="satellite",
            )

        def add_latest_accepted_satellite_candidate(
            *,
            at_or_before: int,
            evidence: str,
        ) -> None:
            accepted_requests = [
                row for row in simulator
                if _event_type(row) == "SATELLITE_REQUEST_ACCEPTED"
                and _step(row) <= at_or_before
            ]
            if not accepted_requests:
                raise CausalCandidateError(
                    f"{evidence} has a satellite effect without an accepted request"
                )
            accepted_request = max(
                enumerate(accepted_requests),
                key=lambda item: (_step(item[1]), item[0]),
            )[1]
            add_satellite_candidate(
                accepted_request.get("entity_id"),
                at_or_before=_step(accepted_request),
                evidence=evidence,
            )

        direct_hits = [
            row for row in simulator
            if _event_type(row) == "DIRECT_HIT"
            and int(row.get("target_entity_id", -1)) == target_id
            and _step(row) == timestep
            and float(row.get("actual_damage", 0.0)) > epsilon
        ]
        if not direct_hits:
            raise CausalCandidateError(
                f"positive reward for target {target_id} at t={timestep} has no direct-hit evidence"
            )

        for hit in direct_hits:
            attacker_id = int(hit["attacking_entity_id"])
            attack_decision = _latest(
                decisions,
                entity_id=attacker_id,
                at_or_before=timestep - 1,
                event_types={"LAUNCH", "RETARGET", "TARGET_SELECT"},
                target_id=target_id,
            )
            if attack_decision is None:
                raise CausalCandidateError(
                    f"damage entity {attacker_id} lacks LAUNCH/RETARGET anchor for target {target_id}"
                )
            add_candidate(
                entity_id=attacker_id,
                anchor=attack_decision,
                category="damage",
                direct_damage=float(hit["actual_damage"]),
            )
            if hit.get("satellite_hit_rate_effect_entity_id") is not None:
                add_latest_accepted_satellite_candidate(
                    at_or_before=timestep,
                    evidence=f"direct hit on target {target_id}",
                )
            add_satellite_candidate(
                attack_decision.get("satellite_source"),
                at_or_before=_step(attack_decision),
                evidence=(
                    f"attack decision by entity {attacker_id} for target {target_id}"
                ),
            )
            search_source = attack_decision.get("search_source")
            if search_source is not None:
                search_id = int(search_source)
                search_anchor = _latest(
                    decisions,
                    entity_id=search_id,
                    at_or_before=_step(attack_decision),
                    event_types={"SEARCH", "REGION_SEARCH", "AREA_SEARCH"},
                )
                if search_anchor is None:
                    raise CausalCandidateError(
                        f"attack cites search source {search_id} without a SEARCH anchor"
                    )
                add_candidate(
                    entity_id=search_id,
                    anchor=search_anchor,
                    category="search",
                )

        for intercept in simulator:
            if _event_type(intercept) != "INTERCEPTOR_LAUNCH":
                continue
            intercept_step = _step(intercept)
            if intercept_step < 0 or intercept_step > timestep:
                continue
            red_entity_id = int(intercept["intercepted_red_entity_id"])
            target_option = _latest(
                decisions,
                entity_id=red_entity_id,
                at_or_before=intercept_step,
                event_types={"LAUNCH", "RETARGET", "TARGET_SELECT"},
                target_id=target_id,
            )
            if target_option is None:
                continue
            maneuver = _latest(
                decisions,
                entity_id=red_entity_id,
                at_or_before=intercept_step,
                after_or_at=_step(target_option),
                event_types={"MANEUVER"},
                nonzero_action=True,
            )
            add_candidate(
                entity_id=red_entity_id,
                anchor=maneuver or target_option,
                category="intercept",
            )
            if (
                maneuver is not None
                and maneuver.get("satellite_source") is not None
            ):
                add_latest_accepted_satellite_candidate(
                    at_or_before=_step(maneuver),
                    evidence=(
                        f"intercept maneuver by entity {red_entity_id}"
                    ),
                )

        normalized = []
        for entity_id in sorted(candidates):
            item = candidates[entity_id]
            item["categories"] = sorted(item["categories"])
            normalized.append(item)
        if not normalized:
            raise CausalCandidateError(
                f"positive reward for target {target_id} has an empty candidate set"
            )
        events.append({
            "event_id": f"{episode_id}:{timestep}:{target_id}",
            "target_id": target_id,
            "timestep": timestep,
            "reward": reward,
            "candidates": normalized,
        })
    return tuple(events)


__all__ = ["CausalCandidateError", "build_factual_target_events"]
