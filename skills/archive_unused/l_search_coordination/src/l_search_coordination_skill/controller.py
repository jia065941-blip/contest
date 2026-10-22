"""Pure coordination rules for L search members; no simulator access."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchMemberSnapshot:
    entity_id: int
    search_cell: int
    alive: bool
    launched: bool
    report_step: int
    discovered_target_ids: tuple[int, ...] = ()
    baseline_action: int = 1


@dataclass(frozen=True)
class SearchReassignmentRequest:
    lost_entity_id: int
    uncovered_cell: int
    reason: str


@dataclass(frozen=True)
class SatelliteRequest:
    reason: str
    missing_discoveries: int
    last_discovery_step: int | None


@dataclass(frozen=True)
class SearchCoordinationStatus:
    assigned_cells: tuple[int, ...]
    live_cells: tuple[int, ...]
    uncovered_cells: tuple[int, ...]
    discovered_target_ids: tuple[int, ...]
    required_discoveries: int
    response_ready: bool


@dataclass(frozen=True)
class SearchCoordinationDecision:
    actions: tuple[tuple[int, int], ...]
    status: SearchCoordinationStatus
    reassignment_requests: tuple[SearchReassignmentRequest, ...]
    satellite_request: SatelliteRequest | None


class LSearchCoordinationSkill:
    def __init__(
        self,
        *,
        expected_cells: int = 32,
        discovery_deadline_step: int = 80,
        quiet_steps_before_satellite: int = 20,
        maximum_report_age_steps: int = 1,
    ) -> None:
        self.expected_cells = int(expected_cells)
        self.discovery_deadline_step = int(discovery_deadline_step)
        self.quiet_steps_before_satellite = int(quiet_steps_before_satellite)
        self.maximum_report_age_steps = int(maximum_report_age_steps)
        self._discovered_ids: set[int] = set()
        self._last_discovery_step: int | None = None

    def reset(self) -> None:
        self._discovered_ids.clear()
        self._last_discovery_step = None

    def decide(
        self,
        *,
        step: int,
        members: tuple[SearchMemberSnapshot, ...],
        required_discoveries: int,
    ) -> SearchCoordinationDecision:
        ordered = tuple(sorted(members, key=lambda item: (item.search_cell, item.entity_id)))
        cells = [item.search_cell for item in ordered]
        if len(cells) != len(set(cells)):
            raise ValueError("search-cell ownership must be unique")
        if any(cell < 0 or cell >= self.expected_cells for cell in cells):
            raise ValueError("search cell is outside the legal grid")
        if any(item.baseline_action not in (0, 1, 2) for item in ordered):
            raise ValueError("baseline actions must be 0, 1, or 2")

        before = len(self._discovered_ids)
        for item in ordered:
            self._discovered_ids.update(int(value) for value in item.discovered_target_ids)
        if len(self._discovered_ids) > before:
            self._last_discovery_step = int(step)

        live = tuple(
            item.search_cell
            for item in ordered
            if item.alive and step - item.report_step <= self.maximum_report_age_steps
        )
        uncovered = tuple(sorted(set(cells) - set(live)))
        reassignments = tuple(
            SearchReassignmentRequest(
                lost_entity_id=item.entity_id,
                uncovered_cell=item.search_cell,
                reason="member_lost_or_stale",
            )
            for item in ordered
            if item.search_cell in uncovered
        )
        response_ready = len(self._discovered_ids) >= int(required_discoveries)
        satellite = None
        quiet_for = (
            step if self._last_discovery_step is None
            else step - self._last_discovery_step
        )
        if (
            not response_ready
            and step >= self.discovery_deadline_step
            and quiet_for >= self.quiet_steps_before_satellite
        ):
            satellite = SatelliteRequest(
                reason="discovery_deadline_without_required_targets",
                missing_discoveries=max(
                    0, int(required_discoveries) - len(self._discovered_ids)
                ),
                last_discovery_step=self._last_discovery_step,
            )
        return SearchCoordinationDecision(
            actions=tuple((item.entity_id, item.baseline_action) for item in ordered),
            status=SearchCoordinationStatus(
                assigned_cells=tuple(cells),
                live_cells=live,
                uncovered_cells=uncovered,
                discovered_target_ids=tuple(sorted(self._discovered_ids)),
                required_discoveries=int(required_discoveries),
                response_ready=response_ready,
            ),
            reassignment_requests=reassignments,
            satellite_request=satellite,
        )
