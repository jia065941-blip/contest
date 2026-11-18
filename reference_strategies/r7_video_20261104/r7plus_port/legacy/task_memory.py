from dataclasses import dataclass

from .contracts import Position


@dataclass(frozen=True)
class PlatformTask:
    entity_id: int
    target_id: int | None
    guidance: tuple[int | None, float, float] | None
    planned_step: int
    search_destination: Position | None = None
    search_retargeted: bool = False
    confirmed_losses: tuple[tuple[int, int], ...] = ()
    redirects: tuple[tuple[int, int, int], ...] = ()
    search_claims: tuple[tuple[int, int], ...] = ()
