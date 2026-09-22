"""Legal red-side track fusion from isolated detectInfo observations."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .baselines import TargetPrior
from .contracts import Position
from .objectives import objective_value


@dataclass(frozen=True)
class DetectionProvenance:
    """One legally received objective track and the platform that produced it."""

    target_id: int
    detect_from: int
    step: int
    position: Position
    via_satellite: bool


class InitialCatalogueTrackFusion:
    """Refine public targets and admit legally detected hidden unmanned ships.

    DetectInfo carries an ID, type and kinematics, but no target health.
    This component updates identity and location only for objective tracks
    received through isolated platform observations.
    """

    OBJECTIVE_TYPES = frozenset({9400, 9500, 9600})

    def __init__(self, targets: tuple[TargetPrior, ...]) -> None:
        self._targets = {item.entity_id: item for item in targets}
        self._provenance: dict[int, list[DetectionProvenance]] = {}
        self._current_source: dict[int, int] = {}

    @property
    def targets(self) -> tuple[TargetPrior, ...]:
        return tuple(sorted(self._targets.values(), key=lambda item: item.entity_id))

    def provenance_for(self, target_id: int) -> tuple[DetectionProvenance, ...]:
        """Return every distinct legal observation retained for ``target_id``."""

        return tuple(self._provenance.get(int(target_id), ()))

    def source_for(self, target_id: int) -> int | None:
        """Return the platform supplying the currently fused target position."""

        return self._current_source.get(int(target_id))

    @property
    def provenance(self) -> dict[int, tuple[DetectionProvenance, ...]]:
        return {
            target_id: tuple(records)
            for target_id, records in self._provenance.items()
        }

    def ingest(self, observation: dict) -> bool:
        """Fuse one platform's received tracks and report whether a track changed."""

        changed = False
        detect_info = observation.get("self", {}).get("detectInfo") or {}
        observation_source = int(observation.get("entity_id", -1))
        observation_step = int(observation.get("step", 0))
        for raw_id, track in detect_info.items():
            entity_id = int(self._field(track, "entity_id", raw_id))
            current = self._targets.get(entity_id)
            entity_type = int(
                self._field(
                    track,
                    "entity_type",
                    current.entity_type if current is not None else -1,
                )
            )
            if entity_type not in self.OBJECTIVE_TYPES:
                continue
            if current is not None and entity_type != current.entity_type:
                continue
            lla = self._field(track, "lla", None)
            if lla is None:
                continue
            position = Position(
                lon=float(
                    self._field(lla, "x", current.position.lon if current else 0.0)
                ),
                lat=float(
                    self._field(lla, "y", current.position.lat if current else 0.0)
                ),
                alt=float(
                    self._field(lla, "z", current.position.alt if current else 0.0)
                ),
            )
            raw_detect_from = self._field(
                track,
                "detect_from",
                self._field(track, "source_entity_id", observation_source),
            )
            if raw_detect_from is None:
                raw_detect_from = observation_source
            raw_detection_time = self._field(track, "time", 0.0)
            detect_from = int(raw_detect_from)
            if "sim_time" not in observation or "sim_step" not in observation:
                # Legacy callers supplied detection time directly in steps.
                detection_step = max(
                    0,
                    observation_step
                    if raw_detection_time is None
                    else int(raw_detection_time),
                )
            else:
                sim_step = max(1.0, float(observation["sim_step"]))
                sim_time = float(observation["sim_time"])
                detection_time = float(
                    sim_time if raw_detection_time is None else raw_detection_time
                )
                age_steps = max(
                    0,
                    int(
                        math.ceil(
                            max(0.0, sim_time - detection_time) / sim_step
                        )
                    ) - 1,
                )
                detection_step = max(0, observation_step - age_steps)
            via_satellite = bool(
                self._field(track, "via_satellite", False)
            )
            if detect_from >= 0:
                record = DetectionProvenance(
                    target_id=entity_id,
                    detect_from=detect_from,
                    step=detection_step,
                    position=position,
                    via_satellite=via_satellite,
                )
                records = self._provenance.setdefault(entity_id, [])
                if not records or records[-1] != record:
                    records.append(record)
                self._current_source[entity_id] = detect_from
            if current is not None and position == current.position:
                continue
            self._targets[entity_id] = TargetPrior(
                entity_id=entity_id,
                entity_type=entity_type,
                position=position,
                value=(
                    current.value
                    if current is not None
                    else objective_value(entity_type)
                ),
                alive=current.alive if current is not None else True,
            )
            changed = True
        return changed

    @staticmethod
    def _field(value, name: str, default):
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)
