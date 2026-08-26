# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class ThreatTrack:
    target_id: int
    name: str
    type: int
    lla: Vector3
    pos_ecf: Vector3
    vel_ecf: Vector3
    detect_time: int


@dataclass(frozen=True)
class InterceptorState:
    interceptor_id: int
    lla: Vector3
    pos_ecf: Vector3
    vel_ecf: Vector3
    available: bool


@dataclass(frozen=True)
class DefendedAsset:
    asset_id: int
    name: str
    lla: Vector3
    pos_ecf: Vector3
    health: float
    value: float


@dataclass
class BlueObservation:
    sim_time: float
    targets: list[ThreatTrack]
    interceptors: list[InterceptorState]
    assets: list[DefendedAsset]
    launched_map: dict[int, list[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class InterceptorAssignment:
    interceptor_id: int
    target_id: int
    target_pos_ecf: Vector3
    target_vel_ecf: Vector3
    priority: float = 0.0


class BluePolicy(ABC):
    name = "base"

    def __init__(self, max_shots_per_target: int = 5):
        self.max_shots_per_target = max(1, int(max_shots_per_target))

    @abstractmethod
    def decide(self, observation: BlueObservation) -> list[InterceptorAssignment]:
        pass

    def available_interceptors(self, observation: BlueObservation) -> list[InterceptorState]:
        return [item for item in observation.interceptors if item.available]

    def remaining_need(self, observation: BlueObservation, target_id: int) -> int:
        launched_count = len(observation.launched_map.get(target_id, []))
        return max(0, self.max_shots_per_target - launched_count)


def distance_m(a: Vector3, b: Vector3, ignore_z: bool = False) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = 0.0 if ignore_z else a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def speed_mps(v: Vector3, default: float = 1.0) -> float:
    speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    return speed if speed > 1e-6 else default
