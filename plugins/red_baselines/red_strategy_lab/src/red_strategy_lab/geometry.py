"""Geometry helpers for valid deployment generation."""

from __future__ import annotations

from math import cos, hypot, radians
from typing import Iterable, Sequence

from .domain import Position


def distance_km(a: Position, b: Position) -> float:
    """Equirectangular distance; sufficient for regional assignment ranking."""
    lat_scale = 111.32
    x = (a.lon - b.lon) * lat_scale * cos(radians((a.lat + b.lat) / 2))
    y = (a.lat - b.lat) * lat_scale
    return hypot(x, y)


def point_in_polygon(point: Position, polygon: Sequence[Position]) -> bool:
    if len(polygon) < 3:
        raise ValueError("a deployment polygon needs at least three points")
    inside = False
    previous = polygon[-1]
    for current in polygon:
        crosses = (current.lat > point.lat) != (previous.lat > point.lat)
        if crosses:
            boundary_lon = (previous.lon - current.lon) * (point.lat - current.lat) / (previous.lat - current.lat) + current.lon
            if point.lon < boundary_lon:
                inside = not inside
        previous = current
    return inside


def spaced_points(polygon: Sequence[Position], count: int, min_distance_km: float) -> list[Position]:
    """Deterministic lattice sampler with polygon and separation constraints."""
    if count < 0 or min_distance_km < 0:
        raise ValueError("count and min_distance_km must be non-negative")
    if count == 0:
        return []
    lon_min, lon_max = min(p.lon for p in polygon), max(p.lon for p in polygon)
    lat_min, lat_max = min(p.lat for p in polygon), max(p.lat for p in polygon)
    resolution = max(4, count * 6)
    candidates: list[Position] = []
    for row in range(resolution + 1):
        for column in range(resolution + 1):
            point = Position(
                lon_min + (lon_max - lon_min) * column / resolution,
                lat_min + (lat_max - lat_min) * row / resolution,
            )
            if point_in_polygon(point, polygon):
                candidates.append(point)
    selected: list[Position] = []
    for candidate in candidates:
        if all(distance_km(candidate, existing) >= min_distance_km for existing in selected):
            selected.append(candidate)
            if len(selected) == count:
                return selected
    raise ValueError("polygon cannot provide enough separated deployment points")

