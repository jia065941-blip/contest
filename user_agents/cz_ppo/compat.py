"""Public Cartesian/legacy-geographic compatibility for the frozen CZ policy.

The frozen policy was authored against the previous public lon/lat contract.
The new official maps are a deterministic affine projection of that contract.
Only public map bounds and actor observations are converted here; no hidden
scene entity is read by this adapter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


# Archived public map bounds.  These are the only map-specific geometry priors
# used by the adapter.  Objective and dark-hive coordinates are not included.
OLD_BOUNDS = {
    "E01": (11.381719629926717, 12.51, 1.9625, 2.6975),
    "E02": (11.40906223175471, 12.51, 1.9625, 2.6975),
    "E03": (11.339404629926717, 12.51, 1.9625, 2.6975),
    "E04": (11.407545169158729, 12.51, 1.9625, 2.6975),
    "E05": (11.376297169158729, 12.51, 1.9625, 2.6975),
    "E06": (11.380017169158729, 12.51, 1.9625, 2.6975),
    "M01": (11.328648576212728, 12.665, 1.90125, 2.75875),
    "M02": (11.28215551528645, 12.665, 1.90125, 2.75875),
    "M03": (11.279411916825357, 12.665, 1.90125, 2.75875),
    "M04": (11.218651916825357, 12.665, 1.90125, 2.75875),
    "M05": (11.311468598619096, 12.665, 1.90125, 2.75875),
    "M06": (11.314289598619096, 12.665, 1.90125, 2.75875),
    "H01": (11.17165229401687, 12.82, 1.84, 2.82),
    "H02": (11.200771192119555, 12.82, 1.84, 2.82),
    "H03": (11.160439447726873, 12.82, 1.84, 2.82),
    "H04": (11.18454734291579, 12.82, 1.84, 2.82),
    "H05": (11.178940573762202, 12.82, 1.84, 2.82),
    "H06": (11.15060229401687, 12.82, 1.84, 2.82),
    "H07": (11.159439447726873, 12.82, 1.84, 2.82),
    "H08": (11.184060897370242, 12.82, 1.84, 2.82),
}

# Public Cartesian bounds serialized by the official scene migration.  Keeping
# the serialized values (rather than recomputing the chord and rounding again)
# is required because the archived source bounds themselves have more precision
# than the compact compatibility table above on H03.
NEW_BOUNDS = {
    "E01": (-62747.211386, 62747.211386, -40636.447751, 40636.447751),
    "E02": (-61226.650878, 61226.650878, -40636.447751, 40636.447751),
    "E03": (-65100.400922, 65100.400922, -40636.447751, 40636.447751),
    "E04": (-61311.016953, 61311.016953, -40636.447751, 40636.447751),
    "E05": (-63048.761627, 63048.761627, -40636.447751, 40636.447751),
    "E06": (-62841.887504, 62841.887504, -40636.447751, 40636.447751),
    "M01": (-74318.211776, 74318.211776, -47409.074005, 47409.074005),
    "M02": (-76903.696535, 76903.696535, -47409.074005, 47409.074005),
    "M03": (-77056.267987, 77056.267987, -47409.074005, 47409.074005),
    "M04": (-80435.118812, 80435.118812, -47409.074005, 47409.074005),
    "M05": (-75273.593862, 75273.593862, -47409.074005, 47409.074005),
    "M06": (-75116.717588, 75116.717588, -47409.074005, 47409.074005),
    "H01": (-91668.107512, 91668.107512, -54181.647165, 54181.647165),
    "H02": (-90048.852658, 90048.852658, -54181.647165, 54181.647165),
    "H03": (-92291.634196, 92291.634196, -54181.647165, 54181.647165),
    "H04": (-90951.035411, 90951.035411, -54181.647165, 54181.647165),
    "H05": (-91262.818608, 91262.818608, -54181.647165, 54181.647165),
    "H06": (-92838.660306, 92838.660306, -54181.647165, 54181.647165),
    "H07": (-92347.242397, 92347.242397, -54181.647165, 54181.647165),
    "H08": (-90978.085855, 90978.085855, -54181.647165, 54181.647165),
}


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _ecef(lon, lat, altitude=0.0):
    longitude, latitude = math.radians(lon), math.radians(lat)
    major, flattening = 6378137.0, 1.0 / 298.257223563
    eccentricity = flattening * (2.0 - flattening)
    radius = major / math.sqrt(1.0 - eccentricity * math.sin(latitude) ** 2)
    return {
        "x": (radius + altitude) * math.cos(latitude) * math.cos(longitude),
        "y": (radius + altitude) * math.cos(latitude) * math.sin(longitude),
        "z": (radius * (1.0 - eccentricity) + altitude) * math.sin(latitude),
    }


class CoordinateAdapter:
    """Invert the official affine map projection and rebuild the old local ABI."""

    def __init__(self, map_name: str):
        if map_name not in OLD_BOUNDS:
            raise ValueError(f"Unsupported frozen CZ map: {map_name}")
        self.map_name = map_name
        self.lon_min, self.lon_max, self.lat_min, self.lat_max = OLD_BOUNDS[map_name]
        self.x_min, self.x_max, self.y_min, self.y_max = NEW_BOUNDS[map_name]

    @property
    def old_map_area(self):
        return {
            "lonMin": self.lon_min,
            "lonMax": self.lon_max,
            "latMin": self.lat_min,
            "latMax": self.lat_max,
        }

    @property
    def new_map_area(self):
        return {
            "xMin": self.x_min,
            "xMax": self.x_max,
            "yMin": self.y_min,
            "yMax": self.y_max,
        }

    def to_old_point(self, point):
        if isinstance(point, Mapping) or hasattr(point, "x"):
            x, y, z = (
                float(field(point, "x", 0.0)),
                float(field(point, "y", 0.0)),
                float(field(point, "z", 0.0)),
            )
        else:
            x, y = float(point[0]), float(point[1])
            z = float(point[2]) if len(point) > 2 else 0.0
        lon = self.lon_min + (x - self.x_min) / (self.x_max - self.x_min) * (self.lon_max - self.lon_min)
        lat = self.lat_min + (y - self.y_min) / (self.y_max - self.y_min) * (self.lat_max - self.lat_min)
        return [lon, lat, z]

    def to_new_point(self, point):
        lon, lat = float(point[0]), float(point[1])
        x = self.x_min + (lon - self.lon_min) / (self.lon_max - self.lon_min) * (self.x_max - self.x_min)
        y = self.y_min + (lat - self.lat_min) / (self.lat_max - self.lat_min) * (self.y_max - self.y_min)
        return [x, y]

    def bound_new_point(self, point):
        """Apply the final public waypoint contract without altering a track."""
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("The public waypoint must be finite")
        return [
            max(self.x_min, min(self.x_max, x)),
            max(self.y_min, min(self.y_max, y)),
        ]

    @staticmethod
    def _ecef_velocity(lon, lat, velocity):
        east = float(field(velocity, "x", 0.0))
        north = float(field(velocity, "y", 0.0))
        up = float(field(velocity, "z", 0.0))
        longitude, latitude = math.radians(lon), math.radians(lat)
        sl, cl = math.sin(longitude), math.cos(longitude)
        sp, cp = math.sin(latitude), math.cos(latitude)
        return {
            "x": -sl * east - sp * cl * north + cp * cl * up,
            "y": cl * east - sp * sl * north + cp * sl * up,
            "z": cp * north + sp * up,
        }

    def _legacy_vector_state(self, position, velocity):
        lon, lat, altitude = self.to_old_point(position)
        return (
            {"lon": lon, "lat": lat, "alt": altitude},
            _ecef(lon, lat, altitude),
            self._ecef_velocity(lon, lat, velocity or {}),
        )

    def public_initial_observation(self, entities):
        converted = {}
        for raw_id, entity in entities.items():
            lon, lat, altitude = self.to_old_point(entity["pos"])
            converted[int(raw_id)] = {
                "position": {"lon": lon, "lat": lat, "alt": altitude},
                "health": float(entity.get("health", 0.0)),
                "type": int(entity["type"]),
            }
        return {"entities": converted}

    def observation(self, observation):
        own = observation["self"]
        position, pos_ecf, vel_ecf = self._legacy_vector_state(own["pos"], own.get("vel", {}))
        detections = {}
        for raw_id, track in (own.get("detectInfo") or {}).items():
            track_pos = field(track, "pos", {})
            track_vel = field(track, "vel", {})
            lla, track_ecf, track_vel_ecf = self._legacy_vector_state(track_pos, track_vel)
            detected_step = int(field(track, "detected_step", field(track, "step", observation["step"])))
            entity_id = int(field(track, "entity_id", raw_id))
            detections[entity_id] = {
                "detect_from": int(field(track, "detect_from", -1)),
                "entity_id": entity_id,
                "entity_type": int(field(track, "entity_type", -1)),
                "nameChn": field(track, "nameChn", ""),
                "lla": {"x": lla["lon"], "y": lla["lat"], "z": lla["alt"]},
                "pos_ecf": track_ecf,
                "vel_ecf": track_vel_ecf,
                "time": detected_step * 1000,
            }
        communications = {}
        for raw_id, record in (own.get("commRangeInfo") or {}).items():
            _, peer_pos, peer_vel = self._legacy_vector_state(record.get("pos", {}), record.get("vel", {}))
            communications[int(raw_id)] = {"posEcf": peer_pos, "velEcf": peer_vel}
        return {
            "step": int(observation["step"]),
            "entity_id": int(observation["entity_id"]),
            "is_using_satellite": bool(observation.get("is_using_satellite", False)),
            "self": {
                "nameChn": own.get("nameChn", ""),
                "position": position,
                "pos_ecf": pos_ecf,
                "vel_ecf": vel_ecf,
                "stage": own.get("stage", 0),
                "health": own.get("health", 0),
                "isVisible": own.get("isVisible", True),
                "type": int(own["type"]),
                "side": own.get("side", 0),
                "commRangeInfo": communications,
                "detectInfo": detections,
            },
        }
