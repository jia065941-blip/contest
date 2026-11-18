"""Internal conversion of archived scenarios into the public local-metre schema."""

from copy import deepcopy
import math
from statistics import median


def _ecef(lon, lat, altitude=0.0):
    lon, lat = math.radians(lon), math.radians(lat)
    major, flattening = 6378137.0, 1.0 / 298.257223563
    eccentricity = flattening * (2.0 - flattening)
    radius = major / math.sqrt(1.0 - eccentricity * math.sin(lat) ** 2)
    return ((radius + altitude) * math.cos(lat) * math.cos(lon),
            (radius + altitude) * math.cos(lat) * math.sin(lon),
            (radius * (1.0 - eccentricity) + altitude) * math.sin(lat))


def _rings(coordinates):
    if not coordinates:
        return []
    if isinstance(coordinates[0][0], (int, float)):
        return [coordinates]
    return coordinates


def migrate_scene(data):
    """Copy a scene, preserving topology and archived layout proportions.

    Fully converted input is returned unchanged. Archived metric positions are
    recognized by their magnitude, so metric practice scenes are never
    projected a second time. Public geometry uses the former map drawing's
    centre-line chord lengths; no arbitrary model scale is applied.
    """
    result = deepcopy(data)
    profile = result["imagineProfile"]
    entities = [row["entity"] for row in profile["entityList"]]
    if all("pos" in entity for entity in entities) and "xMin" in profile["mapArea"]:
        for entity in entities:
            for key in ("lla", "posEcf", "velEcf", "velNue", "posNue"):
                entity.pop(key, None)
            entity.setdefault("vel", dict.fromkeys("xyz", 0.0))
        profile["redArea"]["coordinates"] = _rings(profile["redArea"]["coordinates"])
        return result
    local = median(math.sqrt(sum(float(entity.get("pos", entity.get("posEcf"))[axis]) ** 2
                                 for axis in "xyz")) for entity in entities) < 1_000_000
    area = profile["mapArea"]
    geographic_map = "lonMin" in area
    if geographic_map:
        lon = (area["lonMin"] + area["lonMax"]) / 2
        lat = (area["latMin"] + area["latMax"]) / 2
        width = math.dist(_ecef(area["lonMin"], lat), _ecef(area["lonMax"], lat))
        height = math.dist(_ecef(lon, area["latMin"]), _ecef(lon, area["latMax"]))
        def point(x, y):
            return [(x - lon) / (area["lonMax"] - area["lonMin"]) * width,
                    (y - lat) / (area["latMax"] - area["latMin"]) * height]
        bounds = {"xMin": -width / 2, "xMax": width / 2,
                  "yMin": -height / 2, "yMax": height / 2}
        red = [[point(*vertex) for vertex in ring]
               for ring in _rings(profile["redArea"]["coordinates"])]
    else:
        bounds = {key: float(area[key]) for key in ("xMin", "xMax", "yMin", "yMax")}
        red = _rings(profile["redArea"]["coordinates"])
    for entity in entities:
        if local:
            pos = dict(entity.get("pos", entity.get("posEcf")))
            vel = dict(entity.get("vel", entity.get("velEcf", dict.fromkeys("xyz", 0.0))))
        else:
            old = entity["lla"]
            x, y = point(old["x"], old["y"])
            pos = {"x": x, "y": y, "z": old["z"]}
            vx, vy, vz = (entity.get("velEcf", {}).get(axis, 0.0) for axis in "xyz")
            sl, cl = math.sin(math.radians(lon)), math.cos(math.radians(lon))
            sp, cp = math.sin(math.radians(lat)), math.cos(math.radians(lat))
            vel = {"x": -sl * vx + cl * vy,
                   "y": -sp * cl * vx - sp * sl * vy + cp * vz,
                   "z": cp * cl * vx + cp * sl * vy + sp * vz}
        for key in ("lla", "posEcf", "velEcf", "velNue", "posNue"):
            entity.pop(key, None)
        entity["pos"] = {axis: round(float(pos[axis]), 6) for axis in "xyz"}
        entity["vel"] = {axis: round(float(vel[axis]), 9) for axis in "xyz"}
    profile["mapArea"] = {key: round(value, 6) for key, value in bounds.items()}
    profile["redArea"]["coordinates"] = [[[round(float(value), 6) for value in vertex]
                                             for vertex in ring] for ring in red]
    return result
