"""Translate the public Cartesian scene contract to the private model SDK."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass

from envengine.common.BaseStruct import Vector3d


def entity_position(entity) -> Vector3d:
    """Copy an engine entity position in public metres."""
    return Vector3d(entity.posEcf.x, entity.posEcf.y, entity.posEcf.z)


def vector_dict(vector) -> dict[str, float]:
    if isinstance(vector, Mapping):
        return {axis: float(vector[axis]) for axis in ("x", "y", "z")}
    return {axis: float(getattr(vector, axis)) for axis in ("x", "y", "z")}


def profile_from_scene(data):
    """Create an independent SDK profile without exporting SDK field names."""
    from envengine.sdk.base_struct.profile.profile import Profile
    from envengine.simulator.util.scaled_coordinates import xyz_to_lla_position
    from envengine.simulator.interfaces import ISimulator

    native = deepcopy(data)
    regions = native["imagineProfile"]["redArea"]["coordinates"]
    if len(regions) != 1:
        raise ValueError("A scene must contain exactly one deployment polygon")
    native["imagineProfile"]["redArea"]["coordinates"] = regions[0]
    for row in native["imagineProfile"]["entityList"]:
        entity = row["entity"]
        entity["posEcf"] = entity.pop("pos")
        entity["velEcf"] = entity.pop("vel", {"x": 0.0, "y": 0.0, "z": 0.0})
    profile = Profile.from_dict(native)
    # for row in profile.imagineProfile.entityList:
    #     entity = row.entity
    #     entity.lla = xyz_to_lla_position(entity.posEcf * ISimulator.MODEL_SCALE)
    return profile


def detection_dict(track):
    """Export documented track fields using acquisition steps."""
    if is_dataclass(track):
        source = asdict(track)
    elif isinstance(track, Mapping):
        source = track
    else:
        raise TypeError("探测记录必须是字典或探测数据对象")
    result = {
        name: deepcopy(source[name])
        for name in ("detect_from", "entity_id", "entity_type", "nameChn")
        if name in source
    }
    for name in ("pos", "vel"):
        if name in source:
            result[name] = vector_dict(source[name])
    if "detected_step" in source:
        result["detected_step"] = max(0, int(source["detected_step"]))
    elif "step" in source:
        result["detected_step"] = max(0, int(source["step"]))
    return result


def public_detections(tracks):
    return {int(identifier): track
            for identifier, track in (tracks or {}).items()}


def public_communications(records):
    """Copy the Cartesian summaries rather than serializing internal objects."""
    return {
        int(identifier): {
            name: vector_dict(record[name]) for name in ("pos", "vel") if name in record
        }
        for identifier, record in (records or {}).items()
    }
