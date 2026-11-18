"""从场景初态提取部署区域和高度约束。"""

import math

from envengine.environment.scene_adapter import entity_position

RED_PLATFORM_TYPES = (21000, 21001, 21002)


def resolve_deployment_constraints(profile):
    """使用场景部署区；高度一致时保留共用高度，否则各单位沿用自身高度。"""
    coordinates = profile.imagineProfile.redArea.coordinates
    if coordinates and coordinates[0] and isinstance(coordinates[0][0], (int, float)):
        coordinates = [coordinates]
    altitudes = [
        float(entity_position(item.entity).z)
        for item in profile.imagineProfile.entityList
        if item.entity.entityType in RED_PLATFORM_TYPES
    ]
    if any(not math.isfinite(altitude) for altitude in altitudes):
        raise ValueError("人族单位的初始部署高度必须是有限数值")
    launch_altitude = None
    if altitudes and all(abs(altitude - altitudes[0]) <= 1e-6 for altitude in altitudes):
        launch_altitude = altitudes[0]
    return coordinates, launch_altitude
