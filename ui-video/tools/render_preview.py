"""Render the initial state of a recorded trajectory as a dual-view PNG."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY = ROOT / "sample_data/M01.trajectory.jsonl.gz"
DEFAULT_OUTPUT = ROOT / "output/M01_dual_view_preview.png"
MAIN_BACKGROUND = ROOT / "assets/backgrounds/background.png"
TOP_BACKGROUND = ROOT / "assets/backgrounds/background.png"
RED_TYPES = (21000, 21001, 21002)
BLUE_ANCHOR_TYPES = (9400, 9500, 9600)
DISPLAY_TYPES = RED_TYPES + BLUE_ANCHOR_TYPES + (24000,)
MAIN_ALONG_FACTOR = 4.80
MAIN_RED_ALONG_FACTOR = 0.35
MAIN_BLUE_ALONG_FACTOR = 1.20
MAIN_CROSS_FACTOR = 1.00
MAIN_BLUE_CROSS_FACTOR = 1.85
MAIN_BLUE_SHIFT_LON = -0.03
MAIN_BLUE_SHIFT_LAT = -0.16
MAIN_OUTPUT_AXIS = (1.0, 0.0)
TOP_ALONG_FACTOR = 4.50
TOP_RED_ALONG_FACTOR = 0.35
TOP_BLUE_ALONG_FACTOR = 1.15
TOP_CROSS_FACTOR = 1.35
TOP_BLUE_CROSS_FACTOR = 1.90
TOP_BLUE_SHIFT_LON = 0.15
TOP_BLUE_SHIFT_LAT = 0.15
MAIN_VIEW_FRACTION = 0.75


def load_renderer_class():
    path = ROOT / "envengine/render/renderer.py"
    spec = importlib.util.spec_from_file_location("competition_ui_renderer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Renderer


def read_initial_state(path: Path):
    metadata = None
    area = []
    frame = None
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            kind = record.get("record_type")
            if kind == "metadata":
                metadata = record
            elif kind == "area":
                area = record.get("red_area", [])
            elif kind == "frame":
                frame = record
                break
    if metadata is None or frame is None:
        raise ValueError("trajectory does not contain metadata and an initial frame")
    entities = {}
    for identifier, entity in frame["entities"].items():
        try:
            identifier = int(identifier)
        except (TypeError, ValueError):
            pass
        entities[identifier] = entity
    return tuple(metadata["bounds"]), area, int(frame["step"]), entities


def build_battlefield_transform(
    entities: dict, along_factor, cross_factor=1.0,
    blue_cross_factor=None, red_along_factor=None,
    blue_along_factor=None, blue_shift_lon=0.0, blue_shift_lat=0.0,
    output_axis=None,
):
    """Build one affine transform shared by units, trajectories and effects."""
    red = [
        entity for entity in entities.values()
        if entity.get("side") == 0 and entity.get("type") in RED_TYPES
    ]
    blue = [
        entity for entity in entities.values()
        if entity.get("type") in BLUE_ANCHOR_TYPES
    ]
    if not red or not blue:
        return None

    def centroid(group):
        return (
            sum(float(item["position"]["lon"]) for item in group) / len(group),
            sum(float(item["position"]["lat"]) for item in group) / len(group),
        )

    red_center = centroid(red)
    blue_center = centroid(blue)
    axis_x = blue_center[0] - red_center[0]
    axis_y = blue_center[1] - red_center[1]
    length = math.hypot(axis_x, axis_y)
    if length <= 1e-12:
        axis_x, axis_y = 1.0, 0.0
    else:
        axis_x /= length
        axis_y /= length
    center = (
        (red_center[0] + blue_center[0]) * 0.5,
        (red_center[1] + blue_center[1]) * 0.5,
    )

    def along_of(entity):
        return (
            (float(entity["position"]["lon"]) - center[0]) * axis_x
            + (float(entity["position"]["lat"]) - center[1]) * axis_y
        )

    red_alongs = [along_of(entity) for entity in red]
    blue_alongs = [along_of(entity) for entity in blue]
    red_front_along = max(red_alongs)
    blue_rear_along = min(blue_alongs)
    along_midpoint = 0.0
    if red_along_factor is not None:
        def raw_warp(value):
            if value <= red_front_along:
                return (value - red_front_along) * float(red_along_factor)
            gap = (blue_rear_along - red_front_along) * float(along_factor)
            if value <= blue_rear_along or blue_along_factor is None:
                return (value - red_front_along) * float(along_factor)
            return gap + (value - blue_rear_along) * float(blue_along_factor)

        mapped_red = sum(raw_warp(value) for value in red_alongs) / len(red_alongs)
        mapped_blue = sum(raw_warp(value) for value in blue_alongs) / len(blue_alongs)
        along_midpoint = (mapped_red + mapped_blue) * 0.5

    if output_axis is None:
        output_axis_x, output_axis_y = axis_x, axis_y
    else:
        output_axis_x, output_axis_y = map(float, output_axis)
        output_length = math.hypot(output_axis_x, output_axis_y)
        if output_length <= 1e-12:
            raise ValueError("output_axis must be non-zero")
        output_axis_x /= output_length
        output_axis_y /= output_length

    return {
        "center": center,
        "axis": (axis_x, axis_y),
        "output_axis": (output_axis_x, output_axis_y),
        "along_factor": float(along_factor),
        "cross_factor": float(cross_factor),
        "blue_cross_factor": (
            float(blue_cross_factor)
            if blue_cross_factor is not None else None
        ),
        "red_along": -length * 0.5,
        "blue_along": length * 0.5,
        "red_along_factor": (
            float(red_along_factor)
            if red_along_factor is not None else None
        ),
        "red_front_along": red_front_along,
        "blue_rear_along": blue_rear_along,
        "blue_along_factor": (
            float(blue_along_factor)
            if blue_along_factor is not None else None
        ),
        "along_midpoint": along_midpoint,
        "blue_shift_lon": float(blue_shift_lon),
        "blue_shift_lat": float(blue_shift_lat),
    }


def apply_battlefield_transform(entities: dict, transform) -> dict:
    """Apply a coherent spatial transform to every positional entity."""
    result = deepcopy(entities)
    if transform is None:
        return result
    center_x, center_y = transform["center"]
    axis_x, axis_y = transform["axis"]
    output_axis_x, output_axis_y = transform.get(
        "output_axis", (axis_x, axis_y),
    )
    output_cross_x, output_cross_y = -output_axis_y, output_axis_x
    along_factor = transform["along_factor"]
    base_cross_factor = transform["cross_factor"]
    blue_cross_factor = transform.get("blue_cross_factor")
    for entity in result.values():
        position = entity.get("position")
        if not isinstance(position, dict):
            continue
        try:
            dx = float(position["lon"]) - center_x
            dy = float(position["lat"]) - center_y
        except (KeyError, TypeError, ValueError):
            continue
        along = dx * axis_x + dy * axis_y
        cross_factor = base_cross_factor
        if blue_cross_factor is not None:
            red_along = transform["red_along"]
            blue_along = transform["blue_along"]
            ratio = (along - red_along) / max(1e-12, blue_along - red_along)
            ratio = max(0.0, min(1.0, ratio))
            cross_factor += ratio * (blue_cross_factor - base_cross_factor)
        cross_signed = dx * (-axis_y) + dy * axis_x
        red_along_factor = transform.get("red_along_factor")
        if red_along_factor is None:
            scaled_along = along * along_factor
        else:
            red_front_along = transform["red_front_along"]
            blue_rear_along = transform["blue_rear_along"]
            blue_along_factor = transform.get("blue_along_factor")
            if along <= red_front_along:
                raw_along = (along - red_front_along) * red_along_factor
            elif along <= blue_rear_along or blue_along_factor is None:
                raw_along = (along - red_front_along) * along_factor
            else:
                gap = (blue_rear_along - red_front_along) * along_factor
                raw_along = gap + (along - blue_rear_along) * blue_along_factor
            scaled_along = raw_along - transform["along_midpoint"]
        shift_ratio = (along - transform["red_front_along"]) / max(
            1e-12,
            transform["blue_rear_along"] - transform["red_front_along"],
        )
        shift_ratio = max(0.0, min(1.0, shift_ratio))
        position["lon"] = (
            center_x + scaled_along * output_axis_x
            + cross_signed * output_cross_x * cross_factor
            + transform["blue_shift_lon"] * shift_ratio
        )
        position["lat"] = (
            center_y + scaled_along * output_axis_y
            + cross_signed * output_cross_y * cross_factor
            + transform["blue_shift_lat"] * shift_ratio
        )
    return result


def spread_main_view_entities(
    entities: dict, bounds, factor=MAIN_ALONG_FACTOR,
    cross_factor=MAIN_CROSS_FACTOR,
    blue_cross_factor=MAIN_BLUE_CROSS_FACTOR,
    red_along_factor=MAIN_RED_ALONG_FACTOR,
    blue_along_factor=MAIN_BLUE_ALONG_FACTOR,
):
    """Increase separation without breaking trajectory-to-target alignment."""
    del bounds
    return apply_battlefield_transform(
        entities, build_battlefield_transform(
            entities, factor, cross_factor=cross_factor,
            blue_cross_factor=blue_cross_factor,
            red_along_factor=red_along_factor,
            blue_along_factor=blue_along_factor,
            blue_shift_lon=MAIN_BLUE_SHIFT_LON,
            blue_shift_lat=MAIN_BLUE_SHIFT_LAT,
            output_axis=MAIN_OUTPUT_AXIS,
        ),
    )


def prepare_top_view_entities(
    entities: dict, bounds, factor=TOP_ALONG_FACTOR,
    cross_factor=TOP_CROSS_FACTOR,
    blue_cross_factor=TOP_BLUE_CROSS_FACTOR,
    red_along_factor=TOP_RED_ALONG_FACTOR,
    blue_along_factor=TOP_BLUE_ALONG_FACTOR,
):
    """Open both the force separation and within-force spacing in top view."""
    del bounds
    return apply_battlefield_transform(
        entities,
        build_battlefield_transform(
            entities, factor, cross_factor=cross_factor,
            blue_cross_factor=blue_cross_factor,
            red_along_factor=red_along_factor,
            blue_along_factor=blue_along_factor,
            blue_shift_lon=TOP_BLUE_SHIFT_LON,
            blue_shift_lat=TOP_BLUE_SHIFT_LAT,
        ),
    )


def configure_renderer(
    Renderer, bounds, size, background, *, perspective,
    overview_rotation_deg=90.0, overview_zoom=0.62,
    background_opacity=None, background_haze_alpha=None,
):
    renderer = Renderer(*bounds, fps=30, sim_step_seconds=1.0)
    renderer._configure_viewport(*size)
    renderer.show_satellite = False
    renderer.show_altitude_labels = False
    renderer.geographic_layout = "urban"
    renderer.background_rotation_deg = 0.0
    renderer.background_offset_y = 0.0
    renderer.background_opacity = (
        float(background_opacity)
        if background_opacity is not None
        else (0.92 if perspective else 0.34)
    )
    renderer.background_haze_alpha = background_haze_alpha
    renderer._background_master = pygame.image.load(str(background)).convert()
    renderer.view_3d = perspective
    renderer.view_rotation_deg = 0.0
    if perspective:
        renderer.zoom = 0.77
        renderer.offset_x = size[0] * 0.034
        renderer.offset_y = size[1] * 0.02
        renderer.range_ring_radius_px = 64
        renderer.camera_yaw_deg = 0.0
        renderer.camera_pitch_deg = 30.0
        renderer.altitude_pixels_per_km = 26.0
    else:
        renderer.zoom = float(overview_zoom)
        renderer.view_rotation_deg = float(overview_rotation_deg)
        renderer.offset_x = 0.0
        renderer.offset_y = 0.0
        renderer.range_ring_radius_px = 46
        renderer.background_rotation_deg = float(overview_rotation_deg)
    font_path = str(renderer.ui_font_path) if renderer.ui_font_path.is_file() else None
    renderer.font = pygame.font.Font(font_path, 14)
    renderer.font_legend = pygame.font.Font(font_path, 18)
    return renderer


def center_top_view_on_entities(renderer, entities):
    """Center the final rotated/scaled entity envelope in the top-view pixels."""
    points = []
    for entity in entities.values():
        if entity.get("type") not in DISPLAY_TYPES:
            continue
        if not entity.get("isVisible", False) or entity.get("health", 0) <= 0:
            continue
        position = entity.get("position", {})
        points.append(renderer._geo_to_screen_linear_float(
            float(position.get("lon", 0.0)),
            float(position.get("lat", 0.0)),
        ))
    if not points:
        return
    center_x = (min(x for x, _ in points) + max(x for x, _ in points)) * 0.5
    center_y = (min(y for _, y in points) + max(y for _, y in points)) * 0.5
    renderer.offset_x += renderer.MAP_W * 0.5 - center_x
    renderer.offset_y += renderer.MAP_H * 0.5 - center_y


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--main-background", type=Path, default=MAIN_BACKGROUND)
    parser.add_argument("--top-background", type=Path, default=TOP_BACKGROUND)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1377)
    args = parser.parse_args(argv)

    width, height = args.width, args.height
    divider = 4
    main_width = int((width - divider) * MAIN_VIEW_FRACTION)
    top_width = width - divider - main_width
    bounds, red_area, step, entities = read_initial_state(args.trajectory)
    Renderer = load_renderer_class()

    pygame.init()
    pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
    main_view = configure_renderer(
        Renderer, bounds, (main_width, height), args.main_background,
        perspective=False, overview_rotation_deg=0.0, overview_zoom=0.48,
        background_opacity=0.92, background_haze_alpha=0,
    )
    top_view = configure_renderer(
        Renderer, bounds, (top_width, height), args.top_background,
        perspective=False,
    )
    main_transform = build_battlefield_transform(
        entities, MAIN_ALONG_FACTOR, cross_factor=MAIN_CROSS_FACTOR,
        blue_cross_factor=MAIN_BLUE_CROSS_FACTOR,
        red_along_factor=MAIN_RED_ALONG_FACTOR,
        blue_along_factor=MAIN_BLUE_ALONG_FACTOR,
        blue_shift_lon=MAIN_BLUE_SHIFT_LON,
        blue_shift_lat=MAIN_BLUE_SHIFT_LAT,
        output_axis=MAIN_OUTPUT_AXIS,
    )
    top_transform = build_battlefield_transform(
        entities, TOP_ALONG_FACTOR, cross_factor=TOP_CROSS_FACTOR,
        blue_cross_factor=TOP_BLUE_CROSS_FACTOR,
        red_along_factor=TOP_RED_ALONG_FACTOR,
        blue_along_factor=TOP_BLUE_ALONG_FACTOR,
        blue_shift_lon=TOP_BLUE_SHIFT_LON,
        blue_shift_lat=TOP_BLUE_SHIFT_LAT,
    )
    main_entities = apply_battlefield_transform(entities, main_transform)
    top_entities = apply_battlefield_transform(entities, top_transform)
    main_view.update_data(main_entities, step=step, wait=False)
    top_view.update_data(top_entities, step=step, wait=False)
    center_top_view_on_entities(top_view, top_view.entities_data)

    main_surface = pygame.Surface((main_width, height))
    top_surface = pygame.Surface((top_width, height))
    composed = pygame.Surface((width, height))

    main_view.current_step = step
    top_view.current_step = step
    main_view._display_phase = 0.0
    top_view._display_phase = 0.0
    main_view._draw_live_view(
        main_surface, main_view.entities_data, red_area,
        show_coordinate=False,
        title="MAIN VIEW",
        show_legend=False,
        show_hud=True,
    )
    top_view._draw_live_view(
        top_surface, top_view.entities_data, red_area,
        show_coordinate=False,
        title="TOP VIEW",
        show_legend=True,
        show_hud=False,
    )
    composed.blit(main_surface, (0, 0))
    pygame.draw.rect(composed, (46, 65, 76), (main_width, 0, divider, height))
    composed.blit(top_surface, (main_width + divider, 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(composed, str(args.output))
    pygame.quit()
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
