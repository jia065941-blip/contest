"""Render a recorded trajectory as the approved dual-view MP4."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from render_preview import (
    MAIN_BACKGROUND,
    MAIN_ALONG_FACTOR,
    MAIN_BLUE_ALONG_FACTOR,
    MAIN_BLUE_CROSS_FACTOR,
    MAIN_BLUE_SHIFT_LAT,
    MAIN_BLUE_SHIFT_LON,
    MAIN_CROSS_FACTOR,
    MAIN_OUTPUT_AXIS,
    MAIN_RED_ALONG_FACTOR,
    MAIN_VIEW_FRACTION,
    TOP_BACKGROUND,
    TOP_ALONG_FACTOR,
    TOP_BLUE_ALONG_FACTOR,
    TOP_BLUE_CROSS_FACTOR,
    TOP_BLUE_SHIFT_LAT,
    TOP_BLUE_SHIFT_LON,
    TOP_CROSS_FACTOR,
    TOP_RED_ALONG_FACTOR,
    apply_battlefield_transform,
    build_battlefield_transform,
    center_top_view_on_entities,
    configure_renderer,
    load_renderer_class,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY = ROOT / "sample_data/M01.trajectory.jsonl.gz"
DEFAULT_OUTPUT = ROOT / "output/M01_dual_view.mp4"


def restore_ids(entities: dict) -> dict:
    restored = {}
    for identifier, entity in entities.items():
        try:
            identifier = int(identifier)
        except (TypeError, ValueError):
            pass
        restored[identifier] = entity
    return restored


def sync_explosion_clock(renderer, birth_frames: dict, frame_index: int, fps: int):
    """Make explosion duration follow video time instead of render wall time."""
    now = time.perf_counter()
    for identifier, (position, _started) in tuple(renderer._explosions.items()):
        birth_frames.setdefault(identifier, frame_index)
        age = (frame_index - birth_frames[identifier]) / max(1, fps)
        renderer._explosions[identifier] = (position, now - age)
    for identifier in tuple(birth_frames):
        if identifier not in renderer._explosions:
            birth_frames.pop(identifier, None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--main-background", type=Path, default=MAIN_BACKGROUND)
    parser.add_argument("--top-background", type=Path, default=TOP_BACKGROUND)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="optional frame limit for smoke tests",
    )
    args = parser.parse_args(argv)

    if args.width % 2 or args.height % 2:
        raise ValueError("H.264 yuv420p output requires even width and height")

    metadata = None
    red_area = []
    initial = None
    with gzip.open(args.trajectory, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("record_type") == "metadata":
                metadata = record
            elif record.get("record_type") == "area":
                red_area = record.get("red_area", [])
            elif record.get("record_type") == "frame":
                initial = restore_ids(record["entities"])
                break
    if metadata is None or initial is None:
        raise ValueError("trajectory does not contain metadata and frames")

    bounds = tuple(metadata["bounds"])
    fps = int(metadata.get("video_fps", 30))
    divider = 4
    main_width = int((args.width - divider) * MAIN_VIEW_FRACTION)
    top_width = args.width - divider - main_width
    Renderer = load_renderer_class()

    pygame.init()
    pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
    main_view = configure_renderer(
        Renderer, bounds, (main_width, args.height), args.main_background,
        perspective=False, overview_rotation_deg=0.0, overview_zoom=0.48,
        background_opacity=0.92, background_haze_alpha=0,
    )
    top_view = configure_renderer(
        Renderer, bounds, (top_width, args.height), args.top_background,
        perspective=False,
    )

    main_transform = build_battlefield_transform(
        initial, along_factor=MAIN_ALONG_FACTOR,
        cross_factor=MAIN_CROSS_FACTOR,
        blue_cross_factor=MAIN_BLUE_CROSS_FACTOR,
        red_along_factor=MAIN_RED_ALONG_FACTOR,
        blue_along_factor=MAIN_BLUE_ALONG_FACTOR,
        blue_shift_lon=MAIN_BLUE_SHIFT_LON,
        blue_shift_lat=MAIN_BLUE_SHIFT_LAT,
        output_axis=MAIN_OUTPUT_AXIS,
    )
    top_transform = build_battlefield_transform(
        initial, along_factor=TOP_ALONG_FACTOR,
        cross_factor=TOP_CROSS_FACTOR,
        blue_cross_factor=TOP_BLUE_CROSS_FACTOR,
        red_along_factor=TOP_RED_ALONG_FACTOR,
        blue_along_factor=TOP_BLUE_ALONG_FACTOR,
        blue_shift_lon=TOP_BLUE_SHIFT_LON,
        blue_shift_lat=TOP_BLUE_SHIFT_LAT,
    )
    initial_main = apply_battlefield_transform(initial, main_transform)
    initial_top = apply_battlefield_transform(initial, top_transform)
    main_view.update_data(initial_main, step=0, wait=False)
    top_view.update_data(initial_top, step=0, wait=False)
    center_top_view_on_entities(top_view, top_view.entities_data)

    main_surface = pygame.Surface((main_width, args.height))
    top_surface = pygame.Surface((top_width, args.height))
    composed = pygame.Surface((args.width, args.height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{args.width}x{args.height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    main_births = {}
    top_births = {}
    frame_count = 0
    last_step = None
    started = time.time()
    try:
        with gzip.open(args.trajectory, "rt", encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if record.get("record_type") != "frame":
                    continue
                entities = restore_ids(record["entities"])
                step = int(record["step"])
                main_entities = apply_battlefield_transform(
                    entities, main_transform,
                )
                top_entities = apply_battlefield_transform(
                    entities, top_transform,
                )
                main_view.update_data(main_entities, step=step, wait=False)
                top_view.update_data(top_entities, step=step, wait=False)
                main_view.current_step = step
                top_view.current_step = step
                main_view._display_phase = 0.0
                top_view._display_phase = 0.0
                sync_explosion_clock(main_view, main_births, frame_count, fps)
                sync_explosion_clock(top_view, top_births, frame_count, fps)

                main_view._draw_live_view(
                    main_surface, main_view.entities_data, red_area,
                    show_coordinate=False, title="MAIN VIEW",
                    show_legend=False, show_hud=True,
                )
                top_view._draw_live_view(
                    top_surface, top_view.entities_data, red_area,
                    show_coordinate=False, title="TOP VIEW",
                    show_legend=True, show_hud=False,
                )
                composed.blit(main_surface, (0, 0))
                pygame.draw.rect(
                    composed, (46, 65, 76),
                    (main_width, 0, divider, args.height),
                )
                composed.blit(top_surface, (main_width + divider, 0))
                if encoder.stdin is None:
                    raise RuntimeError("FFmpeg stdin is unavailable")
                encoder.stdin.write(pygame.image.tostring(composed, "RGB"))
                frame_count += 1
                last_step = step
                if frame_count % 100 == 0:
                    print(f"rendered {frame_count} frames", flush=True)
                if args.max_frames is not None and frame_count >= args.max_frames:
                    break
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
        stderr = encoder.stderr.read().decode("utf-8", errors="replace") \
            if encoder.stderr is not None else ""
        return_code = encoder.wait()
        pygame.quit()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed with code {return_code}: {stderr}")

    status = {
        "completed": True,
        "trajectory": str(args.trajectory.resolve()),
        "main_background": str(args.main_background.resolve()),
        "top_background": str(args.top_background.resolve()),
        "video": str(args.output.resolve()),
        "resolution": [args.width, args.height],
        "video_fps": fps,
        "frames_written": frame_count,
        "last_step": last_step,
        "duration_seconds": frame_count / max(1, fps),
        "render_seconds": time.time() - started,
        "top_view_axes": False,
    }
    args.output.with_suffix(".status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"completed {frame_count} frames -> {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
