# Trajectory Video Renderer

This directory contains the minimal code and assets required to generate the
approved dual-view UI video from an existing recorded trajectory. It does not
contain the simulator, policies, checkpoints, rendered frames, previews, or
finished videos.

## Included

- `sample_data/M01.trajectory.jsonl.gz`: recorded M01 trajectory
- `tools/render_preview.py`: render the initial frame for a quick layout check
- `tools/render_video.py`: stream trajectory frames directly to FFmpeg
- `envengine/render/renderer.py`: standalone Pygame renderer
- `assets/`: runtime background, font, and hive sprites

Optional satellite-style backgrounds are available in
`assets/backgrounds/`. To use one in both views, pass its path through
`--main-background` and `--top-background`, for example:

```bash
python tools/render_video.py \
  --main-background assets/backgrounds/300km虚构海峡卫星底图_v8_hq_1920x1080.png \
  --top-background assets/backgrounds/300km虚构海峡卫星底图_v8_hq_1920x1080.png \
  --output output/run.mp4
```

The bundled Noto Sans CJK font is distributed under the SIL Open Font License
1.1; see `assets/fonts/OFL.txt`.

## Requirements

- Python 3.10 or newer
- FFmpeg with H.264 (`libx264`) support

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Render an initial-state preview:

```bash
python tools/render_preview.py
```

Render the complete MP4:

```bash
python tools/render_video.py
```

Outputs are written to `output/` and are intentionally ignored by Git.

Custom trajectories using the same JSONL schema can be supplied explicitly:

```bash
python tools/render_video.py \
  --trajectory /path/to/run.trajectory.jsonl.gz \
  --output output/run.mp4
```

For a short pipeline smoke test:

```bash
python tools/render_video.py --max-frames 10 --output output/smoke.mp4
```

## Docker

```bash
docker build -t contest-ui-video .
docker run --rm -v "$PWD/output:/app/output" contest-ui-video
```

The renderer reads the compressed trajectory incrementally and writes raw RGB
frames to FFmpeg, so it does not create an intermediate frame directory.
