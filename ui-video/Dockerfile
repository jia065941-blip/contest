FROM python:3.11-slim

ENV SDL_VIDEODRIVER=dummy \
    PYGAME_HIDE_SUPPORT_PROMPT=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . ./

CMD ["python", "tools/render_video.py"]
