"""Best-effort metadata-only monitoring; never a numerical decision maker."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from uuid import uuid4

from . import __version__


class Monitor:
    def __init__(self, *, enabled: bool = True, jsonl: Path | None = None,
                 logger: logging.Logger | None = None):
        self.enabled = enabled
        self.jsonl = Path(jsonl) if jsonl is not None else None
        self.logger = logger or logging.getLogger("skill_generation")
        self.run_id = uuid4().hex

    def record(self, event: str, **metadata: object) -> None:
        if not self.enabled:
            return
        record = dict(metadata, event=event, run_id=self.run_id,
                      timestamp=datetime.now(timezone.utc).isoformat(),
                      application_version=__version__)
        try:
            text = json.dumps(record, ensure_ascii=True, allow_nan=False)
            self.logger.info(text)
            if self.jsonl is not None:
                with self.jsonl.open("a", encoding="utf-8") as stream:
                    stream.write(text + "\n")
        except Exception as exc:
            try:
                print(f"MONITOR_WRITE_FAILED: {type(exc).__name__}", file=sys.stderr)
            except Exception:
                pass
