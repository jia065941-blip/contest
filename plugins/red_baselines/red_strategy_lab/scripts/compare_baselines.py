"""Run all deterministic demonstration policies and print their assignments."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).with_name("run_baseline.py")
    for baseline in ("b0", "b1", "b2", "b3"):
        print(f"\n[{baseline}]")
        subprocess.run([sys.executable, str(script), "--baseline", baseline], check=True)


if __name__ == "__main__":
    main()

