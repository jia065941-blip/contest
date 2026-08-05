"""Run B0--B3 against one scenario and write a compact comparison report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_one(baseline: str, scenario: str, max_steps: int, seed: int) -> dict:
    runner = Path(__file__).with_name("run_scenario.py")
    completed = subprocess.run(
        [sys.executable, str(runner), "--scenario", scenario, "--baseline", baseline, "--max-steps", str(max_steps), "--seed", str(seed)],
        cwd=ROOT.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{baseline} failed:\n{completed.stdout}\n{completed.stderr}")
    summary_path = Path(next(line for line in reversed(completed.stdout.splitlines()) if line.endswith("summary.json")))
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=str(ROOT.parent / "scenarios" / "platform.json"))
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    reports = [run_one(name, args.scenario, args.max_steps, args.seed) for name in ("b0", "b1", "b2", "b3")]
    comparison = {
        "scenario": args.scenario,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "baselines": [
            {
                "baseline": report["baseline"],
                "launched_platform_ids": report["launched_platform_ids"],
                "metrics": report["metrics"],
            }
            for report in reports
        ],
    }
    output_dir = ROOT.parent / "results" / "red_strategy_lab"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
