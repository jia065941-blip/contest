"""Run nine maps x five seeds x V13/mission-aware package in parallel."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess


SCENARIOS = (
    "final20/easy/E01",
    "final20/easy/E02",
    "final20/easy/E03",
    "final20/medium/M01",
    "final20/medium/M02",
    "final20/medium/M03",
    "final20/hard/H01",
    "final20/hard/H02",
    "final20/hard/H03",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, default=Path("../../.venv/bin/h-single-maneuver-plan"))
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--max-jobs", type=int, default=48)
    parser.add_argument("--prefix", default="missionbench")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    logs = output_root / "missionbench_driver_logs"
    logs.mkdir(parents=True, exist_ok=True)
    tasks = []
    for scenario in SCENARIOS:
        case = scenario.rsplit("/", 1)[-1]
        for seed in range(1, 6):
            for mode in ("v13", "joint"):
                run_id = f"{args.prefix}_{case}_{mode}_s{seed}"
                existing = list((output_root / run_id).glob("*/summary.json"))
                if existing and not args.force:
                    continue
                command = [
                    str(args.runner.resolve()),
                    "--environment-root", str(args.environment_root.resolve()),
                    "--plan-root", str(args.plan_root.resolve()),
                    "--scenario", scenario,
                    "--plan", "auto",
                    "--launch-mode", "native-waves",
                    "--lower-policy", "local-all" if mode == "v13" else "joint-h-v13",
                    "--seed", str(seed),
                    "--run-id", run_id,
                    "--output-root", str(output_root),
                    "--render-mode", "none",
                ]
                if mode == "joint":
                    command += ["--joint-rule-profile", "mission-aware"]
                command += ["--", "--disable-log-color"]
                tasks.append((run_id, command))

    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOPFORMER_CPU_THREADS": "1",
        }
    )

    def run_one(task):
        run_id, command = task
        log_path = logs / f"{run_id}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        return run_id, result.returncode, str(log_path)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_jobs) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"run_id": result[0], "returncode": result[1]}), flush=True)

    failures = [item for item in results if item[1] != 0]
    index_path = logs / "run_index.json"
    index_path.write_text(
        json.dumps(
            {
                "requested": len(tasks),
                "completed": len(results) - len(failures),
                "failed": failures,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
