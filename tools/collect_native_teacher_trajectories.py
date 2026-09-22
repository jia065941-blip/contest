"""Collect complete contest-teacher trajectories in the current simulator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = "scenarios/cases/final20/easy/E01/scenario.json"
DEFAULT_TEACHER_MODEL = Path(
    "/home/ubuntu/yuanlei/cz/contest/models/r9_mappo_e01.pt"
)
DEFAULT_SEEDS = (
    108, 222, 11, 141, 144, 126, 101, 18,
    142, 232, 19, 26, 84, 159, 33, 243,
    77, 40, 250, 60, 38, 72, 212, 177,
    83, 204, 131, 170, 8, 87, 96, 241,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--teacher-model", type=Path, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def write_manifest(
    path: Path,
    rows: list[dict],
    *,
    scenario: str,
    teacher_model: Path,
    seeds: tuple[int, ...],
) -> None:
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "teacher_model": str(teacher_model.resolve()),
        "blue_policy": "b0_fixed_ratio_random",
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "completed": len(rows),
        "trajectories": sorted(rows, key=lambda row: row["seed"]),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_seed(
    seed: int,
    output_dir: Path,
    scenario: str,
    teacher_model: Path,
) -> dict:
    traces = output_dir / "trajectories"
    logs = output_dir / "logs"
    traces.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    trace_path = traces / f"seed_{seed:04d}.json"
    log_path = logs / f"seed_{seed:04d}.log"
    run_id = f"teacher_distill_e01_s{seed:04d}_20260911"
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "RED_RECORD_NATIVE_TRAJECTORY": "1",
        "RED_NATIVE_TRAJECTORY_PATH": str(trace_path.resolve()),
    })
    libraries = [
        str(ROOT / "core" / "envengine" / "simulator" / "models" / "HXDMissileModel"),
        str(Path(sys.prefix) / "lib"),
    ]
    if environment.get("LD_LIBRARY_PATH"):
        libraries.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(libraries)
    command = [
        sys.executable,
        "run.py",
        "run",
        "--scenario",
        scenario,
        "--red-policy",
        "r9_hierarchical_learning",
        "--blue-policy",
        "b0_fixed_ratio_random",
        "--red-motion-policy",
        "mappo",
        "--red-learning-model",
        str(teacher_model.resolve()),
        "--seed",
        str(seed),
        "--run-id",
        run_id,
        "--",
        "--total-rounds",
        "1",
        "--render-mode",
        "none",
        "--disable-log-color",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"seed {seed} exited {completed.returncode}; log={log_path}"
        )
    summaries = [
        json.loads(line.removeprefix("FINAL_SUMMARY "))
        for line in completed.stdout.splitlines()
        if line.startswith("FINAL_SUMMARY ")
    ]
    if len(summaries) != 1:
        raise RuntimeError(f"seed {seed} emitted {len(summaries)} summaries")
    summary = summaries[0]
    return {
        "seed": seed,
        "score": float(summary["score"]["score"]),
        "destroyed_ids": list(summary["score"]["destroyed_ids"]),
        "trace": str(trace_path.resolve()),
        "log": str(log_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    seeds = tuple(
        int(value.strip())
        for value in args.seeds.split(",")
        if value.strip()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    manifest = args.output_dir / "collection_manifest.json"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_seed,
                seed,
                args.output_dir,
                args.scenario,
                args.teacher_model,
            ): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_manifest(
                manifest,
                rows,
                scenario=args.scenario,
                teacher_model=args.teacher_model,
                seeds=seeds,
            )
            print(
                json.dumps(
                    {
                        "completed": len(rows),
                        "seed": row["seed"],
                        "score": row["score"],
                        "destroyed_ids": row["destroyed_ids"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
