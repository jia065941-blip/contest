#!/usr/bin/env python3
"""Run resumable, parallel fixed-seed evaluations and aggregate summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="easy/E01")
    parser.add_argument("--blue-policy", default="b0_fixed_ratio_random")
    parser.add_argument("--red-policy", default="r9_hierarchical_learning")
    parser.add_argument("--red-motion-policy", default="mappo")
    parser.add_argument("--red-learning-model", required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output-dir", default="results/evaluations")
    return parser.parse_args()


def existing_summary(run_dir: Path) -> Path | None:
    summaries = sorted(run_dir.glob("*/summary.json"), key=lambda path: path.stat().st_mtime)
    return summaries[-1] if summaries else None


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    run_id = f"{args.run_prefix}_s{seed:04d}"
    run_dir = ROOT / "results" / "runs" / run_id
    summary_path = existing_summary(run_dir)
    if summary_path is not None:
        return {"seed": seed, "summary_path": str(summary_path), "cached": True}

    sweep_dir = ROOT / args.output_dir / args.run_prefix
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"seed_{seed:04d}.log"
    model_path = Path(args.red_learning_model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    command = [
        sys.executable,
        str(ROOT / "run.py"),
        "run",
        "--scenario",
        args.scenario,
        "--red-policy",
        args.red_policy,
        "--blue-policy",
        args.blue_policy,
        "--red-motion-policy",
        args.red_motion_policy,
        "--red-learning-model",
        str(model_path),
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
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str((seed - args.seed_start) % args.gpu_count)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    summary_path = existing_summary(run_dir)
    if completed.returncode != 0 or summary_path is None:
        return {
            "seed": seed,
            "error": f"exit={completed.returncode}, summary={summary_path}",
            "log_path": str(log_path),
        }
    return {"seed": seed, "summary_path": str(summary_path), "cached": False}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if "summary_path" not in result:
            continue
        path = Path(result["summary_path"])
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"seed": result["seed"], "summary_path": str(path), "summary": summary})
    return sorted(rows, key=lambda row: row["seed"])


def aggregate(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["summary"]["score"]["score"]) for row in rows]
    if not scores:
        return {"completed_seeds": 0, "failures": failures}
    sample_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    margin = 1.96 * sample_std / math.sqrt(len(scores))
    objective_ids = sorted(
        {int(item["id"]) for row in rows for item in row["summary"].get("objectives", [])}
    )
    objectives: dict[str, Any] = {}
    for objective_id in objective_ids:
        items = [
            next(item for item in row["summary"]["objectives"] if int(item["id"]) == objective_id)
            for row in rows
        ]
        damage_fractions = [
            1.0 - float(item["final_health"]) / max(float(item["initial_health"]), 1e-12)
            for item in items
        ]
        objectives[str(objective_id)] = {
            "damaged_rate": sum(value > 0.0 for value in damage_fractions) / len(items),
            "destroyed_rate": sum(float(item["final_health"]) <= 0.0 for item in items) / len(items),
            "mean_damage_fraction": statistics.mean(damage_fractions),
        }

    histogram = {}
    for lower in range(0, 100, 10):
        upper = lower + 10
        label = f"[{lower},{upper}{']' if upper == 100 else ')'}"
        histogram[label] = sum(
            lower <= score <= upper if upper == 100 else lower <= score < upper
            for score in scores
        )

    return {
        "completed_seeds": len(rows),
        "failed_seeds": len(failures),
        "score": {
            "mean": statistics.mean(scores),
            "sample_std": sample_std,
            "standard_error": sample_std / math.sqrt(len(scores)),
            "confidence_interval_95": [statistics.mean(scores) - margin, statistics.mean(scores) + margin],
            "min": min(scores),
            "p01": percentile(scores, 0.01),
            "p05": percentile(scores, 0.05),
            "p25": percentile(scores, 0.25),
            "median": percentile(scores, 0.50),
            "p75": percentile(scores, 0.75),
            "p95": percentile(scores, 0.95),
            "p99": percentile(scores, 0.99),
            "max": max(scores),
            "histogram": histogram,
        },
        "complete_mission_rate": sum(
            bool(row["summary"]["score"]["completed"]) for row in rows
        ) / len(rows),
        "red": {
            "mean_alive": statistics.mean(float(row["summary"]["red"]["alive"]) for row in rows),
            "mean_lost": statistics.mean(float(row["summary"]["red"]["lost"]) for row in rows),
        },
        "objectives": objectives,
        "failures": failures,
    }


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    output_dir = ROOT / args.output_dir / args.run_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    report.update(
        {
            "scenario": args.scenario,
            "blue_policy": args.blue_policy,
            "red_policy": args.red_policy,
            "red_motion_policy": args.red_motion_policy,
            "red_learning_model": str(Path(args.red_learning_model)),
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
        }
    )
    report_path = output_dir / "aggregate.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report_path)

    csv_path = output_dir / "per_seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["seed", "score", "completed", "red_alive", "red_lost", "destroyed_ids", "summary_path"])
        for row in rows:
            summary = row["summary"]
            writer.writerow(
                [
                    row["seed"],
                    summary["score"]["score"],
                    summary["score"]["completed"],
                    summary["red"]["alive"],
                    summary["red"]["lost"],
                    " ".join(str(value) for value in summary["score"]["destroyed_ids"]),
                    row["summary_path"],
                ]
            )
    print(f"AGGREGATE {report_path}", flush=True)
    print(f"PER_SEED {csv_path}", flush=True)


def main() -> int:
    args = parse_args()
    if args.seed_count <= 0 or args.workers <= 0 or args.gpu_count <= 0:
        raise ValueError("seed-count, workers, and gpu-count must be positive")
    seeds = range(args.seed_start, args.seed_start + args.seed_count)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_seed, args, seed): seed for seed in seeds}
        for completed_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            state = "cached" if result.get("cached") else "failed" if "error" in result else "done"
            print(
                f"PROGRESS {completed_count}/{args.seed_count} seed={result['seed']} {state}",
                flush=True,
            )

    failures = sorted((result for result in results if "error" in result), key=lambda item: item["seed"])
    rows = load_rows(results)
    report = aggregate(rows, failures)
    write_outputs(args, rows, report)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
