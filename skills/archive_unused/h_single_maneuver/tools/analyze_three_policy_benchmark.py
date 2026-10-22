"""Validate and aggregate original/new/new+package benchmark results."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


SCENARIOS = ("E01", "E02", "E03", "M01", "M02", "M03", "H01", "H02", "H03")
CONDITIONS = ("v13", "new", "new_joint")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--prefix", default="compare_v13_v12r3")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    rows = []
    for case in SCENARIOS:
        for seed in seeds:
            for condition in CONDITIONS:
                run = args.results / f"{args.prefix}_{case}_{condition}_s{seed}"
                summaries = list(run.glob("*/summary.json"))
                if len(summaries) != 1:
                    raise RuntimeError(f"Expected one summary for {run}, found {len(summaries)}")
                summary_path = summaries[0]
                hit_path = summary_path.parent / "hit_events.jsonl"
                analysis_path = summary_path.parent / "analysis_summary.json"
                if not hit_path.is_file() or not analysis_path.is_file():
                    raise RuntimeError(f"Missing sidecar for {run}")
                summary = json.loads(summary_path.read_text())
                analysis = json.loads(analysis_path.read_text())
                overrides = 0
                if condition == "new_joint":
                    trace_path = run / "joint_decisions.jsonl"
                    if not trace_path.is_file():
                        raise RuntimeError(f"Missing joint trace for {run}")
                    decisions = [
                        json.loads(line) for line in trace_path.read_text().splitlines() if line
                    ]
                    overrides = sum(
                        bool(item.get("actions"))
                        and item["actions"] != item["baseline_actions"]
                        for item in decisions
                    )
                rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "condition": condition,
                        "score": float(summary["score"]["score"]),
                        "steps": int(summary["steps_executed"]),
                        "red_alive": int(summary["red"]["alive"]),
                        "destroyed_weight": float(summary["score"]["destroyed_weight"]),
                        "hit_event_count": int(analysis["hit_event_count"]),
                        "overrides": overrides,
                        "summary_path": str(summary_path.resolve()),
                    }
                )

    by_map = {}
    for case in SCENARIOS:
        entry = {}
        for condition in CONDITIONS:
            values = [
                row["score"] for row in rows
                if row["case"] == case and row["condition"] == condition
            ]
            entry[condition] = {
                "mean": statistics.mean(values),
                "pstdev": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
            }
        entry["new_minus_v13"] = entry["new"]["mean"] - entry["v13"]["mean"]
        entry["new_joint_minus_new"] = (
            entry["new_joint"]["mean"] - entry["new"]["mean"]
        )
        by_map[case] = entry

    pairwise = {}
    for first, second, label in (
        ("v13", "new", "new_minus_v13"),
        ("new", "new_joint", "new_joint_minus_new"),
    ):
        deltas = []
        for case in SCENARIOS:
            for seed in seeds:
                left = next(
                    row["score"] for row in rows
                    if row["case"] == case and row["seed"] == seed
                    and row["condition"] == first
                )
                right = next(
                    row["score"] for row in rows
                    if row["case"] == case and row["seed"] == seed
                    and row["condition"] == second
                )
                deltas.append(right - left)
        pairwise[label] = {
            "mean": statistics.mean(deltas),
            "wins": sum(value > 0.5 for value in deltas),
            "losses": sum(value < -0.5 for value in deltas),
            "ties": sum(abs(value) <= 0.5 for value in deltas),
        }

    overall = {}
    for condition in CONDITIONS:
        values = [row["score"] for row in rows if row["condition"] == condition]
        overall[condition] = {
            "mean": statistics.mean(values),
            "pstdev": statistics.pstdev(values),
        }
    override_counts = [
        row["overrides"] for row in rows if row["condition"] == "new_joint"
    ]
    result = {
        "schema_version": 1,
        "prefix": args.prefix,
        "scenario_count": len(SCENARIOS),
        "seed_count": len(seeds),
        "condition_count": len(CONDITIONS),
        "run_count": len(rows),
        "conditions": list(CONDITIONS),
        "overall": overall,
        "by_map": by_map,
        "pairwise": pairwise,
        "joint_overrides": {
            "total": sum(override_counts),
            "mean_per_run": statistics.mean(override_counts),
            "zero_override_runs": sum(value == 0 for value in override_counts),
        },
        "rows": rows,
    }
    output = args.output or (
        args.results / f"{args.prefix}_driver_logs" / "aggregate.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "run_count": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
