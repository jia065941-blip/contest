"""Analyze paired V13/package runs, including gate and action quality."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


SCENARIOS = ("E01", "E02", "E03", "M01", "M02", "M03", "H01", "H02", "H03")


def load_run(root: Path, prefix: str, case: str, mode: str, seed: int) -> dict:
    run = root / f"{prefix}_{case}_{mode}_s{seed}"
    summary_path = next(run.glob("*/summary.json"))
    summary = json.loads(summary_path.read_text())
    hits = [
        json.loads(line)
        for line in (summary_path.parent / "hit_events.jsonl").read_text().splitlines()
        if line
    ]
    decisions = []
    trace = run / "joint_decisions.jsonl"
    if trace.is_file():
        decisions = [json.loads(line) for line in trace.read_text().splitlines() if line]
    return {"summary": summary, "hits": hits, "decisions": decisions, "path": str(summary_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--prefix", default="pkgbench")
    parser.add_argument("--seeds", default="1,2,3")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    pairs = []
    for case in SCENARIOS:
        for seed in seeds:
            baseline = load_run(args.results, args.prefix, case, "v13", seed)
            joint = load_run(args.results, args.prefix, case, "joint", seed)
            changed = [
                row for row in joint["decisions"]
                if row.get("actions") and row["actions"] != row["baseline_actions"]
            ]
            transitions = Counter()
            modified_ids = set()
            for row in changed:
                for entity_id, old, new in zip(
                    row["entity_ids"], row["baseline_actions"], row["actions"], strict=True
                ):
                    if old != new:
                        transitions[(old, new)] += 1
                        modified_ids.add(int(entity_id))
            hit_ids = {int(item["missile_id"]) for item in joint["hits"]}
            baseline_targets = {
                int(item["id"]): float(item["final_health"])
                for item in baseline["summary"]["objectives"]
            }
            joint_targets = {
                int(item["id"]): float(item["final_health"])
                for item in joint["summary"]["objectives"]
            }
            pairs.append({
                "case": case,
                "seed": seed,
                "delta": float(joint["summary"]["score"]["score"])
                - float(baseline["summary"]["score"]["score"]),
                "overrides": len(changed),
                "modified_ids": sorted(modified_ids),
                "modified_hit_count": len(modified_ids & hit_ids),
                "transitions": transitions,
                "baseline_separations": [
                    float(row["baseline_min_separation_metres"]) for row in changed
                ],
                "selected_separations": [
                    float(row["selected_min_separation_metres"]) for row in changed
                ],
                "owner_counts": [len(row.get("threat_owners", [])) for row in changed],
                "steps": [int(row["step"]) for row in changed],
                "target_hp_delta": {
                    str(target): round(joint_targets[target] - baseline_targets[target], 6)
                    for target in baseline_targets
                    if joint_targets[target] != baseline_targets[target]
                },
                "v13_path": baseline["path"],
                "joint_path": joint["path"],
            })

    for label, predicate in (
        ("positive", lambda item: item["delta"] > 0.5),
        ("negative", lambda item: item["delta"] < -0.5),
        ("neutral", lambda item: -0.5 <= item["delta"] <= 0.5),
    ):
        selected = [item for item in pairs if predicate(item)]
        transitions = sum((item["transitions"] for item in selected), Counter())
        separations = [value for item in selected for value in item["baseline_separations"]]
        improvements = [
            new - old for item in selected
            for old, new in zip(
                item["baseline_separations"], item["selected_separations"], strict=True
            )
        ]
        print(json.dumps({
            "class": label,
            "pair_count": len(selected),
            "mean_delta": statistics.mean([item["delta"] for item in selected]) if selected else 0.0,
            "mean_overrides": statistics.mean([item["overrides"] for item in selected]) if selected else 0.0,
            "mean_trigger_separation": statistics.mean(separations) if separations else None,
            "mean_predicted_improvement": statistics.mean(improvements) if improvements else None,
            "modified_hit_rate": (
                sum(item["modified_hit_count"] for item in selected)
                / max(1, sum(len(item["modified_ids"]) for item in selected))
            ),
            "owner_protected_fraction": (
                sum(count > 0 for item in selected for count in item["owner_counts"])
                / max(1, sum(len(item["owner_counts"]) for item in selected))
            ),
            "transitions": {f"{old}->{new}": count for (old, new), count in transitions.items()},
        }, ensure_ascii=False))
    print("PAIRS")
    for item in pairs:
        serializable = dict(item)
        serializable["transitions"] = {
            f"{old}->{new}": count for (old, new), count in item["transitions"].items()
        }
        print(json.dumps(serializable, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
