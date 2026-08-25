"""Run paired red-motion comparisons and render Markdown/CSV result tables.

The runner fixes the red allocation policy, blue policy, scenario, and seed
within every pair.  Only the red post-launch motion policy changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MOTIONS = ("straight", "reactive_evasion")
METRICS = (
    ("blue_remaining_health", "蓝方剩余总血量"),
    ("blue_health_loss_rate", "蓝方总血量损耗率"),
    ("objective_remaining_health", "计分目标剩余血量"),
    ("objective_health_loss_rate", "计分目标血量损耗率"),
    ("score", "总分"),
    ("K", "加权毁伤率 K"),
    ("T", "时间效率 T"),
    ("red_alive", "红方存活"),
    ("red_lost", "红方损失"),
    ("blue_destroyed", "蓝方毁伤实体"),
    ("destroyed_objectives", "已毁计分目标"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired straight/evasion baseline comparison")
    parser.add_argument("--scenarios", nargs="+", default=["easy/E01", "medium/M01"])
    parser.add_argument(
        "--red-policies",
        nargs="+",
        default=["b0_random", "b1_priority", "b2_static_assignment", "b3_rolling_rules"],
    )
    parser.add_argument("--blue-policy", default="threat_priority", help="Single blue policy (legacy shorthand)")
    parser.add_argument(
        "--blue-policies",
        nargs="+",
        help="Run one paired motion comparison for each listed blue policy",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "motion_comparison")
    parser.add_argument("--max-steps", type=int, help="Override each competition case's formal step limit")
    parser.add_argument(
        "--from-raw",
        type=Path,
        help="Regenerate report.md and overview.csv from a previous raw_summaries.json without rerunning simulations",
    )
    return parser.parse_args()


def _slug(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


def _latest_summary(run_id: str) -> Path:
    candidates = list((ROOT / "results" / "runs" / run_id).glob("*/summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No summary.json produced for {run_id}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_one(
    *,
    scenario: str,
    red_policy: str,
    blue_policy: str,
    motion: str,
    seed: int,
    max_steps: int | None,
) -> tuple[dict[str, Any], Path]:
    run_id = "motion_{}_{}_{}_{}_s{}".format(
        _slug(scenario), _slug(red_policy), _slug(blue_policy), _slug(motion), seed,
    )
    command = [
        sys.executable,
        "run.py",
        "run",
        "--scenario",
        scenario,
        "--red-policy",
        red_policy,
        "--red-motion-policy",
        motion,
        "--blue-policy",
        blue_policy,
        "--seed",
        str(seed),
        "--run-id",
        run_id,
        "--",
        "--total-rounds",
        "1",
        "--render-mode",
        "none",
    ]
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])

    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        tail = completed.stdout[-4000:]
        raise RuntimeError(f"{scenario} / {red_policy} / {motion} failed:\n{tail}")
    summary_path = _latest_summary(run_id)
    return json.loads(summary_path.read_text(encoding="utf-8")), summary_path


def _values(summary: dict[str, Any]) -> dict[str, float]:
    score = summary.get("score") or {}
    red = summary.get("red") or {}
    blue = summary.get("blue") or {}
    objectives = summary.get("objectives") or []
    blue_initial_health = float(blue.get("initial_health_total", 0.0))
    blue_remaining_health = float(blue.get("remaining_health_total", 0.0))
    objective_initial_health = sum(float(item.get("initial_health", 0.0)) for item in objectives)
    objective_remaining_health = sum(float(item.get("final_health", 0.0)) for item in objectives)
    return {
        "blue_remaining_health": blue_remaining_health,
        "blue_health_loss_rate": (
            1.0 - blue_remaining_health / blue_initial_health
            if blue_initial_health > 0 else 0.0
        ),
        "objective_remaining_health": objective_remaining_health,
        "objective_health_loss_rate": (
            1.0 - objective_remaining_health / objective_initial_health
            if objective_initial_health > 0 else 0.0
        ),
        "score": float(score.get("score", 0.0)),
        "K": float(score.get("K", 0.0)),
        "T": float(score.get("T", 0.0)),
        "red_alive": float(red.get("alive", 0.0)),
        "red_lost": float(red.get("lost", 0.0)),
        "blue_destroyed": float(blue.get("destroyed", 0.0)),
        "destroyed_objectives": float(len(score.get("destroyed_ids", []))),
    }


def _number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}"


def _display(key: str, value: float, *, delta: bool = False) -> str:
    if key.endswith("_rate"):
        if delta:
            return f"{value * 100:+.1f} pp"
        return f"{value:.1%}"
    return _number(value)


def _pair_markdown(case: dict[str, Any]) -> str:
    straight = _values(case["straight"])
    evasion = _values(case["reactive_evasion"])
    rows = [
        f"## {case['scenario']} · {case['red_policy']} vs {case['blue_policy']}",
        "",
        "| 指标 | straight | reactive_evasion | 差值（规避−直飞） |",
        "|---|---:|---:|---:|",
    ]
    for key, label in METRICS:
        rows.append(
            f"| {label} | {_display(key, straight[key])} | {_display(key, evasion[key])} | "
            f"{_display(key, evasion[key] - straight[key], delta=True)} |"
        )
    rows.append("")
    return "\n".join(rows)


def _write_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    fields = ["scenario", "red_policy", "blue_policy", "seed"]
    metric_keys = [key for key, _ in METRICS]
    fields.extend(f"straight_{key}" for key in metric_keys)
    fields.extend(f"reactive_evasion_{key}" for key in metric_keys)
    fields.extend(f"delta_{key}" for key in metric_keys)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            straight = _values(case["straight"])
            evasion = _values(case["reactive_evasion"])
            row = {
                "scenario": case["scenario"],
                "red_policy": case["red_policy"],
                "blue_policy": case["blue_policy"],
                "seed": case["seed"],
            }
            row.update({f"straight_{key}": straight[key] for key in metric_keys})
            row.update({f"reactive_evasion_{key}": evasion[key] for key in metric_keys})
            row.update({f"delta_{key}": evasion[key] - straight[key] for key in metric_keys})
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.from_raw is not None:
        raw_path = args.from_raw.resolve()
        cases = json.loads(raw_path.read_text(encoding="utf-8"))
        output_dir = raw_path.parent
    else:
        output_dir = args.output_dir / datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        cases: list[dict[str, Any]] = []
        blue_policies = args.blue_policies or [args.blue_policy]
        total = len(args.scenarios) * len(args.red_policies) * len(blue_policies) * len(MOTIONS)
        index = 0

        for scenario in args.scenarios:
            for red_policy in args.red_policies:
                for blue_policy in blue_policies:
                    pair: dict[str, Any] = {
                        "scenario": scenario,
                        "red_policy": red_policy,
                        "blue_policy": blue_policy,
                        "seed": args.seed,
                    }
                    for motion in MOTIONS:
                        index += 1
                        print(f"[{index}/{total}] {scenario} {red_policy} {blue_policy} {motion}", flush=True)
                        summary, path = _run_one(
                            scenario=scenario,
                            red_policy=red_policy,
                            blue_policy=blue_policy,
                            motion=motion,
                            seed=args.seed,
                            max_steps=args.max_steps,
                        )
                        pair[motion] = summary
                        pair[f"{motion}_summary"] = str(path)
                    cases.append(pair)

    report = [
        "# 红方 motion 策略对照",
        "",
        f"固定随机种子：`{args.seed}`。每对试验仅切换红方 motion 策略。",
        "",
        "## 总表（以血量为主）",
        "",
        "| 场景 | 红方分配 | 蓝方策略 | 直飞蓝方余血 | 规避蓝方余血 | Δ余血 | 直飞目标余血 | 规避目标余血 | Δ目标余血 | Δ总分 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        straight = _values(case["straight"])
        evasion = _values(case["reactive_evasion"])
        report.append(
            "| {scenario} | {red_policy} | {blue_policy} | {blue_a} | {blue_b} | {delta_blue} | {objective_a} | {objective_b} | {delta_objective} | {delta_score} |".format(
                scenario=case["scenario"],
                red_policy=case["red_policy"],
                blue_policy=case["blue_policy"],
                blue_a=_number(straight["blue_remaining_health"]),
                blue_b=_number(evasion["blue_remaining_health"]),
                objective_a=_number(straight["objective_remaining_health"]),
                objective_b=_number(evasion["objective_remaining_health"]),
                delta_score=_number(evasion["score"] - straight["score"]),
                delta_blue=_number(evasion["blue_remaining_health"] - straight["blue_remaining_health"]),
                delta_objective=_number(evasion["objective_remaining_health"] - straight["objective_remaining_health"]),
            )
        )
    report.append("")
    report.extend(_pair_markdown(case) for case in cases)

    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    if args.from_raw is None:
        (output_dir / "raw_summaries.json").write_text(
            json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    _write_csv(output_dir / "overview.csv", cases)
    print(f"Report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
