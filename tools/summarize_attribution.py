#!/usr/bin/env python3
"""Validate and summarize per-map attribution artifacts from a completed run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", type=Path, help="Earlier summary.json used only for replay comparison")
    args = parser.parse_args()
    root = args.root.resolve()
    run = load(root / "summary.json")
    if run["status"] != "complete":
        raise SystemExit("run summary is not complete")

    maps = []
    missed = []
    cause_counts = Counter()
    cause_weight = Counter()
    kind_counts = Counter()
    kind_weight = Counter()
    terminal_by_role = defaultdict(Counter)
    for run_row in run["rows"]:
        name = run_row["map"]
        folder = root / "maps" / name
        evaluation = load(folder / "example_evaluation.json")
        attribution = load(folder / "attribution.json")
        if attribution["status"] != "complete":
            raise RuntimeError(f"{name}: attribution is not complete")
        objectives = attribution["objectives"]
        total_weight = sum(row["weight"] for row in objectives)
        missing_weight = sum(row["missing_score_weight"] for row in objectives)
        reconstructed = 100.0 * (total_weight - missing_weight) / total_weight
        score = float(evaluation["score"]["score"])
        if abs(reconstructed - score) > 1e-8:
            raise RuntimeError(
                f"{name}: attribution score {reconstructed} != evaluation score {score}")

        map_missed = []
        for row in objectives:
            if row["destroyed"]:
                continue
            primary_cause = row["primary_cause"]
            # Early attribution artifacts used the target's physical map position
            # as an additional guard.  A submitted legacy waypoint can itself be
            # outside the public boundary even when the stationary target is not.
            if (
                row["assignment_count"] > 0
                and row["damage_delivered"] <= 0.0
                and not row["hit_events"]
                and row["boundary_clamp"]
            ):
                primary_cause = "boundary_constrained_no_hit"
            item = {
                "map": name,
                "target_id": row["id"],
                "name": row["name"],
                "kind": row["kind"],
                "type": row["type"],
                "weight": row["weight"],
                "score_loss": 100.0 * row["weight"] / total_weight,
                "initial_health": row["initial_health"],
                "final_health": row["final_health"],
                "damage_delivered": row["damage_delivered"],
                "discovered_step": row["discovered_step"],
                "assignment_count": row["assignment_count"],
                "assigned_unit_ids": row["assigned_unit_ids"],
                "assigned_roles": row["assigned_roles"],
                "intercepted_assigned_unit_ids": row["intercepted_assigned_unit_ids"],
                "arrived_within_100m_unit_ids": row["arrived_within_100m_unit_ids"],
                "hit_count": len(row["hit_events"]),
                "outside_map_distance_m": row["outside_map_distance_m"],
                "boundary_clamp": row["boundary_clamp"],
                "primary_cause": primary_cause,
                "contributing_causes": row["contributing_causes"],
            }
            missed.append(item)
            map_missed.append(item)
            cause_counts[item["primary_cause"]] += 1
            cause_weight[item["primary_cause"]] += item["score_loss"]
            kind_counts[item["kind"]] += 1
            kind_weight[item["kind"]] += item["score_loss"]
        for unit in attribution["units"]:
            terminal_by_role[unit["role"]][unit["terminal_reason"]] += 1
        maps.append({
            "map": name,
            "score": score,
            "total_weight": total_weight,
            "missing_weight": missing_weight,
            "missed_count": len(map_missed),
            "missed_target_ids": [row["target_id"] for row in map_missed],
            "causes": dict(Counter(row["primary_cause"] for row in map_missed)),
        })

    summary = {
        "status": "complete",
        "source_run": str(root),
        "maps": maps,
        "mean_score": statistics.fmean(row["score"] for row in maps),
        "missed_target_count": len(missed),
        "score_loss_points_summed_across_maps": sum(row["score_loss"] for row in missed),
        "mean_score_loss": statistics.fmean(100.0 - row["score"] for row in maps),
        "loss_by_primary_cause": {
            cause: {"target_count": cause_counts[cause], "score_loss": cause_weight[cause]}
            for cause in sorted(cause_counts)
        },
        "loss_by_kind": {
            kind: {"target_count": kind_counts[kind], "score_loss": kind_weight[kind]}
            for kind in sorted(kind_counts)
        },
        "red_terminal_reasons_by_role": {
            role: dict(counts) for role, counts in sorted(terminal_by_role.items())
        },
        "missed_objectives": missed,
    }
    if args.baseline is not None:
        baseline = load(args.baseline.resolve())
        previous = {row["map"]: float(row["score"]) for row in baseline["rows"]}
        differences = [
            {
                "map": row["map"],
                "baseline_score": previous[row["map"]],
                "attribution_replay_score": row["score"],
                "delta": row["score"] - previous[row["map"]],
            }
            for row in maps
            if abs(row["score"] - previous[row["map"]]) > 1e-10
        ]
        summary["baseline_comparison"] = {
            "baseline_path": str(args.baseline.resolve()),
            "baseline_mean_score": float(baseline["mean_score"]),
            "attribution_replay_mean_score": summary["mean_score"],
            "mean_delta": summary["mean_score"] - float(baseline["mean_score"]),
            "identical_map_count": len(maps) - len(differences),
            "different_map_count": len(differences),
            "differences": differences,
            "interpretation": (
                "Official object-address ordering was retained, so the same seed is not a bitwise "
                "replay. Causal labels apply to the fresh attribution run, not retroactively to the baseline."
            ),
        }
    save(root / "attribution_summary.json", summary)
    write_markdown(root / "ATTRIBUTION_REPORT_ZH.md", summary)


def write_markdown(path: Path, summary):
    lines = [
        "# CZ + PPO seed2 新版 20 图失分归因",
        "",
        "本报告由仿真外侧只读评估器生成。引擎真值仅用于赛后归因，未回灌红方策略。",
        "",
        f"- 20 图算术均分：{summary['mean_score']:.6f}",
        f"- 平均失分：{summary['mean_score_loss']:.6f}",
        f"- 未摧毁目标数：{summary['missed_target_count']}",
    ]
    comparison = summary.get("baseline_comparison")
    if comparison is not None:
        lines += [
            "",
            "## 与历史 91.117 分运行的关系",
            "",
            f"- 历史运行均分：{comparison['baseline_mean_score']:.6f}",
            f"- 本次归因重放均分：{comparison['attribution_replay_mean_score']:.6f}",
            f"- 分数完全一致地图：{comparison['identical_map_count']}/20",
            f"- 分数不同地图：{comparison['different_map_count']}/20",
            "- 官方对象地址排序保持不变，因此同 seed 不是位级复现。下面的因果标签只属于本次新鲜重放；历史运行只能做终局目标类型统计，不能倒推出逐单位终止原因。",
        ]
    lines += [
        "",
        "## 按主因归因",
        "",
        "| 主因 | 目标数 | 20 图失分点数之和 |",
        "|---|---:|---:|",
    ]
    for cause, row in summary["loss_by_primary_cause"].items():
        lines.append(f"| `{cause}` | {row['target_count']} | {row['score_loss']:.6f} |")
    lines += [
        "",
        "## 按目标类型归因",
        "",
        "| 类型 | 目标数 | 20 图失分点数之和 |",
        "|---|---:|---:|",
    ]
    for kind, row in summary["loss_by_kind"].items():
        lines.append(f"| {kind} | {row['target_count']} | {row['score_loss']:.6f} |")
    lines += [
        "",
        "## 分图",
        "",
        "| 图 | 得分 | 未摧毁目标 | 缺失权重 | 主因 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary["maps"]:
        causes = ", ".join(f"{key}:{value}" for key, value in sorted(row["causes"].items())) or "无"
        lines.append(
            f"| {row['map']} | {row['score']:.6f} | {row['missed_count']} | "
            f"{row['missing_weight']:.1f} | {causes} |"
        )
    lines += [
        "",
        "## 未摧毁目标明细",
        "",
        "| 图 | ID | 类型 | 扣分 | 末血 | 派发角色 | 被拦截/派发 | 命中 | 主因 |",
        "|---|---:|---|---:|---:|---|---:|---:|---|",
    ]
    for row in summary["missed_objectives"]:
        roles = ",".join(f"{key}:{value}" for key, value in sorted(row["assigned_roles"].items())) or "无"
        lines.append(
            f"| {row['map']} | {row['target_id']} | {row['kind']} | {row['score_loss']:.6f} | "
            f"{row['final_health']:.3f} | {roles} | "
            f"{len(row['intercepted_assigned_unit_ids'])}/{len(row['assigned_unit_ids'])} | "
            f"{row['hit_count']} | `{row['primary_cause']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
