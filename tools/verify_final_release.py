#!/usr/bin/env python3
"""Verify the portable final-branch source, frozen model, and published results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "vendor/cz_merged_hp_seed2_20261101"
RESULT = ROOT / "results/cz_ppo_seed2_final_all20_attribution_20261118"
MAPS = tuple([f"E{i:02d}" for i in range(1, 7)] + [f"M{i:02d}" for i in range(1, 7)] + [f"H{i:02d}" for i in range(1, 9)])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path) -> None:
    if not path.is_file():
        raise AssertionError(f"missing required file: {path.relative_to(ROOT)}")


def main() -> None:
    required = (
        ROOT / "CZ_PPO_20MAP_EVALUATION_REPORT_ZH.md",
        ROOT / "user_agents/cz_ppo/policy.py",
        ROOT / "user_agents/cz_ppo/compat.py",
        RELEASE / "opening/frozen_choices.json",
        RELEASE / "app/design/planner_v2.py",
        RELEASE / "app/skills/hp_tree/runtime.py",
        RELEASE / "app/r3/model.pt",
        RELEASE / "app/r3/r3_runtime/local_runtime.py",
        RESULT / "protocol.json",
        RESULT / "summary.json",
        RESULT / "attribution_summary.json",
        RESULT / "ATTRIBUTION_REPORT_ZH.md",
    )
    for path in required:
        require(path)

    protocol = load(RESULT / "protocol.json")
    expected_hashes = {
        "final_engine": ROOT / "envengine/engine/engine.py",
        "adapter_policy": ROOT / "user_agents/cz_ppo/policy.py",
        "coordinate_adapter": ROOT / "user_agents/cz_ppo/compat.py",
        "frozen_planner": RELEASE / "app/design/planner_v2.py",
        "frozen_choices": RELEASE / "opening/frozen_choices.json",
        "frozen_hp_tree": RELEASE / "app/skills/hp_tree/runtime.py",
        "frozen_ppo": RELEASE / "app/r3/model.pt",
    }
    for name, path in expected_hashes.items():
        actual = sha256(path)
        expected = protocol["sha256"][name]
        if actual != expected:
            raise AssertionError(f"hash mismatch for {name}: {actual} != {expected}")

    summary = load(RESULT / "summary.json")
    if summary["status"] != "complete" or summary["completed_maps"] != 20 or summary["failures"]:
        raise AssertionError("published run is not a clean 20/20 completion")
    rows = {row["map"]: row for row in summary["rows"]}
    if tuple(rows) != MAPS:
        raise AssertionError("published map order or coverage is not final20")

    for map_name in MAPS:
        folder = RESULT / "maps" / map_name
        for filename in (
            "run.yaml", "console.log", "example_evaluation.json", "cz_adapter_audit.json",
            "attribution.json", "result.json",
        ):
            require(folder / filename)
        row = load(folder / "result.json")
        evaluation = load(folder / "example_evaluation.json")
        attribution = load(folder / "attribution.json")
        if attribution["status"] != "complete":
            raise AssertionError(f"{map_name}: attribution incomplete")
        if abs(float(row["score"]) - float(evaluation["score"]["score"])) > 1e-10:
            raise AssertionError(f"{map_name}: result/evaluation score mismatch")
        objectives = attribution["objectives"]
        total = sum(float(item["weight"]) for item in objectives)
        missing = sum(float(item["missing_score_weight"]) for item in objectives)
        reconstructed = 100.0 * (total - missing) / total
        if abs(reconstructed - float(row["score"])) > 1e-8:
            raise AssertionError(f"{map_name}: attribution does not reconstruct score")

    attribution_summary = load(RESULT / "attribution_summary.json")
    if abs(float(attribution_summary["mean_score"]) - float(summary["mean_score"])) > 1e-12:
        raise AssertionError("summary and attribution mean scores differ")
    print(json.dumps({
        "status": "verified",
        "branch_payload": "competition-final portable CZ+PPO release",
        "maps": len(MAPS),
        "mean_score": summary["mean_score"],
        "model_sha256": protocol["sha256"]["frozen_ppo"],
        "result_dir": str(RESULT.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
