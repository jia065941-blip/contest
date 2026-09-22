#!/usr/bin/env python3
"""Build full-grid C0 search returns with the native L motion model.

The actor input remains the saved legal local observation. Hidden ship state
is used only on the training/evaluation side to label counterfactual routes,
consistent with centralized training and decentralized execution.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int, default=3000)
    return parser.parse_args()


def find_map_area(value: Any) -> dict[str, float] | None:
    if isinstance(value, dict):
        required = {"lonMin", "lonMax", "latMin", "latMax"}
        if required.issubset(value):
            return {name: float(value[name]) for name in required}
        for child in value.values():
            result = find_map_area(child)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = find_map_area(child)
            if result is not None:
                return result
    return None


def episode_spec(
    episode: dict[str, Any],
    *,
    map_area: dict[str, float],
    ships: dict[int, tuple[float, float]],
    max_steps: int,
) -> dict[str, Any]:
    rollout_path = Path(episode["rollout"])
    rollout = torch.load(
        rollout_path, map_location="cpu", weights_only=False
    )["rollout"]
    if rollout.batch_size != 1:
        raise ValueError(f"C0 search rollout must contain one row: {rollout_path}")
    factual_paths = tuple(
        rollout_path.parent.glob(
            "sim_results/*/native_handoff_credit/*/result.json.factual"
        )
    )
    if len(factual_paths) != 1:
        raise ValueError(
            f"expected one factual branch for {rollout_path}, got {len(factual_paths)}"
        )
    factual_path = factual_paths[0]
    factual = json.loads(factual_path.read_text(encoding="utf-8"))
    summary = json.loads(
        (factual_path.parents[2] / "summary.json").read_text(encoding="utf-8")
    )
    observation = rollout.observations[0]
    return {
        "seed": int(episode["seed"]),
        "entity_id": int(factual["executor_id"]),
        "branch_step": int(factual["branch_step"]),
        "horizon_seconds": int(factual["search_detection_sample_count"]),
        "start_lla": (
            float(observation[1]) * 180.0,
            float(observation[2]) * 90.0,
            float(observation[3]) * 20_000.0,
        ),
        "destruction_steps": {
            int(key): int(value)
            for key, value in summary["score"]["destruction_steps"].items()
        },
        "ships": ships,
        "map_area": map_area,
        "grid_width": 16,
        "grid_height": 12,
        "max_steps": int(max_steps),
        "selected_index": int(rollout.actions.search_index.item()),
        "actual_nearest_alive_distance_m": float(
            factual["search_nearest_alive_distance_m"]
        ),
        "actual_direct_return": float(
            factual["search_direct_detection_return"]
        ),
        "observation": rollout.observations[0].clone(),
        "target_features": rollout.target_features[0].clone(),
        "target_valid_mask": rollout.target_valid_mask[0].clone(),
    }


def simulate_episode(spec: dict[str, Any]) -> dict[str, Any]:
    from envengine.sdk.Util import UtilsPy
    from envengine.simulator.models.ACMMissileModel import Missile

    def target_ecef(lon: float, lat: float) -> tuple[float, float, float]:
        point = UtilsPy.CoordinateHelper.llaToEcef_py(
            UtilsPy.Vector3D(lon, lat, 0.0)
        )
        return point.x(), point.y(), point.z()

    target_positions = {
        int(target_id): target_ecef(float(lla[0]), float(lla[1]))
        for target_id, lla in spec["ships"].items()
    }
    width = int(spec["grid_width"])
    height = int(spec["grid_height"])
    area = spec["map_area"]
    branch_step = int(spec["branch_step"])
    horizon_seconds = max(1, int(spec["horizon_seconds"]))
    remaining_steps = max(1, int(spec["max_steps"]) - branch_step)
    start_lon, start_lat, start_alt = map(float, spec["start_lla"])
    destruction_steps = {
        int(key): int(value)
        for key, value in spec["destruction_steps"].items()
    }
    distances: list[float] = []
    direct_returns: list[float] = []
    total_returns: list[float] = []
    detection_counts: list[int] = []

    for action_index in range(width * height):
        column = action_index % width
        row = action_index // width
        target_lon = float(area["lonMin"]) + (
            column + 0.5
        ) / width * (float(area["lonMax"]) - float(area["lonMin"]))
        target_lat = float(area["latMin"]) + (
            row + 0.5
        ) / height * (float(area["latMax"]) - float(area["latMin"]))

        model = Missile()
        model.Init(
            0.05,
            UtilsPy.Vector3D(start_lon, start_lat, start_alt + 0.1),
            5,
        )
        model.SetDesiredHeight(10_000)
        model.Save(False, int(spec["entity_id"]))
        model.SetDesiredSpeed(300)
        model.Launch(UtilsPy.Vector3D(target_lon, target_lat, 0.0))
        minimum_by_target = {
            target_id: math.inf for target_id in target_positions
        }
        first_detection_second: dict[int, int] = {}
        for tick in range(horizon_seconds * 20):
            if tick % 20 == 0:
                second = tick // 20
                state = model.getState()
                point = state.posEcf()
                position = point.x(), point.y(), point.z()
                for target_id, target_position in target_positions.items():
                    destruction_step = destruction_steps.get(target_id)
                    if (
                        destruction_step is not None
                        and branch_step + second > destruction_step
                    ):
                        continue
                    distance = math.dist(position, target_position)
                    minimum_by_target[target_id] = min(
                        minimum_by_target[target_id], distance
                    )
                    if distance <= 30_000.0:
                        first_detection_second.setdefault(target_id, second)
            if model.Update() >= 0:
                break

        finite_distances = [
            value for value in minimum_by_target.values()
            if math.isfinite(value)
        ]
        nearest = min(finite_distances) if finite_distances else math.inf
        proximity_return = (
            0.25 * math.exp(-nearest / 100_000.0)
            if math.isfinite(nearest) else 0.0
        )
        direct_return = sum(
            (
                1.0
                + 0.25
                * max(
                    0.0,
                    min(
                        1.0,
                        (
                            int(spec["max_steps"])
                            - (branch_step + detection_second)
                        )
                        / remaining_steps,
                    ),
                )
            )
            / 9.0
            for detection_second in first_detection_second.values()
        )
        distances.append(nearest)
        direct_returns.append(direct_return)
        total_returns.append(direct_return + proximity_return)
        detection_counts.append(len(first_detection_second))

    return {
        "seed": int(spec["seed"]),
        "entity_id": int(spec["entity_id"]),
        "branch_step": branch_step,
        "horizon_seconds": horizon_seconds,
        "selected_index": int(spec["selected_index"]),
        "actual_nearest_alive_distance_m": float(
            spec["actual_nearest_alive_distance_m"]
        ),
        "actual_direct_return": float(spec["actual_direct_return"]),
        "distances_m": distances,
        "direct_returns": direct_returns,
        "total_returns": total_returns,
        "detection_counts": detection_counts,
        "observation": spec["observation"],
        "target_features": spec["target_features"],
        "target_valid_mask": spec["target_valid_mask"],
    }


def main() -> None:
    args = parse_args()
    progress = json.loads(
        args.training_progress.read_text(encoding="utf-8")
    )
    episodes = progress["episodes"]
    if args.limit is not None:
        episodes = episodes[: max(0, int(args.limit))]
    manifest_path = Path(progress["checkpoint"]).parents[1] / "inputs" / (
        "final20_guidance_manifest64.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario_path = Path(manifest["scenario"])
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    map_area = find_map_area(scenario)
    if map_area is None:
        raise ValueError(f"scenario has no mapArea: {scenario_path}")
    case_info = json.loads(
        (scenario_path.parent / "case_info.json").read_text(encoding="utf-8")
    )
    ships = {
        int(row["id"]): (float(row["lon"]), float(row["lat"]))
        for row in case_info["roots"]
        if row["kind"] == "ship"
    }
    specs = [
        episode_spec(
            episode,
            map_area=map_area,
            ships=ships,
            max_steps=args.max_steps,
        )
        for episode in episodes
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(simulate_episode, specs))

    selected_errors = []
    direct_errors = []
    for result in results:
        selected = int(result["selected_index"])
        selected_errors.append(abs(
            float(result["distances_m"][selected])
            - float(result["actual_nearest_alive_distance_m"])
        ))
        direct_errors.append(abs(
            float(result["direct_returns"][selected])
            - float(result["actual_direct_return"])
        ))
    distance_errors = np.asarray(selected_errors, dtype=np.float64)
    validation = {
        "samples": len(results),
        "selected_distance_mae_m": float(distance_errors.mean()),
        "selected_distance_median_error_m": float(np.median(distance_errors)),
        "selected_distance_p95_error_m": float(
            np.quantile(distance_errors, 0.95)
        ),
        "selected_distance_max_error_m": float(distance_errors.max()),
        "selected_direct_return_max_abs_error": float(max(direct_errors)),
    }
    if validation["selected_distance_p95_error_m"] > 500.0:
        raise RuntimeError(
            "native counterfactual replay failed selected-route calibration: "
            + json.dumps(validation)
        )
    payload = {
        "schema_version": 1,
        "return_semantics": (
            "native_L_motion_fixed_factual_survival_and_target_alive_windows"
        ),
        "source_training_progress": str(args.training_progress.resolve()),
        "map_area": map_area,
        "ship_ids": sorted(ships),
        "validation": validation,
        "observations": torch.stack([row["observation"] for row in results]),
        "target_features": torch.stack([
            row["target_features"] for row in results
        ]),
        "target_valid_mask": torch.stack([
            row["target_valid_mask"] for row in results
        ]),
        "returns": torch.tensor([
            row["total_returns"] for row in results
        ], dtype=torch.float32),
        "direct_returns": torch.tensor([
            row["direct_returns"] for row in results
        ], dtype=torch.float32),
        "distances_m": torch.tensor([
            row["distances_m"] for row in results
        ], dtype=torch.float32),
        "detection_counts": torch.tensor([
            row["detection_counts"] for row in results
        ], dtype=torch.long),
        "seeds": torch.tensor([row["seed"] for row in results]),
        "entity_ids": torch.tensor([row["entity_id"] for row in results]),
        "branch_steps": torch.tensor([row["branch_step"] for row in results]),
        "horizon_seconds": torch.tensor([
            row["horizon_seconds"] for row in results
        ]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
