"""Upgrade v6 teacher tensors with the missing continuous SEARCH coordinate factor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch


FACTOR_NAMES = (
    "presence",
    "initial_position",
    "search_position",
    "retarget",
    "target",
    "maneuver",
    "satellite",
)
SEARCH_TARGET_INDEX = 7
LOW_AGENT_START = 85


def trace_commands(path: Path) -> dict[tuple[int, int], dict]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    commands: dict[tuple[int, int], dict] = {}
    for step_row in trace["steps"]:
        step = int(step_row["step"])
        for command in step_row["actions"]:
            command_type = int(command.get("commandType_id", -1))
            if command_type in {200, 3014}:
                commands[(step, int(command["executor_id"]))] = command
    return commands


def search_index(
    command: dict,
    bounds: tuple[float, float, float, float],
    width: int = 16,
    height: int = 12,
) -> int:
    lon_min, lon_max, lat_min, lat_max = bounds
    target = command["target"]
    column = min(
        width - 1,
        max(0, int((float(target["x"]) - lon_min) / (lon_max - lon_min) * width)),
    )
    row = min(
        height - 1,
        max(0, int((float(target["y"]) - lat_min) / (lat_max - lat_min) * height)),
    )
    return row * width + column


def convert_seed(source_manifest_path: Path, output_root: Path) -> dict:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    commands = trace_commands(Path(source_manifest["source_trace"]))
    seed_dir = output_root / f"seed_{int(source_manifest['seed']):04d}"
    chunk_dir = seed_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_chunks: list[dict] = []
    factor_counts = torch.zeros(len(FACTOR_NAMES), dtype=torch.long)
    skipped_target_count = 0
    bounds = (115.8, 128.2, 18.4, 28.2)

    for chunk_number, source_chunk in enumerate(source_manifest["chunks"]):
        payload = torch.load(source_chunk["path"], map_location="cpu", weights_only=True)
        old_mask = payload["factor_mask"].to(dtype=torch.bool)
        count = int(old_mask.shape[0])
        new_mask = torch.zeros((count, len(FACTOR_NAMES)), dtype=torch.bool)
        new_mask[:, 0] = old_mask[:, 0]
        new_mask[:, 1] = old_mask[:, 1]
        new_mask[:, 3] = old_mask[:, 2]
        new_mask[:, 4] = old_mask[:, 3]
        new_mask[:, 5] = old_mask[:, 4]
        new_mask[:, 6] = old_mask[:, 5]
        searches = torch.zeros(count, dtype=torch.long)
        retarget = payload["retarget"].clone()
        target_index = payload["target_index"].clone()

        for row in range(count):
            key = (int(payload["steps"][row]), int(payload["entity_ids"][row]))
            command = commands.get(key)
            if command is None:
                continue
            command_type = int(command["commandType_id"])
            if command_type == 3014:
                retarget[row] = 1.0
                new_mask[row, 3] = True
            index = int(target_index[row])
            if bool(payload["target_valid_mask"][row, index]):
                new_mask[row, 4] = True
                continue
            if int(payload["agent_ids"][row]) >= LOW_AGENT_START:
                target_index[row] = SEARCH_TARGET_INDEX
                new_mask[row, 2] = True
                new_mask[row, 4] = True
                searches[row] = search_index(command, bounds)
            else:
                new_mask[row, 4] = False
                skipped_target_count += 1

        converted = dict(payload)
        converted["factor_mask"] = new_mask
        converted["search_index"] = searches
        converted["retarget"] = retarget
        converted["target_index"] = target_index
        output_path = chunk_dir / f"chunk_{chunk_number:05d}.pt"
        torch.save(converted, output_path)
        output_chunks.append({
            "path": str(output_path.resolve()),
            "sample_count": count,
        })
        factor_counts += new_mask.sum(dim=0)

    manifest = dict(source_manifest)
    manifest.update({
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "r9_to_unified_mappo_teacher_actions_v7_search_grid",
        "factor_counts": dict(zip(FACTOR_NAMES, factor_counts.tolist())),
        "skipped_target_count": skipped_target_count,
        "chunks": output_chunks,
        "source_manifest": str(source_manifest_path.resolve()),
    })
    output_manifest = seed_dir / "manifest.json"
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifests = [
        convert_seed(path, args.output_root)
        for path in sorted(args.source_root.glob("seed_*/manifest.json"))
    ]
    totals = {
        name: sum(int(row["factor_counts"][name]) for row in manifests)
        for name in FACTOR_NAMES
    }
    summary = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "r9_to_unified_mappo_teacher_actions_v7_search_grid",
        "seed_count": len(manifests),
        "sample_count": sum(int(row["sample_count"]) for row in manifests),
        "factor_counts": totals,
        "teacher_score_mean": (
            sum(float(row["teacher_score"]) for row in manifests) / len(manifests)
        ),
    }
    destination = args.output_root / "dataset_manifest.json"
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
