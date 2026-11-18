#!/usr/bin/env python3
"""Run the interface-adapted frozen CZ + V12-R3 policy on final all20."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
MAPS = tuple([f"E{i:02d}" for i in range(1, 7)] + [f"M{i:02d}" for i in range(1, 7)] + [f"H{i:02d}" for i in range(1, 9)])
RELEASE = Path(os.environ.get(
    "CZ_RELEASE",
    str(ROOT / "vendor/cz_merged_hp_seed2_20261101"),
)).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def status(root, maps, rows, running, failures, started):
    complete = {row["map"] for row in rows}
    pending = [name for name in maps if name not in complete and name not in running and name not in failures]
    save(root / "status.json", {
        "status": "failed" if failures else "running" if pending or running else "complete",
        "seed": 2,
        "completed": len(rows),
        "total": len(maps),
        "running": sorted(running),
        "pending": pending,
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
        "updated_at_epoch": time.time(),
    })


def map_run(root, map_name):
    folder = root / "maps" / map_name
    folder.mkdir(parents=True, exist_ok=False)
    config = {
        "agent": str(ROOT / "user_agents/cz_ppo/__init__.py"),
        "scenario": str(ROOT / "scenarios" / f"{map_name}.json"),
        "seed": 2,
        "blue_mode": str(ROOT / "blue_defense/immediate/b0_fixed_ratio_random.py"),
        "render_mode": "none",
        "output_dir": str(folder),
        "logging": {
            "enabled": False,
            "state_interval_steps": 100,
            "events": False,
            "actions": False,
            "example_evaluation": True,
            "attribution_evaluation": True,
        },
    }
    config_path = folder / "run.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    environment = os.environ.copy()
    environment.update({
        "CZ_MAP": map_name,
        "CZ_OUTPUT_DIR": str(folder),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    })
    started = time.monotonic()
    with (folder / "console.log").open("wb") as output:
        process = subprocess.run(
            [sys.executable, "-B", str(ROOT / "main.py"), "--config", str(config_path)],
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    if process.returncode:
        raise RuntimeError(f"{map_name} exited with {process.returncode}; see {folder / 'console.log'}")
    result_path = folder / "example_evaluation.json"
    audit_path = folder / "cz_adapter_audit.json"
    attribution_path = folder / "attribution.json"
    if not result_path.is_file() or not audit_path.is_file() or not attribution_path.is_file():
        raise RuntimeError(f"{map_name} missing terminal result, adapter audit or attribution")
    result = json.loads(result_path.read_text())
    audit = json.loads(audit_path.read_text())
    attribution = json.loads(attribution_path.read_text())
    if audit.get("map") != map_name or audit.get("seed") != 2:
        raise RuntimeError(f"{map_name} adapter audit mismatch")
    return {
        "map": map_name,
        "score": float(result["score"]["score"]),
        "completed": bool(result["score"]["completed"]),
        "steps": int(result["steps_executed"]),
        "termination": result["termination_reason"],
        "choice": audit["choice"],
        "satellite_requests": int(audit["counts"].get("satellite_requests", 0)),
        "motion_calls": audit["motion_calls"],
        "attribution": attribution["summary"],
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maps", nargs="+", choices=MAPS)
    args = parser.parse_args()
    if not 1 <= args.workers <= 20:
        parser.error("--workers must be 1..20")
    maps = tuple(dict.fromkeys(args.maps or MAPS))
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "maps").mkdir()
    started = time.monotonic()
    rows, failures, running = [], {}, set()
    save(root / "protocol.json", {
        "name": "competition-final-cz-ppo-seed2-all20",
        "maps": list(maps),
        "seed": 2,
        "red": "frozen CZ merged+HP with V12-R3 PPO",
        "blue": "b0_fixed_ratio_random",
        "official_object_address_order_retained": True,
        "algorithm_changes": [],
        "interface_adaptations": [
            "Cartesian metres to legacy geographic policy ABI",
            "new BaseAgent and Deployment contracts",
            "legacy actions 4/2/3/0 to public actions 1/1/2/3",
            "final submitted waypoints projected to the official closed map boundary while internal tracks remain unchanged",
            "actual final-environment inventory and public health passed to unchanged planner",
        ],
        "workers": args.workers,
        "python": sys.version,
        "sha256": {
            "final_engine": sha256(ROOT / "envengine/engine/engine.py"),
            "adapter_policy": sha256(ROOT / "user_agents/cz_ppo/policy.py"),
            "coordinate_adapter": sha256(ROOT / "user_agents/cz_ppo/compat.py"),
            "frozen_planner": sha256(RELEASE / "app/design/planner_v2.py"),
            "frozen_choices": sha256(RELEASE / "opening/frozen_choices.json"),
            "frozen_hp_tree": sha256(RELEASE / "app/skills/hp_tree/runtime.py"),
            "frozen_ppo": sha256(RELEASE / "app/r3/model.pt"),
        },
    })
    status(root, maps, rows, running, failures, started)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for map_name in maps:
            future = pool.submit(map_run, root, map_name)
            futures[future] = map_name
            running.add(map_name)
        status(root, maps, rows, running, failures, started)
        for future in concurrent.futures.as_completed(futures):
            map_name = futures[future]
            running.remove(map_name)
            try:
                row = future.result()
                rows.append(row)
                save(root / "maps" / map_name / "result.json", row)
            except BaseException as error:
                failures[map_name] = repr(error)
            status(root, maps, rows, running, failures, started)
    ordered = sorted(rows, key=lambda row: maps.index(row["map"]))
    summary = {
        "status": "complete" if len(ordered) == len(maps) and not failures else "failed",
        "seed": 2,
        "completed_maps": len(ordered),
        "missing_maps": [name for name in maps if name not in {row["map"] for row in ordered}],
        "failures": failures,
        "mean_score": statistics.fmean(row["score"] for row in ordered) if ordered else None,
        "full_score_maps": [row["map"] for row in ordered if abs(row["score"] - 100.0) < 1e-10],
        "rows": ordered,
        "elapsed_seconds": time.monotonic() - started,
    }
    save(root / "summary.json", summary)
    with (root / "returns.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "map", "score", "completed", "steps", "termination", "choice",
            "satellite_requests", "elapsed_seconds",
        ))
        writer.writeheader()
        for row in ordered:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    status(root, maps, rows, running, failures, started)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
