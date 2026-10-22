"""Run original V13, new local policy, and new policy plus package skill."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess


SCENARIOS = (
    "final20/easy/E01", "final20/easy/E02", "final20/easy/E03",
    "final20/medium/M01", "final20/medium/M02", "final20/medium/M03",
    "final20/hard/H01", "final20/hard/H02", "final20/hard/H03",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, default=Path("../../.venv/bin/h-single-maneuver-plan"))
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--old-runtime-root", type=Path, required=True)
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--new-runtime-root", type=Path, required=True)
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--prefix", default="network_compare")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument(
        "--blue-policies",
        default="b0_fixed_ratio_random",
        help="comma-separated blue policy names",
    )
    parser.add_argument("--blue-interceptor-ratio", type=int, default=2)
    parser.add_argument(
        "--conditions",
        default="v13,new,new_joint",
        help="comma-separated conditions: v13,new,new_joint,new_all",
    )
    parser.add_argument("--max-jobs", type=int, default=48)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    blue_policies = tuple(
        value.strip() for value in args.blue_policies.split(",") if value.strip()
    )
    selected_conditions = tuple(
        value.strip() for value in args.conditions.split(",") if value.strip()
    )
    output_root = args.output_root.resolve()
    logs = output_root / f"{args.prefix}_driver_logs"
    logs.mkdir(parents=True, exist_ok=True)
    conditions = {
        "v13": {
            "mode": "local-all",
            "runtime": args.old_runtime_root,
            "checkpoint": args.old_checkpoint,
            "gate": "mission_v12",
        },
        "new": {
            "mode": "local-all",
            "runtime": args.new_runtime_root,
            "checkpoint": args.new_checkpoint,
            "gate": "none",
        },
        "new_joint": {
            "mode": "joint-h-v13",
            "runtime": args.new_runtime_root,
            "checkpoint": args.new_checkpoint,
            "gate": "none",
        },
        "new_all": {
            "mode": "joint-hml-local",
            "runtime": args.new_runtime_root,
            "checkpoint": args.new_checkpoint,
            "gate": "none",
        },
    }
    unknown_conditions = sorted(set(selected_conditions) - set(conditions))
    if unknown_conditions:
        raise ValueError(f"Unknown benchmark conditions: {unknown_conditions}")
    tasks = []
    for scenario in SCENARIOS:
        case = scenario.rsplit("/", 1)[-1]
        for blue_policy in blue_policies:
            blue_short = blue_policy.split("_", 1)[0]
            for seed in seeds:
                for condition in selected_conditions:
                    config = conditions[condition]
                    run_id = (
                        f"{args.prefix}_{case}_{blue_short}_{condition}_s{seed}"
                    )
                    if list((output_root / run_id).glob("*/summary.json")) and not args.force:
                        continue
                    command = [
                        str(args.runner.resolve()),
                        "--environment-root", str(args.environment_root.resolve()),
                        "--plan-root", str(args.plan_root.resolve()),
                        "--scenario", scenario,
                        "--plan", "auto",
                        "--launch-mode", "native-waves",
                        "--lower-policy", config["mode"],
                        "--local-runtime-root", str(config["runtime"].resolve()),
                        "--local-checkpoint", str(config["checkpoint"].resolve()),
                        "--local-gate-mode", config["gate"],
                        "--blue-policy", blue_policy,
                        "--blue-interceptor-ratio", str(args.blue_interceptor_ratio),
                        "--seed", str(seed),
                        "--run-id", run_id,
                        "--output-root", str(output_root),
                        "--render-mode", "none",
                    ]
                    if condition in {"new_joint", "new_all"}:
                        command += ["--joint-rule-profile", "mission-aware"]
                    if condition == "new_all":
                        command += ["--mission-replan", "all-packages"]
                    command += ["--", "--disable-log-color"]
                    tasks.append((run_id, command))

    if args.dry_run:
        print(
            json.dumps(
                {
                    "task_count": len(tasks),
                    "blue_policies": blue_policies,
                    "seeds": seeds,
                    "first_run_id": tasks[0][0] if tasks else None,
                    "last_run_id": tasks[-1][0] if tasks else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOPFORMER_CPU_THREADS": "1"})

    def run_one(task):
        run_id, command = task
        log_path = logs / f"{run_id}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        return run_id, result.returncode, str(log_path)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_jobs) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            item = future.result()
            results.append(item)
            print(json.dumps({"run_id": item[0], "returncode": item[1]}), flush=True)
    failures = [item for item in results if item[1] != 0]
    index_path = logs / "run_index.json"
    index_path.write_text(
        json.dumps(
            {
                "requested": len(tasks),
                "completed": len(results) - len(failures),
                "failed": failures,
                "blue_policies": blue_policies,
                "seeds": seeds,
                "conditions": selected_conditions,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
