"""Unified local entry point for the competition workspace."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from scenarios.cases import load_reward_policy


ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
SCENARIOS = ROOT / "scenarios" / "cases"
BLUE_POLICY_CHOICES = (
    "b0_fixed_ratio_random",
    "b1_nearest_interceptor",
    "b2_threat_priority",
    "b3_min_cost_assignment",
)
RED_POLICY_CHOICES = (
    "r0_random",
    "r1_priority",
    "r2_static_assignment",
    "r3_rolling_rules",
)


def scenario_path(case_id: str) -> Path:
    if case_id == "default":
        return CORE / "scenarios" / "platform.json"

    # Bundled case IDs always win. This keeps the current nine-map design
    # authoritative even when a similarly named legacy path exists.
    candidate = SCENARIOS / case_id / "scenario.json"
    if candidate.is_file():
        return candidate

    external = Path(case_id).expanduser()
    if external.is_file() and external.suffix.lower() == ".json":
        return external.resolve()

    raise ValueError(
        f"Unknown scenario '{case_id}'. Use e.g. easy/E01, medium/M01, "
        "hard/H01, or an existing scenario.json path."
    )


def with_competition_step_limit(scenario: Path, core_args: list[str]) -> list[str]:
    has_explicit_limit = any(
        value == "--max-steps" or value.startswith("--max-steps=")
        for value in core_args
    )
    if has_explicit_limit:
        return core_args

    policy = load_reward_policy(scenario)
    if policy is None:
        return core_args
    return ["--max-steps", str(policy.max_steps), *core_args]


def cmd_run(args: argparse.Namespace) -> int:
    try:
        scenario = scenario_path(args.scenario)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    output = ROOT / "results" / "runs" / args.run_id
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "main.py", "--scenario", str(scenario), "--output-dir", str(output)]
    # argparse keeps the separator used before REMAINDER.  It is meaningful to
    # this wrapper only; forwarding it makes core/main.py reject its arguments.
    core_args = args.core_args[1:] if args.core_args[:1] == ["--"] else args.core_args
    command.extend(with_competition_step_limit(scenario, core_args))
    env = os.environ.copy()
    env["BLUE_POLICY"] = args.blue_policy
    env["RED_POLICY"] = args.red_policy
    env["RED_MOTION_POLICY"] = args.red_motion_policy
    if args.seed is not None:
        env["BLUE_POLICY_SEED"] = str(args.seed)
        env["RED_POLICY_SEED"] = str(args.seed)
        env["SIMULATION_SEED"] = str(args.seed)
    return subprocess.call(command, cwd=CORE, env=env)


def cmd_ui(_: argparse.Namespace) -> int:
    from ui.server import serve
    serve(ROOT)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Competition platform workspace")
    subparsers = parser.add_subparsers(required=True)
    run_parser = subparsers.add_parser("run", help="Run a blue policy against a scenario")
    run_parser.add_argument(
        "--blue-policy",
        default="b0_fixed_ratio_random",
        choices=BLUE_POLICY_CHOICES,
        help="Blue interception policy (default: b0_fixed_ratio_random)",
    )
    run_parser.add_argument(
        "--red-policy",
        default="r0_random",
        choices=RED_POLICY_CHOICES,
        help="Red initial-catalogue baseline (default: r0_random)",
    )
    run_parser.add_argument(
        "--red-motion-policy",
        default="reactive_evasion",
        choices=["straight", "reactive_evasion"],
        help="Red post-launch motion policy (default: reactive_evasion)",
    )
    run_parser.add_argument("--scenario", default="default", help="default or easy/E01 … hard/H03")
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--run-id", default="manual")
    run_parser.add_argument("core_args", nargs=argparse.REMAINDER, help="Arguments forwarded to core/main.py")
    run_parser.set_defaults(handler=cmd_run)
    ui_parser = subparsers.add_parser("ui", help="Start the local plugin control page")
    ui_parser.set_defaults(handler=cmd_ui)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    raise SystemExit(arguments.handler(arguments))
