"""Unified local entry point for the plugin-based competition workspace."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from plugins.scenarios.competition_cases import load_reward_policy


ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
PLUGINS = ROOT / "plugins"
BLUE_POLICY_CHOICES = (
    "fixed_ratio_random",
    "nearest_interceptor",
    "threat_priority",
    "min_cost_assignment",
)


def load_registry() -> dict:
    return json.loads((PLUGINS / "registry.json").read_text(encoding="utf-8"))


def scenario_path(case_id: str) -> Path:
    if case_id == "default":
        return CORE / "scenarios" / "platform.json"

    # Bundled case IDs always win. This keeps the current nine-map design
    # authoritative even when a similarly named legacy path exists.
    candidate = PLUGINS / "scenarios" / "competition_cases" / case_id / "scenario.json"
    if candidate.is_file():
        return candidate

    external = Path(case_id).expanduser()
    if external.is_file() and external.suffix.lower() == ".json":
        return external.resolve()

    raise ValueError(
        f"Unknown scenario '{case_id}'. Use e.g. easy/E01, medium/M01, "
        "hard/H01, or an existing scenario.json path."
    )


def plugin_env() -> dict[str, str]:
    env = os.environ.copy()
    blue_src = PLUGINS / "blue_baselines" / "src"
    red_src = PLUGINS / "red_baselines" / "red_strategy_lab" / "src"
    paths = [str(blue_src), str(red_src), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
    return env


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


def cmd_list(_: argparse.Namespace) -> int:
    for plugin in load_registry()["plugins"]:
        print(f"{plugin['id']:18} {plugin['status']:20} {plugin['path']}")
    return 0


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
    env = plugin_env()
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
    parser = argparse.ArgumentParser(description="Competition platform plugin workspace")
    subparsers = parser.add_subparsers(required=True)
    list_parser = subparsers.add_parser("list", help="List registered plugins")
    list_parser.set_defaults(handler=cmd_list)
    run_parser = subparsers.add_parser("run", help="Run a blue baseline against a scenario")
    run_parser.add_argument(
        "--blue-policy",
        default="fixed_ratio_random",
        choices=BLUE_POLICY_CHOICES,
        help="Blue interception policy (default: fixed_ratio_random)",
    )
    run_parser.add_argument("--red-policy", default="b0_random", choices=["b0_random", "b1_priority", "b2_static_assignment", "b3_rolling_rules"])
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
