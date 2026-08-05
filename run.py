"""Unified local entry point for the plugin-based competition workspace."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
PLUGINS = ROOT / "plugins"


def load_registry() -> dict:
    return json.loads((PLUGINS / "registry.json").read_text(encoding="utf-8"))


def scenario_path(case_id: str) -> Path:
    if case_id == "default":
        return CORE / "scenarios" / "platform.json"
    candidate = PLUGINS / "scenarios" / "competition_cases" / case_id / "scenario.json"
    if not candidate.is_file():
        raise ValueError(f"Unknown scenario '{case_id}'. Use e.g. easy/E01, medium/M01, hard/H01.")
    return candidate


def plugin_env() -> dict[str, str]:
    env = os.environ.copy()
    blue_src = PLUGINS / "blue_baselines" / "src"
    red_src = PLUGINS / "red_baselines" / "red_strategy_lab" / "src"
    paths = [str(blue_src), str(red_src), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
    return env


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
    command.extend(core_args)
    env = plugin_env()
    env["BLUE_POLICY"] = args.blue_policy
    env["RED_POLICY"] = args.red_policy
    if args.seed is not None:
        env["BLUE_POLICY_SEED"] = str(args.seed)
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
    run_parser.add_argument("--blue-policy", default="fixed_ratio_random")
    run_parser.add_argument("--red-policy", default="b0_random", choices=["b0_random", "b1_priority", "b2_static_assignment", "b3_rolling_rules"])
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
