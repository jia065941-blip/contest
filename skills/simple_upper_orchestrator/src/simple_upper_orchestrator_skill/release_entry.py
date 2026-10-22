"""CLI wiring into the unmodified official release."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys

from .config import OrchestratorConfig


DEFAULT_COMPETITION_ROOT = Path("/home/amax/ry/competition/competition-platform-env-release20261021")
DEFAULT_RUNTIME_ROOT = Path("/home/amax/ry/competition/jmy_ppo/contest-jmy")
DEFAULT_CHECKPOINT = DEFAULT_RUNTIME_ROOT / "models/local_avoid_set_attention_v12_r3_timely_arrival.pt"


def _bootstrap(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--competition-root", type=Path, default=DEFAULT_COMPETITION_ROOT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    return parser.parse_known_args(argv)[0]


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Simple upper orchestrator")
    group.add_argument("--competition-root", type=Path, default=DEFAULT_COMPETITION_ROOT)
    group.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    group.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    group.add_argument("--output-root", type=Path)
    group.add_argument("--run-id")
    group.add_argument("--deploy-front-axis", choices=("east", "west", "north", "south"), default="east")
    group.add_argument("--deploy-gap-km", type=float, default=30.0)
    group.add_argument("--deploy-l-depth-km", type=float, default=5.0)
    group.add_argument("--deploy-hm-depth-km", type=float, default=10.0)
    group.add_argument("--deploy-spacing-km", type=float, default=1.0)
    group.add_argument("--deploy-batch-size", type=int, default=10)
    group.add_argument("--l-satellite-trigger-step", type=int, default=1)
    group.add_argument("--l-package-size", type=int)
    group.add_argument("--l-package-count", type=int)
    group.add_argument("--l-reserve-count", type=int, default=0)
    group.add_argument("--l-expected-target-count", type=int)
    group.add_argument("--hm-safety-factor", type=float, default=1.25)
    group.add_argument("--hm-reserve-h", type=int, default=0)
    group.add_argument("--hm-reserve-m", type=int, default=0)


def build_parser(competition_root: Path = DEFAULT_COMPETITION_ROOT) -> argparse.ArgumentParser:
    root = competition_root.expanduser().resolve(strict=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    official = importlib.import_module("main")
    parser = official.build_parser()
    _add_arguments(parser)
    return parser


def _config(args) -> OrchestratorConfig:
    from hm_public_strike_packages_skill import PublicStrikeConfig
    from l_satellite_strike_packages_skill import SatelliteStrikeConfig
    from staggered_deployment_skill import StaggeredDeploymentConfig

    allocation_mode = "fixed_size" if args.l_package_size is not None else "equal"
    return OrchestratorConfig(
        runtime_root=args.runtime_root.expanduser().resolve(strict=True),
        checkpoint=args.checkpoint.expanduser().resolve(strict=True), max_steps=args.max_steps,
        deployment=StaggeredDeploymentConfig(
            front_axis=args.deploy_front_axis, deployment_gap_km=args.deploy_gap_km,
            l_depth_km=args.deploy_l_depth_km, hm_depth_km=args.deploy_hm_depth_km,
            lateral_spacing_km=args.deploy_spacing_km,
        ),
        deployment_batch_size=args.deploy_batch_size,
        l_strike=SatelliteStrikeConfig(
            satellite_trigger_step=args.l_satellite_trigger_step,
            allocation_mode=allocation_mode, package_size=args.l_package_size,
            package_count=args.l_package_count, reserve_count=args.l_reserve_count,
            expected_target_count=args.l_expected_target_count,
        ),
        hm_strike=PublicStrikeConfig(
            safety_factor=args.hm_safety_factor,
            reserve_h=args.hm_reserve_h, reserve_m=args.hm_reserve_m,
        ),
    )


def migrate_legacy_comm_range(value) -> bool:
    """In-place profile migration from legacy empty arrays to empty objects."""
    changed = False
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "commRangeInfo" and child == []:
                value[key] = {}
                changed = True
            else:
                changed = migrate_legacy_comm_range(child) or changed
    elif isinstance(value, list):
        for child in value:
            changed = migrate_legacy_comm_range(child) or changed
    return changed


def _migrated_scenario(args) -> None:
    source = Path(str(args.scenario)).expanduser()
    if not source.is_file():
        return
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not migrate_legacy_comm_range(payload):
        return
    output = (args.output_root or args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    migrated = output / f"{source.stem}.comm-profile.json"
    migrated.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.scenario = str(migrated)


def _compatible_read_profile(original):
    def read_profile(path: str):
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not migrate_legacy_comm_range(payload):
            return original(path)
        from envengine import Profile
        return Profile.from_dict(payload)
    return read_profile


@contextmanager
def patched_runtime(config, official=None):
    import user_agents.red
    from .factory import SimpleUpperFactory

    official = official or importlib.import_module("main")
    original = user_agents.red.RedAgentFactory
    original_read_profile = official.read_profile
    configured = type("ConfiguredSimpleUpperFactory", (SimpleUpperFactory,), {
        "orchestrator_config": config,
        "instances": [],
    })
    user_agents.red.RedAgentFactory = configured
    official.read_profile = _compatible_read_profile(original_read_profile)
    try:
        yield
    finally:
        for instance in configured.instances:
            instance.close()
        user_agents.red.RedAgentFactory = original
        official.read_profile = original_read_profile


patched_factory = patched_runtime


def main(argv: list[str] | None = None) -> int:
    bootstrap = _bootstrap(argv)
    parser = build_parser(bootstrap.competition_root)
    args = parser.parse_args(argv)
    if args.deployment_plan is not None:
        parser.error("20261021 simple upper runs do not accept --deployment-plan")
    if args.output_root is not None:
        args.output_dir = args.output_root.expanduser()
    if args.run_id:
        args.output_dir = args.output_dir / args.run_id
    try:
        _migrated_scenario(args)
        official = importlib.import_module("main")
        bundle = official.prepare_run(args)
        config = _config(args)
        os.environ["LOCAL_AVOID_GATE_MODE"] = "none"
        with patched_runtime(config, official):
            official.run_simulation(args, bundle)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
