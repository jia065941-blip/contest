"""CLI wiring into the unmodified official release."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import importlib
import json
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
    group.add_argument("--deploy-gap-km", type=float, default=15.0)
    group.add_argument("--deploy-l-depth-km", type=float, default=5.0)
    group.add_argument("--deploy-l-reserve-gap-km", type=float, default=8.0)
    group.add_argument("--deploy-l-reserve-depth-km", type=float, default=5.0)
    group.add_argument("--deploy-hm-depth-km", type=float, default=10.0)
    group.add_argument("--deploy-h-depth-km", type=float, default=5.0)
    group.add_argument("--deploy-m-depth-km", type=float, default=10.0)
    group.add_argument("--deploy-hm-follow-gap-km", type=float, default=0.0)
    group.add_argument("--deploy-mother-crossfire-angle", type=float, default=35.0)
    group.add_argument("--deploy-mother-crossfire-max-offset-km", type=float, default=35.0)
    group.add_argument("--deploy-sentry-m-lead-km", type=float, default=7.5)
    group.add_argument(
        "--deploy-sentry-rendezvous-mode",
        choices=("adaptive", "fixed"),
        default="adaptive",
    )
    group.add_argument("--deploy-h-speed", type=float, default=200.0)
    group.add_argument("--deploy-m-speed", type=float, default=120.0)
    group.add_argument("--deploy-m-arrival-lead-seconds", type=float, default=5.0)
    group.add_argument("--deploy-min-m-lead", type=float, default=10.0)
    group.add_argument("--deploy-max-m-lead", type=float)
    group.add_argument("--deploy-spacing-km", type=float, default=1.0)
    group.add_argument("--deploy-l-first-wave-fraction", type=float, default=0.70)
    group.add_argument("--deploy-batch-size", type=int, default=10)
    group.add_argument("--l-satellite-trigger-step", type=int, default=1)
    group.add_argument(
        "--l-allocation-mode",
        choices=("distance_balanced", "equal", "fixed_size"),
        default="distance_balanced",
    )
    group.add_argument("--l-package-size", type=int)
    group.add_argument("--l-package-count", type=int)
    group.add_argument("--l-reserve-count", type=int, default=0)
    group.add_argument("--l-expected-target-count", type=int)
    group.add_argument("--hm-safety-factor", type=float, default=1.25)
    group.add_argument(
        "--hm-allocation-mode",
        choices=(
            "full_coverage_mixed", "deterministic_thresholds", "threshold_packages",
            "kill_probability", "expected_damage",
        ),
        default="full_coverage_mixed",
    )
    group.add_argument("--hm-target-kill-probability", type=float, default=0.85)
    group.add_argument("--hm-min-h-per-mother", type=int, default=2)
    group.add_argument("--hm-min-m-per-mother", type=int, default=0)
    group.add_argument("--hm-min-h-per-sentry", type=int, default=1)
    group.add_argument("--hm-max-h-per-mother", type=int, default=2)
    group.add_argument("--hm-h-per-sentry-package", type=int, default=2)
    group.add_argument("--hm-m-per-h-mixed-package", type=int, default=2)
    group.add_argument("--hm-m-only-sentry-package", type=int, default=6)
    group.add_argument("--hm-allow-m-only-sentry-packages", action="store_true")
    group.add_argument(
        "--hm-m-target-policy",
        choices=("complete_h_sentries", "incomplete_h_sentries", "mothers"),
        default="complete_h_sentries",
    )
    group.add_argument(
        "--hm-target-priority-axis",
        choices=("east", "west", "north", "south", "none"),
        default="east",
    )
    group.add_argument("--hm-reserve-h", type=int, default=0)
    group.add_argument("--hm-reserve-m", type=int, default=0)
    group.add_argument("--reserve-h-count", type=int, default=0)
    group.add_argument("--l-first-wave-per-target", type=int, default=0)
    group.add_argument("--reserve-deadline-step", type=int, default=450)
    group.add_argument("--reserve-l-deadline-step", type=int, default=250)
    group.add_argument("--reserve-max-stage", type=int)
    group.add_argument("--reserve-h-max-distance", type=float, default=80.0)
    group.add_argument("--reserve-l-max-distance", type=float, default=100.0)
    group.add_argument("--reserve-h-max-turn", type=float)
    group.add_argument("--reserve-l-max-turn", type=float)
    group.add_argument("--reserve-h-turn-rate", type=float, default=20.0)
    group.add_argument("--reserve-l-turn-rate", type=float, default=15.0)
    group.add_argument("--reserve-turn-safety-seconds", type=float, default=5.0)
    group.add_argument("--reserve-holding-half-width", type=float, default=6.0)
    group.add_argument("--reserve-holding-heading-offset", type=float, default=35.0)
    group.add_argument("--reserve-holding-max-turn-steps", type=int, default=6)
    group.add_argument("--reserve-holding-straight-steps", type=int, default=2)
    group.add_argument("--dynamic-retarget-threshold", type=float, default=0.70)
    group.add_argument("--dynamic-retarget-max-stage", type=int, default=3)
    group.add_argument("--dynamic-retarget-min-distance", type=float, default=15.0)
    group.add_argument("--dynamic-retarget-cooldown", type=int, default=25)
    group.add_argument("--dynamic-retarget-max-retarget", type=int, default=1)
    group.add_argument("--dynamic-retarget-max-fraction", type=float, default=0.20)
    group.add_argument("--dynamic-retarget-distance-weight", type=float, default=0.05)
    group.add_argument("--dynamic-retarget-track-max-age", type=float, default=3)
    group.add_argument(
        "--dynamic-retarget-max-target-separation", type=float, default=10.0,
    )
    group.add_argument("--dynamic-retarget-max-turn-angle", type=float, default=45.0)
    group.add_argument(
        "--dynamic-retarget-max-candidate-distance", type=float, default=30.0,
    )
    group.add_argument("--dynamic-retarget-max-per-old-target", type=int, default=1)
    group.add_argument("--disable-dynamic-retarget", action="store_true")
    group.add_argument("--enable-dynamic-retarget", action="store_true")
    group.add_argument("--disable-terminal-guidance", action="store_true")
    group.add_argument("--enable-terminal-guidance", action="store_true")
    group.add_argument(
        "--terminal-types", default="H,M,L",
        help="Comma-separated terminal-guidance profiles: H,M,L",
    )
    group.add_argument("--terminal-min-stage", type=int, default=4)
    group.add_argument("--terminal-enter-distance", type=float)
    group.add_argument("--terminal-capture-distance", type=float, default=600.0)
    group.add_argument("--terminal-final-straight-distance", type=float, default=200.0)
    group.add_argument(
        "--terminal-h-capture", "--terminal-h-capture300",
        dest="terminal_h_capture", type=float,
    )
    group.add_argument(
        "--terminal-h-final", "--terminal-h-final150",
        dest="terminal_h_final", type=float,
    )
    group.add_argument(
        "--terminal-m-capture", "--terminal-m-capture600",
        dest="terminal_m_capture", type=float,
    )
    group.add_argument(
        "--terminal-m-final", "--terminal-m-final200",
        dest="terminal_m_final", type=float,
    )
    group.add_argument(
        "--terminal-l-capture", "--terminal-l-capture600",
        dest="terminal_l_capture", type=float,
    )
    group.add_argument(
        "--terminal-l-final", "--terminal-l-final200",
        dest="terminal_l_final", type=float,
    )
    group.add_argument("--terminal-straight-distance", type=float)
    group.add_argument("--terminal-exit-distance", type=float, default=3500.0)
    group.add_argument("--terminal-max-track-age", type=int, default=5)
    group.add_argument("--terminal-stale-grace-age", type=int, default=8)
    group.add_argument("--terminal-threat-distance", type=float, default=3000.0)
    group.add_argument("--terminal-heading-enter", type=float, default=6.0)
    group.add_argument("--terminal-heading-exit", type=float, default=10.0)
    group.add_argument("--terminal-heading-deadband", type=float)
    group.add_argument("--terminal-predicted-miss", type=float, default=80.0)
    group.add_argument("--terminal-turn-pulse-steps", type=int, default=1)
    group.add_argument("--terminal-straight-recovery-steps", type=int, default=2)
    group.add_argument("--terminal-no-improvement-limit", type=int, default=3)
    group.add_argument("--terminal-respect-interceptors-after-latch", action="store_true")
    group.add_argument("--disable-retarget-audit", action="store_true")
    group.add_argument("--formation-audit-interval", type=int, default=25)
    group.add_argument("--missile-trace-interval", type=int, default=0)
    group.add_argument("--missile-terminal-history", type=int, default=20)
    group.add_argument("--seconds-per-step", type=float, default=1.0)
    group.add_argument("--velocity-window-steps", type=int, default=5)


def build_parser(competition_root: Path = DEFAULT_COMPETITION_ROOT) -> argparse.ArgumentParser:
    root = competition_root.expanduser().resolve(strict=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    official = importlib.import_module("main")
    parser = official.build_parser()
    _add_arguments(parser)
    return parser


def _config(args) -> OrchestratorConfig:
    if args.enable_dynamic_retarget and args.disable_dynamic_retarget:
        raise ValueError(
            "--enable-dynamic-retarget and --disable-dynamic-retarget are mutually exclusive"
        )
    if args.enable_terminal_guidance and args.disable_terminal_guidance:
        raise ValueError(
            "--enable-terminal-guidance and --disable-terminal-guidance are mutually exclusive"
        )
    if args.reserve_h_count and args.hm_reserve_h:
        raise ValueError(
            "--reserve-h-count and legacy --hm-reserve-h cannot both be nonzero"
        )
    if args.hm_reserve_m:
        raise ValueError("reserve echelon 0.4.0 does not support M reserves")
    reserve_h_count = args.reserve_h_count or args.hm_reserve_h
    from dynamic_retarget_skill import DynamicRetargetConfig
    from hm_public_strike_packages_skill import PublicStrikeConfig
    from l_satellite_strike_packages_skill import SatelliteStrikeConfig
    from staggered_deployment_skill import StaggeredDeploymentConfig
    from terminal_guidance_override_skill import TerminalGuidanceConfig

    allocation_mode = "fixed_size" if args.l_package_size is not None else args.l_allocation_mode
    terminal_kwargs = {
        "min_stage": args.terminal_min_stage,
        "capture_distance": args.terminal_capture_distance,
        "final_straight_distance": args.terminal_final_straight_distance,
        "max_track_age": args.terminal_max_track_age,
        "stale_grace_age": args.terminal_stale_grace_age,
        "heading_enter_straight_deg": args.terminal_heading_enter,
        "heading_exit_straight_deg": args.terminal_heading_exit,
        "predicted_miss_threshold_m": args.terminal_predicted_miss,
        "turn_pulse_steps": args.terminal_turn_pulse_steps,
        "straight_recovery_steps": args.terminal_straight_recovery_steps,
        "no_improvement_limit": args.terminal_no_improvement_limit,
        "enter_distance_m": args.terminal_enter_distance,
        "straight_distance_m": args.terminal_straight_distance,
        "exit_distance_m": args.terminal_exit_distance,
        "threat_distance_m": args.terminal_threat_distance,
        "heading_deadband_degrees": args.terminal_heading_deadband,
        "ignore_interceptors_after_latch": (
            not args.terminal_respect_interceptors_after_latch
        ),
    }
    terminal_guidance = TerminalGuidanceConfig(**terminal_kwargs)
    profile_overrides = {
        21000: (args.terminal_h_capture, args.terminal_h_final),
        21001: (args.terminal_m_capture, args.terminal_m_final),
        21002: (args.terminal_l_capture, args.terminal_l_final),
    }
    if any(value is not None for values in profile_overrides.values() for value in values):
        profiles = {}
        for missile_type, profile in terminal_guidance.profiles:
            capture, final = profile_overrides[missile_type]
            profiles[missile_type] = replace(
                profile,
                capture_distance=(
                    profile.capture_distance if capture is None else capture
                ),
                final_straight_distance=(
                    profile.final_straight_distance if final is None else final
                ),
            )
        terminal_guidance = TerminalGuidanceConfig(
            **terminal_kwargs, profiles=profiles,
        )
    type_names = {"H": 21000, "M": 21001, "L": 21002}
    requested_type_names = tuple(
        item.strip().upper()
        for item in args.terminal_types.split(",")
        if item.strip()
    )
    unknown_type_names = set(requested_type_names) - set(type_names)
    if unknown_type_names:
        raise ValueError(
            "--terminal-types accepts only H,M,L; got "
            + ",".join(sorted(unknown_type_names))
        )
    enabled_type_ids = {type_names[name] for name in requested_type_names}
    terminal_guidance = TerminalGuidanceConfig(
        **terminal_kwargs,
        profiles={
            missile_type: profile
            for missile_type, profile in terminal_guidance.profiles
            if missile_type in enabled_type_ids
        },
    )

    return OrchestratorConfig(
        runtime_root=args.runtime_root.expanduser().resolve(strict=True),
        checkpoint=args.checkpoint.expanduser().resolve(strict=True), max_steps=args.max_steps,
        deployment=StaggeredDeploymentConfig(
            front_axis=args.deploy_front_axis, deployment_gap_km=args.deploy_gap_km,
            l_depth_km=args.deploy_l_depth_km, hm_depth_km=args.deploy_hm_depth_km,
            l_reserve_gap_km=args.deploy_l_reserve_gap_km,
            l_reserve_depth_km=args.deploy_l_reserve_depth_km,
            h_depth_km=args.deploy_h_depth_km,
            m_depth_km=args.deploy_m_depth_km,
            hm_follow_gap_km=args.deploy_hm_follow_gap_km,
            mother_crossfire_angle_degrees=args.deploy_mother_crossfire_angle,
            mother_crossfire_max_offset_km=args.deploy_mother_crossfire_max_offset_km,
            sentry_m_lead_km=args.deploy_sentry_m_lead_km,
            sentry_rendezvous_mode=args.deploy_sentry_rendezvous_mode,
            h_speed_mps=args.deploy_h_speed,
            m_speed_mps=args.deploy_m_speed,
            desired_m_arrival_lead_seconds=args.deploy_m_arrival_lead_seconds,
            min_sentry_m_lead_km=args.deploy_min_m_lead,
            max_sentry_m_lead_km=args.deploy_max_m_lead,
            lateral_spacing_km=args.deploy_spacing_km,
            l_first_wave_fraction=args.deploy_l_first_wave_fraction,
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
            reserve_h=reserve_h_count, reserve_m=0,
            allocation_mode=args.hm_allocation_mode,
            target_kill_probability=args.hm_target_kill_probability,
            min_h_per_mother=args.hm_min_h_per_mother,
            min_m_per_mother=args.hm_min_m_per_mother,
            min_h_per_sentry=args.hm_min_h_per_sentry,
            max_h_per_mother=args.hm_max_h_per_mother,
            h_per_sentry_package=args.hm_h_per_sentry_package,
            m_per_h_mixed_package=args.hm_m_per_h_mixed_package,
            m_only_sentry_package=args.hm_m_only_sentry_package,
            allow_m_only_sentry_packages=args.hm_allow_m_only_sentry_packages,
            m_target_policy=args.hm_m_target_policy,
            target_priority_axis=args.hm_target_priority_axis,
        ),
        reserve_h_count=reserve_h_count,
        l_first_wave_per_target=args.l_first_wave_per_target,
        reserve_deadline_step=args.reserve_deadline_step,
        reserve_l_deadline_step=args.reserve_l_deadline_step,
        reserve_max_stage=args.reserve_max_stage,
        reserve_h_max_distance_km=args.reserve_h_max_distance,
        reserve_l_max_distance_km=args.reserve_l_max_distance,
        reserve_h_max_turn_angle_degrees=args.reserve_h_max_turn,
        reserve_l_max_turn_angle_degrees=args.reserve_l_max_turn,
        reserve_h_turn_rate_degrees_per_second=args.reserve_h_turn_rate,
        reserve_l_turn_rate_degrees_per_second=args.reserve_l_turn_rate,
        reserve_turn_safety_seconds=args.reserve_turn_safety_seconds,
        reserve_holding_half_width_km=args.reserve_holding_half_width,
        reserve_holding_heading_offset_degrees=args.reserve_holding_heading_offset,
        reserve_holding_max_turn_steps=args.reserve_holding_max_turn_steps,
        reserve_holding_straight_steps=args.reserve_holding_straight_steps,
        dynamic_retarget=DynamicRetargetConfig(
            destruction_threshold=args.dynamic_retarget_threshold,
            max_stage=args.dynamic_retarget_max_stage,
            min_current_distance_km=args.dynamic_retarget_min_distance,
            cooldown_steps=args.dynamic_retarget_cooldown,
            max_retargets=args.dynamic_retarget_max_retarget,
            max_fraction_per_step=args.dynamic_retarget_max_fraction,
            distance_weight=args.dynamic_retarget_distance_weight,
            max_target_separation_km=args.dynamic_retarget_max_target_separation,
            max_turn_angle_degrees=args.dynamic_retarget_max_turn_angle,
            max_candidate_distance_km=args.dynamic_retarget_max_candidate_distance,
            max_per_old_target_per_step=args.dynamic_retarget_max_per_old_target,
        ),
        terminal_guidance=terminal_guidance,
        retarget_track_max_age_steps=args.dynamic_retarget_track_max_age,
        dynamic_retarget_enabled=(
            args.enable_dynamic_retarget and not args.disable_dynamic_retarget
        ),
        terminal_guidance_enabled=(
            args.enable_terminal_guidance and not args.disable_terminal_guidance
        ),
        audit_path=(
            None
            if args.disable_retarget_audit
            else args.output_dir / "retarget_audit.jsonl"
        ),
        formation_audit_interval_steps=args.formation_audit_interval,
        missile_trace_interval_steps=args.missile_trace_interval,
        missile_terminal_history=args.missile_terminal_history,
        seconds_per_step=args.seconds_per_step,
        velocity_window_steps=args.velocity_window_steps,
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
        with patched_runtime(config, official):
            official.run_simulation(args, bundle)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
