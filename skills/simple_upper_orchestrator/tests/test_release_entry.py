from pathlib import Path

from simple_upper_orchestrator_skill.release_entry import (
    _compatible_read_profile, _config, build_parser, migrate_legacy_comm_range,
    patched_factory,
)


OFFICIAL = Path("/home/amax/ry/competition/competition-platform-env-release20261021")


def test_parser_contains_official_and_adapter_options():
    parser = build_parser(OFFICIAL)
    args = parser.parse_args([
        "--scenario", "launch20/easy/E01", "--seed", "7", "--max-steps", "12",
        "--render-mode", "none", "--blue-policy", "b0_fixed_ratio_random",
        "--blue-interceptor-ratio", "3", "--runtime-root", "/tmp/runtime",
        "--checkpoint", "/tmp/checkpoint", "--output-root", "/tmp/results",
        "--run-id", "case-7", "--l-package-count", "3", "--hm-reserve-h", "2",
    ])
    assert args.scenario == "launch20/easy/E01"
    assert args.blue_interceptor_ratio == 3
    assert args.run_id == "case-7"
    assert args.l_package_count == 3
    assert args.hm_reserve_h == 2
    assert args.reserve_h_count == 0
    assert args.l_first_wave_per_target == 0
    assert args.reserve_deadline_step == 300
    assert args.reserve_max_stage is None
    assert args.reserve_h_max_distance == 80.0
    assert args.reserve_l_max_distance == 100.0
    assert args.reserve_h_max_turn == 60.0
    assert args.reserve_l_max_turn == 75.0
    assert args.reserve_holding_half_width == 6.0
    assert args.reserve_holding_heading_offset == 35.0
    assert args.reserve_holding_max_turn_steps == 6
    assert args.reserve_holding_straight_steps == 2
    assert args.deploy_gap_km == 15.0
    assert args.deploy_l_first_wave_fraction == 0.70
    assert args.deploy_h_depth_km == 5.0
    assert args.deploy_m_depth_km == 10.0
    assert args.deploy_hm_follow_gap_km == 0.0
    assert args.deploy_mother_crossfire_angle == 35.0
    assert args.deploy_mother_crossfire_max_offset_km == 35.0
    assert args.deploy_sentry_m_lead_km == 7.5
    assert args.deploy_sentry_rendezvous_mode == "adaptive"
    assert args.deploy_h_speed == 200.0
    assert args.deploy_m_speed == 120.0
    assert args.deploy_m_arrival_lead_seconds == 5.0
    assert args.deploy_min_m_lead == 10.0
    assert args.deploy_max_m_lead is None
    assert args.l_allocation_mode == "distance_balanced"
    assert args.hm_allocation_mode == "full_coverage_mixed"
    assert args.hm_target_kill_probability == 0.85
    assert args.hm_min_h_per_mother == 2
    assert args.hm_min_m_per_mother == 0
    assert args.hm_h_per_sentry_package == 2
    assert args.hm_m_per_h_mixed_package == 2
    assert args.hm_m_only_sentry_package == 6
    assert args.hm_allow_m_only_sentry_packages is False
    assert args.hm_m_target_policy == "complete_h_sentries"
    assert args.hm_target_priority_axis == "east"
    assert args.dynamic_retarget_threshold == 0.70
    assert args.dynamic_retarget_max_stage == 3
    assert args.dynamic_retarget_min_distance == 15.0
    assert args.dynamic_retarget_cooldown == 25
    assert args.dynamic_retarget_max_retarget == 1
    assert args.dynamic_retarget_max_fraction == 0.20
    assert args.dynamic_retarget_distance_weight == 0.05
    assert args.dynamic_retarget_track_max_age == 3
    assert args.dynamic_retarget_max_target_separation == 10.0
    assert args.dynamic_retarget_max_turn_angle == 45.0
    assert args.dynamic_retarget_max_candidate_distance == 30.0
    assert args.dynamic_retarget_max_per_old_target == 1
    assert args.disable_dynamic_retarget is False
    assert args.disable_terminal_guidance is False
    assert args.terminal_min_stage == 4
    assert args.terminal_enter_distance is None
    assert args.terminal_capture_distance == 2000.0
    assert args.terminal_final_straight_distance == 300.0
    assert args.terminal_straight_distance is None
    assert args.terminal_exit_distance == 3500.0
    assert args.terminal_max_track_age == 5
    assert args.terminal_stale_grace_age == 8
    assert args.terminal_threat_distance == 3000.0
    assert args.terminal_heading_enter == 6.0
    assert args.terminal_heading_exit == 10.0
    assert args.terminal_heading_deadband is None
    assert args.terminal_predicted_miss == 80.0
    assert args.terminal_turn_pulse_steps == 1
    assert args.terminal_straight_recovery_steps == 2
    assert args.terminal_no_improvement_limit == 3
    assert args.terminal_respect_interceptors_after_latch is False
    assert args.disable_retarget_audit is False
    assert args.formation_audit_interval == 25
    assert args.missile_trace_interval == 0
    assert args.missile_terminal_history == 20
    assert args.seconds_per_step == 1.0
    assert args.velocity_window_steps == 5


def test_reserve_options_are_wired_into_config():
    parser = build_parser(OFFICIAL)
    args = parser.parse_args([
        "--scenario", "launch20/easy/E01", "--max-steps", "12",
        "--reserve-h-count", "2",
        "--l-first-wave-per-target", "6",
        "--reserve-deadline-step", "240",
        "--reserve-max-stage", "1",
        "--reserve-h-max-distance", "70",
        "--reserve-l-max-distance", "90",
        "--reserve-h-max-turn", "50",
        "--reserve-l-max-turn", "65",
        "--reserve-holding-half-width", "5",
        "--reserve-holding-heading-offset", "30",
        "--reserve-holding-max-turn-steps", "4",
        "--reserve-holding-straight-steps", "3",
    ])

    configured = _config(args)

    assert configured.hm_strike.reserve_h == 2
    assert configured.reserve_h_count == 2
    assert configured.l_first_wave_per_target == 6
    assert configured.reserve_deadline_step == 240
    assert configured.reserve_max_stage == 1
    assert configured.reserve_h_max_distance_km == 70
    assert configured.reserve_l_max_distance_km == 90
    assert configured.reserve_h_max_turn_angle_degrees == 50
    assert configured.reserve_l_max_turn_angle_degrees == 65
    assert configured.reserve_holding_half_width_km == 5
    assert configured.reserve_holding_heading_offset_degrees == 30
    assert configured.reserve_holding_max_turn_steps == 4
    assert configured.reserve_holding_straight_steps == 3


def test_dynamic_retarget_can_be_disabled_from_release_cli():
    parser = build_parser(OFFICIAL)
    args = parser.parse_args([
        "--scenario", "launch20/easy/E01", "--max-steps", "12",
        "--disable-dynamic-retarget",
    ])
    assert args.disable_dynamic_retarget is True
    assert args.enable_dynamic_retarget is False


def test_terminal_guidance_options_are_wired_and_can_be_disabled():
    parser = build_parser(OFFICIAL)
    args = parser.parse_args([
        "--scenario", "launch20/easy/E01", "--max-steps", "12",
        "--disable-terminal-guidance",
        "--terminal-min-stage", "3",
        "--terminal-enter-distance", "1200",
        "--terminal-exit-distance", "1800",
        "--terminal-max-track-age", "7",
        "--terminal-threat-distance", "2500",
    ])

    configured = _config(args)

    assert configured.terminal_guidance_enabled is False
    assert configured.terminal_guidance.min_stage == 3
    assert configured.terminal_guidance.enter_distance_m == 1200.0
    assert configured.terminal_guidance.exit_distance_m == 1800.0
    assert configured.terminal_guidance.max_track_age_steps == 7
    assert configured.terminal_guidance.threat_distance_m == 2500.0


def test_terminal_guidance_030_and_velocity_window_options_are_wired():
    parser = build_parser(OFFICIAL)
    args = parser.parse_args([
        "--scenario", "launch20/easy/E01", "--max-steps", "12",
        "--terminal-capture-distance", "1800",
        "--terminal-final-straight-distance", "250",
        "--terminal-stale-grace-age", "9",
        "--terminal-heading-enter", "5",
        "--terminal-heading-exit", "11",
        "--terminal-predicted-miss", "70",
        "--terminal-turn-pulse-steps", "2",
        "--terminal-straight-recovery-steps", "3",
        "--terminal-no-improvement-limit", "4",
        "--velocity-window-steps", "7",
    ])

    configured = _config(args)
    terminal = configured.terminal_guidance
    assert terminal.capture_distance == 1800.0
    assert terminal.final_straight_distance == 250.0
    assert terminal.stale_grace_age == 9
    assert terminal.heading_enter_straight_deg == 5.0
    assert terminal.heading_exit_straight_deg == 11.0
    assert terminal.predicted_miss_threshold_m == 70.0
    assert terminal.turn_pulse_steps == 2
    assert terminal.straight_recovery_steps == 3
    assert terminal.no_improvement_limit == 4
    assert configured.velocity_window_steps == 7


def test_legacy_terminal_straight_and_deadband_options_remain_compatible():
    parser = build_parser(OFFICIAL)
    args = parser.parse_args([
        "--scenario", "launch20/easy/E01", "--max-steps", "12",
        "--terminal-straight-distance", "240",
        "--terminal-heading-deadband", "7",
    ])

    terminal = _config(args).terminal_guidance
    assert terminal.final_straight_distance == 240.0
    assert terminal.straight_distance_m == 240.0
    assert terminal.heading_enter_straight_deg == 7.0
    assert terminal.heading_exit_straight_deg == 7.0


def test_dynamic_retarget_030_options_are_wired_into_config():
    parser = build_parser(OFFICIAL)
    args = parser.parse_args([
        "--scenario", "launch20/easy/E01", "--max-steps", "12",
        "--dynamic-retarget-track-max-age", "7.5",
        "--dynamic-retarget-max-target-separation", "16",
        "--dynamic-retarget-max-turn-angle", "55",
        "--dynamic-retarget-max-candidate-distance", "45",
        "--dynamic-retarget-max-per-old-target", "2",
    ])

    configured = _config(args)

    assert configured.retarget_track_max_age_steps == 7.5
    assert configured.dynamic_retarget.max_stage == 3
    assert configured.dynamic_retarget.max_target_separation_km == 16
    assert configured.dynamic_retarget.max_turn_angle_degrees == 55
    assert configured.dynamic_retarget.max_candidate_distance_km == 45
    assert configured.dynamic_retarget.max_per_old_target_per_step == 2


def test_staggered_deployment_070_adaptive_options_are_wired_into_config():
    parser = build_parser(OFFICIAL)
    args = parser.parse_args([
        "--scenario", "launch20/easy/E01", "--max-steps", "12",
        "--deploy-sentry-rendezvous-mode", "fixed",
        "--deploy-h-speed", "210",
        "--deploy-m-speed", "130",
        "--deploy-m-arrival-lead-seconds", "7",
        "--deploy-min-m-lead", "11",
        "--deploy-max-m-lead", "19",
    ])

    deployment = _config(args).deployment

    assert deployment.sentry_rendezvous_mode == "fixed"
    assert deployment.h_speed_mps == 210
    assert deployment.m_speed_mps == 130
    assert deployment.desired_m_arrival_lead_seconds == 7
    assert deployment.min_sentry_m_lead_km == 11
    assert deployment.max_sentry_m_lead_km == 19


def test_retarget_audit_defaults_to_prepared_run_directory_and_can_be_disabled(tmp_path):
    parser = build_parser(OFFICIAL)
    args = parser.parse_args([
        "--scenario", "launch20/easy/E01", "--max-steps", "12",
        "--output-dir", str(tmp_path / "run-7"),
    ])

    configured = _config(args)
    assert configured.audit_path == tmp_path / "run-7" / "retarget_audit.jsonl"

    disabled_args = parser.parse_args([
        "--scenario", "launch20/easy/E01", "--max-steps", "12",
        "--output-dir", str(tmp_path / "run-8"), "--disable-retarget-audit",
    ])
    assert _config(disabled_args).audit_path is None
    assert not (tmp_path / "run-8" / "retarget_audit.jsonl").exists()


def test_formation_audit_interval_is_wired_into_config():
    parser = build_parser(OFFICIAL)
    args = parser.parse_args([
        "--scenario", "launch20/easy/E01", "--max-steps", "12",
        "--formation-audit-interval", "7",
    ])

    assert _config(args).formation_audit_interval_steps == 7


def test_missile_trace_options_are_wired_into_config():
    parser = build_parser(OFFICIAL)
    args = parser.parse_args([
        "--scenario", "launch20/easy/E01", "--max-steps", "12",
        "--missile-trace-interval", "5",
        "--missile-terminal-history", "7",
        "--seconds-per-step", "0.5",
    ])

    configured = _config(args)
    assert configured.missile_trace_interval_steps == 5
    assert configured.missile_terminal_history == 7
    assert configured.seconds_per_step == 0.5


def test_legacy_empty_comm_range_profile_migration():
    profile = {"entityList": [{"entity": {"commRangeInfo": []}}], "other": []}
    assert migrate_legacy_comm_range(profile)
    assert profile["entityList"][0]["entity"]["commRangeInfo"] == {}
    assert profile["other"] == []
    assert not migrate_legacy_comm_range(profile)


def test_factory_patch_changes_only_export_and_restores_it():
    import user_agents.red

    original = user_agents.red.RedAgentFactory
    marker = object()
    with patched_factory(marker):
        replacement = user_agents.red.RedAgentFactory
        assert replacement is not original
        assert replacement.orchestrator_config is marker
    assert user_agents.red.RedAgentFactory is original


def test_compatible_profile_reader_migrates_builtin_bundle_file(tmp_path, monkeypatch):
    class Profile:
        @staticmethod
        def from_dict(value):
            return value

    monkeypatch.setitem(__import__("sys").modules, "envengine", type("Env", (), {"Profile": Profile}))
    source = tmp_path / "scenario.json"
    source.write_text('{"entity":{"commRangeInfo":[]}}', encoding="utf-8")
    reader = _compatible_read_profile(lambda path: (_ for _ in ()).throw(AssertionError(path)))

    result = reader(str(source))

    assert result["entity"]["commRangeInfo"] == {}
