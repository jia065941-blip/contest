from pathlib import Path

from simple_upper_orchestrator_skill.release_entry import (
    _compatible_read_profile, build_parser, migrate_legacy_comm_range, patched_factory,
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
