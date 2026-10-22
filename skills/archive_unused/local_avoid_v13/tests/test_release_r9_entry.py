from pathlib import Path

import pytest

from local_avoid_v13_skill.release_r9_entry import (
    _deployment_plan,
    _read_release_profile,
    _scenario_path,
    build_parser,
)


def test_release_parser_requires_external_runtime_and_checkpoint(tmp_path: Path) -> None:
    args = build_parser().parse_args([
        "--competition-root", str(tmp_path),
        "--jmy-contest-root", str(tmp_path / "contest-jmy"),
        "--checkpoint", str(tmp_path / "network.pt"),
        "--scenario", "final20/easy/E01",
        "--run-id", "e01_s1",
        "--output-root", str(tmp_path / "results"),
    ])

    assert args.deployment_plan == "preset"
    assert args.blue_policy == "b0_fixed_ratio_random"
    assert args.blue_interceptor_ratio == 2


def test_release_paths_resolve_bundled_final20_inputs(tmp_path: Path) -> None:
    case = tmp_path / "scenarios/cases/final20/easy/E01"
    case.mkdir(parents=True)
    scenario = case / "scenario.json"
    scenario.write_text("{}", encoding="utf-8")
    plan = case / "deployment_plan.example.json"
    plan.write_text("{}", encoding="utf-8")

    assert _scenario_path(tmp_path, "final20/easy/E01") == scenario.resolve()
    assert _deployment_plan(scenario, "preset") == plan.resolve()


def test_release_paths_fail_closed_when_plan_is_missing(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario.json"
    scenario.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Deployment plan is unavailable"):
        _deployment_plan(scenario, "preset")


def test_release_profile_migrates_legacy_communication_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Profile:
        @staticmethod
        def from_dict(data):
            return data

    monkeypatch.setitem(__import__("sys").modules, "envengine", type("Env", (), {"Profile": Profile}))
    scenario = tmp_path / "scenario.json"
    scenario.write_text(
        '{"imagineProfile":{"entityList":[{"entity":{"commRangeInfo":[]}}]}}',
        encoding="utf-8",
    )

    loaded = _read_release_profile(str(scenario))

    assert loaded["imagineProfile"]["entityList"][0]["entity"]["commRangeInfo"] == {}
