from pathlib import Path

import pytest

from local_avoid_v13_skill.official_entry import (_case_max_steps, _deployment_plan,
                                                   _scenario_path, build_parser)


OFFICIAL = Path(__file__).resolve().parents[4] / "competition_envs"


@pytest.mark.parametrize("case,steps", [("easy/E01", 1200), ("easy/E02", 1500),
                                         ("easy/E03", 1600)])
def test_official_scenario_resolution(case, steps):
    if not OFFICIAL.is_dir():
        pytest.skip("local official environment is unavailable")
    scenario = _scenario_path(OFFICIAL, case)
    assert scenario.is_file()
    assert scenario.is_relative_to(OFFICIAL)
    assert _case_max_steps(scenario) == steps


def test_cli_requires_explicit_run_id():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--competition-root", str(OFFICIAL),
                           "--scenario", "easy/E01"])


@pytest.mark.parametrize("case", ["E01", "E02", "E03"])
def test_final20_preset_resolves_next_to_scenario(case):
    if not OFFICIAL.is_dir():
        pytest.skip("local official environment is unavailable")
    scenario = _scenario_path(OFFICIAL, f"final20/easy/{case}")
    plan = _deployment_plan(scenario, "preset")
    assert plan == scenario.with_name("deployment_plan.example.json")
    assert plan.is_file()


def test_legacy_case_cannot_claim_final20_preset():
    if not OFFICIAL.is_dir():
        pytest.skip("local official environment is unavailable")
    scenario = _scenario_path(OFFICIAL, "easy/E01")
    with pytest.raises(ValueError, match="Deployment plan is unavailable"):
        _deployment_plan(scenario, "preset")
