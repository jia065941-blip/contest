from pathlib import Path

import pytest

from local_avoid_v13_skill.plan_anchor_entry import (_max_steps, _resolve_plan,
                                                      _resolve_scenario,
                                                      build_parser)


PLAN_ROOT = Path(__file__).resolve().parents[4] / "plan-anchor-final20-v2"


@pytest.mark.parametrize("case", ["E01", "E02", "E03"])
def test_plan_anchor_a00_resolution(case):
    if not PLAN_ROOT.is_dir():
        pytest.skip("Plan Anchor checkout is unavailable")
    release = PLAN_ROOT / "github_release/plan_anchor_final20_v2"
    snapshot = release / "snapshot"
    scenario = _resolve_scenario(snapshot, f"final20/easy/{case}")
    plan = _resolve_plan(release, scenario, "A00_balanced")
    assert scenario.name == "scenario.json"
    assert plan == release / f"plans/{case}/A00_balanced.json"
    assert _max_steps(scenario) == 3000


def test_plan_anchor_cli_requires_explicit_roots_and_output():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--scenario", "final20/easy/E01", "--run-id", "missing"])
