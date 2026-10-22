import pytest

from l_search_coordination_skill import LSearchCoordinationSkill, SearchMemberSnapshot


def member(entity_id, cell, *, alive=True, report_step=10, discoveries=()):
    return SearchMemberSnapshot(
        entity_id, cell, alive, True, report_step, tuple(discoveries), 1
    )


def test_discovery_quorum_releases_response_without_cell_completion():
    skill = LSearchCoordinationSkill()
    result = skill.decide(
        step=31,
        members=(member(1, 0, discoveries=(100, 101, 102, 103)), member(2, 1)),
        required_discoveries=4,
    )
    assert result.status.response_ready is True
    assert result.satellite_request is None


def test_lost_member_requests_cell_reassignment():
    result = LSearchCoordinationSkill().decide(
        step=20,
        members=(member(1, 0, alive=False), member(2, 1, report_step=20)),
        required_discoveries=4,
    )
    assert result.status.uncovered_cells == (0,)
    assert result.reassignment_requests[0].uncovered_cell == 0


def test_deadline_requests_satellite_but_preserves_actions():
    result = LSearchCoordinationSkill().decide(
        step=80,
        members=(member(1, 0, report_step=80), member(2, 1, report_step=80)),
        required_discoveries=4,
    )
    assert result.actions == ((1, 1), (2, 1))
    assert result.satellite_request is not None
    assert result.satellite_request.missing_discoveries == 4


def test_duplicate_or_invalid_cells_are_rejected():
    with pytest.raises(ValueError):
        LSearchCoordinationSkill().decide(
            step=0,
            members=(member(1, 0), member(2, 0)),
            required_discoveries=4,
        )
