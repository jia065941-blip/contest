from types import SimpleNamespace

from core.main import (
    _attach_l_resource_outcomes,
    _rasterize_l_sensor_coverage,
    _team_detection_auc,
)
from core.evaluation.run_summary import RunSummary


def _entity(side, entity_type, *, health=1.0, detect_info=None):
    return {
        "side": side,
        "type": entity_type,
        "health": health,
        "detectInfo": detect_info or {},
    }


def test_run_summary_records_legal_9500_detection_source_and_first_step():
    initial = {
        "entities": {
            100: _entity(0, 21000),
            101: _entity(0, 21002),
            168: _entity(1, 9500),
        }
    }
    summary = RunSummary(scenario="E01", policies={})
    summary.start(initial)

    observed = {
        "entities": {
            100: _entity(
                0,
                21000,
                detect_info={
                    168: {
                        "entity_id": 168,
                        "entity_type": 9500,
                        "detect_from": 101,
                    }
                },
            ),
            101: _entity(0, 21002),
            168: _entity(1, 9500),
        }
    }
    summary.update(7, observed)
    summary.update(9, observed)

    result = summary.build(
        observed,
        termination_reason="time_limit",
        red_launched=2,
    )
    detection = result["red"]["detection"]
    assert detection["detected_9500_ids"] == [168]
    assert detection["first_9500_step_by_target"] == {"168": 7}
    assert detection["first_9500_source_by_target"] == {"168": 101}
    assert detection["9500_targets_by_source"] == {"101": [168]}
    assert detection["9500_first_step_by_source"] == {"101": {"168": 7}}


def test_run_summary_ignores_blue_truth_without_red_detect_info():
    observation = {
        "entities": {
            100: _entity(0, 21000),
            168: _entity(1, 9500),
        }
    }
    summary = RunSummary(scenario="E01", policies={})
    summary.start(observation)
    summary.update(3, observation)
    result = summary.build(
        observation,
        termination_reason="time_limit",
        red_launched=0,
    )
    assert result["red"]["detection"]["detected_9500_ids"] == []


def test_l_resource_outcomes_partition_intercept_hit_timeout_and_explorer():
    def simulator(entity_id, reason, direct_detections):
        return SimpleNamespace(
            entity_ext=SimpleNamespace(
                entity=SimpleNamespace(id=entity_id),
            ),
            launch=10,
            search_termination_reason=reason,
            direct_9500_first_detection_step=direct_detections,
        )

    simulators = [
        simulator(101, "external_damage", {}),
        simulator(102, "waypoint_completion", {169: 5}),
        simulator(103, "timeout", {168: 6}),
        simulator(104, "model_completion_away_from_waypoint", {}),
    ]
    events = (
        {
            "event_type": "direct_hit",
            "attacking_entity_type": 24000,
            "target_entity_type": 21002,
            "target_entity_id": 101,
            "actual_damage": 1.0,
        },
        {
            "event_type": "direct_hit",
            "attacking_entity_type": 21002,
            "attacking_entity_id": 102,
            "target_entity_type": 9500,
            "target_entity_id": 169,
            "actual_damage": 0.8,
        },
    )
    factory = SimpleNamespace(
        get_simulators_by_type=lambda entity_type: (
            simulators if entity_type == 21002 else []
        ),
        causal_event_ledger=events,
    )
    training_env = SimpleNamespace(
        engine=SimpleNamespace(simulator_factory=factory),
        agent_manager=SimpleNamespace(
            get_all_agents=lambda: [
                SimpleNamespace(entity_id=entity_id, agent_id=index)
                for index, entity_id in enumerate((101, 102, 103, 104), 1)
            ]
        ),
    )
    summary = {
        "red": {
            "detection": {
                "9500_targets_by_source": {
                    "102": [169],
                    "103": [168],
                }
            }
        },
        "objectives": [
            {"id": 168, "type": 9500, "final_health": 0.8},
            {"id": 169, "type": 9500, "final_health": 0.0},
        ],
    }

    _attach_l_resource_outcomes(summary, training_env)
    result = summary["red"]["l_resource_outcomes"]

    assert result["total"] == 4
    assert result["intercepted"] == 1
    assert result["nonintercepted"] == 3
    assert result["timeout"] == 1
    assert result["hit_9500_entities"] == 1
    assert result["hit_9500_targets"] == 1
    assert result["completion_without_9500_hit"] == 1
    assert result["nonintercepted_nonhitter"] == 2
    assert result["search_reserve_after_required_hits"] == 1
    assert result["l_9500_coverage"] == 1.0
    assert result["explorer_9500_coverage"] == 0.5
    assert result["direct_9500_first_step_by_entity"] == {
        "102": {"169": 5},
        "103": {"168": 6},
    }


def test_l_resource_outcomes_does_not_use_cluster_source_as_direct_credit():
    simulator = SimpleNamespace(
        entity_ext=SimpleNamespace(entity=SimpleNamespace(id=101)),
        launch=10,
        search_termination_reason="timeout",
        direct_9500_first_detection_step={},
    )
    factory = SimpleNamespace(
        get_simulators_by_type=lambda entity_type: (
            [simulator] if entity_type == 21002 else []
        ),
        causal_event_ledger=(),
    )
    training_env = SimpleNamespace(
        engine=SimpleNamespace(simulator_factory=factory),
        agent_manager=SimpleNamespace(
            get_all_agents=lambda: [
                SimpleNamespace(entity_id=101, agent_id=1)
            ]
        ),
    )
    summary = {
        "red": {
            "detection": {
                "9500_targets_by_source": {"101": [168]},
            }
        },
        "objectives": [
            {"id": 168, "type": 9500, "final_health": 1.0},
        ],
    }

    _attach_l_resource_outcomes(summary, training_env)
    result = summary["red"]["l_resource_outcomes"]

    assert result["l_9500_coverage"] == 0.0
    assert result["discovered_9500_ids"] == []
    assert result["cluster_source_discovered_9500_ids"] == [168]



def test_true_sensor_coverage_and_detection_auc_use_only_diagnostic_events():
    def simulator(entity_id, step):
        return SimpleNamespace(
            entity_ext=SimpleNamespace(entity=SimpleNamespace(id=entity_id)),
            sensor_footprint_samples=[{
                "step": step,
                "lon": 0.25,
                "lat": 0.25,
                "alt": 0.0,
            }],
        )

    coverage = _rasterize_l_sensor_coverage(
        [simulator(101, 10), simulator(102, 20)],
        (0.0, 0.0, 1.0, 1.0),
        grid_width=2,
        grid_height=2,
        sensor_radius_m=100.0,
    )
    assert coverage["covered_cell_count"] == 1
    assert coverage["coverage_fraction"] == 0.25
    assert coverage["duplicate_cell_count"] == 1
    assert coverage["duplicate_coverage_ratio"] == 0.5
    assert coverage["first_step_by_entity_and_cell"] == {
        "101": {"0": 10},
        "102": {"0": 20},
    }

    auc = _team_detection_auc(
        {"101": {"168": 20}, "102": {"168": 10}},
        {168, 169},
        100,
    )
    assert auc == 0.45
