import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from target_damage_belief_skill import TargetPriorEvidence

from h_single_maneuver_skill.plan_entry import (
    BEST_B0_PLANS,
    _flatten_wave_releases,
    _install_exact_motion_clock,
    _shift_wave_releases,
    _max_steps,
    _retarget_auto_launched,
    _resolve_plan,
    _resolve_scenario,
    build_parser,
)
from h_single_maneuver_skill.runtime_adapter import (
    HSingleManeuverRuntimePolicy,
    JointGroup,
    LResponseGroup,
    MissileTypeDispatchRuntimePolicy,
    PlanMotionConfig,
    default_profile_path,
)
from h_single_maneuver_skill.r9_entry import build_parser as build_r9_parser


ROOT = Path(__file__).resolve().parents[4]
ENVIRONMENT_ROOT = ROOT / "competition_envs"
PLAN_ROOT = ROOT / "plan-anchor-final20-v2"


def own_observation(step=0, entity_type=21000, threats=None):
    return {
        "step": step,
        "self": {
            "type": entity_type,
            "position": {"lon": 120.0, "lat": 30.0, "alt": 1000.0},
            "pos_ecf": {"x": 1_000_000.0 + 2000.0 * step, "y": 0.0, "z": 0.0},
            "detectInfo": threats or {},
            "commRangeInfo": {},
        },
    }


def interceptor(threat_id=900, y=1000.0):
    return {
        "entity_id": threat_id,
        "entity_type": 24000,
        "pos_ecf": {"x": 1_012_000.0, "y": y, "z": 0.0},
        "vel_ecf": {"x": -2000.0, "y": 0.0, "z": 0.0},
    }


class FakeLocalPolicy:
    def __init__(self, *, max_steps):
        self.max_steps = max_steps
        self.loaded = None
        self.reset_count = 0
        self.closed = False
        self.training = False

    def load(self, path):
        self.loaded = path

    def encode_motion_observation(self, observation, **context):
        return (observation["self"]["type"], context["agent_id"])

    def prepare_action_mask(self, observation, action_mask):
        return np.asarray(action_mask, dtype=np.bool_)

    def select_action(self, observation, action_mask):
        return 2

    def set_training(self, training):
        if training:
            raise ValueError("evaluation only")
        self.training = False

    def reset_episode(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


def dispatch_policy(mode="hybrid", pulse_steps=10, mission_replan_mode="off"):
    policy = MissileTypeDispatchRuntimePolicy(
        mode=mode,
        max_steps=3000,
        h_profile=default_profile_path(),
        local_factory=FakeLocalPolicy,
        local_pulse_steps=pulse_steps,
        mission_replan_mode=mission_replan_mode,
    )
    policy.load("/tmp/fake-local.pt")
    return policy


def motion_context(agent_id=1, entity_id=10):
    return {
        "agent_id": agent_id,
        "entity_id": entity_id,
        "launch_step": 0,
        "maneuver_state": 0,
        "goal_position": {"lon": 121.0, "lat": 30.0},
    }


def test_profile_load_and_non_h_stop(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(default_profile_path().read_text(encoding="utf-8"), encoding="utf-8")
    policy = HSingleManeuverRuntimePolicy(PlanMotionConfig(max_steps=3000))
    policy.load(profile)
    encoded = policy.encode_motion_observation(
        own_observation(entity_type=21001),
        agent_id=1,
        entity_id=10,
        launch_step=0,
        maneuver_state=0,
        goal_position={"lon": 121.0, "lat": 30.0},
    )
    assert encoded.action == 1
    assert encoded.decision is None


def test_h_policy_outputs_one_hot_mask_for_child_skill_action():
    policy = HSingleManeuverRuntimePolicy()
    policy.load(default_profile_path())
    encoded = policy.encode_motion_observation(
        own_observation(threats={900: interceptor()}),
        agent_id=1,
        entity_id=10,
        launch_step=0,
        maneuver_state=0,
        goal_position={"lon": 121.0, "lat": 30.0},
    )
    mask = policy.prepare_action_mask(encoded, np.ones(3, dtype=np.bool_))
    assert mask.dtype == np.bool_
    assert int(mask.sum()) == 1
    assert bool(mask[encoded.action])
    assert policy.select_action(encoded, mask) == encoded.action


def test_runtime_trace_contains_rule_decision(tmp_path, monkeypatch):
    trace = tmp_path / "decisions.jsonl"
    monkeypatch.setenv("H_SINGLE_MANEUVER_TRACE", str(trace))
    policy = HSingleManeuverRuntimePolicy()
    policy.load(default_profile_path())
    policy.encode_motion_observation(
        own_observation(),
        agent_id=1,
        entity_id=10,
        launch_step=0,
        maneuver_state=0,
        goal_position={"lon": 121.0, "lat": 30.0},
    )
    policy.close()
    record = json.loads(trace.read_text(encoding="utf-8"))
    assert record["entity_id"] == 10
    assert record["action"] == 1
    assert record["phase"] == "CRUISE"


def test_hybrid_routes_h_to_rule_and_m_l_to_local():
    policy = dispatch_policy("hybrid")
    h = policy.encode_motion_observation(own_observation(entity_type=21000), **motion_context())
    m = policy.encode_motion_observation(own_observation(entity_type=21001), **motion_context(2, 11))
    l = policy.encode_motion_observation(own_observation(entity_type=21002), **motion_context(3, 12))
    assert h.delegate is policy.h_policy
    assert m.delegate is policy.local_policy
    assert l.delegate is policy.local_policy


def test_local_all_routes_every_missile_type_to_local():
    policy = dispatch_policy("local-all")
    for index, entity_type in enumerate((21000, 21001, 21002), start=1):
        encoded = policy.encode_motion_observation(
            own_observation(entity_type=entity_type),
            **motion_context(index, 20 + index),
        )
        assert encoded.delegate is policy.local_policy


def test_joint_mode_builds_deterministic_h_triple_and_caches_actions():
    policy = dispatch_policy("joint-h-v13")
    goal = SimpleNamespace(
        kind=SimpleNamespace(value="public_target"),
        target_id=52,
        target_type=9400,
    )
    bindings = tuple(
        SimpleNamespace(
            entity_id=entity_id,
            role=SimpleNamespace(value="H"),
            goal=goal,
        )
        for entity_id in (10, 11, 12, 13)
    )
    agents = {}
    observations = {}
    for agent_id, entity_id in enumerate((10, 11, 12, 13), start=1):
        agent = SimpleNamespace(
            agent_id=agent_id,
            entity_id=entity_id,
            launch_step=0,
            maneuver_state=0,
            current_goal_id=52,
            _last_stage=3,
            _current_goal_position=lambda: (121.0, 30.0),
        )
        agents[agent_id] = agent
        observation = own_observation(step=5, entity_type=21000)
        observation["self"]["pos_ecf"]["y"] = (
            float((entity_id - 10) * 100) if entity_id < 13 else 5000.0
        )
        observations[agent_id] = observation

    policy.prepare_decision_step(
        step=5,
        decision_order=(1, 2, 3, 4),
        agent_observations=observations,
        agents_by_id=agents,
        bindings=bindings,
        entity_types={entity_id: 21000 for entity_id in (10, 11, 12, 13)},
    )

    assert [group.entity_ids for group in policy._joint_groups] == [(10, 11, 12, 13)]
    assert set(policy._prepared_actions) == {10, 11, 12, 13}
    prepared = policy.encode_motion_observation(
        observations[1], **motion_context(1, 10)
    )
    assert prepared.forced_action in (0, 1, 2)


def test_joint_mode_leaves_single_h_on_direct_v13_path():
    policy = dispatch_policy("joint-h-v13")
    goal = SimpleNamespace(
        kind=SimpleNamespace(value="public_target"),
        target_id=52,
        target_type=9400,
    )
    binding = SimpleNamespace(
        entity_id=10,
        role=SimpleNamespace(value="H"),
        goal=goal,
        planned_release_step=0,
    )
    policy.configure_plan_bindings((binding,), {10: 21000})
    assert policy._joint_groups == ()
    encoded = policy.encode_motion_observation(
        own_observation(step=1, entity_type=21000), **motion_context(1, 10)
    )
    assert encoded.delegate is policy.local_policy


def test_release_order_never_changes_entity_action_order():
    policy = dispatch_policy("joint-h-v13")
    goal = SimpleNamespace(
        kind=SimpleNamespace(value="public_target"),
        target_id=52,
        target_type=9400,
    )
    bindings = (
        SimpleNamespace(
            entity_id=12,
            role=SimpleNamespace(value="H"),
            goal=goal,
            planned_release_step=0,
        ),
        SimpleNamespace(
            entity_id=10,
            role=SimpleNamespace(value="H"),
            goal=goal,
            planned_release_step=5,
        ),
    )
    policy.configure_plan_bindings(bindings, {10: 21000, 12: 21000})
    assert policy._joint_groups[0].entity_ids == (10, 12)


@pytest.mark.parametrize(
    ("allocations", "expected_groups"),
    [
        (
            {
                52: [(2576, 0), (2577, 0), (2582, 120), (2588, 240)],
                53: [(2578, 0), (2583, 120), (2584, 120), (2589, 240)],
                2552: [(2579, 0), (2585, 120), (2586, 120), (2590, 240)],
            },
            [(2576, 2577), (2583, 2584), (2585, 2586)],
        ),
        (
            {
                51: [(2576, 16), (2577, 16), (2582, 136), (2588, 256)],
                53: [(2578, 16), (2583, 136), (2584, 136), (2589, 256)],
                2551: [(2579, 16), (2580, 16), (2585, 136), (2590, 256)],
            },
            [(2576, 2577), (2579, 2580), (2583, 2584)],
        ),
        (
            {
                51: [(2576, 0), (2577, 0), (2582, 120), (2588, 240)],
                52: [(2578, 0), (2583, 120), (2584, 120), (2589, 240)],
                2551: [(2579, 0), (2580, 0), (2585, 120), (2590, 240)],
            },
            [(2576, 2577), (2579, 2580), (2583, 2584)],
        ),
    ],
)
def test_e01_e03_release_window_packages(allocations, expected_groups):
    policy = dispatch_policy("joint-h-v13")
    bindings = []
    entity_types = {}
    for target_id, members in allocations.items():
        goal = SimpleNamespace(
            kind=SimpleNamespace(value="public_target"),
            target_id=target_id,
            target_type=9400,
        )
        for entity_id, release_step in members:
            bindings.append(
                SimpleNamespace(
                    entity_id=entity_id,
                    role=SimpleNamespace(value="H"),
                    goal=goal,
                    planned_release_step=release_step,
                )
            )
            entity_types[entity_id] = 21000
    policy.configure_plan_bindings(tuple(bindings), entity_types)
    assert sorted(group.entity_ids for group in policy._joint_groups) == expected_groups


def test_easy_l_bindings_create_eight_search_teams_and_response_packages():
    policy = dispatch_policy("joint-l-local")
    bindings = []
    entity_types = {}
    for cell in range(32):
        entity_id = 2660 + cell
        bindings.append(
            SimpleNamespace(
                entity_id=entity_id,
                role=SimpleNamespace(value="L"),
                goal=SimpleNamespace(kind=SimpleNamespace(value="hidden_search")),
                reserve_response=False,
                search_cell=cell,
                response_group_index=None,
            )
        )
        entity_types[entity_id] = 21002
    for offset in range(14):
        entity_id = 2700 + offset
        bindings.append(
            SimpleNamespace(
                entity_id=entity_id,
                role=SimpleNamespace(value="L"),
                goal=None,
                reserve_response=True,
                search_cell=None,
                response_group_index=0,
            )
        )
        entity_types[entity_id] = 21002
    policy.configure_plan_bindings(tuple(bindings), entity_types)
    assert len(policy._l_search_groups) == 8
    assert {len(group.entity_ids) for group in policy._l_search_groups} == {4}
    assert [len(group.entity_ids) for group in policy._l_response_groups] == [4, 3, 3, 2, 2]


def test_hard_l_response_groups_split_eight_as_four_plus_four():
    policy = dispatch_policy("joint-l-local")
    bindings = []
    entity_types = {}
    for cell in range(32):
        for rank in range(3 if cell < 8 else 2):
            entity_id = 3000 + cell * 3 + rank
            bindings.append(
                SimpleNamespace(
                    entity_id=entity_id,
                    role=SimpleNamespace(value="L"),
                    goal=SimpleNamespace(kind=SimpleNamespace(value="hidden_search")),
                    reserve_response=False,
                    search_cell=cell,
                    response_group_index=None,
                )
            )
            entity_types[entity_id] = 21002
    for offset in range(8):
        entity_id = 4000 + offset
        bindings.append(
            SimpleNamespace(
                entity_id=entity_id,
                role=SimpleNamespace(value="L"),
                goal=None,
                reserve_response=True,
                search_cell=None,
                response_group_index=0,
            )
        )
        entity_types[entity_id] = 21002
    policy.configure_plan_bindings(tuple(bindings), entity_types)
    assert len(policy._l_search_groups) == 18
    assert [len(group.entity_ids) for group in policy._l_response_groups] == [4, 4]


def test_m_bindings_form_release_window_pairs_and_mass_packages():
    goal = SimpleNamespace(
        kind=SimpleNamespace(value="public_target"),
        target_id=54,
        target_type=9600,
    )
    easy = dispatch_policy("joint-m-local")
    easy_bindings = tuple(
        SimpleNamespace(
            entity_id=entity_id,
            role=SimpleNamespace(value="M"),
            goal=goal,
            planned_release_step=release_step,
        )
        for entity_id, release_step in (
            (2611, 10), (2612, 10), (2619, 130), (2629, 250)
        )
    )
    easy.configure_plan_bindings(
        easy_bindings, {item.entity_id: 21001 for item in easy_bindings}
    )
    assert [group.entity_ids for group in easy._m_groups] == [(2611, 2612)]

    massed = dispatch_policy("joint-m-local")
    massed_bindings = tuple(
        SimpleNamespace(
            entity_id=entity_id,
            role=SimpleNamespace(value="M"),
            goal=goal,
            planned_release_step=20 + index * 2,
        )
        for index, entity_id in enumerate((3001, 3002, 3003, 3004))
    )
    massed.configure_plan_bindings(
        massed_bindings,
        {item.entity_id: 21001 for item in massed_bindings},
    )
    assert [group.entity_ids for group in massed._m_groups] == [
        (3001, 3002, 3003, 3004)
    ]


def test_all_packages_mission_filter_can_stop_unlaunched_m_member():
    policy = dispatch_policy(
        "joint-m-local", mission_replan_mode="all-packages"
    )
    policy._mission_m_stopped_entities.add(20)
    agent = SimpleNamespace(entity_id=20)
    binding = SimpleNamespace(reserve_response=False)
    plan_policy = SimpleNamespace(
        _agent_by_id={1: agent},
        _binding_by_entity={20: binding},
    )
    decision = {"launch": 0, "goal": 4, "motion": 1, "resource": None}
    filtered = policy.filter_plan_decision(
        plan_policy=plan_policy, agent_id=1, decision=decision
    )
    assert filtered["launch"] is None
    assert filtered["goal"] is None


def test_shared_damage_belief_tracks_accepted_launch_retarget_and_terminal():
    policy = dispatch_policy("joint-m-local")
    targets = (
        SimpleNamespace(entity_id=52, entity_type=9400),
        SimpleNamespace(entity_id=53, entity_type=9400),
    )
    info_center = SimpleNamespace(
        fusion=SimpleNamespace(targets=targets),
        initial_health={52: 24.0, 53: 24.0},
        dead_target_ids=set(),
    )
    agent = SimpleNamespace(
        entity_id=20,
        missile_type=21001,
        current_goal_id=52,
        death_reason=None,
        _last_stage=3,
        _min_goal_dist=10.0,
        sat_used=False,
        policy=SimpleNamespace(_current_step=10),
    )
    plan_policy = SimpleNamespace(
        info_center=info_center,
        _agents=(agent,),
        _current_step=10,
    )
    launch = SimpleNamespace(
        commands=np.asarray([[1.0, 20.0, 120.0, 20.0]])
    )
    policy.observe_command_batch((launch,), plan_policy)
    assert policy._damage_belief.estimate(52).committed_missiles == (20,)

    agent.current_goal_id = 53
    retarget = SimpleNamespace(
        commands=np.asarray([[2.0, 20.0, 121.0, 20.0]])
    )
    policy.observe_command_batch((retarget,), plan_policy)
    assert policy._damage_belief.estimate(52).committed_missiles == ()
    assert policy._damage_belief.estimate(53).committed_missiles == (20,)

    agent.death_reason = "intercept"
    policy.observe_terminal(agent)
    assert policy._damage_belief.estimate(53).committed_missiles == ()


def test_completed_target_retargets_live_package_to_nearest_legal_target():
    policy = dispatch_policy(
        "joint-h-v13", mission_replan_mode="all-packages"
    )
    policy._joint_groups = (
        JointGroup("h-package", 21000, 9400, 52, (10, 11)),
    )
    targets = (
        SimpleNamespace(
            entity_id=52,
            entity_type=9400,
            position=SimpleNamespace(lon=120.0, lat=20.0),
        ),
        SimpleNamespace(
            entity_id=53,
            entity_type=9400,
            position=SimpleNamespace(lon=120.2, lat=20.0),
        ),
    )
    info_center = SimpleNamespace(
        fusion=SimpleNamespace(targets=targets),
        initial_health={52: 24.0, 53: 24.0},
        dead_target_ids=set(),
        targets=lambda: targets,
        target_by_id=lambda target_id: next(
            (item for item in targets if item.entity_id == target_id), None
        ),
        coordinate_scale=1.0,
    )
    for target in targets:
        policy._damage_belief.register_target(
            TargetPriorEvidence(target.entity_id, target.entity_type, 24.0)
        )
        policy._belief_registered_targets.add(target.entity_id)
    for entity_id in (10, 11):
        policy._damage_belief.commit_missile(
            missile_id=entity_id,
            missile_type=21000,
            target_id=52,
        )
    policy._damage_belief.confirm_destroyed(52)
    agents = {}
    for agent_id, entity_id in enumerate((10, 11), start=1):
        agents[agent_id] = SimpleNamespace(
            agent_id=agent_id,
            entity_id=entity_id,
            missile_type=21000,
            launch_step=0,
            death_reason=None,
            _last_stage=3,
            current_goal_id=52,
            latest_observation={
                "self": {
                    "position": {"lon": 120.0, "lat": 20.0, "alt": 0.0}
                }
            },
            _sorted_targets=(targets[1],),
        )
    plan_policy = SimpleNamespace(
        info_center=info_center,
        _agents=tuple(agents.values()),
        _agent_by_id=agents,
        _binding_by_entity={
            10: SimpleNamespace(reserve_response=False),
            11: SimpleNamespace(reserve_response=False),
        },
    )
    policy._prepare_package_retargets(
        step=100,
        agents_by_id=agents,
        plan_policy=plan_policy,
    )
    assert policy._mission_retarget_by_entity == {10: 53, 11: 53}
    decision = {"launch": None, "goal": None, "motion": 1, "resource": None}
    filtered = policy.filter_plan_decision(
        plan_policy=plan_policy,
        agent_id=1,
        decision=decision,
    )
    assert filtered["goal"] == 0
    agents[1].current_goal_id = 53
    record = SimpleNamespace(
        commands=np.asarray([[2.0, 10.0, 120.2, 20.0]])
    )
    plan_policy._current_step = 100
    policy.observe_command_batch((record,), plan_policy)
    assert policy._damage_belief.estimate(52).committed_missiles == (11,)
    assert policy._damage_belief.estimate(53).committed_missiles == (10,)


def test_mission_replan_releases_first_l_package_and_holds_next():
    policy = dispatch_policy(
        "joint-l-local", mission_replan_mode="l-response"
    )
    policy._l_response_groups = (
        LResponseGroup("p0", 0, (10, 11, 12)),
        LResponseGroup("p1", 0, (13, 14)),
    )
    info_center = SimpleNamespace(
        discovery_events=[{"step": 31, "target_id": 168}],
        initial_health={168: 1.2},
        expected_damage={168: 0.0},
        inflight_commit={},
        dead_target_ids=set(),
        fusion=SimpleNamespace(
            targets=(SimpleNamespace(entity_id=168, entity_type=9500),)
        ),
    )
    plan_policy = SimpleNamespace(
        info_center=info_center,
        spec=SimpleNamespace(
            hidden_response=SimpleNamespace(
                required_discoveries=1,
                release_delay_steps=20,
            )
        ),
    )
    policy._prepare_mission_replan(51, plan_policy)
    assert policy._mission_released_entities == {10, 11, 12}
    assert policy._mission_cancelled_entities == set()

    released_agent = SimpleNamespace(entity_id=10)
    held_agent = SimpleNamespace(entity_id=13)
    response_binding = SimpleNamespace(reserve_response=True)
    plan_policy._agent_by_id = {1: released_agent, 2: held_agent}
    plan_policy._binding_by_entity = {10: response_binding, 13: response_binding}
    decision = {"launch": 0, "goal": 4, "motion": 1, "resource": None}
    assert policy.filter_plan_decision(
        plan_policy=plan_policy, agent_id=1, decision=decision
    )["launch"] == 0
    held = policy.filter_plan_decision(
        plan_policy=plan_policy, agent_id=2, decision=decision
    )
    assert held["launch"] is None
    assert held["goal"] is None


def test_unknown_type_fails_closed_to_stop():
    policy = dispatch_policy()
    encoded = policy.encode_motion_observation(
        own_observation(entity_type=99999), **motion_context()
    )
    mask = policy.prepare_action_mask(encoded, np.ones(3, dtype=np.bool_))
    assert mask.tolist() == [False, True, False]
    assert policy.select_action(encoded, mask) == 1


def test_local_delegate_retains_original_ten_step_pulse():
    policy = dispatch_policy(pulse_steps=10)
    context = motion_context(agent_id=7, entity_id=17)
    actions = []
    for step in range(12):
        encoded = policy.encode_motion_observation(
            own_observation(step=step, entity_type=21001), **context
        )
        mask = policy.prepare_action_mask(encoded, np.ones(3, dtype=np.bool_))
        actions.append(policy.select_action(encoded, mask))
    assert actions == [2] * 10 + [1, 2]


def test_dispatch_load_reset_close_and_training_contract():
    policy = dispatch_policy()
    assert policy.local_policy.loaded == "/tmp/fake-local.pt"
    initial_resets = policy.local_policy.reset_count
    policy._local_pulses[1] = (2, 5)
    policy.reset_episode()
    assert policy._local_pulses == {}
    assert policy.local_policy.reset_count == initial_resets + 1
    with pytest.raises(ValueError, match="evaluation-only"):
        policy.set_training(True)
    policy.close()
    policy.close()
    assert policy.local_policy.closed is True


def test_auto_plan_resolution_for_official_final20_cases():
    if not ENVIRONMENT_ROOT.is_dir() or not PLAN_ROOT.is_dir():
        return
    release = PLAN_ROOT / "github_release/plan_anchor_final20_v2"
    for case, plan_name in BEST_B0_PLANS.items():
        tier = {"E": "easy", "M": "medium", "H": "hard"}[case[0]]
        scenario = _resolve_scenario(ENVIRONMENT_ROOT, f"final20/{tier}/{case}")
        plan = _resolve_plan(release, scenario, "auto")
        assert plan == release / f"plans/{case}/{plan_name}.json"
        assert _max_steps(scenario) == 3000


def test_cli_requires_explicit_environment_plan_and_output_roots():
    parser = build_parser()
    try:
        parser.parse_args(["--scenario", "final20/easy/E01", "--run-id", "missing"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("parser unexpectedly accepted missing roots")


def test_cli_comparison_defaults_and_modes():
    parser = build_parser()
    common = [
        "--environment-root", str(ENVIRONMENT_ROOT),
        "--plan-root", str(PLAN_ROOT),
        "--scenario", "final20/easy/E01",
        "--run-id", "comparison",
        "--output-root", "/tmp/comparison",
    ]
    defaults = parser.parse_args(common)
    assert defaults.lower_policy == "hybrid"
    assert defaults.local_pulse_steps == 10
    assert defaults.launch_mode == "official-auto"
    assert defaults.joint_rule_profile == "safe"
    assert defaults.joint_release_window_steps == 10
    assert defaults.joint_max_members == 6
    control = parser.parse_args([*common, "--lower-policy", "local-all"])
    assert control.lower_policy == "local-all"
    joint = parser.parse_args([*common, "--lower-policy", "joint-h-v13"])
    assert joint.lower_policy == "joint-h-v13"
    experimental = parser.parse_args(
        [
            *common,
            "--lower-policy", "joint-h-v13",
            "--joint-rule-profile", "experimental",
        ]
    )
    assert experimental.joint_rule_profile == "experimental"
    mission_aware = parser.parse_args(
        [
            *common,
            "--lower-policy", "joint-h-v13",
            "--joint-rule-profile", "mission-aware",
        ]
    )
    assert mission_aware.joint_rule_profile == "mission-aware"
    l_mode = parser.parse_args([*common, "--lower-policy", "joint-l-local"])
    assert l_mode.lower_policy == "joint-l-local"
    hl_mode = parser.parse_args([*common, "--lower-policy", "joint-hl-local"])
    assert hl_mode.lower_policy == "joint-hl-local"
    m_mode = parser.parse_args([*common, "--lower-policy", "joint-m-local"])
    assert m_mode.lower_policy == "joint-m-local"
    hml_mode = parser.parse_args([*common, "--lower-policy", "joint-hml-local"])
    assert hml_mode.lower_policy == "joint-hml-local"
    all_packages = parser.parse_args(
        [*common, "--mission-replan", "all-packages"]
    )
    assert all_packages.mission_replan == "all-packages"
    mission = parser.parse_args([*common, "--mission-replan", "l-response"])
    assert mission.mission_replan == "l-response"
    waves = parser.parse_args([*common, "--launch-mode", "patched-waves"])
    assert waves.launch_mode == "patched-waves"
    native = parser.parse_args([*common, "--launch-mode", "native-waves"])
    assert native.launch_mode == "native-waves"


def test_r9_cli_requires_explicit_v13_source_and_mode():
    parser = build_r9_parser()
    args = parser.parse_args(
        [
            "--environment-root", "/tmp/environment",
            "--jmy-contest-root", "/tmp/jmy",
            "--checkpoint", "/tmp/v13.pt",
            "--scenario", "final20/easy/E01",
            "--lower-policy", "hybrid",
            "--run-id", "r9-comparison",
            "--output-root", "/tmp/results",
        ]
    )
    assert args.lower_policy == "hybrid"
    assert args.blue_interceptor_ratio == 2
    assert args.seed == 701


def test_official_auto_launch_rows_become_supported_retarget_rows():
    commands = np.asarray(
        [[1.0, 2576.0, 12.2, 2.4], [0.0, 2576.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    rewritten = _retarget_auto_launched(commands)
    assert rewritten.tolist() == [
        [2.0, 2576.0, 12.2, 2.4],
        [0.0, 2576.0, -1.0, 0.0],
    ]
    assert commands[0, 0] == 1.0


def test_composite_owns_exact_motion_step_duration(monkeypatch):
    module = ModuleType("user_agents.unified_missile_agent")
    module.MOTION_PULSE_STEPS = 10
    monkeypatch.setitem(sys.modules, module.__name__, module)
    installed, previous = _install_exact_motion_clock()
    assert installed is module
    assert previous == 10
    assert module.MOTION_PULSE_STEPS == 0


def test_auto_launch_plan_flattens_delayed_waves_without_touching_source(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "effective.json"
    source.write_text(
        json.dumps(
            [
                {"op": "PLAN", "schema_version": 2},
                {"op": "WAVE", "wave_id": 1, "release_step": 0},
                {"op": "WAVE", "wave_id": 2, "release_step": 120},
                {"op": "END"},
            ]
        ),
        encoding="utf-8",
    )
    assert _flatten_wave_releases(source, destination) == destination
    assert json.loads(source.read_text(encoding="utf-8"))[2]["release_step"] == 120
    assert [
        token["release_step"]
        for token in json.loads(destination.read_text(encoding="utf-8"))
        if token["op"] == "WAVE"
    ] == [0, 0]


def test_auto_launch_plan_concentrates_public_h_and_m_salvos(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "effective.json"
    tokens = [
        {"op": "PLAN", "schema_version": 2},
        {"op": "WAVE", "wave_id": 1, "release_step": 0},
        {"op": "WAVE", "wave_id": 2, "release_step": 120},
        {"op": "ASSIGN", "role": "H", "count": 2, "goal_kind": "public_target", "target_id": 10, "target_type": 9400, "wave_id": 1},
        {"op": "ASSIGN", "role": "H", "count": 1, "goal_kind": "public_target", "target_id": 20, "target_type": 9600, "wave_id": 1},
        {"op": "ASSIGN", "role": "M", "count": 3, "goal_kind": "public_target", "target_id": 20, "target_type": 9600, "wave_id": 1},
        {"op": "END"},
    ]
    source.write_text(json.dumps(tokens), encoding="utf-8")
    _flatten_wave_releases(source, destination, concentrate_public_salvos=True)
    compiled = json.loads(destination.read_text(encoding="utf-8"))
    assignments = [token for token in compiled if token.get("op") == "ASSIGN"]
    assert [(token["role"], token["target_id"], token["count"]) for token in assignments] == [
        ("H", 10, 2),
        ("H", 20, 1),
        ("M", 10, 3),
    ]
    assert [token["wave_id"] for token in compiled if token.get("op") == "WAVE"] == [1]


def test_native_wave_delay_shifts_every_wave_without_touching_source(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "effective.json"
    source.write_text(
        json.dumps(
            [
                {"op": "PLAN", "schema_version": 2},
                {"op": "WAVE", "wave_id": 1, "release_step": 0},
                {"op": "WAVE", "wave_id": 2, "release_step": 120},
                {"op": "END"},
            ]
        ),
        encoding="utf-8",
    )
    assert _shift_wave_releases(source, destination, 1) == destination
    assert [
        token["release_step"]
        for token in json.loads(destination.read_text(encoding="utf-8"))
        if token["op"] == "WAVE"
    ] == [1, 121]
    assert json.loads(source.read_text(encoding="utf-8"))[1]["release_step"] == 0
