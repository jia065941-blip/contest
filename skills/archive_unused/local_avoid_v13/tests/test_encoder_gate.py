import numpy as np

from local_avoid_v13_skill.encoder import EnvironmentConfig, LocalAvoidRuntimeEncoder
from local_avoid_v13_skill.gate import LEFT, RIGHT, STOP, prepare_action_mask
from local_avoid_v13_skill.policy import LocalAvoidControlPolicy


CONFIG = EnvironmentConfig(
    unit_speeds_mps=(200.0, 120.0, 100.0), goal_distance_scale_m=100_000.0,
    threat_distance_scale_m=10_000.0, relative_speed_scale_mps=500.0,
    t_cpa_scale_s=60.0, horizon_steps=900, risk_distance_m=1_000.0,
    risk_t_cpa_s=45.0, risk_d_cpa_m=3_000.0, actor_risk_gate=True,
    actor_action_history=False, encode_track_source=False,
)


def raw_observation(step=10):
    return {
        "step": step,
        "self": {
            "type": 21000, "health": 100.0, "stage": 1,
            "position": {"lon": 0.0, "lat": 0.0, "alt": 1000.0},
            "pos_ecf": {"x": 6_378_137.0, "y": 0.0, "z": 0.0},
            "detectInfo": {}, "commRangeInfo": [],
        },
    }


def test_encoder_layout_without_threats():
    encoder = LocalAvoidRuntimeEncoder(entity_id=7, max_steps=1200, config=CONFIG)
    result = encoder.encode(raw_observation(), launch_step=5, maneuver_state=0,
                            goal_position={"lon": 0.1, "lat": 0.0})
    assert result.shape == (33,) and result.dtype == np.float32
    np.testing.assert_allclose(result[:6], [1.0, 0.0, 1.0, 0.11132, 0.0, 1.0],
                               atol=1e-6)
    np.testing.assert_array_equal(result[6:29], np.zeros(23, dtype=np.float32))
    assert result[29] == np.float32(5 / 900)
    np.testing.assert_array_equal(result[30:33], [1.0, 0.0, 0.0])


def test_mission_gate_straight_urgent_and_recovery():
    observation = np.zeros(33, dtype=np.float32)
    observation[3] = 1.0
    observation[30] = 1.0
    mask, pending = prepare_action_mask(observation, CONFIG, False)
    np.testing.assert_array_equal(mask, [False, True, False])
    assert not pending
    observation[6] = 1.0
    observation[7] = 0.05
    mask, _ = prepare_action_mask(observation, CONFIG, False)
    np.testing.assert_array_equal(mask, [True, True, True])
    observation[6] = 0.0
    observation[4], observation[5] = 1.0, 0.0
    mask, pending = prepare_action_mask(observation, CONFIG, True)
    np.testing.assert_array_equal(mask, [True, False, False])
    assert pending
    observation[4], observation[5] = -1.0, 0.0
    mask, _ = prepare_action_mask(observation, CONFIG, True)
    np.testing.assert_array_equal(mask, [False, False, True])


def test_policy_keeps_agent_state_isolated_and_resets():
    policy = LocalAvoidControlPolicy(max_steps=1200)
    try:
        assert policy.skill.monitor.enabled is False
        first = policy.action_for(agent_id=1, entity_id=7,
                                  observation=raw_observation(), launch_step=5,
                                  maneuver_state=0, goal_position={"lon": 0.1, "lat": 0.0})
        second = policy.action_for(agent_id=2, entity_id=8,
                                   observation=raw_observation(), launch_step=5,
                                   maneuver_state=0, goal_position={"lon": 0.1, "lat": 0.0})
        assert first == second == STOP
        assert policy._encoders[1] is not policy._encoders[2]
        policy._recovery[1] = True
        policy.reset_agent(1)
        assert 1 not in policy._recovery and 2 in policy._encoders
    finally:
        policy.close()


def test_unified_agent_compatibility_interface():
    policy = LocalAvoidControlPolicy(max_steps=1200)
    try:
        encoded = policy.encode_motion_observation(
            raw_observation(), agent_id=1, entity_id=7, launch_step=5,
            maneuver_state=0, goal_position={"lon": 0.1, "lat": 0.0})
        mask = policy.prepare_action_mask(encoded, np.ones(3, dtype=np.bool_))
        assert policy.select_action(encoded, mask) == STOP
        policy.reset_episode()
        assert policy._current_agent_id is None and not policy._recovery
    finally:
        policy.close()
