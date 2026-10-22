import numpy as np
import pytest

from staggered_deployment_skill import StaggeredDeploymentConfig

from simple_upper_orchestrator_skill.deploy_agent import StaggeredDeployAgent, rectangle_bounds


RECTANGLE = [[[0, 0], [1, 0], [1, 1], [0, 1]]]


def test_deploy_returns_nx5_batches_then_completion():
    agent = StaggeredDeployAgent(
        RECTANGLE, RECTANGLE,
        config=StaggeredDeploymentConfig(deployment_gap_km=1, l_depth_km=1, hm_depth_km=1),
        batch_size=2, launch_altitude=1000,
    )
    agent.set_observation({"entities": {
        1: {"type": 21000, "position": {"alt": 1000}},
        2: {"type": 21001, "position": {"alt": 1000}},
        3: {"type": 21002, "position": {"alt": 1000}},
        99: {"type": 9202, "position": {"alt": 0}},
    }})
    first, second, complete = agent.get_action(), agent.get_action(), agent.get_action()
    assert first.shape == (2, 5)
    assert second.shape == (1, 5)
    assert np.array_equal(complete, np.array([[1, -1, 0, 0, 0]], dtype=np.float64))
    assert set(np.concatenate((first, second))[:, 1].astype(int)) == {1, 2, 3}


def test_deployment_geometry_fails_closed():
    with pytest.raises(ValueError, match="axis-aligned"):
        rectangle_bounds([[[0, 0], [1, 0], [0.5, 1], [0, 1]]])
    with pytest.raises(ValueError, match="identical"):
        StaggeredDeployAgent(
            RECTANGLE, [[[0, 0], [2, 0], [2, 1], [0, 1]]],
            config=StaggeredDeploymentConfig(), batch_size=10, launch_altitude=1000,
        )
