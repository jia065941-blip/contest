from __future__ import annotations

import torch

from tools.apply_option_boundary_return_candidate import (
    BoundaryEpisode,
    OptionBoundaryReturnModel,
    seed_centered_labels,
    uncertainty_shrunk_returns,
)


def _episode(seed: int, score: float) -> BoundaryEpisode:
    return BoundaryEpisode(
        rollout_path=None,  # type: ignore[arg-type]
        progress_path=None,  # type: ignore[arg-type]
        seed=seed,
        score=score,
        suffix_return=0.0,
        tokens=torch.zeros(1, 3),
    )


def test_seed_centering_removes_scenario_difficulty() -> None:
    labels = seed_centered_labels((
        _episode(1, 80.0),
        _episode(1, 84.0),
        _episode(2, 50.0),
        _episode(2, 54.0),
    ))
    assert torch.allclose(labels, torch.tensor([-0.02, 0.02, -0.02, 0.02]))


def test_uncertainty_shrink_preserves_pair_direction_and_loo_scale() -> None:
    predictions = torch.tensor([
        [0.03, -0.03, 0.02, -0.02],
        [0.02, -0.02, 0.01, -0.01],
        [0.04, -0.04, -0.01, 0.01],
    ])
    returns, metrics = uncertainty_shrunk_returns(
        predictions,
        (1, 1, 2, 2),
    )
    assert returns[0] > returns[1]
    assert abs(returns[2] - returns[3]) < abs(returns[0] - returns[1])
    assert 0.0 < metrics["ensemble_reliability_mean"] < 1.0


def test_return_model_accepts_variable_length_padding() -> None:
    model = OptionBoundaryReturnModel(input_dim=3, hidden_dim=8)
    inputs = torch.randn(2, 3, 3)
    outputs = model(inputs, torch.tensor([1, 3]))
    assert outputs.shape == (2,)
    assert bool(torch.isfinite(outputs).all().item())
