import numpy as np

from acceleration_forecasting.datasets.residual_dataset import masked_softmax_baseline


def test_masked_softmax_baseline_weights_and_masks():
    values = np.array([[1, 2, 3], [3, 4, 5], [9, 9, 9]], dtype=float)
    masks = np.array([[1, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=float)
    baseline, weights, sources = masked_softmax_baseline(
        values, masks, [0.9, 0.8, 0.7], [1, 1, 0], 2.0, temperature=0.1
    )
    assert np.allclose(weights.sum(axis=0), 1)
    assert weights[0, 0] > weights[1, 0]
    assert baseline[1] == 2
    assert baseline[2] == 5
    assert sources.tolist() == ["weighted_guides"] * 3


def test_baseline_interpolates_internal_and_fills_edges():
    values = np.array([[1, np.nan, 3, np.nan]], dtype=float)
    masks = np.array([[1, 0, 1, 0]], dtype=float)
    baseline, weights, sources = masked_softmax_baseline(
        values, masks, [0.9], [1], 2.0
    )
    assert baseline.tolist() == [1, 2, 3, 3]
    assert sources.tolist() == ["weighted_guides", "interpolated", "weighted_guides", "edge_filled"]
    assert np.allclose(weights[:, [0, 2]], 1)


def test_all_missing_baseline_uses_current_without_changing_weights():
    baseline, weights, sources = masked_softmax_baseline(
        np.full((3, 18), np.nan), np.zeros((3, 18)), [0.9, 0.8, 0.7], [0, 0, 0], 1.75
    )
    assert np.allclose(baseline, 1.75)
    assert np.all(weights == 0)
    assert set(sources) == {"current_value_fallback"}
