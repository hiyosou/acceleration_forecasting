import numpy as np
import pandas as pd

from acceleration_forecasting.datasets.build_generation_dataset import select_first_eligible_anchors
from acceleration_forecasting.datasets.history_builder import sanitize_future, select_monthly_history
from acceleration_forecasting.datasets.normalization import fit_normalization
from acceleration_forecasting.evaluation.bootstrap import dataset_bootstrap
from acceleration_forecasting.evaluation.metrics import evaluate_target


def test_normalization_uses_unique_observations():
    frame = pd.DataFrame({
        "dataset_id": ["a", "a", "a"],
        "measurement_date": ["2024-01-01", "2024-01-01", "2024-02-01"],
        "current_acc_z_max": [1.0, 1.0, 3.0],
    })
    norm = fit_normalization(frame)
    assert norm.fitted_observation_count == 2
    assert norm.mean == 2.0
    values = np.array([1.0, 3.0])
    assert np.allclose(norm.denormalize(norm.normalize(values)), values)


def test_history_prefers_past_on_equal_distance():
    group = pd.DataFrame({
        "measurement_date": pd.to_datetime(["2024-01-31", "2024-02-02", "2024-03-01"]),
        "current_acc_z_max": [1.0, 2.0, 3.0],
    })
    values, mask, dates = select_monthly_history(group, "2024-05-10", months=3)
    assert dates[0] == "2024-01-31"
    assert mask.sum() >= 2


def test_cutoff_masks_target_month_and_after():
    values, mask = sanitize_future(
        "2024-01-15", list(range(18)), [1] * 18, "2024-04-01"
    )
    assert mask[:2].tolist() == [1, 1]
    assert mask[2:].sum() == 0
    assert np.isnan(values[2:]).all()


def test_metrics_respect_mask():
    actual = np.array([1.0, 2.0, 100.0])
    mask = np.array([1, 1, 0])
    result = evaluate_target(actual, mask, [1.0, 3.0, 0.0], [0.5, 2.5, 0], [1.5, 3.5, 0])
    assert result["MAE"] == 0.5
    assert result["coverage_p10_p90"] == 0.5


def test_bootstrap_resamples_dataset_groups_reproducibly():
    frame = pd.DataFrame({
        "dataset_id": ["a", "a", "b"], "MAE": [1.0, 2.0, 3.0],
        "RMSE": [1.0, 2.0, 3.0], "coverage_p10_p90": [1, 0, 1],
        "mean_interval_width": [1, 1, 1], "peak_value_error": [1, 2, 3],
    })
    one = dataset_bootstrap(frame, iterations=10, seed=42)
    two = dataset_bootstrap(frame, iterations=10, seed=42)
    pd.testing.assert_frame_equal(one, two)


def _anchor_frame(rows):
    frame = pd.DataFrame(rows)
    frame["measurement_date"] = pd.to_datetime(frame["measurement_date"])
    frame["future_values"] = [str([1.0] * 18).replace("'", '"')] * len(frame)
    frame["cutoff_maintenance_date"] = ""
    return frame


def test_selects_first_eligible_anchor_per_dataset():
    frame = _anchor_frame([
        {"dataset_id": "a", "target_id": "a1", "measurement_date": "2024-01-01", "current_acc_z_max": 1.0, "future_mask": str([1] * 11 + [0] * 7)},
        {"dataset_id": "a", "target_id": "a2", "measurement_date": "2024-02-01", "current_acc_z_max": 1.1, "future_mask": str([1] * 12 + [0] * 6)},
        {"dataset_id": "a", "target_id": "a3", "measurement_date": "2024-03-01", "current_acc_z_max": 1.2, "future_mask": str([1] * 18)},
        {"dataset_id": "b", "target_id": "b1", "measurement_date": "2024-01-01", "current_acc_z_max": 2.0, "future_mask": str([1] * 18)},
    ])
    vectors = {target: [np.ones(256)] for target in ("a1", "a2", "a3", "b1")}
    selected, diagnostics = select_first_eligible_anchors(
        frame, "model_train", lambda _split, target: vectors.get(target, []),
        progress=False,
    )
    assert [item["row"]["target_id"] for item in selected] == ["a2", "b1"]
    assert diagnostics["candidate_datasets"] == 2
    assert diagnostics["adopted_datasets"] == 2


def test_missing_vector_uses_next_anchor_and_inference_allows_zero_future():
    frame = _anchor_frame([
        {"dataset_id": "a", "target_id": "a1", "measurement_date": "2024-01-01", "current_acc_z_max": 1.0, "future_mask": str([0] * 18)},
        {"dataset_id": "a", "target_id": "a2", "measurement_date": "2024-02-01", "current_acc_z_max": 1.1, "future_mask": str([0] * 18)},
    ])
    selected, diagnostics = select_first_eligible_anchors(
        frame, "inference",
        lambda _split, target: [np.ones(256)] if target == "a2" else [],
        progress=False,
    )
    assert len(selected) == 1
    assert selected[0]["row"]["target_id"] == "a2"
    assert selected[0]["valid_target_months"] == 0
    assert diagnostics["datasets_without_eligible_anchor"] == 0
