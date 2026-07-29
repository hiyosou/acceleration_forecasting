from __future__ import annotations

import numpy as np


def evaluate_target(actual, mask, median, p10, p90):
    actual = np.asarray(actual, dtype=float)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(actual)
    median, p10, p90 = map(lambda value: np.asarray(value, dtype=float), (median, p10, p90))
    if not valid.any():
        return None
    truth, prediction = actual[valid], median[valid]
    error = prediction - truth
    correlation = np.nan
    if len(truth) >= 2 and np.std(truth) > 0 and np.std(prediction) > 0:
        correlation = float(np.corrcoef(truth, prediction)[0, 1])
    valid_indices = np.flatnonzero(valid)
    actual_peak_local = int(np.argmax(truth))
    predicted_peak_local = int(np.argmax(prediction))
    return {
        "valid_months": int(valid.sum()),
        "MAE": float(np.mean(np.abs(error))),
        "MSE": float(np.mean(error**2)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "correlation": correlation,
        "peak_value_error": float(abs(prediction[predicted_peak_local] - truth[actual_peak_local])),
        "peak_month_error": int(abs(valid_indices[predicted_peak_local] - valid_indices[actual_peak_local])),
        "coverage_p10_p90": float(np.mean((truth >= p10[valid]) & (truth <= p90[valid]))),
        "mean_interval_width": float(np.mean((p90 - p10)[valid])),
    }
