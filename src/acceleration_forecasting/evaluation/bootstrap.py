from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_METRICS = (
    "MAE", "RMSE", "coverage_p10_p90", "mean_interval_width", "peak_value_error"
)


def dataset_bootstrap(frame, iterations=1000, seed=42, metrics=DEFAULT_METRICS):
    dataset_ids = frame["dataset_id"].dropna().astype(str).unique()
    if not len(dataset_ids):
        return pd.DataFrame()
    groups = {key: group for key, group in frame.groupby(frame["dataset_id"].astype(str))}
    generator = np.random.default_rng(seed)
    rows = []
    for iteration in range(int(iterations)):
        selected = generator.choice(dataset_ids, size=len(dataset_ids), replace=True)
        sampled = pd.concat([groups[key] for key in selected], ignore_index=True)
        row = {"iteration": iteration}
        for metric in metrics:
            row[metric] = float(sampled[metric].mean(skipna=True))
        rows.append(row)
    return pd.DataFrame(rows)


def confidence_intervals(bootstrap_frame, metrics=DEFAULT_METRICS):
    result = {}
    for metric in metrics:
        values = bootstrap_frame[metric].dropna().to_numpy(dtype=float)
        if values.size:
            result[f"{metric}_ci_lower"] = float(np.percentile(values, 2.5))
            result[f"{metric}_ci_upper"] = float(np.percentile(values, 97.5))
    return result

