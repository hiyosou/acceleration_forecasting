from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from acceleration_forecasting.datasets.normalization import AccelerationNormalization

from .bootstrap import confidence_intervals, dataset_bootstrap
from .metrics import evaluate_target
from .visualize import load_sample_map, plot_evaluation


def _actual_history(retrieval_dir):
    retrieval_dir = Path(retrieval_dir)
    development = pd.read_csv(retrieval_dir / "development_trends.csv", encoding="utf-8-sig")
    inference_targets = pd.read_csv(retrieval_dir / "inference_targets.csv", encoding="utf-8-sig")
    inference_targets = inference_targets.rename(columns={"target_id": "trend_id"})
    trends = pd.concat([development, inference_targets], ignore_index=True)
    manifest = pd.read_csv(retrieval_dir / "dataset_split_manifest.csv", encoding="utf-8-sig")
    speed = manifest.groupby("trend_id", as_index=False)["mean_velocity_kmh"].mean().rename(columns={"mean_velocity_kmh": "velocity"})
    trends = trends.merge(speed, on="trend_id", how="left")
    return trends


def evaluate(
    dataset_dir,
    prediction_dir,
    output_dir,
    *,
    bootstrap_iterations=1000,
    seed=42,
    plot=False,
    plot_max_targets=100,
    y_max=5.0,
    dpi=150,
):
    dataset_dir, prediction_dir, output_dir = map(Path, (dataset_dir, prediction_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(dataset_dir / "inference" / "inputs" / "metadata.csv", encoding="utf-8-sig")
    target_ids = pd.read_csv(dataset_dir / "inference" / "targets" / "target_ids.csv", encoding="utf-8-sig")
    target_values = np.load(dataset_dir / "inference" / "targets" / "target_values.npy", mmap_mode="r")
    target_masks = np.load(dataset_dir / "inference" / "targets" / "target_masks.npy", mmap_mode="r")
    target_lookup = {str(value): index for index, value in enumerate(target_ids["target_id"].astype(str))}
    predictions = pd.read_csv(prediction_dir / "predictions.csv", encoding="utf-8-sig")
    per_target, monthly = [], []
    for target_id, group in predictions.groupby(predictions["target_id"].astype(str), sort=False):
        if target_id not in target_lookup:
            continue
        index = target_lookup[target_id]
        ordered = group.sort_values("month_index")
        metrics = evaluate_target(
            target_values[index], target_masks[index], ordered["prediction_median"],
            ordered["prediction_p10"], ordered["prediction_p90"],
        )
        if metrics is None:
            continue
        meta = metadata.loc[metadata["target_id"].astype(str) == target_id].iloc[0]
        per_target.append({
            "target_id": target_id, "dataset_id": meta["dataset_id"],
            "direction": meta["direction"], "bin_start_m": meta["bin_start_m"],
            "current_acc_z_max": meta.get("current_acc_z_max", np.nan),
            "guide_count": meta["guide_count"], **metrics,
        })
        for month, row in ordered.iterrows():
            mi = int(row["month_index"]) - 1
            if target_masks[index][mi] > 0 and np.isfinite(target_values[index][mi]):
                actual = float(target_values[index][mi])
                monthly.append({
                    "target_id": target_id, "month_index": mi + 1, "actual": actual,
                    "prediction_median": row["prediction_median"], "prediction_p10": row["prediction_p10"],
                    "prediction_p90": row["prediction_p90"],
                    "absolute_error": abs(float(row["prediction_median"]) - actual),
                    "inside_interval": int(float(row["prediction_p10"]) <= actual <= float(row["prediction_p90"])),
                })
    per_target_frame = pd.DataFrame(per_target)
    monthly_frame = pd.DataFrame(monthly)
    per_target_frame.to_csv(output_dir / "evaluation_per_target.csv", index=False, encoding="utf-8-sig")
    monthly_frame.to_csv(output_dir / "evaluation_per_month_records.csv", index=False, encoding="utf-8-sig")
    month_summary = monthly_frame.groupby("month_index").agg(
        MAE=("absolute_error", "mean"), coverage_p10_p90=("inside_interval", "mean"), target_count=("target_id", "count")
    ).reset_index()
    month_summary.to_csv(output_dir / "evaluation_per_month.csv", index=False, encoding="utf-8-sig")
    guide_path = prediction_dir / "prediction_guides.csv"
    if guide_path.exists():
        guide_audit = pd.read_csv(guide_path, encoding="utf-8-sig")
        selected_guides = guide_audit.loc[guide_audit.get("selection_status", "") == "selected"].copy()
        selected_guides["cosine_similarity"] = pd.to_numeric(selected_guides.get("cosine_similarity"), errors="coerce")
        mean_similarity = selected_guides.groupby(selected_guides["target_id"].astype(str))["cosine_similarity"].mean()
        per_target_frame["mean_guide_similarity"] = per_target_frame["target_id"].astype(str).map(mean_similarity)
    else:
        per_target_frame["mean_guide_similarity"] = np.nan
    per_target_frame["current_value_band"] = pd.cut(
        pd.to_numeric(per_target_frame["current_acc_z_max"], errors="coerce"),
        bins=[-np.inf, 1, 2, 3, 4, np.inf], right=False,
    ).astype(str)
    per_target_frame["similarity_band"] = pd.cut(
        per_target_frame["mean_guide_similarity"],
        bins=[-np.inf, 0.5, 0.7, 0.85, 0.95, np.inf], right=False,
    ).astype(str)
    grouped_rows = []
    for column in ("direction", "bin_start_m", "current_value_band", "guide_count", "similarity_band"):
        for value, group in per_target_frame.groupby(column, dropna=False):
            grouped_rows.append({
                "group_type": column, "group_value": value, "target_count": len(group),
                "MAE": group["MAE"].mean(), "RMSE": group["RMSE"].mean(),
                "coverage_p10_p90": group["coverage_p10_p90"].mean(),
                "mean_interval_width": group["mean_interval_width"].mean(),
            })
    pd.DataFrame(grouped_rows).to_csv(output_dir / "evaluation_by_group.csv", index=False, encoding="utf-8-sig")
    bootstrap = dataset_bootstrap(per_target_frame, bootstrap_iterations, seed)
    bootstrap.to_csv(output_dir / "bootstrap_results.csv", index=False, encoding="utf-8-sig")
    metric_columns = ["MAE", "RMSE", "correlation", "peak_value_error", "peak_month_error", "coverage_p10_p90", "mean_interval_width"]
    summary = {column: float(per_target_frame[column].mean(skipna=True)) for column in metric_columns}
    summary.update(confidence_intervals(bootstrap))
    summary.update({"target_count": int(len(per_target_frame)), "bootstrap_iterations": int(bootstrap_iterations), "seed": seed})
    pd.DataFrame([summary]).to_csv(output_dir / "evaluation_summary.csv", index=False, encoding="utf-8-sig")

    if plot and not per_target_frame.empty:
        source = json.loads((dataset_dir / "source_retrieval_artifacts.json").read_text(encoding="utf-8"))
        retrieval_dir = Path(source["retrieval_artifact_dir"])
        history = _actual_history(retrieval_dir)
        guide_values = np.load(dataset_dir / "inference" / "inputs" / "guide_values.npy", mmap_mode="r")
        guide_masks = np.load(dataset_dir / "inference" / "inputs" / "guide_masks.npy", mmap_mode="r")
        guides = pd.read_csv(prediction_dir / "prediction_guides.csv", encoding="utf-8-sig")
        selected = per_target_frame.sort_values("RMSE", ascending=False).head(int(plot_max_targets))["target_id"].astype(str).tolist()
        sample_map = load_sample_map(prediction_dir / "samples", selected)
        meta_lookup = {str(row.target_id): (index, row._asdict()) for index, row in enumerate(metadata.itertuples(index=False))}
        for target_id in selected:
            if target_id not in sample_map or target_id not in meta_lookup:
                continue
            data_index, meta = meta_lookup[target_id]
            relevant = history.loc[
                (history["direction"].astype(str) == str(meta["direction"]))
                & (pd.to_numeric(history["bin_start_m"], errors="coerce") == float(meta["bin_start_m"]))
            ].copy()
            relevant = relevant.loc[pd.to_numeric(relevant["velocity"], errors="coerce").between(50, 75, inclusive="both")]
            target_index = target_lookup[target_id]
            plot_evaluation(
                output_dir / "plots" / str(meta["direction"]) / f"{float(meta['bin_start_m']):.0f}-{float(meta['bin_end_m']):.0f}m" / f"{target_id}_evaluation.png",
                meta, relevant, predictions.loc[predictions["target_id"].astype(str) == target_id],
                target_values[target_index], target_masks[target_index], guide_values[data_index], guide_masks[data_index],
                guides.loc[guides["target_id"].astype(str) == target_id], sample_map[target_id], y_max, dpi,
            )
    return summary
