from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compare_predictions(
    baseline_evaluation_dir, residual_evaluation_dir, baseline_prediction_dir,
    residual_prediction_dir, dataset_dir, output_dir, *, max_images=100, dpi=150,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(Path(baseline_evaluation_dir) / "evaluation_summary.csv", encoding="utf-8-sig").iloc[0]
    new = pd.read_csv(Path(residual_evaluation_dir) / "evaluation_summary.csv", encoding="utf-8-sig").iloc[0]
    old_mse = float(np.mean(pd.read_csv(
        Path(baseline_evaluation_dir) / "evaluation_per_month_records.csv", encoding="utf-8-sig"
    )["absolute_error"].to_numpy(float) ** 2))
    rows = []
    for metric in ("MAE", "MSE", "RMSE", "coverage_p10_p90", "mean_interval_width"):
        old_value = old_mse if metric == "MSE" and "MSE" not in old.index else float(old.get(metric, np.nan))
        new_value = float(new.get(metric, np.nan))
        rows.append({
            "metric": metric, "one_anchor": old_value, "residual": new_value,
            "change": new_value - old_value,
            "change_percent": (new_value - old_value) / old_value * 100 if old_value else np.nan,
        })
    pd.DataFrame(rows).to_csv(output / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    old_target = pd.read_csv(Path(baseline_evaluation_dir) / "evaluation_per_target.csv", encoding="utf-8-sig")
    new_target = pd.read_csv(Path(residual_evaluation_dir) / "evaluation_per_target.csv", encoding="utf-8-sig")
    merged = old_target.merge(new_target, on="target_id", suffixes=("_one_anchor", "_residual"))
    merged.to_csv(output / "comparison_per_target.csv", index=False, encoding="utf-8-sig")
    old_prediction = pd.read_csv(Path(baseline_prediction_dir) / "predictions.csv", encoding="utf-8-sig")
    new_prediction = pd.read_csv(Path(residual_prediction_dir) / "predictions.csv", encoding="utf-8-sig")
    metadata = pd.read_csv(Path(dataset_dir) / "inference" / "inputs" / "metadata.csv", encoding="utf-8-sig")
    target_ids = pd.read_csv(Path(dataset_dir) / "inference" / "targets" / "target_ids.csv", encoding="utf-8-sig")
    targets = np.load(Path(dataset_dir) / "inference" / "targets" / "target_values.npy", mmap_mode="r")
    masks = np.load(Path(dataset_dir) / "inference" / "targets" / "target_masks.npy", mmap_mode="r")
    lookup = {str(value): index for index, value in enumerate(target_ids["target_id"].astype(str))}
    selected_ids = merged.sort_values("RMSE_residual", ascending=False).head(int(max_images))["target_id"].astype(str)
    image_count = 0
    for target_id in selected_ids:
        if target_id not in lookup:
            continue
        first = old_prediction.loc[old_prediction["target_id"].astype(str) == target_id].sort_values("month_index")
        second = new_prediction.loc[new_prediction["target_id"].astype(str) == target_id].sort_values("month_index")
        if len(first) != 18 or len(second) != 18:
            continue
        index, months = lookup[target_id], np.arange(1, 19)
        valid = np.asarray(masks[index], dtype=bool)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
        for axis, frame, title, color in (
            (axes[0], first, "One-anchor absolute diffusion", "tab:blue"),
            (axes[1], second, "Guide-baseline residual diffusion", "tab:red"),
        ):
            p10, p90 = frame["prediction_p10"].to_numpy(float), frame["prediction_p90"].to_numpy(float)
            axis.fill_between(months, p10, p90, color=color, alpha=0.2)
            axis.plot(months, frame["prediction_median"], "o-", color=color, label="median")
            axis.plot(months[valid], np.asarray(targets[index])[valid], "ko-", markerfacecolor="white", label="actual")
            axis.set_ylim(0, 5)
            axis.set_ylabel("acc_z max [m/s²]")
            axis.set_title(title)
            axis.grid(True, color="0.85")
            axis.legend()
        axes[1].set_xlabel("Forecast month")
        meta = metadata.loc[metadata["target_id"].astype(str) == target_id].iloc[0]
        fig.suptitle(f"{meta['direction']} {float(meta['bin_start_m']):.0f}-{float(meta['bin_end_m']):.0f}m / {meta['anchor_date']}")
        path = output / "plots" / str(meta["direction"]) / f"{target_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, transparent=True)
        plt.close(fig)
        image_count += 1
    return {
        "common_targets": int(len(merged)), "images": image_count,
        "one_anchor_interval_width": float(old["mean_interval_width"]),
        "residual_interval_width": float(new["mean_interval_width"]),
        "interval_width_below_2": bool(float(new["mean_interval_width"]) < 2.0),
    }
