from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from acceleration_forecasting.common.progress import progress_bar


def plot_guide_progressions(dataset_dir, prediction_dir, output_dir, *, y_max=5.0, dpi=150, progress=True):
    dataset_dir, prediction_dir, output_dir = map(Path, (dataset_dir, prediction_dir, output_dir))
    inputs = dataset_dir / "inference" / "inputs"
    metadata = pd.read_csv(inputs / "metadata.csv", encoding="utf-8-sig")
    values = np.load(inputs / "guide_values.npy", mmap_mode="r")
    masks = np.load(inputs / "guide_masks.npy", mmap_mode="r")
    baselines = np.load(inputs / "guide_baselines.npy", mmap_mode="r")
    weights = np.load(inputs / "guide_softmax_weights.npy", mmap_mode="r")
    assignments = pd.read_csv(prediction_dir / "prediction_guides.csv", encoding="utf-8-sig")
    image_count = 0
    for index, meta in progress_bar(
        list(metadata.iterrows()), enabled=progress, total=len(metadata),
        desc="ガイド進展画像を生成", unit="image",
    ):
        target_id = str(meta["target_id"])
        selected = assignments.loc[
            (assignments["target_id"].astype(str) == target_id)
            & (assignments["selection_status"] == "selected")
        ].copy()
        fig, axes = plt.subplots(4, 1, figsize=(12, 11), constrained_layout=True)
        colors = ("tab:blue", "tab:green", "tab:purple")
        for rank in range(3):
            axis = axes[rank]
            info = selected.loc[pd.to_numeric(selected["guide_rank"], errors="coerce") == rank + 1]
            valid = np.asarray(masks[index, rank], dtype=bool)
            if info.empty:
                axis.text(0.5, 0.5, f"Guide {rank + 1}: not found", transform=axis.transAxes, ha="center", va="center")
                axis.set_ylim(0, y_max)
                axis.grid(True, color="0.85")
                continue
            row = info.iloc[0]
            guide_date = pd.Timestamp(row["guide_date"])
            future_dates = pd.date_range(guide_date.replace(day=1) + pd.DateOffset(months=1), periods=18, freq="MS")
            current = float(row["guide_current_acc_z_max"])
            axis.plot([guide_date], [current], "o", color=colors[rank], markersize=7, label="guide anchor")
            axis.plot(future_dates[valid], np.asarray(values[index, rank])[valid], "o-", color=colors[rank], label="observed progression")
            axis.plot(future_dates[~valid], np.zeros((~valid).sum()), "x", color="0.75", alpha=0.0)
            mean_weight = float(weights[index, rank][valid].mean()) if valid.any() else 0.0
            axis.set_title(
                f"G{rank + 1} / {row['candidate_dataset_id']} / {row['candidate_direction']} "
                f"{float(row['candidate_bin_start_m']):.0f}-{float(row['candidate_bin_end_m']):.0f}m / "
                f"anchor {row['guide_date']} / similarity={float(row['cosine_similarity']):.3f} / "
                f"mean weight={mean_weight:.3f}"
            )
            axis.set_ylim(0, y_max)
            axis.set_ylabel("acc_z max [m/s²]")
            axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
            axis.grid(True, color="0.85")
            axis.legend(loc="upper right")
        query_date = pd.Timestamp(meta["anchor_date"])
        query_months = pd.date_range(query_date.replace(day=1) + pd.DateOffset(months=1), periods=18, freq="MS")
        axes[3].plot(query_months, baselines[index], "o-", color="darkorange", linewidth=2, label="Softmax guide baseline")
        axes[3].set_title(
            f"Query baseline / {meta['dataset_id']} / {meta['direction']} "
            f"{float(meta['bin_start_m']):.0f}-{float(meta['bin_end_m']):.0f}m / anchor {meta['anchor_date']}"
        )
        axes[3].set_ylim(0, y_max)
        axes[3].set_ylabel("acc_z max [m/s²]")
        axes[3].set_xlabel("Calendar date")
        axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
        axes[3].grid(True, color="0.85")
        axes[3].legend(loc="upper right")
        path = output_dir / str(meta["direction"]) / f"{target_id}_guide_progressions.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, transparent=True)
        plt.close(fig)
        image_count += 1
    return {"target_count": int(len(metadata)), "image_count": image_count, "output_dir": str(output_dir.resolve())}
