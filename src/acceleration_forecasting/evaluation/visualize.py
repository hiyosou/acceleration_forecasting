from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "MS Gothic"


def load_sample_map(sample_dir, wanted=None):
    wanted = set(wanted or [])
    output = {}
    for path in sorted(Path(sample_dir).glob("samples_*.npz")):
        data = np.load(path)
        for target_id, samples in zip(data["target_ids"].astype(str), data["samples"]):
            if not wanted or target_id in wanted:
                output[target_id] = samples
    return output


def plot_evaluation(
    output_path,
    metadata,
    actual_history,
    prediction,
    actual_future,
    target_mask,
    guide_values,
    guide_masks,
    guide_info,
    samples,
    y_max=5.0,
    dpi=150,
    plot_style="detailed",
    single_sample_index=None,
):
    if plot_style not in {"detailed", "clean"}:
        raise ValueError(f"Unsupported plot style: {plot_style}")
    show_guide_details = plot_style == "detailed"
    anchor = pd.Timestamp(metadata["anchor_date"])
    future_dates = pd.date_range(anchor.replace(day=1) + pd.DateOffset(months=1), periods=18, freq="MS")
    forecast_start = future_dates[0]
    prediction = prediction.sort_values("month_index")
    median = prediction["prediction_median"].to_numpy(float)
    p10 = prediction["prediction_p10"].to_numpy(float)
    p90 = prediction["prediction_p90"].to_numpy(float)
    selected_sample = None
    if single_sample_index is not None:
        sample_index = int(single_sample_index)
        sample_array = np.asarray(samples)
        if sample_array.ndim != 2 or sample_array.shape[1] != 18:
            raise ValueError("Generated samples must have shape [N, 18]")
        if sample_index < 0 or sample_index >= sample_array.shape[0]:
            raise IndexError(
                f"single_sample_index {sample_index} is outside 0..{sample_array.shape[0] - 1}"
            )
        selected_sample = sample_array[sample_index]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    figure_color = fig.patch.get_facecolor()
    top.set_facecolor("none")
    bottom.set_facecolor("none")
    if not actual_history.empty:
        scatter = top.scatter(
            pd.to_datetime(actual_history["measurement_date"]), actual_history["current_acc_z_max"],
            c=actual_history["velocity"], cmap="turbo", vmin=50, vmax=75,
            s=32, alpha=0.75, label="実測最大加速度",
        )
        colorbar = fig.colorbar(scatter, ax=top, pad=0.01)
        colorbar.set_label("走行速度 [km/h]")
    colors = ("tab:blue", "tab:green", "tab:purple")
    notes = []
    for rank in range(3):
        valid = guide_masks[rank].astype(bool)
        values = np.where(valid, guide_values[rank], np.nan)
        info = guide_info.loc[guide_info["guide_rank"] == rank + 1]
        selected_info = (
            info.iloc[0]
            if not info.empty and info.iloc[0].get("selection_status") == "selected"
            else None
        )
        if selected_info is not None:
            guide_label = (
                f"G{rank + 1}: {selected_info.get('candidate_direction', '?')} "
                f"{float(selected_info.get('candidate_bin_start_m', np.nan)):.0f}-"
                f"{float(selected_info.get('candidate_bin_end_m', np.nan)):.0f}m / "
                f"{selected_info.get('guide_date', '')}起点"
            )
        else:
            guide_label = f"Guide {rank + 1}"
        if valid.any():
            top.plot(
                future_dates, values, "--", color=colors[rank], linewidth=1.5,
                label=guide_label if show_guide_details else "_nolegend_",
            )
        if selected_info is not None and show_guide_details:
            row = selected_info
            notes.append(
                f"G{rank + 1}: {row.get('candidate_direction', '?')} "
                f"{float(row.get('candidate_bin_start_m', np.nan)):.0f}-"
                f"{float(row.get('candidate_bin_end_m', np.nan)):.0f}m "
                f"{row.get('guide_date','')} sim={row.get('cosine_similarity',np.nan):.3f} "
                f"diff={row.get('current_max_difference',np.nan):+.3f} valid={int(row.get('guide_valid_months',0))}/18"
            )
    top.fill_between(future_dates, p10, p90, color="red", alpha=0.15, label="予測 p10–p90")
    top.plot(future_dates, median, "o-", color="red", linewidth=2, markersize=4, label="予測中央値")
    if selected_sample is not None:
        top.plot(
            future_dates, selected_sample, "x-.", color="darkorange",
            linewidth=2.0, markersize=4, zorder=7,
            label=f"生成例 #{int(single_sample_index) + 1}",
        )
    valid_target = np.asarray(target_mask, dtype=bool)
    top.plot(future_dates[valid_target], np.asarray(actual_future)[valid_target], "o-", color="black", markerfacecolor="white", label="未来正解")
    top.axvline(anchor, color="steelblue", linewidth=1.5, label="予測起点")
    top.axvline(
        forecast_start, color="navy", linestyle="--", linewidth=2.5,
        zorder=8, label="予測開始",
    )
    top.text(
        forecast_start, 0.50, "予測開始",
        transform=top.get_xaxis_transform(), rotation=90,
        va="center", ha="right", color="navy", fontsize=9, fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
        zorder=9,
    )
    if "cutoff_maintenance_date" in actual_history:
        maintenance = actual_history.copy()
        maintenance["_cutoff"] = pd.to_datetime(maintenance["cutoff_maintenance_date"], errors="coerce")
        maintenance = maintenance.dropna(subset=["_cutoff"]).drop_duplicates("_cutoff")
        for number, (_, row) in enumerate(maintenance.iterrows()):
            date = row["_cutoff"]
            description = str(row.get("maintenance_description", "")).strip()
            top.axvline(date, color="0.35", linestyle="--", linewidth=1.0,
                        label="施工日" if number == 0 else None)
            if description and description.lower() != "nan":
                top.text(date, y_max * 0.98, description, rotation=90,
                         va="top", ha="right", fontsize=7, color="0.25")
    top.set_ylim(0, y_max)
    top.set_ylabel("絶対値上下加速度区間最大値 [m/s²]")
    top.set_title(
        f"{metadata['direction']}方向 / {float(metadata['bin_start_m']):.0f}-{float(metadata['bin_end_m']):.0f}m / "
        f"起点日 {metadata['anchor_date']} / 18か月予測"
    )
    top.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    top.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    top.grid(True, color="0.85", linewidth=0.7)
    if plot_style == "detailed":
        top.legend(loc="upper left", fontsize=8, ncol=2)
    if notes and show_guide_details:
        top.text(0.99, 0.98, "\n".join(notes), transform=top.transAxes, ha="right", va="top", fontsize=8)
    if show_guide_details and metadata.get("guide_search_mode") == "hybrid_spatiotemporal":
        top.text(
            0.01, 0.02,
            "Guide search: hybrid spatiotemporal\n"
            f"Near range: same/adjacent +/-{float(metadata.get('near_distance_m', 100.0)):.0f}m requires completed past\n"
            "Far range: all periods",
            transform=top.transAxes, ha="left", va="bottom", fontsize=7, color="0.25",
        )

    months = np.arange(1, 19)
    for sample in samples:
        bottom.plot(months, sample, color="red", alpha=0.04, linewidth=0.7)
    bottom.fill_between(months, p10, p90, color="red", alpha=0.15)
    bottom.plot(months, median, color="red", linewidth=2, label="予測中央値")
    if selected_sample is not None:
        bottom.plot(
            months, selected_sample, "x-.", color="darkorange",
            linewidth=2.0, markersize=4, zorder=7,
            label=f"生成例 #{int(single_sample_index) + 1}",
        )
    bottom.plot(months, p10, "--", color="firebrick", linewidth=1, label="p10 / p90")
    bottom.plot(months, p90, "--", color="firebrick", linewidth=1)
    bottom.plot(months[valid_target], np.asarray(actual_future)[valid_target], "o-", color="black", markerfacecolor="white", label="未来正解")
    bottom.set_xlim(1, 18)
    bottom.set_ylim(0, y_max)
    bottom.set_xlabel("予測先 [か月]")
    bottom.set_ylabel("絶対値上下加速度区間最大値 [m/s²]")
    bottom.grid(True, color="0.85", linewidth=0.7)
    if plot_style == "detailed":
        bottom.legend(loc="upper left")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, transparent=True)
    plt.close(fig)
