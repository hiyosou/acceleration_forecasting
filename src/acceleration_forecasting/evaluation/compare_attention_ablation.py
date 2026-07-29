from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from acceleration_forecasting.generation.diffusion import create_model


def _load_prediction(path):
    return pd.read_csv(Path(path) / "predictions.csv", encoding="utf-8-sig")


def _load_summary(path):
    return pd.read_csv(Path(path) / "evaluation_summary.csv", encoding="utf-8-sig").iloc[0]


def _sample_adjacent_difference(path):
    values = []
    for file in sorted((Path(path) / "samples").glob("samples_*.npz")):
        samples = np.load(file)["samples"]
        values.append(np.abs(np.diff(samples, axis=-1)).reshape(-1))
    return float(np.concatenate(values).mean()) if values else np.nan


def _model_stats(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_model(checkpoint["model_name"], **checkpoint.get("model_kwargs", {}))
    return sum(parameter.numel() for parameter in model.parameters()), checkpoint_path.stat().st_size


def _draw_panel(axis, frame, actual, mask, baseline, title, color):
    frame = frame.sort_values("month_index")
    months = np.arange(1, 19)
    p10 = frame["prediction_p10"].to_numpy(float)
    p90 = frame["prediction_p90"].to_numpy(float)
    axis.fill_between(months, p10, p90, color=color, alpha=0.2)
    axis.plot(months, frame["prediction_median"], "o-", color=color, label="median")
    axis.plot(months, baseline, "--", color="darkorange", linewidth=1.7, label="guide baseline")
    axis.plot(months[mask], actual[mask], "ko-", markerfacecolor="white", label="actual")
    axis.set_ylim(0, 5)
    axis.set_ylabel("acc_z max [m/s²]")
    axis.set_title(title)
    axis.grid(True, color="0.85")
    axis.legend(loc="upper right")


def compare_attention_ablation(
    dataset_dir, one_anchor_prediction_dir, attention_prediction_dir,
    no_attention_prediction_dir, one_anchor_evaluation_dir, attention_evaluation_dir,
    no_attention_evaluation_dir, attention_selection_file, no_attention_selection_file,
    output_dir, *, max_images=100, dpi=150,
):
    dataset_dir, output = Path(dataset_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = {
        "one_anchor": _load_prediction(one_anchor_prediction_dir),
        "residual_attention": _load_prediction(attention_prediction_dir),
        "residual_no_attention": _load_prediction(no_attention_prediction_dir),
    }
    summaries = {
        "one_anchor": _load_summary(one_anchor_evaluation_dir),
        "residual_attention": _load_summary(attention_evaluation_dir),
        "residual_no_attention": _load_summary(no_attention_evaluation_dir),
    }
    selections = {
        "residual_attention": json.loads(Path(attention_selection_file).read_text(encoding="utf-8")),
        "residual_no_attention": json.loads(Path(no_attention_selection_file).read_text(encoding="utf-8")),
    }
    rows = []
    for name, summary in summaries.items():
        row = {"model": name}
        for metric in ("MAE", "MSE", "RMSE", "correlation", "peak_value_error", "peak_month_error", "coverage_p10_p90", "mean_interval_width"):
            row[metric] = float(summary.get(metric, np.nan))
        row["median_adjacent_abs_difference"] = float(
            predictions[name].sort_values(["target_id", "month_index"])
            .groupby("target_id")["prediction_median"].apply(lambda x: np.abs(np.diff(x)).mean()).mean()
        )
        row["sample_adjacent_abs_difference"] = _sample_adjacent_difference(
            {"one_anchor": one_anchor_prediction_dir, "residual_attention": attention_prediction_dir,
             "residual_no_attention": no_attention_prediction_dir}[name]
        )
        if name in selections:
            row["parameter_count"], row["checkpoint_size_bytes"] = _model_stats(selections[name]["selected_checkpoint"])
            run = json.loads((Path({"residual_attention": attention_prediction_dir, "residual_no_attention": no_attention_prediction_dir}[name]) / "prediction_run.json").read_text(encoding="utf-8"))
            row["inference_seconds"] = float(run["elapsed_seconds"])
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "attention_ablation_summary.csv", index=False, encoding="utf-8-sig")
    current = comparison.set_index("model").loc["residual_attention"]
    candidate = comparison.set_index("model").loc["residual_no_attention"]
    equivalent = bool(
        abs(candidate.MAE - current.MAE) < 0.01
        and abs(candidate.mean_interval_width - current.mean_interval_width) < 0.05
        and abs(candidate.coverage_p10_p90 - current.coverage_p10_p90) < 0.02
    )
    accepted = bool(
        candidate.MAE <= current.MAE
        and candidate.mean_interval_width < 2.0
        and candidate.coverage_p10_p90 >= 0.75
    )
    decision = {
        "recommended_model": "residual_no_attention" if accepted or equivalent else "residual_attention",
        "accepted_as_improvement": accepted, "statistically_equivalent_by_rule": equivalent,
        "mae_change": float(candidate.MAE - current.MAE),
        "interval_width_change": float(candidate.mean_interval_width - current.mean_interval_width),
        "coverage_change": float(candidate.coverage_p10_p90 - current.coverage_p10_p90),
        "inference_speedup_percent": float((current.inference_seconds - candidate.inference_seconds) / current.inference_seconds * 100),
    }
    (output / "attention_ablation_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = pd.read_csv(dataset_dir / "inference" / "inputs" / "metadata.csv", encoding="utf-8-sig")
    target_ids = pd.read_csv(dataset_dir / "inference" / "targets" / "target_ids.csv", encoding="utf-8-sig")
    actual = np.load(dataset_dir / "inference" / "targets" / "target_values.npy", mmap_mode="r")
    masks = np.load(dataset_dir / "inference" / "targets" / "target_masks.npy", mmap_mode="r")
    baselines = np.load(dataset_dir / "inference" / "inputs" / "guide_baselines.npy", mmap_mode="r")
    lookup = {str(value): index for index, value in enumerate(target_ids["target_id"].astype(str))}
    ranking = pd.read_csv(Path(no_attention_evaluation_dir) / "evaluation_per_target.csv", encoding="utf-8-sig")
    selected_ids = ranking.sort_values("RMSE", ascending=False).head(int(max_images))["target_id"].astype(str)
    image_count = 0
    for target_id in selected_ids:
        if target_id not in lookup:
            continue
        index = lookup[target_id]
        valid = np.asarray(masks[index], dtype=bool)
        meta = metadata.loc[metadata["target_id"].astype(str) == target_id].iloc[0]
        title = f"{meta['direction']} {float(meta['bin_start_m']):.0f}-{float(meta['bin_end_m']):.0f}m / {meta['anchor_date']}"
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
        _draw_panel(axes[0], predictions["residual_attention"].loc[predictions["residual_attention"]["target_id"].astype(str) == target_id], actual[index], valid, baselines[index], "Residual + Cross-Attention", "tab:red")
        _draw_panel(axes[1], predictions["residual_no_attention"].loc[predictions["residual_no_attention"]["target_id"].astype(str) == target_id], actual[index], valid, baselines[index], "Residual without Cross-Attention", "tab:green")
        axes[-1].set_xlabel("Forecast month")
        fig.suptitle(title)
        path = output / "comparison_with_attention" / str(meta["direction"]) / f"{target_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, transparent=True)
        plt.close(fig)
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
        for axis, name, panel_title, color in (
            (axes[0], "one_anchor", "One-anchor absolute diffusion", "tab:blue"),
            (axes[1], "residual_attention", "Residual + Cross-Attention", "tab:red"),
            (axes[2], "residual_no_attention", "Residual without Cross-Attention", "tab:green"),
        ):
            _draw_panel(axis, predictions[name].loc[predictions[name]["target_id"].astype(str) == target_id], actual[index], valid, baselines[index], panel_title, color)
        axes[-1].set_xlabel("Forecast month")
        fig.suptitle(title)
        path = output / "comparison_all_models" / str(meta["direction"]) / f"{target_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, transparent=True)
        plt.close(fig)
        image_count += 1
    return {**decision, "common_targets": int(len(ranking)), "images_per_comparison": image_count}
