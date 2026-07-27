from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from acceleration_forecasting.common.progress import progress_bar
from acceleration_forecasting.datasets.generation_dataset import GenerationDataset
from acceleration_forecasting.evaluation.metrics import evaluate_target

from .sampling import load_ema_checkpoint, sample_one


def _evaluate_checkpoint(
    checkpoint, dataset, normalization, device, num_samples, max_records, seed,
    progress=True,
):
    model, name, _ = load_ema_checkpoint(checkpoint, device)
    rows = []
    count = min(len(dataset), max_records or len(dataset))
    for index in progress_bar(
        range(count), enabled=progress, total=count,
        desc=f"{name} validation", unit="target", leave=False,
    ):
        item = dataset[index]
        target_id = str(dataset.metadata.iloc[index]["target_id"])
        batch = {key: value.unsqueeze(0) if hasattr(value, "ndim") and value.ndim > 0 else value for key, value in item.items() if key not in ("target", "target_mask", "index")}
        normalized = sample_one(model, batch, target_id, num_samples=num_samples, seed=seed)
        samples = normalization.denormalize(normalized, clip_nonnegative=True)
        median = np.median(samples, axis=0)
        actual = normalization.denormalize(item["target"].numpy(), clip_nonnegative=True)
        mask = item["target_mask"].numpy().astype(bool)
        p10, p90 = np.percentile(samples, [10, 90], axis=0)
        metrics = evaluate_target(actual, mask, median, p10, p90)
        rows.append({"model_name": name, "target_id": target_id, **metrics})
    frame = pd.DataFrame(rows)
    return frame, {
        "model_name": name, "MAE": float(frame["MAE"].mean()),
        "RMSE": float(frame["RMSE"].mean()),
        "coverage_p10_p90": float(frame["coverage_p10_p90"].mean()),
        "mean_interval_width": float(frame["mean_interval_width"].mean()),
        "peak_value_error": float(frame["peak_value_error"].mean()),
        "peak_month_error": float(frame["peak_month_error"].mean()),
        "target_count": int(len(frame)),
    }


def select_model(
    dataset_dir, model_dir, output_dir, *, device=None, num_samples=100,
    max_records=None, equivalence_threshold=0.01, seed=42, progress=True,
):
    dataset_dir, model_dir, output_dir = Path(dataset_dir), Path(model_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = GenerationDataset(dataset_dir / "model_validation", dataset_dir / "normalization.json")
    results = []
    model_progress = progress_bar(
        ("mlp", "unet"), enabled=progress, total=2,
        desc="モデル選択", unit="model",
    )
    for name in model_progress:
        model_progress.set_postfix(model=name, refresh=False)
        frame, summary = _evaluate_checkpoint(
            model_dir / name / "best_ema_model.pt", dataset, dataset.normalization,
            device, num_samples, max_records, seed, progress=progress,
        )
        frame.to_csv(output_dir / f"{name}_metrics.csv", index=False, encoding="utf-8-sig")
        results.append(summary)
    comparison = pd.DataFrame(results)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    mlp, unet = results
    difference = abs(mlp["MAE"] - unet["MAE"])
    if difference < equivalence_threshold:
        selected, reason = "mlp", "MAE difference below threshold; simpler MLP preferred"
    else:
        selected = min(results, key=lambda item: item["MAE"])["model_name"]
        reason = f"{selected} had lower validation MAE by at least {equivalence_threshold}"
    selected_summary = next(item for item in results if item["model_name"] == selected)
    payload = {
        "selected_model": selected,
        "selected_checkpoint": str((model_dir / selected / "best_ema_model.pt").resolve()),
        "selection_split": "model_validation", "primary_metric": "MAE",
        "mlp_mae": mlp["MAE"], "unet_mae": unet["MAE"],
        "mae_difference": difference, "equivalence_threshold": equivalence_threshold,
        "selection_reason": reason, "selected_metrics": selected_summary, "seed": seed,
    }
    (output_dir / "selected_model.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "selected_model.txt").write_text(
        f"Selected model: {selected}\nCheckpoint: {payload['selected_checkpoint']}\n"
        f"Validation MAE: {selected_summary['MAE']:.6f} m/s^2\nReason: {reason}\n",
        encoding="utf-8",
    )
    return payload
