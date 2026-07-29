from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from acceleration_forecasting.common.progress import progress_bar
from acceleration_forecasting.datasets.generation_dataset import GenerationDataset
from acceleration_forecasting.evaluation.metrics import evaluate_target

from .sampling import load_ema_checkpoint, sample_one
from .sampling_bounds import fit_sampling_bounds, fit_residual_sampling_bounds


def _evaluate_checkpoint(
    checkpoint, dataset, normalization, device, num_samples, max_records, seed,
    clean_clip, progress=True,
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
        normalized = sample_one(
            model, batch, target_id, num_samples=num_samples, seed=seed,
            clean_clip=clean_clip,
        )
        samples = dataset.denormalize_prediction(normalized, index)
        median = np.median(samples, axis=0)
        actual = dataset.physical_target(index)
        mask = item["target_mask"].numpy().astype(bool)
        p10, p90 = np.percentile(samples, [10, 90], axis=0)
        metrics = evaluate_target(actual, mask, median, p10, p90)
        rows.append({
            "model_name": name, "target_id": target_id,
            "samples_finite": bool(np.isfinite(samples).all()),
            "sample_min": float(np.min(samples)), "sample_max": float(np.max(samples)),
            "quantiles_ordered": bool(np.all(p10 <= median) and np.all(median <= p90)),
            **metrics,
        })
    frame = pd.DataFrame(rows)
    return frame, {
        "model_name": name, "MAE": float(frame["MAE"].mean()),
        "MSE": float(frame["MSE"].mean()),
        "RMSE": float(frame["RMSE"].mean()),
        "coverage_p10_p90": float(frame["coverage_p10_p90"].mean()),
        "mean_interval_width": float(frame["mean_interval_width"].mean()),
        "peak_value_error": float(frame["peak_value_error"].mean()),
        "peak_month_error": float(frame["peak_month_error"].mean()),
        "samples_finite": bool(frame["samples_finite"].all()),
        "quantiles_ordered": bool(frame["quantiles_ordered"].all()),
        "sample_min": float(frame["sample_min"].min()),
        "sample_max": float(frame["sample_max"].max()),
        "target_count": int(len(frame)),
    }


def select_model(
    dataset_dir, model_dir, output_dir, *, device=None, num_samples=100,
    max_records=None, equivalence_threshold=0.01, seed=42, progress=True,
    candidates=("mlp", "unet"), mae_limit=None, coverage_min=None,
    interval_width_limit=None,
):
    dataset_dir, model_dir, output_dir = Path(dataset_dir), Path(model_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    search_config_path = dataset_dir / "guide_search_config.json"
    search_config = json.loads(search_config_path.read_text(encoding="utf-8"))
    dataset_build_id = search_config["dataset_build_id"]
    candidates = tuple(candidates)
    for name in candidates:
        checkpoint_path = model_dir / name / "best_ema_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("dataset_build_id") != dataset_build_id:
            raise ValueError(
                f"{name} checkpoint was trained with a different guide search configuration"
            )
    dataset = GenerationDataset(dataset_dir / "model_validation", dataset_dir / "normalization.json")
    residual_mode = (dataset_dir / "residual_config.json").is_file()
    sampling_bounds = (
        fit_residual_sampling_bounds(dataset_dir) if residual_mode
        else fit_sampling_bounds(dataset_dir)
    )
    sampling_bounds.save(output_dir / "sampling_bounds.json")
    results = []
    model_progress = progress_bar(
        candidates, enabled=progress, total=len(candidates),
        desc="モデル選択", unit="model",
    )
    for name in model_progress:
        model_progress.set_postfix(model=name, refresh=False)
        frame, summary = _evaluate_checkpoint(
            model_dir / name / "best_ema_model.pt", dataset, dataset.normalization,
            device, num_samples, max_records, seed, sampling_bounds.normalized,
            progress=progress,
        )
        frame.to_csv(output_dir / f"{name}_metrics.csv", index=False, encoding="utf-8-sig")
        results.append(summary)
    comparison = pd.DataFrame(results)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    if len(results) == 1:
        selected = results[0]["model_name"]
        reason = f"only {selected} was requested"
        difference = 0.0
        mlp_mae = None
        unet_mae = results[0]["MAE"] if selected == "unet" else None
    else:
        mlp, unet = results
        mlp_mae, unet_mae = mlp["MAE"], unet["MAE"]
        difference = abs(mlp_mae - unet_mae)
        if difference < equivalence_threshold:
            selected, reason = "mlp", "MAE difference below threshold; simpler MLP preferred"
        else:
            selected = min(results, key=lambda item: item["MAE"])["model_name"]
            reason = f"{selected} had lower validation MAE by at least {equivalence_threshold}"
    selected_summary = next(item for item in results if item["model_name"] == selected)
    tolerance = 1e-6
    quality_checks = {
        "all_samples_finite": bool(selected_summary["samples_finite"]),
        "quantiles_ordered": bool(selected_summary["quantiles_ordered"]),
        "samples_within_physical_bounds": bool(
            selected_summary["sample_min"] >= 0.3 - tolerance
            and selected_summary["sample_max"] <= 5.0 + tolerance
        ),
        "mae_within_limit": bool(selected_summary["MAE"] <= (mae_limit if mae_limit is not None else dataset.condition_normalization.std)),
        "coverage_within_limit": bool(coverage_min is None or selected_summary["coverage_p10_p90"] >= coverage_min),
        "interval_width_within_limit": bool(selected_summary["mean_interval_width"] < (interval_width_limit if interval_width_limit is not None else 4.7)),
    }
    quality_gate = {
        "passed": bool(all(quality_checks.values())),
        "checks": quality_checks,
        "mae_limit": float(mae_limit if mae_limit is not None else dataset.condition_normalization.std),
        "coverage_min": coverage_min,
        "interval_width_limit": float(interval_width_limit if interval_width_limit is not None else 4.7),
    }
    payload = {
        "selected_model": selected,
        "selected_checkpoint": str((model_dir / selected / "best_ema_model.pt").resolve()),
        "selection_split": "model_validation", "primary_metric": "MAE",
        "mlp_mae": mlp_mae, "unet_mae": unet_mae,
        "mae_difference": difference, "equivalence_threshold": equivalence_threshold,
        "selection_reason": reason, "selected_metrics": selected_summary, "seed": seed,
        "sampling_bounds": sampling_bounds.to_dict(),
        "clean_prediction_clipping": True,
        "quality_gate": quality_gate,
        "dataset_build_id": dataset_build_id,
        "guide_search_settings": search_config,
        "residual_mode": residual_mode,
        "final_physical_bounds": [0.3, 5.0],
    }
    (output_dir / "selected_model.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "selected_model.txt").write_text(
        f"Selected model: {selected}\nCheckpoint: {payload['selected_checkpoint']}\n"
        f"Validation MAE: {selected_summary['MAE']:.6f} m/s^2\nReason: {reason}\n",
        encoding="utf-8",
    )
    return payload
