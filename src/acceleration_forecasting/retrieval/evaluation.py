from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate_predictions(artifact_dir, prediction_file, output_file=None):
    artifact_dir = Path(artifact_dir)
    target_path = artifact_dir / "inference_targets.csv"
    if not target_path.is_file():
        raise FileNotFoundError(f"評価用正解ファイルがありません: {target_path}")
    predictions = pd.read_csv(prediction_file, encoding="utf-8-sig")
    required = {"target_id", "predicted_values"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError("予測CSVに必要な列がありません: " + ", ".join(sorted(missing)))
    targets = pd.read_csv(target_path, encoding="utf-8-sig")
    joined = predictions.merge(
        targets.loc[:, ["target_id", "future_values", "future_mask"]],
        on="target_id",
        how="left",
        validate="many_to_one",
    )
    if joined["future_values"].isna().any():
        raise ValueError("inference_targetsに存在しないtarget_idがあります。")
    records = []
    for _, row in joined.iterrows():
        predicted = np.asarray(json.loads(row["predicted_values"]), dtype=float)
        actual_json = json.loads(row["future_values"])
        actual = np.asarray(
            [np.nan if value is None else value for value in actual_json], dtype=float
        )
        mask = np.asarray(json.loads(row["future_mask"]), dtype=bool)
        if predicted.shape != (18,) or actual.shape != (18,) or mask.shape != (18,):
            raise ValueError("予測値・正解値・maskは18要素である必要があります。")
        valid = mask & np.isfinite(actual) & np.isfinite(predicted)
        if valid.any():
            error = predicted[valid] - actual[valid]
            mae = float(np.mean(np.abs(error)))
            rmse = float(np.sqrt(np.mean(np.square(error))))
        else:
            mae = np.nan
            rmse = np.nan
        records.append(
            {
                "target_id": row["target_id"],
                "valid_months": int(valid.sum()),
                "mae": mae,
                "rmse": rmse,
            }
        )
    result = pd.DataFrame.from_records(records)
    output_path = (
        Path(output_file)
        if output_file
        else artifact_dir / "inference_evaluation.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "evaluated_targets": int(len(result)),
        "targets_with_valid_months": int((result["valid_months"] > 0).sum()),
        "mean_mae": float(result["mae"].mean()),
        "mean_rmse": float(result["rmse"].mean()),
        "output_path": str(output_path),
    }
