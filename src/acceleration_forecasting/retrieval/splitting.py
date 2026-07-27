from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DATABASE_RATIO, RANDOM_SEED, SAMPLES_PER_BIN


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _split_groups(groups, first_ratio, seed):
    values = sorted({str(value) for value in groups})
    if len(values) < 2:
        raise ValueError("分割には2件以上のdataset_idが必要です。")
    random.Random(int(seed)).shuffle(values)
    first_count = min(
        max(int(np.floor(len(values) * float(first_ratio))), 1),
        len(values) - 1,
    )
    return set(values[:first_count]), set(values[first_count:])


def _atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _atomic_json(value, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def create_dataset_split(
    source_artifact_dir,
    artifact_dir,
    development_ratio=DATABASE_RATIO,
    train_ratio=0.9,
    seed=RANDOM_SEED,
):
    source_artifact_dir = Path(source_artifact_dir).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    if source_artifact_dir == artifact_dir:
        raise ValueError("sourceと出力artifact_dirは別フォルダーにしてください。")
    source_manifest_path = source_artifact_dir / "split_manifest.csv"
    source_trend_path = source_artifact_dir / "trend_catalog.csv"
    waveform_path = source_artifact_dir / "waveforms.bin"
    for path in (source_manifest_path, source_trend_path, waveform_path):
        if not path.is_file():
            raise FileNotFoundError(f"既存成果物がありません: {path}")

    manifest = pd.read_csv(source_manifest_path, encoding="utf-8-sig")
    trends = pd.read_csv(source_trend_path, encoding="utf-8-sig")
    if manifest["record_id"].duplicated().any():
        raise ValueError("既存manifestのrecord_idが重複しています。")
    trend_to_dataset = trends.loc[:, ["trend_id", "dataset_id"]].drop_duplicates()
    if trend_to_dataset["trend_id"].duplicated().any():
        raise ValueError("1つのtrend_idに複数のdataset_idがあります。")
    merged = manifest.merge(
        trend_to_dataset, on="trend_id", how="left", validate="many_to_one"
    )
    if merged["dataset_id"].isna().any():
        raise ValueError("dataset_idへ結合できない波形があります。")

    development_ids, inference_ids = _split_groups(
        merged["dataset_id"], development_ratio, seed
    )
    model_train_ids, validation_ids = _split_groups(
        development_ids, train_ratio, int(seed) + 1
    )
    merged = merged.rename(columns={"split": "legacy_date_split"})
    merged["outer_split"] = np.where(
        merged["dataset_id"].isin(development_ids),
        "development",
        "inference",
    )
    merged["model_split"] = np.select(
        [
            merged["dataset_id"].isin(model_train_ids),
            merged["dataset_id"].isin(validation_ids),
        ],
        ["model_train", "model_validation"],
        default="inference",
    )

    expected_size = len(merged) * SAMPLES_PER_BIN * np.dtype(np.float32).itemsize
    actual_size = waveform_path.stat().st_size
    if expected_size != actual_size:
        raise ValueError(
            f"waveforms.binのサイズが不正です: expected={expected_size}, "
            f"actual={actual_size}"
        )

    development_trend_ids = set(
        merged.loc[merged["outer_split"] == "development", "trend_id"]
    )
    inference_trend_ids = set(
        merged.loc[merged["outer_split"] == "inference", "trend_id"]
    )
    if development_trend_ids & inference_trend_ids:
        raise ValueError("trend_idがdevelopmentとinferenceに混在しています。")
    development_trends = trends.loc[
        trends["trend_id"].isin(development_trend_ids)
    ].copy()
    inference_trends = trends.loc[
        trends["trend_id"].isin(inference_trend_ids)
    ].copy()

    input_columns = [
        "record_id",
        "waveform_index",
        "measurement_id",
        "measurement_date",
        "direction",
        "bin_start_m",
        "bin_end_m",
        "mean_velocity_kmh",
        "source_csv_path",
        "waveform_sha256",
        "dataset_id",
    ]
    inference_inputs = merged.loc[
        merged["outer_split"] == "inference", input_columns
    ].copy()
    inference_inputs["target_id"] = merged.loc[
        merged["outer_split"] == "inference", "trend_id"
    ].to_numpy()
    inference_targets = inference_trends.rename(
        columns={"trend_id": "target_id"}
    ).copy()

    artifact_dir.mkdir(parents=True, exist_ok=True)
    source_config = {
        "format": "dataset_id_split_v1",
        "source_artifact_dir": str(source_artifact_dir),
        "waveforms_path": str(waveform_path.resolve()),
        "waveforms_size_bytes": int(actual_size),
        "waveforms_sha256": _sha256_file(waveform_path),
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_trend_catalog_path": str(source_trend_path.resolve()),
        "record_count": int(len(merged)),
        "samples_per_waveform": SAMPLES_PER_BIN,
    }
    summary = {
        "seed": int(seed),
        "development_ratio": float(development_ratio),
        "model_train_ratio_within_development": float(train_ratio),
        "dataset_count": int(merged["dataset_id"].nunique()),
        "development_dataset_count": int(len(development_ids)),
        "inference_dataset_count": int(len(inference_ids)),
        "model_train_dataset_count": int(len(model_train_ids)),
        "model_validation_dataset_count": int(len(validation_ids)),
        "record_count": int(len(merged)),
        "development_record_count": int(
            (merged["outer_split"] == "development").sum()
        ),
        "inference_record_count": int(
            (merged["outer_split"] == "inference").sum()
        ),
        "model_train_record_count": int(
            (merged["model_split"] == "model_train").sum()
        ),
        "model_validation_record_count": int(
            (merged["model_split"] == "model_validation").sum()
        ),
        "development_trend_count": int(len(development_trends)),
        "inference_target_count": int(len(inference_targets)),
    }
    _atomic_csv(merged, artifact_dir / "dataset_split_manifest.csv")
    _atomic_csv(development_trends, artifact_dir / "development_trends.csv")
    _atomic_csv(inference_inputs, artifact_dir / "inference_inputs.csv")
    _atomic_csv(inference_targets, artifact_dir / "inference_targets.csv")
    _atomic_json(source_config, artifact_dir / "source_artifacts.json")
    _atomic_json(summary, artifact_dir / "split_summary.json")
    return summary
