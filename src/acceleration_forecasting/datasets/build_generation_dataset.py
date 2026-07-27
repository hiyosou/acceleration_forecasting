from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from acceleration_forecasting.common.progress import progress_bar

from acceleration_forecasting.retrieval.guide_index import GuideIndex
from acceleration_forecasting.retrieval.guide_search import GuideSearchConfig
from acceleration_forecasting.retrieval.model import WaveformAutoencoder, l2_normalize

from .history_builder import sanitize_future, select_monthly_history
from .normalization import fit_normalization


ARRAY_NAMES = (
    "current_values", "history_values", "history_masks", "guide_values",
    "guide_deltas", "guide_masks", "guide_similarities", "retrieval_masks",
)


def _json_list(value):
    return json.loads(value) if isinstance(value, str) else list(value)


def _read_source_config(artifact_dir):
    return json.loads((Path(artifact_dir) / "source_artifacts.json").read_text(encoding="utf-8"))


def _load_embeddings_from_db(connection):
    rows = connection.execute("SELECT record_id, embedding, embedding_dim FROM waveform_records").fetchall()
    return {
        str(record_id): np.frombuffer(blob, dtype=np.float32, count=int(dim)).copy()
        for record_id, blob, dim in rows
    }


def _encode_missing_embeddings(rows, retrieval_dir, source_config, device, batch_size=512, progress=True):
    if rows.empty:
        return {}
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(
        Path(retrieval_dir) / "autoencoder.pt", map_location=device, weights_only=True
    )
    model = WaveformAutoencoder(int(checkpoint.get("embedding_dim", 256))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean, std = float(checkpoint["mean"]), float(checkpoint["std"])
    count = int(source_config["record_count"])
    waveforms = np.memmap(
        source_config["waveforms_path"], dtype=np.float32, mode="r", shape=(count, 500)
    )
    output = {}
    records = list(rows.loc[:, ["record_id", "waveform_index"]].itertuples(index=False))
    with torch.no_grad():
        for start in progress_bar(
            range(0, len(records), batch_size), enabled=progress,
            total=(len(records) + batch_size - 1) // batch_size,
            desc="inference波形をベクトル化", unit="batch",
        ):
            batch = records[start:start + batch_size]
            values = np.stack([waveforms[int(item.waveform_index)] for item in batch]).copy()
            tensor = torch.from_numpy((values - mean) / std).unsqueeze(1).to(device)
            embedded = l2_normalize(model.encode(tensor)).cpu().numpy().astype(np.float32)
            for item, vector in zip(batch, embedded):
                output[str(item.record_id)] = vector
    return output


def _target_frames(retrieval_dir):
    retrieval_dir = Path(retrieval_dir)
    manifest = pd.read_csv(retrieval_dir / "dataset_split_manifest.csv", encoding="utf-8-sig")
    development = pd.read_csv(retrieval_dir / "development_trends.csv", encoding="utf-8-sig")
    inference = pd.read_csv(retrieval_dir / "inference_targets.csv", encoding="utf-8-sig")
    dataset_splits = manifest.drop_duplicates("dataset_id").set_index("dataset_id")["model_split"]
    development["model_split"] = development["dataset_id"].map(dataset_splits)
    development["target_id"] = development["trend_id"].astype(str)
    inference["model_split"] = "inference"
    if "trend_id" not in inference:
        inference["trend_id"] = inference["target_id"]
    columns = sorted(set(development.columns) | set(inference.columns))
    return manifest, development.reindex(columns=columns), inference.reindex(columns=columns)


def _prepare_trends(frame):
    frame = frame.copy()
    frame["measurement_date"] = pd.to_datetime(frame["measurement_date"], errors="raise").dt.normalize()
    frame["current_acc_z_max"] = pd.to_numeric(frame["current_acc_z_max"], errors="coerce")
    return frame


def _query_rows(manifest, split, target_id):
    if split == "inference":
        raise RuntimeError("Inference query rows are provided separately")
    return manifest.loc[
        (manifest["model_split"] == split)
        & (manifest["trend_id"].astype(str) == str(target_id))
    ]


def _save_split(split_dir, metadata, arrays, targets=None):
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata).to_csv(split_dir / "metadata.csv", index=False, encoding="utf-8-sig")
    for name, values in arrays.items():
        np.save(split_dir / f"{name}.npy", np.asarray(values, dtype=np.float32))
    if targets is not None:
        np.save(split_dir / "target_values.npy", np.asarray(targets[0], dtype=np.float32))
        np.save(split_dir / "target_masks.npy", np.asarray(targets[1], dtype=np.float32))


def prepare_generation_dataset(
    retrieval_artifact_dir,
    output_dir,
    *,
    device=None,
    top_k=3,
    max_current_difference=0.5,
    min_valid_guide_months=12,
    min_valid_target_months=12,
    max_train=None,
    max_validation=None,
    max_inference=None,
    progress=True,
):
    retrieval_dir = Path(retrieval_artifact_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, development, inference = _target_frames(retrieval_dir)
    development = _prepare_trends(development)
    inference = _prepare_trends(inference)
    all_trends = pd.concat([development, inference], ignore_index=True)
    groups = {key: group.copy() for key, group in all_trends.groupby("dataset_id", sort=False)}

    train_observations = development.loc[development["model_split"] == "model_train"]
    normalization = fit_normalization(train_observations)
    normalization.save(output_dir / "normalization.json")

    connection = sqlite3.connect(f"file:{(retrieval_dir / 'vector_database.sqlite').as_posix()}?mode=ro", uri=True)
    try:
        index = GuideIndex(connection)
        embeddings = _load_embeddings_from_db(connection)
    finally:
        connection.close()
    source_config = _read_source_config(retrieval_dir)
    inference_inputs = pd.read_csv(retrieval_dir / "inference_inputs.csv", encoding="utf-8-sig")
    inference_to_encode = inference_inputs
    if max_inference is not None:
        selected_targets = set(
            inference.sort_values(["dataset_id", "measurement_date", "target_id"], kind="mergesort")
            .head(int(max_inference))["target_id"].astype(str)
        )
        inference_to_encode = inference_inputs.loc[
            inference_inputs["target_id"].astype(str).isin(selected_targets)
        ]
    embeddings.update(_encode_missing_embeddings(
        inference_to_encode, retrieval_dir, source_config, device, progress=progress
    ))

    train_ids = set(development.loc[development["model_split"] == "model_train", "dataset_id"].astype(str))
    development_ids = set(development["dataset_id"].astype(str))
    search_config = GuideSearchConfig(
        top_k=int(top_k), max_current_difference=float(max_current_difference),
        min_valid_months=int(min_valid_guide_months), strict_time=True,
    )
    inference_rows_by_target = {
        key: group for key, group in inference_inputs.groupby("target_id", sort=False)
    }
    assignments = []
    summary = {}
    split_specs = (
        ("model_train", development.loc[development["model_split"] == "model_train"], max_train),
        ("model_validation", development.loc[development["model_split"] == "model_validation"], max_validation),
        ("inference", inference, max_inference),
    )
    for split, target_frame, limit in split_specs:
        target_frame = target_frame.sort_values(["dataset_id", "measurement_date", "target_id"], kind="mergesort")
        if limit is not None:
            target_frame = target_frame.head(int(limit))
        metadata, arrays = [], {name: [] for name in ARRAY_NAMES}
        target_values, target_masks = [], []
        target_progress = progress_bar(
            target_frame.iterrows(), enabled=progress, total=len(target_frame),
            desc=f"{split}生成データ", unit="target",
        )
        for _, row in target_progress:
            current = float(row["current_acc_z_max"])
            if not np.isfinite(current):
                continue
            raw_future, raw_mask = sanitize_future(
                row["measurement_date"], _json_list(row["future_values"]),
                _json_list(row["future_mask"]), row.get("cutoff_maintenance_date", ""),
            )
            valid_target_months = int(raw_mask.sum())
            if split != "inference" and valid_target_months < int(min_valid_target_months):
                continue
            history, history_mask, history_dates = select_monthly_history(
                groups[str(row["dataset_id"])], row["measurement_date"]
            )
            if split == "inference":
                query_rows = inference_rows_by_target.get(str(row["target_id"]), pd.DataFrame())
            else:
                query_rows = _query_rows(manifest, split, row["target_id"])
            query_vectors = [embeddings[str(record_id)] for record_id in query_rows.get("record_id", []) if str(record_id) in embeddings]
            if not query_vectors:
                continue
            allowed = development_ids if split == "inference" else train_ids
            guides = index.search(
                np.stack(query_vectors), query_date=pd.Timestamp(row["measurement_date"]).strftime("%Y-%m-%d"),
                query_dataset_id=row["dataset_id"], query_current=current,
                allowed_dataset_ids=allowed, config=search_config,
            )
            guide_values = np.full((top_k, 18), np.nan, dtype=np.float32)
            guide_deltas = np.full((top_k, 18), np.nan, dtype=np.float32)
            guide_masks = np.zeros((top_k, 18), dtype=np.float32)
            similarities = np.zeros(top_k, dtype=np.float32)
            retrieval_masks = np.zeros(top_k, dtype=np.float32)
            for rank, guide in enumerate(guides):
                values, mask = sanitize_future(
                    guide["measurement_date"], guide["future_values"], guide["future_mask"],
                    guide["cutoff_maintenance_date"],
                )
                guide_values[rank] = values
                guide_deltas[rank] = values - float(guide["current_acc_z_max"])
                guide_masks[rank] = mask
                similarities[rank] = float(guide["similarity"])
                retrieval_masks[rank] = 1
                assignments.append({
                    "split": split, "target_id": row["target_id"], "guide_rank": rank + 1,
                    "query_waveform_count": len(query_vectors), "guide_record_id": guide["record_id"],
                    "guide_trend_id": guide["trend_id"], "guide_date": guide["measurement_date"],
                    "cosine_similarity": guide["similarity"], "query_current_acc_z_max": current,
                    "guide_current_acc_z_max": guide["current_acc_z_max"],
                    "current_max_difference": guide["current_max_difference"],
                    "guide_valid_months": int(mask.sum()), "guide_available_date": guide["guide_available_date"],
                    "selection_status": "selected",
                })
            for rank in range(len(guides), top_k):
                assignments.append({"split": split, "target_id": row["target_id"], "guide_rank": rank + 1, "query_waveform_count": len(query_vectors), "selection_status": "not_found"})
            metadata.append({
                "target_id": row["target_id"], "dataset_id": row["dataset_id"],
                "anchor_date": pd.Timestamp(row["measurement_date"]).strftime("%Y-%m-%d"),
                "current_acc_z_max": current,
                "direction": row["direction"], "bin_start_m": float(row["bin_start_m"]),
                "bin_end_m": float(row["bin_end_m"]), "history_dates": json.dumps(history_dates),
                "valid_target_months": valid_target_months, "guide_count": len(guides),
                "cutoff_maintenance_date": row.get("cutoff_maintenance_date", "") or "",
                "maintenance_description": row.get("maintenance_description", "") or "",
            })
            for name, value in {
                "current_values": [current], "history_values": history, "history_masks": history_mask,
                "guide_values": guide_values, "guide_deltas": guide_deltas,
                "guide_masks": guide_masks, "guide_similarities": similarities,
                "retrieval_masks": retrieval_masks,
            }.items():
                arrays[name].append(value)
            target_values.append(raw_future)
            target_masks.append(raw_mask)
        target_progress.set_postfix(adopted=len(metadata), refresh=progress)
        split_dir = output_dir / split
        if split == "inference":
            _save_split(split_dir / "inputs", metadata, arrays)
            target_dir = split_dir / "targets"
            target_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"target_id": [item["target_id"] for item in metadata]}).to_csv(target_dir / "target_ids.csv", index=False, encoding="utf-8-sig")
            np.save(target_dir / "target_values.npy", np.asarray(target_values, dtype=np.float32))
            np.save(target_dir / "target_masks.npy", np.asarray(target_masks, dtype=np.float32))
        else:
            _save_split(split_dir, metadata, arrays, (target_values, target_masks))
        summary[split] = len(metadata)
    pd.DataFrame(assignments).to_csv(output_dir / "guide_assignments.csv", index=False, encoding="utf-8-sig")
    source = {
        "retrieval_artifact_dir": str(retrieval_dir),
        "database_path": str(retrieval_dir / "vector_database.sqlite"),
        "waveforms_path": source_config["waveforms_path"],
        "waveforms_size_bytes": source_config["waveforms_size_bytes"],
        "waveforms_sha256": source_config["waveforms_sha256"],
        "read_only": True,
    }
    (output_dir / "source_retrieval_artifacts.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
